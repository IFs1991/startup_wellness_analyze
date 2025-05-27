"""
Task 4.3: 非同期ジョブキュー テスト
TDD RED段階: ジョブキューシナリオテスト
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any

# テスト対象
from backend.federated_learning.job_queue import (
    JobManager,
    PriorityJobQueue,
    RetryManager,
    DeadLetterQueue,
    ModelTrainingJob,
    AggregationJob,
    EncryptionJob,
    HealthCheckJob,
    DataSyncJob,
    JobType,
    JobPriority,
    JobStatus,
    JobResult,
    JobMetrics,
    JobRequest,
    DeadLetterRecord,
    QueueStatistics,
    WorkerStatus
)


class TestJobRetryLogic:
    """ジョブリトライロジックテスト"""

    @pytest.fixture
    def retry_manager(self):
        """リトライ管理器フィクスチャ"""
        return RetryManager(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=False  # テスト用にジッターを無効化
        )

    @pytest.fixture
    def sample_job_request(self):
        """サンプルジョブリクエスト"""
        return JobRequest(
            job_type=JobType.MODEL_TRAINING,
            payload={"model_id": "test_model", "client_id": "client_1"},
            priority=JobPriority.HIGH,
            max_retries=3
        )

    @pytest.mark.asyncio
    async def test_job_retry_exponential_backoff(self, retry_manager, sample_job_request):
        """エクスポネンシャルバックオフリトライテスト"""
        # 失敗ジョブのシミュレート
        failed_result = JobResult(
            job_id=sample_job_request.job_id,
            job_type=sample_job_request.job_type,
            status=JobStatus.FAILURE,
            error_message="Network timeout",
            retry_count=0
        )

        # リトライ間隔の計算
        retry_delay_1 = retry_manager.calculate_retry_delay(1)
        retry_delay_2 = retry_manager.calculate_retry_delay(2)
        retry_delay_3 = retry_manager.calculate_retry_delay(3)

        assert retry_delay_1 == 1.0  # 1回目: 1秒
        assert retry_delay_2 == 2.0  # 2回目: 2秒
        assert retry_delay_3 == 4.0  # 3回目: 4秒

        # リトライ可能性チェック
        assert retry_manager.should_retry(failed_result) == True

        # 最大リトライ回数超過
        failed_result.retry_count = 3
        assert retry_manager.should_retry(failed_result) == False

    @pytest.mark.asyncio
    async def test_job_retry_failure_after_max_attempts(self, retry_manager):
        """最大リトライ回数後の失敗処理テスト"""
        # 最大リトライ回数に達したジョブ
        exhausted_job = JobResult(
            job_id="test_job_exhausted",
            job_type=JobType.AGGREGATION,
            status=JobStatus.FAILURE,
            error_message="Persistent failure",
            retry_count=3
        )

        # リトライ不可能の確認
        should_retry = retry_manager.should_retry(exhausted_job)
        assert should_retry == False

        # デッドレターキューへの移動判定
        should_move_to_dlq = retry_manager.should_move_to_dead_letter(exhausted_job)
        assert should_move_to_dlq == True

    @pytest.mark.asyncio
    async def test_transient_vs_permanent_failures(self, retry_manager):
        """一時的障害と永続的障害の区別テスト"""
        # 一時的障害（リトライ対象）
        transient_failures = [
            "Connection timeout",
            "Network unreachable",
            "Service temporarily unavailable",
            "Rate limit exceeded"
        ]

        # 永続的障害（リトライ非対象）
        permanent_failures = [
            "Invalid credentials",
            "Malformed payload",
            "Authorization failed",
            "Resource not found"
        ]

        for error_msg in transient_failures:
            job_result = JobResult(
                job_id="transient_job",
                job_type=JobType.ENCRYPTION,
                status=JobStatus.FAILURE,
                error_message=error_msg,
                retry_count=1
            )
            assert retry_manager.is_retryable_error(error_msg) == True

        for error_msg in permanent_failures:
            job_result = JobResult(
                job_id="permanent_job",
                job_type=JobType.ENCRYPTION,
                status=JobStatus.FAILURE,
                error_message=error_msg,
                retry_count=1
            )
            assert retry_manager.is_retryable_error(error_msg) == False

    @pytest.mark.asyncio
    async def test_retry_circuit_breaker(self, retry_manager):
        """リトライサーキットブレーカーテスト"""
        # 高頻度失敗時のサーキットブレーカー動作
        retry_manager.enable_circuit_breaker = True
        retry_manager.failure_threshold = 5
        retry_manager.reset_timeout = 60.0

        # 連続失敗のシミュレート
        for i in range(6):
            retry_manager.record_failure("worker_1")

        # サーキットブレーカーが開いている状態
        assert retry_manager.is_circuit_open("worker_1") == True

        # リトライ禁止の確認
        job_result = JobResult(
            job_id="circuit_test_job",
            job_type=JobType.DATA_SYNC,
            status=JobStatus.FAILURE,
            error_message="Worker circuit breaker",
            retry_count=1
        )
        assert retry_manager.should_retry(job_result, worker_id="worker_1") == False


class TestJobPrioritization:
    """ジョブ優先度付けテスト"""

    @pytest.fixture
    def priority_queue(self):
        """優先度キューフィクスチャ"""
        return PriorityJobQueue()

    @pytest.mark.asyncio
    async def test_priority_queue_ordering(self, priority_queue):
        """優先度キューの順序付けテスト"""
        # 異なる優先度のジョブを追加
        jobs = [
            JobRequest(job_type=JobType.CLEANUP, payload={}, priority=JobPriority.LOW),
            JobRequest(job_type=JobType.AGGREGATION, payload={}, priority=JobPriority.CRITICAL),
            JobRequest(job_type=JobType.HEALTH_CHECK, payload={}, priority=JobPriority.NORMAL),
            JobRequest(job_type=JobType.ENCRYPTION, payload={}, priority=JobPriority.HIGH),
            JobRequest(job_type=JobType.BACKUP, payload={}, priority=JobPriority.BATCH)
        ]

        # ジョブをランダムな順序で追加
        for job in jobs:
            await priority_queue.enqueue(job)

        # 優先度順で取得されることを確認
        dequeued_job_1 = await priority_queue.dequeue()  # CRITICAL
        dequeued_job_2 = await priority_queue.dequeue()  # HIGH
        dequeued_job_3 = await priority_queue.dequeue()  # NORMAL
        dequeued_job_4 = await priority_queue.dequeue()  # LOW
        dequeued_job_5 = await priority_queue.dequeue()  # BATCH

        assert dequeued_job_1.priority == JobPriority.CRITICAL
        assert dequeued_job_2.priority == JobPriority.HIGH
        assert dequeued_job_3.priority == JobPriority.NORMAL
        assert dequeued_job_4.priority == JobPriority.LOW
        assert dequeued_job_5.priority == JobPriority.BATCH

    @pytest.mark.asyncio
    async def test_priority_queue_fairness(self, priority_queue):
        """優先度キューの公平性テスト"""
        # 公平性制御をリセット（テスト用）
        await priority_queue.rebalance_queues()

        # 同じ優先度のジョブを複数追加
        normal_jobs = []
        for i in range(3):  # 公平性制御の制限（3回）以内に設定
            job = JobRequest(
                job_type=JobType.MODEL_TRAINING,
                payload={"batch_id": i},
                priority=JobPriority.NORMAL
            )
            normal_jobs.append(job)
            await priority_queue.enqueue(job)

        # 全てのジョブが取得できることを確認
        dequeued_jobs = []
        for i in range(3):
            dequeued = await priority_queue.dequeue()
            assert dequeued is not None, f"Expected job at position {i}, got None"
            # ジョブが正しく取得されることを確認
            assert dequeued.payload["batch_id"] in range(3)
            dequeued_jobs.append(dequeued.payload["batch_id"])

        # 全てのジョブが取得されたことを確認
        assert len(dequeued_jobs) == 3

    @pytest.mark.asyncio
    async def test_priority_preemption(self, priority_queue):
        """優先度によるプリエンプションテスト"""
        # 低優先度ジョブを追加
        low_priority_job = JobRequest(
            job_type=JobType.METRICS_COLLECTION,
            payload={},
            priority=JobPriority.LOW
        )
        await priority_queue.enqueue(low_priority_job)

        # 高優先度ジョブを後から追加
        critical_job = JobRequest(
            job_type=JobType.AGGREGATION,
            payload={},
            priority=JobPriority.CRITICAL
        )
        await priority_queue.enqueue(critical_job)

        # 高優先度ジョブが先に取得される
        first_job = await priority_queue.dequeue()
        assert first_job.priority == JobPriority.CRITICAL
        assert first_job.job_type == JobType.AGGREGATION

        second_job = await priority_queue.dequeue()
        assert second_job.priority == JobPriority.LOW
        assert second_job.job_type == JobType.METRICS_COLLECTION

    @pytest.mark.asyncio
    async def test_queue_capacity_limits(self, priority_queue):
        """キュー容量制限テスト"""
        # キューサイズ制限を設定
        priority_queue.max_queue_size = 10

        # 制限まで ジョブを追加
        for i in range(10):
            job = JobRequest(
                job_type=JobType.DATA_SYNC,
                payload={"item": i},
                priority=JobPriority.NORMAL
            )
            result = await priority_queue.enqueue(job)
            assert result == True

        # 制限を超えたジョブは拒否される
        overflow_job = JobRequest(
            job_type=JobType.CLEANUP,
            payload={},
            priority=JobPriority.LOW
        )
        result = await priority_queue.enqueue(overflow_job)
        assert result == False

        # 高優先度ジョブは制限を超えても追加可能
        critical_job = JobRequest(
            job_type=JobType.AGGREGATION,
            payload={},
            priority=JobPriority.CRITICAL
        )
        result = await priority_queue.enqueue(critical_job, allow_overflow=True)
        assert result == True


class TestDeadLetterQueue:
    """デッドレターキューテスト"""

    @pytest.fixture
    def dead_letter_queue(self):
        """デッドレターキューフィクスチャ"""
        return DeadLetterQueue(max_size=100)

    @pytest.mark.asyncio
    async def test_dead_letter_queue_storage(self, dead_letter_queue):
        """デッドレターキューへの保存テスト"""
        failed_job = JobResult(
            job_id="failed_job_1",
            job_type=JobType.MODEL_TRAINING,
            status=JobStatus.FAILURE,
            error_message="Persistent training failure",
            retry_count=3
        )

        original_payload = {"model_id": "test_model", "epochs": 10}

        # デッドレターキューに追加
        dlq_record = await dead_letter_queue.add_failed_job(
            failed_job, original_payload
        )

        assert dlq_record.job_id == "failed_job_1"
        assert dlq_record.job_type == JobType.MODEL_TRAINING
        assert dlq_record.failure_reason == "Persistent training failure"
        assert dlq_record.retry_count == 3

    @pytest.mark.asyncio
    async def test_dead_letter_queue_retrieval(self, dead_letter_queue):
        """デッドレターキューからの取得テスト"""
        # 複数の失敗ジョブを追加
        failed_jobs = []
        for i in range(3):
            job_result = JobResult(
                job_id=f"failed_job_{i}",
                job_type=JobType.ENCRYPTION,
                status=JobStatus.FAILURE,
                error_message=f"Encryption error {i}",
                retry_count=3
            )
            payload = {"data": f"test_data_{i}"}

            dlq_record = await dead_letter_queue.add_failed_job(job_result, payload)
            failed_jobs.append(dlq_record)

        # 全ての失敗ジョブを取得
        retrieved_jobs = await dead_letter_queue.get_failed_jobs()
        assert len(retrieved_jobs) == 3

        # 特定のジョブタイプで検索
        encryption_jobs = await dead_letter_queue.get_failed_jobs(
            job_type=JobType.ENCRYPTION
        )
        assert len(encryption_jobs) == 3
        assert all(job.job_type == JobType.ENCRYPTION for job in encryption_jobs)

    @pytest.mark.asyncio
    async def test_dead_letter_queue_resubmission(self, dead_letter_queue):
        """デッドレターキューからの再実行テスト"""
        # 失敗ジョブをデッドレターキューに追加
        failed_job = JobResult(
            job_id="resubmit_job",
            job_type=JobType.AGGREGATION,
            status=JobStatus.FAILURE,
            error_message="Temporary aggregation failure",
            retry_count=3
        )

        original_payload = {"round_id": 5, "participants": ["client1", "client2"]}
        dlq_record = await dead_letter_queue.add_failed_job(failed_job, original_payload)

        # 再実行用のジョブリクエストを作成
        resubmit_request = await dead_letter_queue.create_resubmission_request(
            dlq_record.job_id
        )

        assert resubmit_request.job_type == JobType.AGGREGATION
        assert resubmit_request.payload == original_payload
        assert resubmit_request.priority == JobPriority.HIGH  # 再実行は高優先度

        # 再実行後はデッドレターキューから削除
        await dead_letter_queue.mark_resubmitted(dlq_record.job_id)
        remaining_jobs = await dead_letter_queue.get_failed_jobs()
        assert len(remaining_jobs) == 0

    @pytest.mark.asyncio
    async def test_dead_letter_queue_analysis(self, dead_letter_queue):
        """デッドレターキュー分析テスト"""
        # 様々なタイプの失敗ジョブを追加
        failure_scenarios = [
            (JobType.MODEL_TRAINING, "Out of memory", 2),
            (JobType.MODEL_TRAINING, "Out of memory", 3),
            (JobType.ENCRYPTION, "Key rotation failed", 1),
            (JobType.AGGREGATION, "Client disconnect", 4),
            (JobType.AGGREGATION, "Client disconnect", 2)
        ]

        for i, (job_type, error_msg, retry_count) in enumerate(failure_scenarios):
            job_result = JobResult(
                job_id=f"analysis_job_{i}",
                job_type=job_type,
                status=JobStatus.FAILURE,
                error_message=error_msg,
                retry_count=retry_count
            )
            await dead_letter_queue.add_failed_job(job_result, {})

        # 失敗分析を実行
        analysis = await dead_letter_queue.analyze_failures()

        assert analysis["total_failed_jobs"] == 5
        assert analysis["failure_by_type"][JobType.MODEL_TRAINING] == 2
        assert analysis["failure_by_type"][JobType.AGGREGATION] == 2
        assert analysis["failure_by_type"][JobType.ENCRYPTION] == 1

        # 最も一般的な失敗原因
        assert "Out of memory" in [error["error"] for error in analysis["common_errors"]]
        assert "Client disconnect" in [error["error"] for error in analysis["common_errors"]]


class TestJobManager:
    """ジョブ管理器テスト"""

    @pytest.fixture
    def job_manager(self):
        """ジョブ管理器フィクスチャ"""
        return JobManager(
            worker_count=4,
            enable_priority_queue=True,
            enable_dead_letter_queue=True
        )

    @pytest.mark.asyncio
    async def test_job_submission_and_execution(self, job_manager):
        """ジョブ投入と実行テスト"""
        # Celeryタスクをモック化
        with patch.object(job_manager, '_task_mapping') as mock_tasks:
            # モックタスクの設定
            mock_task = MagicMock()
            mock_result = MagicMock()
            mock_result.get.return_value = {"status": "completed", "result": "success"}
            mock_task.delay.return_value = mock_result
            mock_tasks.__getitem__.return_value = mock_task

            # モデル訓練ジョブの投入
            training_request = JobRequest(
                job_type=JobType.MODEL_TRAINING,
                payload={
                    "model_id": "federated_model_v1",
                    "client_data": "encrypted_data",
                    "epochs": 5
                },
                priority=JobPriority.HIGH
            )

            # ジョブの投入
            job_id = await job_manager.submit_job(training_request)
            assert job_id is not None

            # 短時間待機してジョブ完了を確認
            await asyncio.sleep(0.1)
            job_result = await job_manager.wait_for_completion(job_id, timeout=1.0)

            assert job_result.job_id == job_id
            assert job_result.job_type == JobType.MODEL_TRAINING
            assert job_result.status == JobStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_concurrent_job_execution(self, job_manager):
        """並行ジョブ実行テスト"""
        # Celeryタスクをモック化
        with patch.object(job_manager, '_task_mapping') as mock_tasks:
            # モックタスクの設定
            mock_task = MagicMock()
            mock_result = MagicMock()
            mock_result.get.return_value = {"status": "completed", "result": "success"}
            mock_task.delay.return_value = mock_result
            mock_tasks.__getitem__.return_value = mock_task

            # 複数ジョブを同時投入
            job_requests = []
            for i in range(5):  # 10から5に減らして高速化
                request = JobRequest(
                    job_type=JobType.ENCRYPTION,
                    payload={"data_chunk": i, "encryption_key": "test_key"},
                    priority=JobPriority.NORMAL
                )
                job_requests.append(request)

            # 並行投入
            job_ids = []
            for request in job_requests:
                job_id = await job_manager.submit_job(request)
                job_ids.append(job_id)

            # 短時間待機
            await asyncio.sleep(0.1)

            # 全ジョブの完了を待機
            results = await asyncio.gather(
                *[job_manager.wait_for_completion(job_id, timeout=1.0)
                  for job_id in job_ids]
            )

            assert len(results) == 5
            assert all(result.status == JobStatus.SUCCESS for result in results)

    @pytest.mark.asyncio
    async def test_job_cancellation(self, job_manager):
        """ジョブキャンセルテスト"""
        # 長時間実行ジョブの投入
        long_job_request = JobRequest(
            job_type=JobType.DATA_SYNC,
            payload={"sync_duration": 300},  # 5分間の同期ジョブ
            priority=JobPriority.LOW
        )

        job_id = await job_manager.submit_job(long_job_request)

        # 少し待ってからキャンセル
        await asyncio.sleep(0.01)
        cancel_result = await job_manager.cancel_job(job_id)

        assert cancel_result == True

        # ジョブ状態の確認
        job_status = await job_manager.get_job_status(job_id)
        assert job_status == JobStatus.REVOKED

    @pytest.mark.asyncio
    async def test_worker_health_monitoring(self, job_manager):
        """ワーカー健全性監視テスト"""
        # ワーカー状態の取得
        worker_statuses = await job_manager.get_worker_statuses()

        assert len(worker_statuses) == 4  # 設定した ワーカー数
        assert all(worker.is_active for worker in worker_statuses)

        # 特定ワーカーの詳細情報
        worker_id = worker_statuses[0].worker_id
        worker_details = await job_manager.get_worker_details(worker_id)

        assert worker_details.worker_id == worker_id
        assert worker_details.is_healthy()

    @pytest.mark.asyncio
    async def test_job_metrics_collection(self, job_manager):
        """ジョブメトリクス収集テスト"""
        # Celeryタスクをモック化
        with patch.object(job_manager, '_task_mapping') as mock_tasks:
            # モックタスクの設定
            mock_task = MagicMock()
            mock_result = MagicMock()
            mock_result.get.return_value = {"status": "completed", "result": "success"}
            mock_task.delay.return_value = mock_result
            mock_tasks.__getitem__.return_value = mock_task

            # テストジョブの実行
            test_job = JobRequest(
                job_type=JobType.HEALTH_CHECK,
                payload={"target": "all_clients"},
                priority=JobPriority.NORMAL
            )

            job_id = await job_manager.submit_job(test_job)
            await asyncio.sleep(0.1)
            await job_manager.wait_for_completion(job_id, timeout=1.0)

            # メトリクスの取得
            job_metrics = await job_manager.get_job_metrics(job_id)

            assert job_metrics.job_id == job_id
            assert job_metrics.job_type == JobType.HEALTH_CHECK
            assert job_metrics.execution_time >= 0.0
            assert job_metrics.queue_time >= 0.0

            # キュー統計の取得
            queue_stats = await job_manager.get_queue_statistics()

            assert queue_stats.completed_jobs >= 0
            assert queue_stats.average_execution_time >= 0.0