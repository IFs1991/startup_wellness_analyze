"""
ジョブ管理器
Task 4.3: 非同期ジョブキュー
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from .models import JobRequest, JobResult, JobMetrics, QueueStatistics, WorkerStatus
from .job_types import JobType, JobPriority, JobStatus
from .priority_queue import PriorityJobQueue
from .retry_manager import RetryManager
from .dead_letter_queue import DeadLetterQueue
from .fl_jobs import (
    model_training_task,
    aggregation_task,
    encryption_task,
    health_check_task,
    data_sync_task,
    metrics_collection_task,
    cleanup_task
)


class JobManager:
    """統合ジョブ管理器"""

    def __init__(
        self,
        worker_count: int = 4,
        enable_priority_queue: bool = True,
        enable_dead_letter_queue: bool = True,
        max_queue_size: int = 1000
    ):
        self.worker_count = worker_count
        self.enable_priority_queue = enable_priority_queue
        self.enable_dead_letter_queue = enable_dead_letter_queue

        # コンポーネント初期化
        self.priority_queue = PriorityJobQueue(max_queue_size) if enable_priority_queue else None
        self.retry_manager = RetryManager()
        self.dead_letter_queue = DeadLetterQueue() if enable_dead_letter_queue else None

        # ジョブ状態管理
        self._active_jobs: Dict[str, JobRequest] = {}
        self._completed_jobs: Dict[str, JobResult] = {}
        self._job_metrics: Dict[str, JobMetrics] = {}

        # ワーカー状態管理
        self._workers: Dict[str, WorkerStatus] = {}

        # 統計情報
        self._statistics = QueueStatistics(queue_name="job_manager")

        # Celeryタスクマッピング
        self._task_mapping = {
            JobType.MODEL_TRAINING: model_training_task,
            JobType.AGGREGATION: aggregation_task,
            JobType.ENCRYPTION: encryption_task,
            JobType.HEALTH_CHECK: health_check_task,
            JobType.DATA_SYNC: data_sync_task,
            JobType.METRICS_COLLECTION: metrics_collection_task,
            JobType.CLEANUP: cleanup_task
        }

        # ワーカー初期化
        self._init_workers()

    def _init_workers(self):
        """ワーカーを初期化"""
        for i in range(self.worker_count):
            worker_id = f"worker_{i+1}"
            self._workers[worker_id] = WorkerStatus(
                worker_id=worker_id,
                hostname=f"host_{i+1}",
                is_active=True
            )

    async def submit_job(self, job_request: JobRequest) -> str:
        """
        ジョブを投入

        Args:
            job_request: ジョブリクエスト

        Returns:
            str: ジョブID
        """
        # ジョブIDが未設定の場合は生成
        if not hasattr(job_request, 'job_id') or not job_request.job_id:
            job_request.job_id = str(uuid.uuid4())

        # メトリクス作成
        now = datetime.now()
        queue_name = f"{job_request.priority.name.lower()}_queue" if job_request.priority else "normal_queue"

        job_metrics = JobMetrics(
            job_id=job_request.job_id,
            job_type=job_request.job_type,
            priority=job_request.priority or JobPriority.NORMAL,
            queue_name=queue_name,
            created_at=now
        )
        self._job_metrics[job_request.job_id] = job_metrics

        # 優先度キューに追加
        if self.priority_queue:
            await self.priority_queue.enqueue(job_request)

        # アクティブジョブに追加
        self._active_jobs[job_request.job_id] = job_request

        # ジョブ実行開始
        asyncio.create_task(self._execute_job(job_request))

        return job_request.job_id

    async def _execute_job(self, job_request: JobRequest):
        """
        ジョブを実行

        Args:
            job_request: ジョブリクエスト
        """
        job_id = job_request.job_id
        start_time = datetime.now()

        try:
            # メトリクス更新
            if job_id in self._job_metrics:
                self._job_metrics[job_id].started_at = start_time
                self._job_metrics[job_id].calculate_timing_metrics()

            # ワーカー割り当て
            worker = self._assign_worker(job_request)
            if worker:
                worker.current_job_id = job_id
                worker.current_job_type = job_request.job_type
                worker.job_started_at = start_time

            # Celeryタスクの実行
            task_func = self._task_mapping.get(job_request.job_type)
            if not task_func:
                raise ValueError(f"Unsupported job type: {job_request.job_type}")

            # タスク実行
            celery_result = task_func.delay(job_request.payload)
            result_data = celery_result.get(timeout=job_request.timeout or 600)

            # 成功結果の作成
            job_result = JobResult(
                job_id=job_id,
                job_type=job_request.job_type,
                status=JobStatus.SUCCESS,
                result_data=result_data,
                started_at=start_time,
                completed_at=datetime.now(),
                worker_id=worker.worker_id if worker else None
            )

            # 統計更新
            if worker:
                worker.successful_jobs += 1
                worker.total_jobs_processed += 1
                self.retry_manager.record_success(worker.worker_id)

        except Exception as e:
            # 失敗結果の作成
            job_result = JobResult(
                job_id=job_id,
                job_type=job_request.job_type,
                status=JobStatus.FAILURE,
                error_message=str(e),
                started_at=start_time,
                completed_at=datetime.now(),
                worker_id=worker.worker_id if worker else None
            )

            # リトライ判定
            should_retry = self.retry_manager.should_retry(job_result, worker.worker_id if worker else None)

            if should_retry:
                # リトライ実行
                job_result.retry_count += 1
                job_result.status = JobStatus.RETRY

                retry_delay = self.retry_manager.calculate_retry_delay(job_result.retry_count)
                await asyncio.sleep(retry_delay)

                # リトライジョブとして再投入
                await self.submit_job(job_request)
                return

            # デッドレターキューへの移動判定
            if self.dead_letter_queue and self.retry_manager.should_move_to_dead_letter(job_result):
                await self.dead_letter_queue.add_failed_job(job_result, job_request.payload)
                job_result.status = JobStatus.DEAD_LETTER

            # 統計更新
            if worker:
                worker.failed_jobs += 1
                worker.total_jobs_processed += 1
                self.retry_manager.record_failure(worker.worker_id)

        finally:
            # 後処理
            await self._finalize_job(job_request, job_result, worker)

    async def _finalize_job(
        self,
        job_request: JobRequest,
        job_result: JobResult,
        worker: Optional[WorkerStatus]
    ):
        """
        ジョブの最終処理

        Args:
            job_request: ジョブリクエスト
            job_result: ジョブ結果
            worker: ワーカー
        """
        job_id = job_request.job_id

        # アクティブジョブから削除
        if job_id in self._active_jobs:
            del self._active_jobs[job_id]

        # 完了ジョブに追加
        self._completed_jobs[job_id] = job_result

        # メトリクス更新
        if job_id in self._job_metrics:
            metrics = self._job_metrics[job_id]
            metrics.completed_at = job_result.completed_at
            metrics.calculate_timing_metrics()

            if job_result.status == JobStatus.FAILURE:
                metrics.error_type = type(Exception).__name__
                metrics.error_message = job_result.error_message

        # ワーカー状態をリセット
        if worker:
            worker.current_job_id = None
            worker.current_job_type = None
            worker.job_started_at = None
            worker.update_heartbeat()

    def _assign_worker(self, job_request: JobRequest) -> Optional[WorkerStatus]:
        """
        ワーカーを割り当て

        Args:
            job_request: ジョブリクエスト

        Returns:
            Optional[WorkerStatus]: 割り当てられたワーカー
        """
        # アクティブで利用可能なワーカーを検索
        available_workers = [
            worker for worker in self._workers.values()
            if worker.is_active and worker.current_job_id is None and worker.is_healthy()
        ]

        if not available_workers:
            return None

        # ワーカー選択ロジック（負荷分散）
        best_worker = min(available_workers, key=lambda w: w.total_jobs_processed)
        return best_worker

    async def wait_for_completion(self, job_id: str, timeout: float = 300.0) -> JobResult:
        """
        ジョブの完了を待機

        Args:
            job_id: ジョブID
            timeout: タイムアウト時間（秒）

        Returns:
            JobResult: ジョブ結果
        """
        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < timeout:
            if job_id in self._completed_jobs:
                return self._completed_jobs[job_id]

            await asyncio.sleep(0.1)

        # タイムアウト時の処理
        if job_id in self._active_jobs:
            await self.cancel_job(job_id)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

    async def cancel_job(self, job_id: str) -> bool:
        """
        ジョブをキャンセル

        Args:
            job_id: ジョブID

        Returns:
            bool: キャンセルに成功したかどうか
        """
        if job_id not in self._active_jobs:
            return False

        job_request = self._active_jobs[job_id]

        # キュー優先度から削除
        if self.priority_queue:
            await self.priority_queue.remove_job(job_id)

        # キャンセル結果を作成
        job_result = JobResult(
            job_id=job_id,
            job_type=job_request.job_type,
            status=JobStatus.REVOKED,
            completed_at=datetime.now()
        )

        # 最終処理
        await self._finalize_job(job_request, job_result, None)

        return True

    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        ジョブ状態を取得

        Args:
            job_id: ジョブID

        Returns:
            Optional[JobStatus]: ジョブ状態
        """
        if job_id in self._completed_jobs:
            return self._completed_jobs[job_id].status
        elif job_id in self._active_jobs:
            return JobStatus.STARTED
        else:
            return None

    async def get_worker_statuses(self) -> List[WorkerStatus]:
        """
        全ワーカーの状態を取得

        Returns:
            List[WorkerStatus]: ワーカー状態一覧
        """
        return list(self._workers.values())

    async def get_worker_details(self, worker_id: str) -> Optional[WorkerStatus]:
        """
        特定ワーカーの詳細情報を取得

        Args:
            worker_id: ワーカーID

        Returns:
            Optional[WorkerStatus]: ワーカー状態
        """
        return self._workers.get(worker_id)

    async def get_job_metrics(self, job_id: str) -> Optional[JobMetrics]:
        """
        ジョブメトリクスを取得

        Args:
            job_id: ジョブID

        Returns:
            Optional[JobMetrics]: ジョブメトリクス
        """
        return self._job_metrics.get(job_id)

    async def get_queue_statistics(self) -> QueueStatistics:
        """
        キュー統計情報を取得

        Returns:
            QueueStatistics: 統計情報
        """
        if self.priority_queue:
            return await self.priority_queue.get_statistics()
        else:
            # 基本統計の計算
            self._statistics.active_jobs = len(self._active_jobs)
            self._statistics.completed_jobs = len(self._completed_jobs)
            self._statistics.last_updated = datetime.now()

            return self._statistics

    async def get_system_health(self) -> Dict[str, Any]:
        """
        システム全体の健全性を取得

        Returns:
            Dict[str, Any]: システム健全性情報
        """
        queue_health = await self.priority_queue.get_queue_health() if self.priority_queue else {}
        worker_health = {
            "total_workers": len(self._workers),
            "active_workers": sum(1 for w in self._workers.values() if w.is_active),
            "healthy_workers": sum(1 for w in self._workers.values() if w.is_healthy()),
            "average_success_rate": sum(w.get_success_rate() for w in self._workers.values()) / len(self._workers)
        }

        dlq_stats = await self.dead_letter_queue.get_statistics() if self.dead_letter_queue else {}
        retry_stats = self.retry_manager.get_retry_statistics()

        return {
            "queue_health": queue_health,
            "worker_health": worker_health,
            "dead_letter_queue": dlq_stats,
            "retry_manager": retry_stats,
            "active_jobs_count": len(self._active_jobs),
            "completed_jobs_count": len(self._completed_jobs),
            "system_uptime": datetime.now().isoformat()
        }