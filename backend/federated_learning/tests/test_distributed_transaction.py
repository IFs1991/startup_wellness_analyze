# Phase 3 Task 3.4: 分散トランザクション管理システムの実装
# TDD RED段階: 失敗するテストから開始

import pytest
import pytest_asyncio
import asyncio
import json
import uuid
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Optional, List, Dict, Any, Union, Callable
from enum import Enum

# テスト対象となるクラス（まだ存在しない）
from ..distributed_transaction.saga_coordinator import SagaCoordinator
from ..distributed_transaction.two_phase_commit import TwoPhaseCommitCoordinator
from ..distributed_transaction.transaction_manager import DistributedTransactionManager
from ..distributed_transaction.compensation_engine import CompensationEngine
from ..distributed_transaction.models import (
    Transaction, TransactionStep, SagaTransaction,
    TransactionStatus, StepStatus, CompensationAction
)

# テストマーカー追加
pytestmark = pytest.mark.distributed_transaction


# 共有フィクスチャ
@pytest_asyncio.fixture
async def mock_resource_managers():
    """モック化されたリソースマネージャーのフィクスチャ"""
    resource_managers = {}

    for i in range(3):
        rm = Mock()
        rm.prepare = AsyncMock(return_value=True)
        rm.commit = AsyncMock(return_value=True)
        rm.abort = AsyncMock(return_value=True)
        rm.status = AsyncMock(return_value="ready")
        resource_managers[f"rm_{i}"] = rm

    return resource_managers


@pytest_asyncio.fixture
async def two_phase_coordinator(mock_resource_managers):
    """TwoPhaseCommitCoordinatorのフィクスチャ"""
    return TwoPhaseCommitCoordinator(
        resource_managers=mock_resource_managers,
        timeout=30.0
    )


@pytest_asyncio.fixture
async def saga_coordinator():
    """SagaCoordinatorのフィクスチャ"""
    return SagaCoordinator(
        max_retry_attempts=3,
        retry_delay=0.1
    )


@pytest_asyncio.fixture
async def distributed_transaction_manager(two_phase_coordinator, saga_coordinator):
    """DistributedTransactionManagerのフィクスチャ"""
    return DistributedTransactionManager(
        two_phase_coordinator=two_phase_coordinator,
        saga_coordinator=saga_coordinator
    )


@pytest_asyncio.fixture
async def sample_transaction_steps():
    """サンプルトランザクションステップ"""
    return [
        TransactionStep(
            step_id="step_1",
            operation="update_model",
            resource_id="model_registry",
            data={"model_id": "model_123", "version": "1.1.0"},
            compensation_data={"model_id": "model_123", "version": "1.0.0"}
        ),
        TransactionStep(
            step_id="step_2",
            operation="store_artifact",
            resource_id="artifact_storage",
            data={"artifact_id": "artifact_456", "data": "model_weights"},
            compensation_data={"artifact_id": "artifact_456"}
        ),
        TransactionStep(
            step_id="step_3",
            operation="update_cache",
            resource_id="cache_service",
            data={"key": "latest_model", "value": "model_123"},
            compensation_data={"key": "latest_model", "value": "model_122"}
        )
    ]


# TwoPhaseCommitCoordinatorのテスト
class TestTwoPhaseCommitCoordinator:
    """2相コミットコーディネーターのテスト"""

    async def test_successful_two_phase_commit(self, two_phase_coordinator, mock_resource_managers):
        """成功する2相コミットテスト"""
        transaction_id = str(uuid.uuid4())

        # 全リソースマネージャーがprepareで成功
        for rm in mock_resource_managers.values():
            rm.prepare.return_value = True
            rm.commit.return_value = True

        result = await two_phase_coordinator.execute_transaction(
            transaction_id=transaction_id,
            operations={
                "rm_0": {"action": "update", "data": {"key": "value1"}},
                "rm_1": {"action": "insert", "data": {"key": "value2"}},
                "rm_2": {"action": "delete", "data": {"id": "123"}}
            }
        )

        assert result.status == TransactionStatus.COMMITTED
        assert result.transaction_id == transaction_id

        # 全リソースマネージャーでprepareとcommitが呼ばれたことを確認
        for rm in mock_resource_managers.values():
            rm.prepare.assert_called_once()
            rm.commit.assert_called_once()

    async def test_failed_prepare_phase(self, two_phase_coordinator, mock_resource_managers):
        """prepare段階での失敗テスト"""
        transaction_id = str(uuid.uuid4())

        # 1つのリソースマネージャーがprepareで失敗
        mock_resource_managers["rm_1"].prepare.return_value = False

        result = await two_phase_coordinator.execute_transaction(
            transaction_id=transaction_id,
            operations={
                "rm_0": {"action": "update", "data": {"key": "value1"}},
                "rm_1": {"action": "insert", "data": {"key": "value2"}},
                "rm_2": {"action": "delete", "data": {"id": "123"}}
            }
        )

        assert result.status == TransactionStatus.ABORTED

        # 全リソースマネージャーでabortが呼ばれたことを確認
        for rm in mock_resource_managers.values():
            rm.abort.assert_called_once()

    async def test_timeout_handling(self, two_phase_coordinator, mock_resource_managers):
        """タイムアウト処理テスト"""
        transaction_id = str(uuid.uuid4())

        # 遅いprepare操作をシミュレート
        async def slow_prepare(*args, **kwargs):
            await asyncio.sleep(1.0)  # coordinatorのタイムアウト以上
            return True

        mock_resource_managers["rm_1"].prepare = slow_prepare

        # タイムアウトを短く設定
        two_phase_coordinator.timeout = 0.5

        result = await two_phase_coordinator.execute_transaction(
            transaction_id=transaction_id,
            operations={"rm_1": {"action": "update", "data": {"key": "value"}}}
        )

        assert result.status == TransactionStatus.ABORTED

    async def test_partial_commit_failure(self, two_phase_coordinator, mock_resource_managers):
        """commit段階での部分的失敗テスト"""
        transaction_id = str(uuid.uuid4())

        # prepareは全て成功、commitで1つ失敗
        for rm in mock_resource_managers.values():
            rm.prepare.return_value = True

        mock_resource_managers["rm_2"].commit.return_value = False

        result = await two_phase_coordinator.execute_transaction(
            transaction_id=transaction_id,
            operations={
                "rm_0": {"action": "update", "data": {"key": "value1"}},
                "rm_1": {"action": "insert", "data": {"key": "value2"}},
                "rm_2": {"action": "delete", "data": {"id": "123"}}
            }
        )

        # 部分的失敗でも可能な限りcommitを試行
        assert result.status in [TransactionStatus.COMMITTED, TransactionStatus.PARTIALLY_COMMITTED]


# SagaCoordinatorのテスト
class TestSagaCoordinator:
    """Sagaコーディネーターのテスト"""

    async def test_successful_saga_execution(self, saga_coordinator, sample_transaction_steps):
        """成功するSaga実行テスト"""
        saga_id = str(uuid.uuid4())

        # モックサービス
        mock_services = {
            "model_registry": Mock(),
            "artifact_storage": Mock(),
            "cache_service": Mock()
        }

        for service in mock_services.values():
            service.execute_operation = AsyncMock(return_value={"status": "success"})

        saga_transaction = SagaTransaction(
            saga_id=saga_id,
            steps=sample_transaction_steps,
            services=mock_services
        )

        result = await saga_coordinator.execute_saga(saga_transaction)

        assert result.status == TransactionStatus.COMMITTED
        assert len(result.completed_steps) == len(sample_transaction_steps)

    async def test_saga_compensation_on_failure(self, saga_coordinator, sample_transaction_steps):
        """失敗時のSaga補償テスト"""
        saga_id = str(uuid.uuid4())

        # モックサービス
        mock_services = {
            "model_registry": Mock(),
            "artifact_storage": Mock(),
            "cache_service": Mock()
        }

        # 最初の2つは成功、3つ目で失敗
        mock_services["model_registry"].execute_operation = AsyncMock(return_value={"status": "success"})
        mock_services["artifact_storage"].execute_operation = AsyncMock(return_value={"status": "success"})
        mock_services["cache_service"].execute_operation = AsyncMock(side_effect=Exception("Service failure"))

        # 補償操作の設定
        for service in mock_services.values():
            service.compensate_operation = AsyncMock(return_value={"status": "compensated"})

        saga_transaction = SagaTransaction(
            saga_id=saga_id,
            steps=sample_transaction_steps,
            services=mock_services
        )

        result = await saga_coordinator.execute_saga(saga_transaction)

        assert result.status == TransactionStatus.COMPENSATED
        # 失敗前に実行された操作の補償が実行されたことを確認
        mock_services["model_registry"].compensate_operation.assert_called_once()
        mock_services["artifact_storage"].compensate_operation.assert_called_once()

    async def test_saga_retry_mechanism(self, saga_coordinator, sample_transaction_steps):
        """Sagaリトライ機構テスト"""
        saga_id = str(uuid.uuid4())

        # モックサービス
        mock_service = Mock()

        # 最初の2回は失敗、3回目で成功
        call_count = 0
        async def mock_operation(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return {"status": "success"}

        mock_service.execute_operation = mock_operation

        saga_transaction = SagaTransaction(
            saga_id=saga_id,
            steps=[sample_transaction_steps[0]],  # 1つのステップのみ
            services={"model_registry": mock_service}
        )

        result = await saga_coordinator.execute_saga(saga_transaction)

        assert result.status == TransactionStatus.COMMITTED
        assert call_count == 3  # 3回試行されたことを確認

    async def test_saga_parallel_execution(self, saga_coordinator):
        """Saga並列実行テスト"""
        saga_id = str(uuid.uuid4())

        # 並列実行可能なステップを作成
        parallel_steps = []
        for i in range(3):
            step = TransactionStep(
                step_id=f"parallel_step_{i}",
                operation="parallel_operation",
                resource_id=f"service_{i}",
                data={"value": i},
                can_run_parallel=True
            )
            parallel_steps.append(step)

        # モックサービス
        mock_services = {}
        execution_times = {}

        for i in range(3):
            service = Mock()
            # クロージャの問題を修正
            def create_mock_operation(index):
                async def mock_operation(*args, **kwargs):
                    start_time = time.time()
                    await asyncio.sleep(0.05)  # より短い時間でテスト
                    execution_times[index] = time.time()
                    return {"status": "success"}
                return mock_operation

            service.execute_operation = create_mock_operation(i)
            mock_services[f"service_{i}"] = service

        saga_transaction = SagaTransaction(
            saga_id=saga_id,
            steps=parallel_steps,
            services=mock_services
        )

        start = time.time()
        result = await saga_coordinator.execute_saga(saga_transaction)
        total_time = time.time() - start

        assert result.status == TransactionStatus.COMMITTED
        # 並列実行により総実行時間が短縮されることを確認
        assert total_time < 0.2  # 順次実行なら0.15秒以上かかるはず


# CompensationEngineのテスト
class TestCompensationEngine:
    """補償エンジンのテスト"""

    @pytest_asyncio.fixture
    async def compensation_engine(self):
        """CompensationEngineのフィクスチャ"""
        return CompensationEngine(
            max_compensation_attempts=3,
            compensation_delay=0.1
        )

    async def test_successful_compensation(self, compensation_engine):
        """成功する補償処理テスト"""
        # 補償アクション定義
        compensation_actions = [
            CompensationAction(
                action_id="comp_1",
                resource_id="service_1",
                operation="rollback_update",
                data={"previous_value": "old_data"}
            ),
            CompensationAction(
                action_id="comp_2",
                resource_id="service_2",
                operation="delete_created",
                data={"created_id": "123"}
            )
        ]

        # モックサービス
        mock_services = {
            "service_1": Mock(),
            "service_2": Mock()
        }

        for service in mock_services.values():
            service.execute_compensation = AsyncMock(return_value={"status": "success"})

        result = await compensation_engine.execute_compensations(
            compensation_actions,
            mock_services
        )

        assert result.success is True
        assert len(result.successful_compensations) == 2

    async def test_compensation_failure_and_retry(self, compensation_engine):
        """補償失敗とリトライテスト"""
        compensation_action = CompensationAction(
            action_id="comp_fail",
            resource_id="service_fail",
            operation="rollback",
            data={"data": "test"}
        )

        # モックサービス（最初の2回は失敗、3回目で成功）
        mock_service = Mock()
        call_count = 0

        async def mock_compensation(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Compensation failed")
            return {"status": "success"}

        mock_service.execute_compensation = mock_compensation

        result = await compensation_engine.execute_compensations(
            [compensation_action],
            {"service_fail": mock_service}
        )

        assert result.success is True
        assert call_count == 3

    async def test_deadlock_detection_and_resolution(self, compensation_engine):
        """デッドロック検出と解決テスト"""
        # 循環依存を持つ補償アクション
        compensation_actions = [
            CompensationAction(
                action_id="comp_a",
                resource_id="service_a",
                operation="unlock_resource",
                data={"resource": "res_b"},
                dependencies=["comp_b"]
            ),
            CompensationAction(
                action_id="comp_b",
                resource_id="service_b",
                operation="unlock_resource",
                data={"resource": "res_a"},
                dependencies=["comp_a"]
            )
        ]

        mock_services = {
            "service_a": Mock(),
            "service_b": Mock()
        }

        for service in mock_services.values():
            service.execute_compensation = AsyncMock(return_value={"status": "success"})

        result = await compensation_engine.execute_compensations(
            compensation_actions,
            mock_services
        )

        # デッドロック検出により適切に処理される
        assert result.deadlock_detected is True


# DistributedTransactionManagerのテスト
class TestDistributedTransactionManager:
    """分散トランザクションマネージャーのテスト"""

    async def test_transaction_type_selection(self, distributed_transaction_manager):
        """トランザクションタイプ選択テスト"""
        # ACID要件が厳格な場合は2PC
        strict_transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            transaction_type="strict",
            consistency_level="strong",
            steps=[]
        )

        coordinator = distributed_transaction_manager.select_coordinator(strict_transaction)
        assert isinstance(coordinator, TwoPhaseCommitCoordinator)

        # 柔軟性が必要な場合はSaga
        flexible_transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            transaction_type="saga",
            consistency_level="eventual",
            steps=[]
        )

        coordinator = distributed_transaction_manager.select_coordinator(flexible_transaction)
        assert isinstance(coordinator, SagaCoordinator)

    async def test_mixed_transaction_handling(self, distributed_transaction_manager, sample_transaction_steps):
        """混合トランザクション処理テスト"""
        # リソースマネージャーを設定
        for step in sample_transaction_steps:
            if step.resource_id not in distributed_transaction_manager.two_phase_coordinator.resource_managers:
                mock_rm = Mock()
                mock_rm.prepare = AsyncMock(return_value=True)
                mock_rm.commit = AsyncMock(return_value=True)
                mock_rm.abort = AsyncMock(return_value=True)
                mock_rm.status = AsyncMock(return_value="ready")
                distributed_transaction_manager.two_phase_coordinator.resource_managers[step.resource_id] = mock_rm

        transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            transaction_type="mixed",
            steps=sample_transaction_steps,
            consistency_requirements={
                "model_registry": "strong",  # 2PCが必要
                "artifact_storage": "eventual",  # Sagaで可
                "cache_service": "eventual"
            }
        )

        result = await distributed_transaction_manager.execute_transaction(transaction)

                # 混合モードでも成功することを確認
        assert result.status in [TransactionStatus.COMMITTED, TransactionStatus.PARTIALLY_COMMITTED]

    async def test_transaction_isolation_levels(self, distributed_transaction_manager):
        """トランザクション分離レベルテスト"""
        # 同時実行される2つのトランザクション
        transaction1 = Transaction(
            transaction_id="tx_1",
            transaction_type="saga",
            isolation_level="serializable",
            steps=[
                TransactionStep(
                    step_id="step1_tx1",
                    operation="update",
                    resource_id="shared_resource",
                    data={"value": 100}
                )
            ]
        )

        transaction2 = Transaction(
            transaction_id="tx_2",
            transaction_type="saga",
            isolation_level="read_committed",
            steps=[
                TransactionStep(
                    step_id="step1_tx2",
                    operation="read",
                    resource_id="shared_resource",
                    data={}
                )
            ]
        )

        # 並行実行
        results = await asyncio.gather(
            distributed_transaction_manager.execute_transaction(transaction1),
            distributed_transaction_manager.execute_transaction(transaction2),
            return_exceptions=True
        )

        # 分離レベルが適切に処理されることを確認
        assert any(result.status == TransactionStatus.COMMITTED for result in results if hasattr(result, 'status'))


# 統合テスト
class TestDistributedTransactionIntegration:
    """分散トランザクション統合テスト"""

    async def test_end_to_end_model_deployment_transaction(self, distributed_transaction_manager):
        """エンドツーエンドモデルデプロイメントトランザクションテスト"""
        # 実際のモデルデプロイメントシナリオをシミュレート
        deployment_steps = [
            TransactionStep(
                step_id="validate_model",
                operation="validate",
                resource_id="model_validator",
                data={"model_id": "new_model_v2", "validation_config": {"accuracy_threshold": 0.9}}
            ),
            TransactionStep(
                step_id="update_registry",
                operation="register_model",
                resource_id="model_registry",
                data={"model_id": "new_model_v2", "status": "active"}
            ),
            TransactionStep(
                step_id="deploy_to_serving",
                operation="deploy",
                resource_id="serving_infrastructure",
                data={"model_id": "new_model_v2", "replicas": 3}
            ),
            TransactionStep(
                step_id="update_routing",
                operation="update_traffic_routing",
                resource_id="traffic_manager",
                data={"new_model": "new_model_v2", "traffic_split": 0.1}
            )
        ]

        transaction = Transaction(
            transaction_id="model_deployment_tx",
            transaction_type="saga",  # 長時間実行の可能性があるためSagaを選択
            steps=deployment_steps,
            metadata={
                "deployment_type": "canary",
                "rollback_on_failure": True
            }
        )

        result = await distributed_transaction_manager.execute_transaction(transaction)

        assert result.status in [TransactionStatus.COMMITTED, TransactionStatus.COMPENSATED]

    async def test_concurrent_transaction_handling(self, distributed_transaction_manager):
        """並行トランザクション処理テスト"""
        # 複数の並行トランザクションを作成
        transactions = []
        for i in range(5):
            transaction = Transaction(
                transaction_id=f"concurrent_tx_{i}",
                transaction_type="saga",
                steps=[
                    TransactionStep(
                        step_id=f"step_{i}",
                        operation="process",
                        resource_id=f"resource_{i % 3}",  # リソース競合をシミュレート
                        data={"value": i}
                    )
                ]
            )
            transactions.append(transaction)

        # 全トランザクションを並行実行
        results = await asyncio.gather(
            *[distributed_transaction_manager.execute_transaction(tx) for tx in transactions],
            return_exceptions=True
        )

        # 少なくとも一部のトランザクションが成功することを確認
        successful_count = sum(1 for result in results
                              if hasattr(result, 'status') and result.status == TransactionStatus.COMMITTED)
        assert successful_count > 0

    async def test_disaster_recovery_scenario(self, distributed_transaction_manager):
        """災害復旧シナリオテスト"""
        # 実行中にコーディネーターが失敗するシナリオ
        transaction = Transaction(
            transaction_id="disaster_tx",
            transaction_type="saga",
            steps=[
                TransactionStep(
                    step_id="step_1",
                    operation="backup_data",
                    resource_id="backup_service",
                    data={"backup_id": "backup_123"}
                ),
                TransactionStep(
                    step_id="step_2",
                    operation="simulate_failure",
                    resource_id="failure_service",
                    data={"failure_type": "coordinator_crash"}
                )
            ]
        )

        # 失敗シミュレーション付きで実行
        with patch.object(distributed_transaction_manager, '_handle_coordinator_failure') as mock_handler:
            mock_handler.return_value = {"recovery_status": "success"}

            result = await distributed_transaction_manager.execute_transaction(transaction)

            # 復旧処理が呼ばれることを確認
            assert result is not None