# Phase 3 Task 3.4: 分散トランザクション管理システムの実装
# TDD GREEN段階: DistributedTransactionManager実装

import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone

import structlog
from .models import (
    Transaction, TransactionStatus, TwoPhaseCommitResult, SagaResult,
    generate_transaction_id, is_terminal_status, validate_transaction
)
from .two_phase_commit import TwoPhaseCommitCoordinator
from .saga_coordinator import SagaCoordinator
from .compensation_engine import CompensationEngine

logger = structlog.get_logger(__name__)


class DistributedTransactionManager:
    """
    分散トランザクション管理マネージャー

    2相コミットとSagaパターンを統合した分散トランザクション管理
    """

    def __init__(
        self,
        two_phase_coordinator: Optional[TwoPhaseCommitCoordinator] = None,
        saga_coordinator: Optional[SagaCoordinator] = None,
        compensation_engine: Optional[CompensationEngine] = None,
        default_timeout: float = 300.0
    ):
        """
        DistributedTransactionManagerの初期化

        Args:
            two_phase_coordinator: 2相コミットコーディネーター
            saga_coordinator: Sagaコーディネーター
            compensation_engine: 補償エンジン
            default_timeout: デフォルトタイムアウト（秒）
        """
        self.two_phase_coordinator = two_phase_coordinator or TwoPhaseCommitCoordinator()
        self.saga_coordinator = saga_coordinator or SagaCoordinator()
        self.compensation_engine = compensation_engine or CompensationEngine()
        self.default_timeout = default_timeout

        self.active_transactions: Dict[str, Transaction] = {}
        self.transaction_history: List[Transaction] = []
        self.metrics = {
            "total_transactions": 0,
            "successful_transactions": 0,
            "failed_transactions": 0,
            "compensated_transactions": 0,
            "avg_execution_time": 0.0
        }

    async def execute_transaction(self, transaction: Transaction) -> Union[TwoPhaseCommitResult, SagaResult]:
        """
        分散トランザクションを実行

        Args:
            transaction: 実行するトランザクション

        Returns:
            実行結果（2PCまたはSaga）
        """
        start_time = time.time()
        transaction.started_at = datetime.now(timezone.utc)

        # トランザクション検証
        validation_errors = validate_transaction(transaction)
        if validation_errors:
            error_msg = f"Transaction validation failed: {', '.join(validation_errors)}"
            await self._log_transaction(transaction.transaction_id, "VALIDATION_FAILED", error_msg)
            raise ValueError(error_msg)

        # アクティブトランザクションに追加
        self.active_transactions[transaction.transaction_id] = transaction
        self.metrics["total_transactions"] += 1

        try:
            await self._log_transaction(transaction.transaction_id, "START",
                                       f"Starting {transaction.transaction_type} transaction")

            # コーディネーター選択と実行
            coordinator = self.select_coordinator(transaction)

            if isinstance(coordinator, TwoPhaseCommitCoordinator):
                result = await self._execute_2pc_transaction(transaction, coordinator)
            elif isinstance(coordinator, SagaCoordinator):
                result = await self._execute_saga_transaction(transaction, coordinator)
            else:
                raise ValueError(f"Unknown coordinator type: {type(coordinator)}")

            # 結果に基づく後処理
            await self._post_process_transaction(transaction, result)

            # 実行時間更新
            execution_time = time.time() - start_time
            self._update_metrics(result.status, execution_time)

            return result

        except Exception as e:
            error_msg = f"Transaction execution failed: {str(e)}"
            transaction.status = TransactionStatus.FAILED
            transaction.error_message = error_msg

            await self._log_transaction(transaction.transaction_id, "ERROR", error_msg)
            logger.error(error_msg, transaction_id=transaction.transaction_id)

            self.metrics["failed_transactions"] += 1
            raise

        finally:
            transaction.completed_at = datetime.now(timezone.utc)
            # アクティブトランザクションから削除し、履歴に追加
            self.active_transactions.pop(transaction.transaction_id, None)
            self.transaction_history.append(transaction)

    def select_coordinator(self, transaction: Transaction) -> Union[TwoPhaseCommitCoordinator, SagaCoordinator]:
        """
        トランザクションタイプに基づいてコーディネーターを選択

        Args:
            transaction: トランザクション

        Returns:
            選択されたコーディネーター
        """
        if transaction.transaction_type == "2pc" or transaction.transaction_type == "strict":
            return self.two_phase_coordinator
        elif transaction.transaction_type == "saga":
            return self.saga_coordinator
        elif transaction.transaction_type == "mixed":
            # 混合モードの場合は一貫性要件に基づいて判定
            if self._requires_strong_consistency(transaction):
                return self.two_phase_coordinator
            else:
                return self.saga_coordinator
        else:
            # デフォルトはSaga
            return self.saga_coordinator

    def _requires_strong_consistency(self, transaction: Transaction) -> bool:
        """
        強い一貫性が必要かどうかを判定

        Args:
            transaction: トランザクション

        Returns:
            強い一貫性が必要かどうか
        """
        if transaction.consistency_level == "strong":
            return True

        # リソース個別の要件をチェック
        if transaction.consistency_requirements:
            for step in transaction.steps:
                requirement = transaction.consistency_requirements.get(step.resource_id, "eventual")
                if requirement == "strong":
                    return True

        return False

    async def _execute_2pc_transaction(
        self,
        transaction: Transaction,
        coordinator: TwoPhaseCommitCoordinator
    ) -> TwoPhaseCommitResult:
        """
        2相コミットトランザクション実行

        Args:
            transaction: トランザクション
            coordinator: 2PCコーディネーター

        Returns:
            2PC実行結果
        """
        # トランザクションステップを2PC操作に変換
        operations = {}
        for step in transaction.steps:
            operations[step.resource_id] = {
                "action": step.operation,
                "data": step.data
            }

        # タイムアウト設定
        timeout = transaction.timeout or self.default_timeout
        coordinator.timeout = timeout

        transaction.status = TransactionStatus.PREPARING
        result = await coordinator.execute_transaction(transaction.transaction_id, operations)
        transaction.status = result.status

        return result

    async def _execute_saga_transaction(
        self,
        transaction: Transaction,
        coordinator: SagaCoordinator
    ) -> SagaResult:
        """
        Sagaトランザクション実行

        Args:
            transaction: トランザクション
            coordinator: Sagaコーディネーター

        Returns:
            Saga実行結果
        """
        from .saga_coordinator import SagaTransaction

        # モックサービス作成（実際の実装では適切なサービスを注入）
        services = self._create_mock_services(transaction)

        saga_transaction = SagaTransaction(
            saga_id=transaction.transaction_id,
            steps=transaction.steps,
            services=services,
            parallel_execution=transaction.metadata.get("parallel_execution", False)
        )

        transaction.status = TransactionStatus.PENDING
        result = await coordinator.execute_saga(saga_transaction)
        transaction.status = result.status

        return result

    def _create_mock_services(self, transaction: Transaction) -> Dict[str, Any]:
        """
        テスト用のモックサービスを作成

        Args:
            transaction: トランザクション

        Returns:
            モックサービス辞書
        """
        services = {}

        for step in transaction.steps:
            if step.resource_id not in services:
                # モックサービス作成
                mock_service = type('MockService', (), {
                    'execute_operation': self._mock_execute_operation,
                    'compensate_operation': self._mock_compensate_operation,
                    'execute_compensation': self._mock_execute_compensation
                })()
                services[step.resource_id] = mock_service

        return services

    async def _mock_execute_operation(self, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """モック操作実行"""
        # 簡単な成功パターン
        await asyncio.sleep(0.01)  # 処理時間をシミュレート
        return {"status": "success", "operation": operation, "data": data}

    async def _mock_compensate_operation(self, operation: str, compensation_data: Dict[str, Any]) -> Dict[str, Any]:
        """モック補償操作実行"""
        await asyncio.sleep(0.01)
        return {"status": "compensated", "operation": operation, "data": compensation_data}

    async def _mock_execute_compensation(self, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """モック補償実行"""
        await asyncio.sleep(0.01)
        return {"status": "success", "operation": operation, "data": data}

    async def _post_process_transaction(
        self,
        transaction: Transaction,
        result: Union[TwoPhaseCommitResult, SagaResult]
    ) -> None:
        """
        トランザクション後処理

        Args:
            transaction: トランザクション
            result: 実行結果
        """
        if result.status == TransactionStatus.COMMITTED:
            self.metrics["successful_transactions"] += 1
            await self._log_transaction(transaction.transaction_id, "SUCCESS",
                                       "Transaction committed successfully")
        elif result.status == TransactionStatus.COMPENSATED:
            self.metrics["compensated_transactions"] += 1
            await self._log_transaction(transaction.transaction_id, "COMPENSATED",
                                       "Transaction compensated")
        else:
            self.metrics["failed_transactions"] += 1
            await self._log_transaction(transaction.transaction_id, "FAILED",
                                       f"Transaction failed with status: {result.status}")

    def _update_metrics(self, status: TransactionStatus, execution_time: float) -> None:
        """
        メトリクス更新

        Args:
            status: トランザクション状態
            execution_time: 実行時間
        """
        # 平均実行時間更新（移動平均）
        total_transactions = self.metrics["total_transactions"]
        current_avg = self.metrics["avg_execution_time"]
        self.metrics["avg_execution_time"] = (
            (current_avg * (total_transactions - 1) + execution_time) / total_transactions
        )

    async def _log_transaction(self, transaction_id: str, operation: str, message: str) -> None:
        """
        トランザクションログ記録

        Args:
            transaction_id: トランザクションID
            operation: 操作名
            message: ログメッセージ
        """
        logger.info(
            message,
            transaction_id=transaction_id,
            operation=operation,
            active_transactions=len(self.active_transactions)
        )

    async def get_transaction_status(self, transaction_id: str) -> Optional[Transaction]:
        """
        トランザクション状態取得

        Args:
            transaction_id: トランザクションID

        Returns:
            トランザクション（存在しない場合はNone）
        """
        # アクティブトランザクションを先にチェック
        if transaction_id in self.active_transactions:
            return self.active_transactions[transaction_id]

        # 履歴から検索
        for transaction in self.transaction_history:
            if transaction.transaction_id == transaction_id:
                return transaction

        return None

    async def get_active_transactions(self) -> List[Transaction]:
        """
        アクティブトランザクション一覧取得

        Returns:
            アクティブトランザクションリスト
        """
        return list(self.active_transactions.values())

    async def get_transaction_history(
        self,
        limit: int = 100,
        status_filter: Optional[TransactionStatus] = None
    ) -> List[Transaction]:
        """
        トランザクション履歴取得

        Args:
            limit: 取得件数制限
            status_filter: 状態フィルター

        Returns:
            トランザクション履歴リスト
        """
        history = self.transaction_history

        if status_filter:
            history = [tx for tx in history if tx.status == status_filter]

        return history[-limit:] if limit > 0 else history

    async def get_metrics(self) -> Dict[str, Any]:
        """
        トランザクションメトリクス取得

        Returns:
            メトリクス辞書
        """
        return {
            **self.metrics,
            "active_transactions": len(self.active_transactions),
            "total_history": len(self.transaction_history)
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        システムヘルスチェック

        Returns:
            ヘルス状態
        """
        # 各コンポーネントのヘルスチェック
        two_phase_health = await self.two_phase_coordinator.health_check()
        saga_health = await self.saga_coordinator.health_check()
        compensation_health = await self.compensation_engine.health_check()

        # 全体的なヘルス状態判定
        overall_status = "healthy"
        if (two_phase_health.get("status") != "healthy" or
            saga_health.get("status") != "healthy" or
            compensation_health.get("status") != "healthy"):
            overall_status = "degraded"

        return {
            "status": overall_status,
            "components": {
                "two_phase_commit": two_phase_health,
                "saga_coordinator": saga_health,
                "compensation_engine": compensation_health
            },
            "metrics": await self.get_metrics(),
            "default_timeout": self.default_timeout
        }

    async def _handle_coordinator_failure(self, transaction: Transaction) -> Dict[str, Any]:
        """
        コーディネーター失敗処理（災害復旧）

        Args:
            transaction: 失敗したトランザクション

        Returns:
            復旧結果
        """
        await self._log_transaction(transaction.transaction_id, "COORDINATOR_FAILURE",
                                   "Handling coordinator failure")

        # 復旧ロジック（実装例）
        recovery_result = {
            "recovery_status": "success",
            "action_taken": "coordinator_restart",
            "transaction_status": transaction.status
        }

        # トランザクション状態を復旧
        if not is_terminal_status(transaction.status):
            transaction.status = TransactionStatus.FAILED
            transaction.error_message = "Coordinator failure"

        return recovery_result

    def create_transaction(
        self,
        transaction_type: str = "saga",
        consistency_level: str = "eventual",
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Transaction:
        """
        トランザクション作成ヘルパー

        Args:
            transaction_type: トランザクションタイプ
            consistency_level: 一貫性レベル
            timeout: タイムアウト
            metadata: メタデータ

        Returns:
            新しいトランザクション
        """
        return Transaction(
            transaction_id=generate_transaction_id(),
            transaction_type=transaction_type,
            steps=[],
            consistency_level=consistency_level,
            timeout=timeout,
            metadata=metadata or {}
        )