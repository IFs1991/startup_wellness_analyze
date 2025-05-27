# Phase 3 Task 3.4: 分散トランザクション管理システムの実装
# TDD GREEN段階: TwoPhaseCommitCoordinator実装

import asyncio
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone

import structlog
from .models import (
    TransactionStatus, TwoPhaseCommitResult, ResourceManager,
    TransactionLog, generate_transaction_id
)

logger = structlog.get_logger(__name__)


class TwoPhaseCommitCoordinator:
    """
    2相コミットコーディネーター

    分散環境での厳密なACID特性を保証するトランザクション管理
    """

    def __init__(
        self,
        resource_managers: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        TwoPhaseCommitCoordinatorの初期化

        Args:
            resource_managers: リソースマネージャーの辞書
            timeout: タイムアウト時間（秒）
            max_retries: 最大リトライ回数
            retry_delay: リトライ間隔（秒）
        """
        self.resource_managers = resource_managers or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.active_transactions: Dict[str, Dict[str, Any]] = {}
        self.transaction_logs: List[TransactionLog] = []

    async def execute_transaction(
        self,
        transaction_id: str,
        operations: Dict[str, Dict[str, Any]]
    ) -> TwoPhaseCommitResult:
        """
        2相コミットトランザクションを実行

        Args:
            transaction_id: トランザクションID
            operations: リソース毎の操作定義

        Returns:
            実行結果
        """
        start_time = time.time()

        result = TwoPhaseCommitResult(
            transaction_id=transaction_id,
            status=TransactionStatus.PENDING
        )

        try:
            # トランザクション開始ログ
            await self._log_transaction(transaction_id, "START", "coordinator",
                                      f"Starting 2PC transaction with {len(operations)} resources")

            # Phase 1: Prepare
            result.status = TransactionStatus.PREPARING
            prepare_start = time.time()

            prepare_success = await self._prepare_phase(transaction_id, operations, result)
            result.prepare_time = time.time() - prepare_start

            if not prepare_success:
                # Prepare失敗 - 全リソースをabort
                result.status = TransactionStatus.ABORTING
                await self._abort_phase(transaction_id, operations, result)
                result.status = TransactionStatus.ABORTED
                await self._log_transaction(transaction_id, "ABORTED", "coordinator",
                                          "Transaction aborted due to prepare failure")
            else:
                # Phase 2: Commit
                result.status = TransactionStatus.COMMITTING
                commit_start = time.time()

                commit_success = await self._commit_phase(transaction_id, operations, result)
                result.commit_time = time.time() - commit_start

                if commit_success:
                    result.status = TransactionStatus.COMMITTED
                    await self._log_transaction(transaction_id, "COMMITTED", "coordinator",
                                              "Transaction committed successfully")
                else:
                    result.status = TransactionStatus.PARTIALLY_COMMITTED
                    await self._log_transaction(transaction_id, "PARTIAL_COMMIT", "coordinator",
                                              "Transaction partially committed")

        except asyncio.TimeoutError:
            result.status = TransactionStatus.ABORTED
            result.error_message = "Transaction timeout"
            await self._log_transaction(transaction_id, "TIMEOUT", "coordinator",
                                      f"Transaction timeout after {self.timeout}s")

        except Exception as e:
            result.status = TransactionStatus.FAILED
            result.error_message = str(e)
            await self._log_transaction(transaction_id, "ERROR", "coordinator",
                                      f"Transaction failed: {str(e)}")
            logger.error(f"2PC transaction failed: {e}", transaction_id=transaction_id)

        finally:
            result.execution_time = time.time() - start_time
            # アクティブトランザクションから削除
            self.active_transactions.pop(transaction_id, None)

        return result

    async def _prepare_phase(
        self,
        transaction_id: str,
        operations: Dict[str, Dict[str, Any]],
        result: TwoPhaseCommitResult
    ) -> bool:
        """
        Prepare段階の実行

        Args:
            transaction_id: トランザクションID
            operations: 操作定義
            result: 結果オブジェクト

        Returns:
            全リソースでprepareが成功したかどうか
        """
        prepare_tasks = []
        resource_timeouts = {}

        for resource_id, operation in operations.items():
            if resource_id not in self.resource_managers:
                logger.warning(f"Resource manager {resource_id} not found")
                result.failed_resources.append(resource_id)
                continue

            rm = self.resource_managers[resource_id]
            task = asyncio.create_task(
                self._prepare_resource(transaction_id, resource_id, operation, rm),
                name=f"prepare_{resource_id}"
            )
            prepare_tasks.append(task)
            resource_timeouts[task] = resource_id

        if not prepare_tasks:
            return False

        try:
            # タイムアウト付きで全prepare操作を実行
            done, pending = await asyncio.wait(
                prepare_tasks,
                timeout=self.timeout,
                return_when=asyncio.ALL_COMPLETED
            )

            # タイムアウトしたタスクをキャンセル
            for task in pending:
                task.cancel()
                resource_id = resource_timeouts.get(task, "unknown")
                result.failed_resources.append(resource_id)
                await self._log_transaction(transaction_id, "PREPARE_TIMEOUT", resource_id,
                                          "Prepare operation timed out")

            # 完了したタスクの結果を確認
            all_prepared = True
            for task in done:
                resource_id = resource_timeouts.get(task, "unknown")
                try:
                    success = await task
                    if success:
                        result.prepared_resources.append(resource_id)
                        await self._log_transaction(transaction_id, "PREPARED", resource_id,
                                                  "Resource prepared successfully")
                    else:
                        result.failed_resources.append(resource_id)
                        all_prepared = False
                        await self._log_transaction(transaction_id, "PREPARE_FAILED", resource_id,
                                                  "Resource prepare failed")
                except Exception as e:
                    result.failed_resources.append(resource_id)
                    all_prepared = False
                    await self._log_transaction(transaction_id, "PREPARE_ERROR", resource_id,
                                              f"Prepare error: {str(e)}")

            return all_prepared and len(pending) == 0

        except Exception as e:
            logger.error(f"Prepare phase error: {e}")
            return False

    async def _prepare_resource(
        self,
        transaction_id: str,
        resource_id: str,
        operation: Dict[str, Any],
        resource_manager: Any
    ) -> bool:
        """
        個別リソースのprepare操作

        Args:
            transaction_id: トランザクションID
            resource_id: リソースID
            operation: 操作定義
            resource_manager: リソースマネージャー

        Returns:
            prepare成功フラグ
        """
        for attempt in range(self.max_retries):
            try:
                success = await resource_manager.prepare(transaction_id, operation)
                return success
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Prepare failed for {resource_id} after {self.max_retries} attempts: {e}")
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        return False

    async def _commit_phase(
        self,
        transaction_id: str,
        operations: Dict[str, Dict[str, Any]],
        result: TwoPhaseCommitResult
    ) -> bool:
        """
        Commit段階の実行

        Args:
            transaction_id: トランザクションID
            operations: 操作定義
            result: 結果オブジェクト

        Returns:
            全リソースでcommitが成功したかどうか
        """
        commit_tasks = []
        resource_timeouts = {}

        # prepareが成功したリソースのみcommit
        for resource_id in result.prepared_resources:
            if resource_id not in self.resource_managers:
                continue

            rm = self.resource_managers[resource_id]
            task = asyncio.create_task(
                self._commit_resource(transaction_id, resource_id, rm),
                name=f"commit_{resource_id}"
            )
            commit_tasks.append(task)
            resource_timeouts[task] = resource_id

        if not commit_tasks:
            return False

        try:
            # タイムアウト付きで全commit操作を実行
            done, pending = await asyncio.wait(
                commit_tasks,
                timeout=self.timeout,
                return_when=asyncio.ALL_COMPLETED
            )

            # タイムアウトしたタスクをキャンセル
            for task in pending:
                task.cancel()
                resource_id = resource_timeouts.get(task, "unknown")
                await self._log_transaction(transaction_id, "COMMIT_TIMEOUT", resource_id,
                                          "Commit operation timed out")

            # 完了したタスクの結果を確認
            all_committed = True
            for task in done:
                resource_id = resource_timeouts.get(task, "unknown")
                try:
                    success = await task
                    if success:
                        result.committed_resources.append(resource_id)
                        await self._log_transaction(transaction_id, "COMMITTED", resource_id,
                                                  "Resource committed successfully")
                    else:
                        all_committed = False
                        await self._log_transaction(transaction_id, "COMMIT_FAILED", resource_id,
                                                  "Resource commit failed")
                except Exception as e:
                    all_committed = False
                    await self._log_transaction(transaction_id, "COMMIT_ERROR", resource_id,
                                              f"Commit error: {str(e)}")

            return all_committed and len(pending) == 0

        except Exception as e:
            logger.error(f"Commit phase error: {e}")
            return False

    async def _commit_resource(
        self,
        transaction_id: str,
        resource_id: str,
        resource_manager: Any
    ) -> bool:
        """
        個別リソースのcommit操作

        Args:
            transaction_id: トランザクションID
            resource_id: リソースID
            resource_manager: リソースマネージャー

        Returns:
            commit成功フラグ
        """
        for attempt in range(self.max_retries):
            try:
                success = await resource_manager.commit(transaction_id)
                return success
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Commit failed for {resource_id} after {self.max_retries} attempts: {e}")
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        return False

    async def _abort_phase(
        self,
        transaction_id: str,
        operations: Dict[str, Dict[str, Any]],
        result: TwoPhaseCommitResult
    ) -> None:
        """
        Abort段階の実行

        Args:
            transaction_id: トランザクションID
            operations: 操作定義
            result: 結果オブジェクト
        """
        abort_tasks = []

        # 全リソースに対してabort操作を実行
        for resource_id in operations.keys():
            if resource_id not in self.resource_managers:
                continue

            rm = self.resource_managers[resource_id]
            task = asyncio.create_task(
                self._abort_resource(transaction_id, resource_id, rm),
                name=f"abort_{resource_id}"
            )
            abort_tasks.append(task)

        if abort_tasks:
            try:
                # タイムアウト付きで全abort操作を実行
                await asyncio.wait(abort_tasks, timeout=self.timeout)

                for task in abort_tasks:
                    if not task.done():
                        task.cancel()

            except Exception as e:
                logger.error(f"Abort phase error: {e}")

    async def _abort_resource(
        self,
        transaction_id: str,
        resource_id: str,
        resource_manager: Any
    ) -> bool:
        """
        個別リソースのabort操作

        Args:
            transaction_id: トランザクションID
            resource_id: リソースID
            resource_manager: リソースマネージャー

        Returns:
            abort成功フラグ
        """
        try:
            success = await resource_manager.abort(transaction_id)
            if success:
                await self._log_transaction(transaction_id, "ABORTED", resource_id,
                                          "Resource aborted successfully")
            else:
                await self._log_transaction(transaction_id, "ABORT_FAILED", resource_id,
                                          "Resource abort failed")
            return success
        except Exception as e:
            await self._log_transaction(transaction_id, "ABORT_ERROR", resource_id,
                                      f"Abort error: {str(e)}")
            logger.error(f"Abort failed for {resource_id}: {e}")
            return False

    async def _log_transaction(
        self,
        transaction_id: str,
        operation: str,
        resource_id: Optional[str],
        message: str,
        status: str = "info"
    ) -> None:
        """
        トランザクションログ記録

        Args:
            transaction_id: トランザクションID
            operation: 操作名
            resource_id: リソースID
            message: ログメッセージ
            status: ログレベル
        """
        log_entry = TransactionLog(
            log_id=f"log_{len(self.transaction_logs) + 1:06d}",
            transaction_id=transaction_id,
            operation=operation,
            resource_id=resource_id,
            status=status,
            message=message
        )
        self.transaction_logs.append(log_entry)

        # 構造化ログ出力
        logger.info(
            message,
            transaction_id=transaction_id,
            operation=operation,
            resource_id=resource_id,
            status=status
        )

    async def get_transaction_status(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """
        トランザクション状態取得

        Args:
            transaction_id: トランザクションID

        Returns:
            トランザクション状態
        """
        return self.active_transactions.get(transaction_id)

    async def get_transaction_logs(
        self,
        transaction_id: Optional[str] = None,
        limit: int = 100
    ) -> List[TransactionLog]:
        """
        トランザクションログ取得

        Args:
            transaction_id: フィルター用トランザクションID
            limit: 取得件数制限

        Returns:
            ログエントリリスト
        """
        logs = self.transaction_logs

        if transaction_id:
            logs = [log for log in logs if log.transaction_id == transaction_id]

        return logs[-limit:] if limit > 0 else logs

    async def health_check(self) -> Dict[str, Any]:
        """
        ヘルスチェック

        Returns:
            ヘルス状態
        """
        healthy_resources = 0
        total_resources = len(self.resource_managers)

        for resource_id, rm in self.resource_managers.items():
            try:
                status = await rm.status()
                if status == "ready":
                    healthy_resources += 1
            except Exception as e:
                logger.warning(f"Health check failed for {resource_id}: {e}")

        return {
            "status": "healthy" if healthy_resources == total_resources else "degraded",
            "healthy_resources": healthy_resources,
            "total_resources": total_resources,
            "active_transactions": len(self.active_transactions),
            "total_logs": len(self.transaction_logs)
        }