# Phase 3 Task 3.4: 分散トランザクション管理システムの実装
# TDD GREEN段階: SagaCoordinator実装

import asyncio
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone

import structlog
from .models import (
    TransactionStatus, SagaTransaction, SagaResult, TransactionStep,
    StepStatus, CompensationAction, generate_saga_id, generate_compensation_id
)

logger = structlog.get_logger(__name__)


class SagaCoordinator:
    """
    Sagaトランザクションコーディネーター

    長時間実行される分散トランザクションを補償ベースで管理
    """

    def __init__(
        self,
        max_retry_attempts: int = 3,
        retry_delay: float = 1.0,
        parallel_execution: bool = False,
        timeout: Optional[float] = None
    ):
        """
        SagaCoordinatorの初期化

        Args:
            max_retry_attempts: 最大リトライ回数
            retry_delay: リトライ間隔（秒）
            parallel_execution: 並列実行フラグ
            timeout: タイムアウト時間（秒）
        """
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay = retry_delay
        self.parallel_execution = parallel_execution
        self.timeout = timeout
        self.active_sagas: Dict[str, SagaTransaction] = {}
        self.saga_logs: List[Dict[str, Any]] = []

    async def execute_saga(self, saga_transaction: SagaTransaction) -> SagaResult:
        """
        Sagaトランザクションを実行

        Args:
            saga_transaction: Sagaトランザクション

        Returns:
            実行結果
        """
        start_time = time.time()
        saga_transaction.started_at = datetime.now(timezone.utc)

        # アクティブSagaに追加
        self.active_sagas[saga_transaction.saga_id] = saga_transaction

        try:
            await self._log_saga(saga_transaction.saga_id, "START",
                               f"Starting Saga with {len(saga_transaction.steps)} steps")

            if saga_transaction.parallel_execution or self.parallel_execution:
                result = await self._execute_parallel_saga(saga_transaction)
            else:
                result = await self._execute_sequential_saga(saga_transaction)

            result.execution_time = time.time() - start_time
            saga_transaction.completed_at = datetime.now(timezone.utc)

            return result

        except Exception as e:
            error_message = f"Saga execution failed: {str(e)}"
            await self._log_saga(saga_transaction.saga_id, "ERROR", error_message)
            logger.error(error_message, saga_id=saga_transaction.saga_id)

            # 失敗時の補償実行
            await self._compensate_saga(saga_transaction)

            return SagaResult(
                saga_id=saga_transaction.saga_id,
                status=TransactionStatus.COMPENSATED,
                execution_time=time.time() - start_time,
                error_message=error_message
            )
        finally:
            # アクティブSagaから削除
            self.active_sagas.pop(saga_transaction.saga_id, None)

    async def _execute_sequential_saga(self, saga_transaction: SagaTransaction) -> SagaResult:
        """
        順次実行Saga

        Args:
            saga_transaction: Sagaトランザクション

        Returns:
            実行結果
        """
        result = SagaResult(
            saga_id=saga_transaction.saga_id,
            status=TransactionStatus.PENDING,
            total_steps=len(saga_transaction.steps)
        )

        for i, step in enumerate(saga_transaction.steps):
            saga_transaction.current_step_index = i

            # 依存関係チェック
            if not await self._check_dependencies(step, saga_transaction):
                await self._log_saga(saga_transaction.saga_id, "DEPENDENCY_FAILED",
                                   f"Step {step.step_id} dependencies not satisfied")
                step.status = StepStatus.SKIPPED
                continue

            step.status = StepStatus.EXECUTING
            step.started_at = datetime.now(timezone.utc)

            try:
                success = await self._execute_step_with_retry(step, saga_transaction)

                if success:
                    step.status = StepStatus.COMPLETED
                    step.completed_at = datetime.now(timezone.utc)
                    saga_transaction.completed_steps.append(step.step_id)
                    result.completed_steps.append(step.step_id)
                    result.successful_steps += 1

                    await self._log_saga(saga_transaction.saga_id, "STEP_COMPLETED",
                                       f"Step {step.step_id} completed successfully")
                else:
                    step.status = StepStatus.FAILED
                    saga_transaction.failed_steps.append(step.step_id)
                    result.failed_steps.append(step.step_id)

                    await self._log_saga(saga_transaction.saga_id, "STEP_FAILED",
                                       f"Step {step.step_id} failed")

                    # 失敗時は即座に補償を開始
                    await self._compensate_saga(saga_transaction)
                    result.status = TransactionStatus.COMPENSATED
                    return result

            except Exception as e:
                step.status = StepStatus.FAILED
                step.error_message = str(e)
                saga_transaction.failed_steps.append(step.step_id)
                result.failed_steps.append(step.step_id)

                await self._log_saga(saga_transaction.saga_id, "STEP_ERROR",
                                   f"Step {step.step_id} error: {str(e)}")

                # 例外時も補償を開始
                await self._compensate_saga(saga_transaction)
                result.status = TransactionStatus.COMPENSATED
                return result

        # 全ステップ成功
        result.status = TransactionStatus.COMMITTED
        saga_transaction.status = TransactionStatus.COMMITTED
        await self._log_saga(saga_transaction.saga_id, "COMMITTED", "All steps completed successfully")

        return result

    async def _execute_parallel_saga(self, saga_transaction: SagaTransaction) -> SagaResult:
        """
        並列実行Saga

        Args:
            saga_transaction: Sagaトランザクション

        Returns:
            実行結果
        """
        result = SagaResult(
            saga_id=saga_transaction.saga_id,
            status=TransactionStatus.PENDING,
            total_steps=len(saga_transaction.steps)
        )

        # 並列実行可能なステップを抽出
        parallel_steps = [step for step in saga_transaction.steps if step.can_run_parallel]
        sequential_steps = [step for step in saga_transaction.steps if not step.can_run_parallel]

        # 並列ステップの実行
        if parallel_steps:
            parallel_results = await self._execute_parallel_steps(parallel_steps, saga_transaction)

            for step_id, success in parallel_results.items():
                if success:
                    saga_transaction.completed_steps.append(step_id)
                    result.completed_steps.append(step_id)
                    result.successful_steps += 1
                else:
                    saga_transaction.failed_steps.append(step_id)
                    result.failed_steps.append(step_id)

        # 順次ステップの実行
        for step in sequential_steps:
            if not await self._check_dependencies(step, saga_transaction):
                step.status = StepStatus.SKIPPED
                continue

            step.status = StepStatus.EXECUTING
            step.started_at = datetime.now(timezone.utc)

            try:
                success = await self._execute_step_with_retry(step, saga_transaction)

                if success:
                    step.status = StepStatus.COMPLETED
                    step.completed_at = datetime.now(timezone.utc)
                    saga_transaction.completed_steps.append(step.step_id)
                    result.completed_steps.append(step.step_id)
                    result.successful_steps += 1
                else:
                    step.status = StepStatus.FAILED
                    saga_transaction.failed_steps.append(step.step_id)
                    result.failed_steps.append(step.step_id)

            except Exception as e:
                step.status = StepStatus.FAILED
                step.error_message = str(e)
                saga_transaction.failed_steps.append(step.step_id)
                result.failed_steps.append(step.step_id)

        # 結果判定
        if result.failed_steps:
            await self._compensate_saga(saga_transaction)
            result.status = TransactionStatus.COMPENSATED
        else:
            result.status = TransactionStatus.COMMITTED
            saga_transaction.status = TransactionStatus.COMMITTED

        return result

    async def _execute_parallel_steps(
        self,
        steps: List[TransactionStep],
        saga_transaction: SagaTransaction
    ) -> Dict[str, bool]:
        """
        並列ステップ実行

        Args:
            steps: 並列実行するステップリスト
            saga_transaction: Sagaトランザクション

        Returns:
            ステップID -> 成功フラグの辞書
        """
        tasks = []
        step_map = {}

        for step in steps:
            step.status = StepStatus.EXECUTING
            step.started_at = datetime.now(timezone.utc)

            task = asyncio.create_task(
                self._execute_step_with_retry(step, saga_transaction),
                name=f"step_{step.step_id}"
            )
            tasks.append(task)
            step_map[task] = step

        results = {}

        try:
            if self.timeout:
                done, pending = await asyncio.wait(tasks, timeout=self.timeout)

                # タイムアウトしたタスクをキャンセル
                for task in pending:
                    task.cancel()
                    step = step_map[task]
                    step.status = StepStatus.FAILED
                    step.error_message = "Step execution timeout"
                    results[step.step_id] = False
            else:
                done = tasks

            # 完了したタスクの結果を収集
            for task in done:
                step = step_map[task]
                try:
                    success = await task
                    if success:
                        step.status = StepStatus.COMPLETED
                        step.completed_at = datetime.now(timezone.utc)
                    else:
                        step.status = StepStatus.FAILED
                    results[step.step_id] = success

                except Exception as e:
                    step.status = StepStatus.FAILED
                    step.error_message = str(e)
                    results[step.step_id] = False

        except Exception as e:
            logger.error(f"Parallel execution error: {e}")
            for step in steps:
                if step.step_id not in results:
                    step.status = StepStatus.FAILED
                    step.error_message = str(e)
                    results[step.step_id] = False

        return results

    async def _execute_step_with_retry(
        self,
        step: TransactionStep,
        saga_transaction: SagaTransaction
    ) -> bool:
        """
        リトライ付きステップ実行

        Args:
            step: 実行するステップ
            saga_transaction: Sagaトランザクション

        Returns:
            実行成功フラグ
        """
        max_attempts = min(step.max_retry_attempts, self.max_retry_attempts)

        for attempt in range(max_attempts):
            try:
                step.retry_attempts = attempt

                # サービス取得
                service = saga_transaction.services.get(step.resource_id)
                if not service:
                    raise ValueError(f"Service {step.resource_id} not found")

                # ステップ実行
                result = await service.execute_operation(step.operation, step.data)

                if result and result.get("status") == "success":
                    await self._log_saga(saga_transaction.saga_id, "STEP_SUCCESS",
                                       f"Step {step.step_id} succeeded on attempt {attempt + 1}")
                    return True

            except Exception as e:
                step.error_message = str(e)

                if attempt == max_attempts - 1:
                    await self._log_saga(saga_transaction.saga_id, "STEP_RETRY_EXHAUSTED",
                                       f"Step {step.step_id} failed after {max_attempts} attempts: {str(e)}")
                    return False
                else:
                    await self._log_saga(saga_transaction.saga_id, "STEP_RETRY",
                                       f"Step {step.step_id} attempt {attempt + 1} failed, retrying: {str(e)}")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        return False

    async def _check_dependencies(
        self,
        step: TransactionStep,
        saga_transaction: SagaTransaction
    ) -> bool:
        """
        ステップ依存関係チェック

        Args:
            step: チェックするステップ
            saga_transaction: Sagaトランザクション

        Returns:
            依存関係が満たされているかどうか
        """
        if not step.dependencies:
            return True

        for dep_step_id in step.dependencies:
            if dep_step_id not in saga_transaction.completed_steps:
                return False

        return True

    async def _compensate_saga(self, saga_transaction: SagaTransaction) -> None:
        """
        Saga補償実行

        Args:
            saga_transaction: Sagaトランザクション
        """
        saga_transaction.status = TransactionStatus.COMPENSATING
        await self._log_saga(saga_transaction.saga_id, "COMPENSATING",
                           f"Starting compensation for {len(saga_transaction.completed_steps)} completed steps")

        # 完了したステップを逆順で補償
        compensation_steps = []
        for step_id in reversed(saga_transaction.completed_steps):
            step = next((s for s in saga_transaction.steps if s.step_id == step_id), None)
            if step and step.compensation_data:
                compensation_steps.append(step)

        for step in compensation_steps:
            try:
                step.status = StepStatus.COMPENSATING

                # 補償操作実行
                service = saga_transaction.services.get(step.resource_id)
                if service:
                    await service.compensate_operation(step.operation, step.compensation_data)
                    step.status = StepStatus.COMPENSATED
                    saga_transaction.compensated_steps.append(step.step_id)

                    await self._log_saga(saga_transaction.saga_id, "STEP_COMPENSATED",
                                       f"Step {step.step_id} compensated successfully")
                else:
                    await self._log_saga(saga_transaction.saga_id, "COMPENSATION_FAILED",
                                       f"Service {step.resource_id} not available for compensation")

            except Exception as e:
                await self._log_saga(saga_transaction.saga_id, "COMPENSATION_ERROR",
                                   f"Compensation failed for step {step.step_id}: {str(e)}")
                logger.error(f"Compensation error for step {step.step_id}: {e}")

        saga_transaction.status = TransactionStatus.COMPENSATED
        await self._log_saga(saga_transaction.saga_id, "COMPENSATED",
                           f"Compensation completed for {len(saga_transaction.compensated_steps)} steps")

    async def _log_saga(self, saga_id: str, operation: str, message: str) -> None:
        """
        Sagaログ記録

        Args:
            saga_id: SagaID
            operation: 操作名
            message: ログメッセージ
        """
        log_entry = {
            "saga_id": saga_id,
            "operation": operation,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.saga_logs.append(log_entry)

        logger.info(message, saga_id=saga_id, operation=operation)

    async def get_saga_status(self, saga_id: str) -> Optional[SagaTransaction]:
        """
        Saga状態取得

        Args:
            saga_id: SagaID

        Returns:
            Sagaトランザクション
        """
        return self.active_sagas.get(saga_id)

    async def get_saga_logs(
        self,
        saga_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Sagaログ取得

        Args:
            saga_id: フィルター用SagaID
            limit: 取得件数制限

        Returns:
            ログエントリリスト
        """
        logs = self.saga_logs

        if saga_id:
            logs = [log for log in logs if log["saga_id"] == saga_id]

        return logs[-limit:] if limit > 0 else logs

    async def health_check(self) -> Dict[str, Any]:
        """
        ヘルスチェック

        Returns:
            ヘルス状態
        """
        active_count = len(self.active_sagas)

        # アクティブSagaの状態集計
        status_counts = {}
        for saga in self.active_sagas.values():
            status = saga.status.name
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "status": "healthy",
            "active_sagas": active_count,
            "saga_status_counts": status_counts,
            "total_logs": len(self.saga_logs),
            "max_retry_attempts": self.max_retry_attempts,
            "parallel_execution": self.parallel_execution
        }