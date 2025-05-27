# Phase 3 Task 3.4: 分散トランザクション管理システムの実装
# TDD GREEN段階: CompensationEngine実装

import asyncio
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timezone
from collections import defaultdict, deque

import structlog
from .models import (
    CompensationAction, CompensationResult, StepStatus,
    generate_compensation_id
)

logger = structlog.get_logger(__name__)


class CompensationEngine:
    """
    補償エンジン

    失敗したトランザクションの補償操作を管理・実行
    """

    def __init__(
        self,
        max_compensation_attempts: int = 3,
        compensation_delay: float = 1.0,
        deadlock_detection_timeout: float = 30.0
    ):
        """
        CompensationEngineの初期化

        Args:
            max_compensation_attempts: 最大補償試行回数
            compensation_delay: 補償試行間隔（秒）
            deadlock_detection_timeout: デッドロック検出タイムアウト（秒）
        """
        self.max_compensation_attempts = max_compensation_attempts
        self.compensation_delay = compensation_delay
        self.deadlock_detection_timeout = deadlock_detection_timeout
        self.active_compensations: Dict[str, CompensationAction] = {}
        self.compensation_logs: List[Dict[str, Any]] = []

    async def execute_compensations(
        self,
        compensation_actions: List[CompensationAction],
        services: Dict[str, Any]
    ) -> CompensationResult:
        """
        補償アクション群を実行

        Args:
            compensation_actions: 補償アクションリスト
            services: サービス辞書

        Returns:
            補償実行結果
        """
        start_time = time.time()

        result = CompensationResult(
            success=False,
            deadlock_detected=False
        )

        try:
            await self._log_compensation("COMPENSATION_START",
                                       f"Starting compensation for {len(compensation_actions)} actions")

            # デッドロック検出
            if self._detect_deadlock(compensation_actions):
                result.deadlock_detected = True
                await self._log_compensation("DEADLOCK_DETECTED",
                                           "Circular dependency detected in compensation actions")

                # デッドロック解決
                compensation_actions = await self._resolve_deadlock(compensation_actions)

            # 依存関係に基づく実行順序決定
            execution_order = self._calculate_execution_order(compensation_actions)

            # 補償実行
            for action in execution_order:
                self.active_compensations[action.action_id] = action

                success = await self._execute_single_compensation(action, services)

                if success:
                    result.successful_compensations.append(action.action_id)
                    await self._log_compensation("COMPENSATION_SUCCESS",
                                               f"Action {action.action_id} compensated successfully")
                else:
                    result.failed_compensations.append(action.action_id)
                    await self._log_compensation("COMPENSATION_FAILED",
                                               f"Action {action.action_id} compensation failed")

                self.active_compensations.pop(action.action_id, None)

            # 結果判定
            result.success = len(result.failed_compensations) == 0

            if result.success:
                await self._log_compensation("COMPENSATION_COMPLETED",
                                           f"All {len(compensation_actions)} compensations successful")
            else:
                await self._log_compensation("COMPENSATION_PARTIAL",
                                           f"{len(result.successful_compensations)}/{len(compensation_actions)} compensations successful")

        except Exception as e:
            result.error_message = str(e)
            await self._log_compensation("COMPENSATION_ERROR", f"Compensation engine error: {str(e)}")
            logger.error(f"Compensation execution failed: {e}")

        finally:
            result.execution_time = time.time() - start_time
            # アクティブ補償をクリア
            self.active_compensations.clear()

        return result

    def _detect_deadlock(self, actions: List[CompensationAction]) -> bool:
        """
        循環依存（デッドロック）を検出

        Args:
            actions: 補償アクションリスト

        Returns:
            デッドロックが検出されたかどうか
        """
        # 依存関係グラフを構築
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        action_ids = {action.action_id for action in actions}

        for action in actions:
            for dep in action.dependencies:
                if dep in action_ids:
                    graph[dep].append(action.action_id)
                    in_degree[action.action_id] += 1

        # トポロジカルソート（Kahn's algorithm）を実行
        queue = deque([action_id for action_id in action_ids if in_degree[action_id] == 0])
        processed = 0

        while queue:
            current = queue.popleft()
            processed += 1

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 全ノードが処理されない場合は循環依存
        return processed != len(action_ids)

    async def _resolve_deadlock(
        self,
        actions: List[CompensationAction]
    ) -> List[CompensationAction]:
        """
        デッドロックを解決

        Args:
            actions: 補償アクションリスト

        Returns:
            解決済み補償アクションリスト
        """
        await self._log_compensation("DEADLOCK_RESOLUTION", "Attempting to resolve circular dependencies")

        # 優先度ベースで依存関係を切断
        # 1. created_atが古いアクションの依存関係を優先
        # 2. リソースタイプに基づく優先度
        resolved_actions = []

        for action in sorted(actions, key=lambda a: a.created_at):
            # 循環依存を形成する依存関係を除去
            filtered_dependencies = []

            for dep in action.dependencies:
                # 依存先が自分より後に作成されたアクションの場合は除去
                dep_action = next((a for a in actions if a.action_id == dep), None)
                if dep_action and dep_action.created_at > action.created_at:
                    await self._log_compensation("DEPENDENCY_REMOVED",
                                               f"Removed dependency {dep} from {action.action_id} to resolve deadlock")
                    continue
                filtered_dependencies.append(dep)

            # 依存関係を更新
            action.dependencies = filtered_dependencies
            resolved_actions.append(action)

        await self._log_compensation("DEADLOCK_RESOLVED",
                                   f"Deadlock resolved, modified {len(actions)} actions")

        return resolved_actions

    def _calculate_execution_order(
        self,
        actions: List[CompensationAction]
    ) -> List[CompensationAction]:
        """
        依存関係に基づく実行順序を計算

        Args:
            actions: 補償アクションリスト

        Returns:
            実行順序でソートされたアクションリスト
        """
        # 依存関係グラフを構築
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        action_map = {action.action_id: action for action in actions}

        for action in actions:
            in_degree[action.action_id] = len(action.dependencies)
            for dep in action.dependencies:
                if dep in action_map:
                    graph[dep].append(action.action_id)

        # トポロジカルソート実行
        execution_order = []
        queue = deque([
            action for action in actions
            if in_degree[action.action_id] == 0
        ])

        while queue:
            current = queue.popleft()
            execution_order.append(current)

            for neighbor_id in graph[current.action_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(action_map[neighbor_id])

        return execution_order

    async def _execute_single_compensation(
        self,
        action: CompensationAction,
        services: Dict[str, Any]
    ) -> bool:
        """
        単一補償アクションを実行

        Args:
            action: 補償アクション
            services: サービス辞書

        Returns:
            実行成功フラグ
        """
        action.status = StepStatus.EXECUTING

        for attempt in range(self.max_compensation_attempts):
            action.current_attempt = attempt + 1

            try:
                # サービス取得
                service = services.get(action.resource_id)
                if not service:
                    raise ValueError(f"Service {action.resource_id} not found")

                # 補償操作実行
                result = await service.execute_compensation(action.operation, action.data)

                if result and result.get("status") == "success":
                    action.status = StepStatus.COMPENSATED
                    action.executed_at = datetime.now(timezone.utc)

                    await self._log_compensation("COMPENSATION_ATTEMPT_SUCCESS",
                                               f"Action {action.action_id} succeeded on attempt {attempt + 1}")
                    return True
                else:
                    await self._log_compensation("COMPENSATION_ATTEMPT_FAILED",
                                               f"Action {action.action_id} failed on attempt {attempt + 1}")

            except Exception as e:
                action.error_message = str(e)

                if attempt == self.max_compensation_attempts - 1:
                    action.status = StepStatus.FAILED
                    await self._log_compensation("COMPENSATION_EXHAUSTED",
                                               f"Action {action.action_id} failed after {self.max_compensation_attempts} attempts: {str(e)}")
                    return False
                else:
                    await self._log_compensation("COMPENSATION_RETRY",
                                               f"Action {action.action_id} attempt {attempt + 1} failed, retrying: {str(e)}")
                    await asyncio.sleep(self.compensation_delay * (attempt + 1))

        action.status = StepStatus.FAILED
        return False

    async def _log_compensation(self, operation: str, message: str) -> None:
        """
        補償ログ記録

        Args:
            operation: 操作名
            message: ログメッセージ
        """
        log_entry = {
            "operation": operation,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_compensations": len(self.active_compensations)
        }
        self.compensation_logs.append(log_entry)

        logger.info(message, operation=operation)

    async def get_compensation_status(self, action_id: str) -> Optional[CompensationAction]:
        """
        補償アクション状態取得

        Args:
            action_id: アクションID

        Returns:
            補償アクション
        """
        return self.active_compensations.get(action_id)

    async def get_compensation_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        補償ログ取得

        Args:
            limit: 取得件数制限

        Returns:
            ログエントリリスト
        """
        return self.compensation_logs[-limit:] if limit > 0 else self.compensation_logs

    async def health_check(self) -> Dict[str, Any]:
        """
        ヘルスチェック

        Returns:
            ヘルス状態
        """
        return {
            "status": "healthy",
            "active_compensations": len(self.active_compensations),
            "total_logs": len(self.compensation_logs),
            "max_compensation_attempts": self.max_compensation_attempts,
            "compensation_delay": self.compensation_delay,
            "deadlock_detection_timeout": self.deadlock_detection_timeout
        }

    def create_compensation_action(
        self,
        resource_id: str,
        operation: str,
        data: Dict[str, Any],
        dependencies: Optional[List[str]] = None
    ) -> CompensationAction:
        """
        補償アクション作成ヘルパー

        Args:
            resource_id: リソースID
            operation: 操作名
            data: 操作データ
            dependencies: 依存関係

        Returns:
            補償アクション
        """
        return CompensationAction(
            action_id=generate_compensation_id(),
            resource_id=resource_id,
            operation=operation,
            data=data,
            dependencies=dependencies or []
        )

    async def validate_compensation_plan(
        self,
        actions: List[CompensationAction]
    ) -> List[str]:
        """
        補償計画の検証

        Args:
            actions: 補償アクションリスト

        Returns:
            検証エラーリスト
        """
        errors = []

        # 基本検証
        if not actions:
            errors.append("No compensation actions provided")
            return errors

        action_ids = {action.action_id for action in actions}

        # 依存関係検証
        for action in actions:
            for dep in action.dependencies:
                if dep not in action_ids:
                    errors.append(f"Action {action.action_id} depends on non-existent action {dep}")

        # 循環依存検証
        if self._detect_deadlock(actions):
            errors.append("Circular dependency detected in compensation actions")

        # リソース検証
        resource_counts = defaultdict(int)
        for action in actions:
            resource_counts[action.resource_id] += 1

        # 同一リソースに対する補償が多すぎる場合は警告
        for resource_id, count in resource_counts.items():
            if count > 10:  # 閾値は調整可能
                errors.append(f"Too many compensation actions ({count}) for resource {resource_id}")

        return errors