"""
フェイルオーバー調整器
Task 4.2: 自動フェイルオーバー機構
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from structlog import get_logger

from .models import (
    ClusterState, NodeState, FailoverConfiguration,
    FailoverStatus, NodeRole, FailoverEvent
)
from .primary_failure_detector import PrimaryFailureDetector
from .auto_failover_manager import AutoFailoverManager, FailoverResult
from .data_consistency_checker import DataConsistencyChecker


logger = get_logger(__name__)


@dataclass
class WorkflowResult:
    """フェイルオーバーワークフロー結果"""
    success: bool
    new_primary_id: Optional[str] = None
    total_execution_time: float = 0.0
    data_consistency_verified: bool = False
    kubernetes_resources_updated: bool = False
    error_message: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class FailoverCoordinator:
    """フェイルオーバー調整システム"""

    def __init__(self, config: FailoverConfiguration):
        """
        Args:
            config: フェイルオーバー設定
        """
        self.config = config

        # 各コンポーネントの初期化
        self.failure_detector = PrimaryFailureDetector(config)
        self.failover_manager = AutoFailoverManager(config)
        self.consistency_checker = DataConsistencyChecker(config)

        # 内部状態管理
        self.workflow_lock = asyncio.Lock()
        self.active_workflows: Dict[str, asyncio.Task] = {}
        self.event_history: List[FailoverEvent] = []
        self.coordination_stats = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_execution_time": 0.0
        }

        logger.info("FailoverCoordinator initialized",
                   automatic_failover=config.enable_automatic_failover,
                   consistency_check=config.consistency_check_enabled)

    async def execute_failover_workflow(self, cluster_state: ClusterState) -> WorkflowResult:
        """
        エンドツーエンドフェイルオーバーワークフローを実行

        Args:
            cluster_state: クラスター状態

        Returns:
            WorkflowResult: ワークフロー実行結果
        """
        start_time = time.time()
        cluster_id = cluster_state.cluster_id
        workflow_id = str(uuid.uuid4())

        # 同時フェイルオーバー防止
        async with self.workflow_lock:
            logger.info("Starting failover workflow",
                       cluster_id=cluster_id,
                       workflow_id=workflow_id)

            try:
                self.coordination_stats["total_workflows"] += 1

                # Phase 1: 障害検知の再確認
                phase1_result = await self._phase1_failure_reconfirmation(cluster_state)
                if not phase1_result:
                    return WorkflowResult(
                        success=False,
                        error_message="Failure reconfirmation failed",
                        total_execution_time=time.time() - start_time
                    )

                # Phase 2: フェイルオーバー実行
                phase2_result = await self._phase2_execute_failover(cluster_state)
                if not phase2_result.success:
                    return WorkflowResult(
                        success=False,
                        error_message=phase2_result.error_message,
                        total_execution_time=time.time() - start_time
                    )

                # Phase 3: データ整合性検証
                phase3_result = await self._phase3_verify_consistency(cluster_state)

                # Phase 4: 最終化とクリーンアップ
                phase4_result = await self._phase4_finalization(cluster_state)

                total_execution_time = time.time() - start_time

                # 成功統計の更新
                self.coordination_stats["successful_workflows"] += 1
                self._update_execution_time_stats(total_execution_time)

                # ワークフロー完了イベント記録
                completion_event = FailoverEvent(
                    event_id=str(uuid.uuid4()),
                    event_type="workflow_completed",
                    source_node=cluster_id,
                    target_node=phase2_result.new_primary_id,
                    status=FailoverStatus.FAILED_OVER,
                    details={
                        "workflow_id": workflow_id,
                        "execution_time": total_execution_time,
                        "data_consistency_verified": phase3_result,
                        "finalization_completed": phase4_result
                    }
                )
                await self._record_event(completion_event)

                logger.info("Failover workflow completed successfully",
                           cluster_id=cluster_id,
                           workflow_id=workflow_id,
                           new_primary=phase2_result.new_primary_id,
                           total_execution_time=total_execution_time)

                return WorkflowResult(
                    success=True,
                    new_primary_id=phase2_result.new_primary_id,
                    total_execution_time=total_execution_time,
                    data_consistency_verified=phase3_result,
                    kubernetes_resources_updated=phase4_result
                )

            except Exception as e:
                self.coordination_stats["failed_workflows"] += 1

                logger.error("Failover workflow failed",
                           cluster_id=cluster_id,
                           workflow_id=workflow_id,
                           error=str(e))

                # 失敗イベント記録
                failure_event = FailoverEvent(
                    event_id=str(uuid.uuid4()),
                    event_type="workflow_failed",
                    source_node=cluster_id,
                    status=FailoverStatus.FAILED,
                    error_message=str(e),
                    details={"workflow_id": workflow_id}
                )
                await self._record_event(failure_event)

                return WorkflowResult(
                    success=False,
                    error_message=str(e),
                    total_execution_time=time.time() - start_time
                )

    async def _phase1_failure_reconfirmation(self, cluster_state: ClusterState) -> bool:
        """
        Phase 1: 障害の再確認

        Args:
            cluster_state: クラスター状態

        Returns:
            bool: 障害が確認された場合True
        """
        logger.info("Phase 1: Failure reconfirmation",
                   cluster_id=cluster_state.cluster_id)

        try:
            # 障害検知の再実行
            failure_confirmed = await self.failure_detector.detect_primary_failure(cluster_state)

            if failure_confirmed:
                # 障害確認イベント記録
                event = FailoverEvent(
                    event_id=str(uuid.uuid4()),
                    event_type="primary_failure",
                    source_node=cluster_state.primary_node or "unknown",
                    status=FailoverStatus.FAILED,
                    details={"phase": "reconfirmation"}
                )
                await self._record_event(event)

                logger.info("Primary failure reconfirmed",
                           cluster_id=cluster_state.cluster_id)
                return True
            else:
                logger.info("Primary failure not confirmed, aborting workflow",
                           cluster_id=cluster_state.cluster_id)
                return False

        except Exception as e:
            logger.error("Phase 1 failed", error=str(e))
            return False

    async def _phase2_execute_failover(self, cluster_state: ClusterState) -> FailoverResult:
        """
        Phase 2: フェイルオーバー実行

        Args:
            cluster_state: クラスター状態

        Returns:
            FailoverResult: フェイルオーバー結果
        """
        logger.info("Phase 2: Execute failover",
                   cluster_id=cluster_state.cluster_id)

        try:
            # フェイルオーバー実行
            result = await self.failover_manager.execute_failover_with_retry(cluster_state)

            if result.success:
                # フェイルオーバー成功イベント記録
                event = FailoverEvent(
                    event_id=str(uuid.uuid4()),
                    event_type="automatic_failover",
                    source_node=cluster_state.cluster_id,
                    target_node=result.new_primary_id,
                    status=FailoverStatus.FAILED_OVER,
                    details={
                        "execution_time": result.execution_time,
                        "retry_count": result.retry_count
                    }
                )
                await self._record_event(event)

                logger.info("Phase 2 completed successfully",
                           new_primary=result.new_primary_id,
                           execution_time=result.execution_time)
            else:
                logger.error("Phase 2 failed", error=result.error_message)

            return result

        except Exception as e:
            logger.error("Phase 2 exception", error=str(e))
            return FailoverResult(
                success=False,
                error_message=str(e)
            )

    async def _phase3_verify_consistency(self, cluster_state: ClusterState) -> bool:
        """
        Phase 3: データ整合性検証

        Args:
            cluster_state: クラスター状態

        Returns:
            bool: 整合性が確認された場合True
        """
        logger.info("Phase 3: Verify data consistency",
                   cluster_id=cluster_state.cluster_id)

        try:
            # データ整合性チェック
            consistency_result = await self.consistency_checker.check_post_failover_consistency(cluster_state)

            if not consistency_result.is_consistent:
                logger.warning("Data inconsistency detected, attempting synchronization",
                              inconsistency_count=consistency_result.inconsistency_count)

                # データ同期試行
                sync_result = await self.consistency_checker.synchronize_cluster_data(cluster_state)

                if sync_result.success:
                    logger.info("Data synchronization successful",
                               synchronized_records=sync_result.synchronized_records)

                    # 再チェック
                    recheck_result = await self.consistency_checker.check_post_failover_consistency(cluster_state)
                    return recheck_result.is_consistent
                else:
                    logger.error("Data synchronization failed",
                                error=sync_result.error_message)
                    return False

            logger.info("Phase 3 completed successfully",
                       check_duration=consistency_result.check_duration)
            return True

        except Exception as e:
            logger.error("Phase 3 failed", error=str(e))
            return False

    async def _phase4_finalization(self, cluster_state: ClusterState) -> bool:
        """
        Phase 4: 最終化とクリーンアップ

        Args:
            cluster_state: クラスター状態

        Returns:
            bool: 最終化が成功した場合True
        """
        logger.info("Phase 4: Finalization and cleanup",
                   cluster_id=cluster_state.cluster_id)

        try:
            # クラスター状態の最終更新
            cluster_state.status = FailoverStatus.HEALTHY

            # 統計情報の更新
            primary_node = cluster_state.get_primary_node()
            if primary_node:
                primary_node.last_heartbeat = datetime.now()
                primary_node.health_score = 1.0

            # Kubernetesリソース更新（オプション）
            k8s_updated = False
            try:
                if primary_node:
                    k8s_result = await self.failover_manager.execute_kubernetes_failover(cluster_state)
                    k8s_updated = getattr(k8s_result, 'kubernetes_resources_updated', False)
            except Exception as k8s_error:
                logger.warning("Kubernetes resources update failed",
                              error=str(k8s_error))

            # 検知履歴のクリーンアップ
            await self.failure_detector.reset_detection_history()

            logger.info("Phase 4 completed successfully",
                       kubernetes_updated=k8s_updated)
            return True

        except Exception as e:
            logger.error("Phase 4 failed", error=str(e))
            return False

    async def _record_event(self, event: FailoverEvent) -> None:
        """
        フェイルオーバーイベントを記録

        Args:
            event: 記録するイベント
        """
        self.event_history.append(event)

        # 履歴サイズ制限（最新100件）
        if len(self.event_history) > 100:
            self.event_history = self.event_history[-100:]

        logger.debug("Failover event recorded",
                    event_type=event.event_type,
                    event_id=event.event_id)

    def _update_execution_time_stats(self, execution_time: float) -> None:
        """
        実行時間統計を更新

        Args:
            execution_time: 実行時間
        """
        current_avg = self.coordination_stats["average_execution_time"]
        total_successful = self.coordination_stats["successful_workflows"]

        # 移動平均の計算
        new_avg = ((current_avg * (total_successful - 1)) + execution_time) / total_successful
        self.coordination_stats["average_execution_time"] = new_avg

    async def get_failover_events(self) -> List[FailoverEvent]:
        """
        フェイルオーバーイベント履歴を取得

        Returns:
            List[FailoverEvent]: イベント履歴
        """
        return self.event_history.copy()

    async def execute_kubernetes_failover(self, cluster_state: ClusterState) -> WorkflowResult:
        """
        Kubernetes統合フェイルオーバーワークフロー

        Args:
            cluster_state: クラスター状態

        Returns:
            WorkflowResult: ワークフロー結果
        """
        logger.info("Starting Kubernetes-integrated failover workflow",
                   cluster_id=cluster_state.cluster_id)

        # 通常のワークフロー実行
        result = await self.execute_failover_workflow(cluster_state)

        if result.success:
            # Kubernetesリソース更新
            try:
                k8s_result = await self.failover_manager.execute_kubernetes_failover(cluster_state)
                result.kubernetes_resources_updated = getattr(k8s_result, 'kubernetes_resources_updated', False)

                logger.info("Kubernetes failover workflow completed",
                           kubernetes_updated=result.kubernetes_resources_updated)
            except Exception as e:
                logger.error("Kubernetes integration failed", error=str(e))
                result.kubernetes_resources_updated = False

        return result

    def get_coordination_statistics(self) -> Dict[str, Any]:
        """
        調整統計情報を取得

        Returns:
            Dict[str, Any]: 統計情報
        """
        stats = self.coordination_stats.copy()

        # 成功率の計算
        if stats["total_workflows"] > 0:
            stats["success_rate"] = stats["successful_workflows"] / stats["total_workflows"]
        else:
            stats["success_rate"] = 0.0

        # 最近のイベント統計
        recent_events = self.event_history[-20:] if self.event_history else []
        stats["recent_events_count"] = len(recent_events)

        if recent_events:
            event_types = {}
            for event in recent_events:
                event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            stats["recent_event_types"] = event_types

        # コンポーネント統計
        stats["failure_detector_stats"] = self.failure_detector.get_detection_statistics()
        stats["failover_manager_stats"] = self.failover_manager.get_failover_statistics()
        stats["consistency_checker_stats"] = self.consistency_checker.get_consistency_statistics()

        return stats

    async def start_monitoring_workflow(self, cluster_state: ClusterState) -> None:
        """
        継続的監視ワークフローを開始

        Args:
            cluster_state: クラスター状態
        """
        logger.info("Starting continuous monitoring workflow",
                   cluster_id=cluster_state.cluster_id)

        while True:
            try:
                # 定期的な健全性チェック
                failure_detected = await self.failure_detector.detect_primary_failure(cluster_state)

                if failure_detected:
                    logger.critical("Primary failure detected, initiating automatic failover",
                                  cluster_id=cluster_state.cluster_id)

                    # 自動フェイルオーバーワークフロー実行
                    workflow_result = await self.execute_failover_workflow(cluster_state)

                    if workflow_result.success:
                        logger.info("Automatic failover completed successfully",
                                   new_primary=workflow_result.new_primary_id)
                    else:
                        logger.error("Automatic failover failed",
                                    error=workflow_result.error_message)

                # 次のチェックまで待機
                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                logger.info("Monitoring workflow cancelled")
                break
            except Exception as e:
                logger.error("Error in monitoring workflow", error=str(e))
                await asyncio.sleep(1.0)  # エラー時は短い間隔で再試行

    async def cleanup_resources(self) -> None:
        """リソースのクリーンアップ"""
        logger.info("Cleaning up failover coordinator resources")

        # アクティブなワークフローの停止
        for workflow_id, task in self.active_workflows.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.active_workflows.clear()

        # 各コンポーネントのクリーンアップ
        await self.failure_detector.reset_detection_history()
        await self.consistency_checker.reset_consistency_cache()

        logger.info("Failover coordinator cleanup completed")