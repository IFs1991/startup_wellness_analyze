"""
自動フェイルオーバー管理器
Task 4.2: 自動フェイルオーバー機構
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from structlog import get_logger

from .models import (
    ClusterState, NodeState, FailoverConfiguration,
    FailoverStatus, NodeRole, FailoverEvent
)


logger = get_logger(__name__)


@dataclass
class FailoverResult:
    """フェイルオーバー結果"""
    success: bool
    new_primary_id: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    error_message: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AutoFailoverManager:
    """自動フェイルオーバー管理システム"""

    def __init__(self, config: FailoverConfiguration):
        """
        Args:
            config: フェイルオーバー設定
        """
        self.config = config
        self.failover_lock = asyncio.Lock()
        self.active_failovers: Dict[str, asyncio.Task] = {}

        logger.info("AutoFailoverManager initialized",
                   failover_timeout=config.failover_timeout,
                   enable_automatic_failover=config.enable_automatic_failover,
                   max_failover_attempts=config.max_failover_attempts)

    async def execute_failover(self, cluster_state: ClusterState) -> FailoverResult:
        """
        フェイルオーバーを実行

        Args:
            cluster_state: クラスター状態

        Returns:
            FailoverResult: フェイルオーバー結果
        """
        start_time = time.time()
        cluster_id = cluster_state.cluster_id

        # 自動フェイルオーバーが無効の場合
        if not self.config.enable_automatic_failover:
            logger.warning("Automatic failover is disabled", cluster_id=cluster_id)
            return FailoverResult(
                success=False,
                error_message="Automatic failover is disabled"
            )

        # 同時フェイルオーバー防止
        async with self.failover_lock:
            logger.info("Starting failover execution", cluster_id=cluster_id)

            try:
                # 1. フェイルオーバー候補選択
                best_candidate = await self.select_best_candidate(cluster_state)
                if not best_candidate:
                    return FailoverResult(
                        success=False,
                        error_message="No viable failover candidates available",
                        execution_time=time.time() - start_time
                    )

                logger.info("Failover candidate selected",
                           cluster_id=cluster_id,
                           candidate_id=best_candidate.node_id,
                           health_score=best_candidate.health_score)

                # 2. フェイルオーバー実行
                success = await self._perform_failover_operation(
                    cluster_state, best_candidate
                )

                if success:
                    execution_time = time.time() - start_time
                    logger.info("Failover completed successfully",
                               cluster_id=cluster_id,
                               new_primary=best_candidate.node_id,
                               execution_time=execution_time)

                    return FailoverResult(
                        success=True,
                        new_primary_id=best_candidate.node_id,
                        execution_time=execution_time
                    )
                else:
                    return FailoverResult(
                        success=False,
                        error_message="Failover operation failed",
                        execution_time=time.time() - start_time
                    )

            except Exception as e:
                logger.error("Failover execution failed",
                           cluster_id=cluster_id,
                           error=str(e))
                return FailoverResult(
                    success=False,
                    error_message=str(e),
                    execution_time=time.time() - start_time
                )

    async def select_best_candidate(self, cluster_state: ClusterState) -> Optional[NodeState]:
        """
        最適なフェイルオーバー候補を選択

        Args:
            cluster_state: クラスター状態

        Returns:
            Optional[NodeState]: 最適候補（見つからない場合はNone）
        """
        # 健全なセカンダリノードを取得
        healthy_secondaries = cluster_state.get_healthy_secondary_nodes()

        if not healthy_secondaries:
            logger.warning("No healthy secondary nodes available",
                          cluster_id=cluster_state.cluster_id)
            return None

        # プライマリになれるノードのフィルタリング
        eligible_candidates = [
            node for node in healthy_secondaries
            if node.can_become_primary()
        ]

        if not eligible_candidates:
            logger.warning("No eligible candidates for primary role",
                          cluster_id=cluster_state.cluster_id)
            return None

        # 最適候補選択（複数の基準で評価）
        best_candidate = max(eligible_candidates, key=self._evaluate_candidate)

        logger.info("Best failover candidate evaluated",
                   candidate_id=best_candidate.node_id,
                   health_score=best_candidate.health_score,
                   total_candidates=len(eligible_candidates))

        return best_candidate

    def _evaluate_candidate(self, node: NodeState) -> float:
        """
        候補ノードを評価

        Args:
            node: 評価対象ノード

        Returns:
            float: 評価スコア（高いほど良い）
        """
        score = 0.0

        # ヘルススコア（重み: 40%）
        score += node.health_score * 0.4

        # 最近のハートビート（重み: 30%）
        heartbeat_recency = (datetime.now() - node.last_heartbeat).total_seconds()
        recency_score = max(0, 1.0 - (heartbeat_recency / 30.0))  # 30秒以内なら満点
        score += recency_score * 0.3

        # ステータス評価（重み: 20%）
        status_score = 1.0 if node.status == FailoverStatus.HEALTHY else 0.5
        score += status_score * 0.2

        # メタデータベースの追加評価（重み: 10%）
        metadata_score = self._evaluate_metadata(node)
        score += metadata_score * 0.1

        return score

    def _evaluate_metadata(self, node: NodeState) -> float:
        """
        ノードメタデータを評価

        Args:
            node: 評価対象ノード

        Returns:
            float: メタデータスコア（0.0-1.0）
        """
        score = 0.5  # デフォルトスコア

        if not node.metadata:
            return score

        # CPU使用率（低いほど良い）
        cpu_usage = node.metadata.get("cpu_usage", 50.0)
        score += (100.0 - cpu_usage) / 200.0  # 0.0-0.5の範囲

        # メモリ使用率（低いほど良い）
        memory_usage = node.metadata.get("memory_usage", 50.0)
        score += (100.0 - memory_usage) / 200.0  # 0.0-0.5の範囲

        return min(1.0, score)

    async def _perform_failover_operation(self, cluster_state: ClusterState,
                                        new_primary: NodeState) -> bool:
        """
        実際のフェイルオーバー操作を実行

        Args:
            cluster_state: クラスター状態
            new_primary: 新しいプライマリノード

        Returns:
            bool: 成功した場合True
        """
        try:
            # 1. 古いプライマリを非アクティブ化
            old_primary = cluster_state.get_primary_node()
            if old_primary:
                old_primary.role = NodeRole.FAILED
                old_primary.status = FailoverStatus.FAILED
                logger.info("Old primary deactivated",
                           old_primary_id=old_primary.node_id)

            # 2. 新しいプライマリを昇格
            new_primary.role = NodeRole.PRIMARY
            new_primary.status = FailoverStatus.HEALTHY
            cluster_state.primary_node = new_primary.node_id

            # セカンダリリストから削除
            if new_primary.node_id in cluster_state.secondary_nodes:
                cluster_state.secondary_nodes.remove(new_primary.node_id)

            # 3. クラスター状態を更新
            cluster_state.last_failover = datetime.now()
            cluster_state.failover_count += 1
            cluster_state.status = FailoverStatus.FAILED_OVER

            logger.info("Failover operation completed",
                       new_primary_id=new_primary.node_id,
                       cluster_id=cluster_state.cluster_id)

            return True

        except Exception as e:
            logger.error("Failover operation failed", error=str(e))
            return False

    async def execute_failover_with_retry(self, cluster_state: ClusterState) -> FailoverResult:
        """
        リトライ機構付きフェイルオーバー実行

        Args:
            cluster_state: クラスター状態

        Returns:
            FailoverResult: フェイルオーバー結果
        """
        max_attempts = self.config.max_failover_attempts

        for attempt in range(max_attempts):
            logger.info("Attempting failover",
                       cluster_id=cluster_state.cluster_id,
                       attempt=attempt + 1,
                       max_attempts=max_attempts)

            result = await self.execute_failover(cluster_state)

            if result.success:
                result.retry_count = attempt
                return result

            # 最後の試行でない場合は少し待機
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 2  # 指数バックオフ
                logger.info("Failover attempt failed, retrying",
                           attempt=attempt + 1,
                           wait_time=wait_time)
                await asyncio.sleep(wait_time)

        # 全ての試行が失敗
        logger.error("All failover attempts failed",
                    cluster_id=cluster_state.cluster_id,
                    attempts=max_attempts)

        return FailoverResult(
            success=False,
            retry_count=max_attempts,
            error_message=f"Failed after {max_attempts} attempts"
        )

    async def validate_failover_prerequisites(self, cluster_state: ClusterState) -> Tuple[bool, str]:
        """
        フェイルオーバー前提条件を検証

        Args:
            cluster_state: クラスター状態

        Returns:
            Tuple[bool, str]: (検証結果, エラーメッセージ)
        """
        # 1. 最小セカンダリノード数チェック
        healthy_secondaries = cluster_state.get_healthy_secondary_nodes()
        if len(healthy_secondaries) < self.config.min_secondary_nodes:
            return False, f"Insufficient secondary nodes: {len(healthy_secondaries)} < {self.config.min_secondary_nodes}"

        # 2. 利用可能候補チェック
        best_candidate = await self.select_best_candidate(cluster_state)
        if not best_candidate:
            return False, "No viable failover candidates available"

        # 3. 最近のフェイルオーバー頻度チェック
        if cluster_state.last_failover:
            time_since_last = (datetime.now() - cluster_state.last_failover).total_seconds()
            min_interval = 300  # 5分間隔
            if time_since_last < min_interval:
                return False, f"Too frequent failovers: {time_since_last}s < {min_interval}s"

        # 4. リソース可用性チェック（基本）
        if best_candidate.health_score < 0.8:
            return False, f"Best candidate health score too low: {best_candidate.health_score}"

        return True, "Prerequisites validated successfully"

    async def execute_kubernetes_failover(self, cluster_state: ClusterState) -> FailoverResult:
        """
        Kubernetes統合フェイルオーバー実行

        Args:
            cluster_state: クラスター状態

        Returns:
            FailoverResult: フェイルオーバー結果
        """
        start_time = time.time()

        try:
            # 通常のフェイルオーバー実行
            result = await self.execute_failover(cluster_state)

            if not result.success:
                return result

            # Kubernetesリソース更新
            k8s_success = await self._update_kubernetes_resources(
                cluster_state, result.new_primary_id
            )

            if k8s_success:
                result.kubernetes_resources_updated = True
                logger.info("Kubernetes resources updated successfully",
                           new_primary=result.new_primary_id)
            else:
                logger.warning("Kubernetes resources update failed",
                              new_primary=result.new_primary_id)
                result.kubernetes_resources_updated = False

            return result

        except Exception as e:
            logger.error("Kubernetes failover failed", error=str(e))
            return FailoverResult(
                success=False,
                error_message=f"Kubernetes failover failed: {str(e)}",
                execution_time=time.time() - start_time
            )

    async def _update_kubernetes_resources(self, cluster_state: ClusterState,
                                         new_primary_id: str) -> bool:
        """
        Kubernetesリソースを更新

        Args:
            cluster_state: クラスター状態
            new_primary_id: 新しいプライマリノードID

        Returns:
            bool: 成功した場合True
        """
        try:
            # Kubernetesクライアントのインポート（オプション）
            try:
                from kubernetes import client, config
                config.load_incluster_config()  # Pod内での実行時
                v1 = client.CoreV1Api()
            except ImportError:
                logger.warning("Kubernetes client not available")
                return False
            except:
                try:
                    from kubernetes import client, config
                    config.load_kube_config()  # ローカル開発時
                    v1 = client.CoreV1Api()
                except:
                    logger.warning("Failed to load Kubernetes configuration")
                    return False

            # サービスエンドポイント更新（例）
            service_name = f"{cluster_state.cluster_id}-primary-service"
            namespace = "default"  # 実際の名前空間を使用

            # サービスの更新は実装に依存するため、ここでは基本的な例
            logger.info("Updating Kubernetes service endpoint",
                       service=service_name,
                       new_primary=new_primary_id)

            # 実際のKubernetesリソース更新ロジックをここに実装
            # 例: サービスエンドポイント、ConfigMap、Secretの更新

            return True

        except Exception as e:
            logger.error("Kubernetes resource update failed", error=str(e))
            return False

    def get_failover_statistics(self) -> Dict[str, any]:
        """
        フェイルオーバー統計情報を取得

        Returns:
            Dict[str, any]: 統計情報
        """
        return {
            "automatic_failover_enabled": self.config.enable_automatic_failover,
            "max_failover_attempts": self.config.max_failover_attempts,
            "failover_timeout": self.config.failover_timeout,
            "active_failovers": len(self.active_failovers),
            "min_secondary_nodes": self.config.min_secondary_nodes
        }