"""
プライマリ障害検知器
Task 4.2: 自動フェイルオーバー機構
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from structlog import get_logger

from .models import (
    ClusterState, NodeState, FailoverConfiguration,
    FailoverStatus, NodeRole
)


logger = get_logger(__name__)


class PrimaryFailureDetector:
    """プライマリノード障害検知システム"""

    def __init__(self, config: FailoverConfiguration):
        """
        Args:
            config: フェイルオーバー設定
        """
        self.config = config
        self.detection_history: Dict[str, List[float]] = {}

        logger.info("PrimaryFailureDetector initialized",
                   failure_detection_timeout=config.failure_detection_timeout,
                   health_check_interval=config.health_check_interval)

    async def detect_primary_failure(self, cluster_state: ClusterState) -> bool:
        """
        プライマリノードの障害を検知

        Args:
            cluster_state: クラスター状態

        Returns:
            bool: 障害が検知された場合True
        """
        primary_node = cluster_state.get_primary_node()
        if not primary_node:
            logger.warning("No primary node found in cluster",
                         cluster_id=cluster_state.cluster_id)
            return True  # プライマリが存在しない場合は障害とみなす

        # 複数の障害検知メカニズム
        failure_indicators = []

        # 1. ハートビートタイムアウトチェック
        heartbeat_timeout = await self.check_heartbeat_timeout(cluster_state)
        failure_indicators.append(heartbeat_timeout)

        # 2. ヘルススコア劣化チェック
        health_degradation = await self.check_health_degradation(cluster_state)
        failure_indicators.append(health_degradation)

        # 3. ステータスチェック
        status_failure = primary_node.status == FailoverStatus.FAILED
        failure_indicators.append(status_failure)

        # 障害判定（いずれかの指標で障害を検知）
        is_failed = any(failure_indicators)

        if is_failed:
            logger.error("Primary failure detected",
                        node_id=primary_node.node_id,
                        heartbeat_timeout=heartbeat_timeout,
                        health_degradation=health_degradation,
                        status_failure=status_failure)

            # プライマリノードを失敗状態に更新
            primary_node.status = FailoverStatus.FAILED
            primary_node.role = NodeRole.FAILED

        return is_failed

    async def check_heartbeat_timeout(self, cluster_state: ClusterState) -> bool:
        """
        ハートビートタイムアウトをチェック

        Args:
            cluster_state: クラスター状態

        Returns:
            bool: タイムアウトが検知された場合True
        """
        primary_node = cluster_state.get_primary_node()
        if not primary_node:
            return True

        time_since_heartbeat = (datetime.now() - primary_node.last_heartbeat).total_seconds()
        is_timeout = time_since_heartbeat > self.config.failure_detection_timeout

        if is_timeout:
            logger.warning("Heartbeat timeout detected",
                          node_id=primary_node.node_id,
                          time_since_heartbeat=time_since_heartbeat,
                          timeout_threshold=self.config.failure_detection_timeout)

        return is_timeout

    async def check_health_degradation(self, cluster_state: ClusterState) -> bool:
        """
        ヘルススコア劣化をチェック

        Args:
            cluster_state: クラスター状態

        Returns:
            bool: ヘルススコア劣化が検知された場合True
        """
        primary_node = cluster_state.get_primary_node()
        if not primary_node:
            return True

        # ヘルススコア劣化の閾値
        health_threshold = 0.7
        is_degraded = primary_node.health_score < health_threshold

        if is_degraded:
            logger.warning("Health score degradation detected",
                          node_id=primary_node.node_id,
                          health_score=primary_node.health_score,
                          threshold=health_threshold)

        # 履歴に追加（トレンド分析用）
        node_id = primary_node.node_id
        if node_id not in self.detection_history:
            self.detection_history[node_id] = []

        self.detection_history[node_id].append(primary_node.health_score)

        # 履歴サイズ制限（最新10件まで）
        if len(self.detection_history[node_id]) > 10:
            self.detection_history[node_id] = self.detection_history[node_id][-10:]

        return is_degraded

    async def check_continuous_degradation(self, cluster_state: ClusterState) -> bool:
        """
        継続的なヘルススコア劣化をチェック

        Args:
            cluster_state: クラスター状態

        Returns:
            bool: 継続的劣化が検知された場合True
        """
        primary_node = cluster_state.get_primary_node()
        if not primary_node:
            return True

        node_id = primary_node.node_id
        if node_id not in self.detection_history or len(self.detection_history[node_id]) < 3:
            return False

        # 最近3回の平均ヘルススコア
        recent_scores = self.detection_history[node_id][-3:]
        average_score = sum(recent_scores) / len(recent_scores)

        # 継続的劣化の判定
        is_degrading = average_score < 0.6

        if is_degrading:
            logger.warning("Continuous health degradation detected",
                          node_id=node_id,
                          recent_average=average_score,
                          recent_scores=recent_scores)

        return is_degrading

    async def start_monitoring(self, cluster_state: ClusterState) -> None:
        """
        継続的な監視を開始

        Args:
            cluster_state: クラスター状態
        """
        logger.info("Starting primary failure monitoring",
                   cluster_id=cluster_state.cluster_id)

        while True:
            try:
                failure_detected = await self.detect_primary_failure(cluster_state)

                if failure_detected:
                    logger.critical("Primary failure detected during monitoring",
                                  cluster_id=cluster_state.cluster_id)
                    # 通常、ここでフェイルオーバー調整器に通知
                    break

                # 次のチェックまで待機
                await asyncio.sleep(self.config.health_check_interval)

            except Exception as e:
                logger.error("Error in failure monitoring loop",
                           error=str(e))
                await asyncio.sleep(1.0)  # エラー時は短い間隔で再試行

    def get_detection_statistics(self) -> Dict[str, dict]:
        """
        検知統計情報を取得

        Returns:
            Dict[str, dict]: ノード別検知統計
        """
        statistics = {}

        for node_id, history in self.detection_history.items():
            if history:
                statistics[node_id] = {
                    "sample_count": len(history),
                    "average_health_score": sum(history) / len(history),
                    "min_health_score": min(history),
                    "max_health_score": max(history),
                    "latest_health_score": history[-1]
                }

        return statistics

    async def reset_detection_history(self, node_id: Optional[str] = None) -> None:
        """
        検知履歴をリセット

        Args:
            node_id: 特定のノードIDが指定された場合、そのノードのみリセット
        """
        if node_id:
            if node_id in self.detection_history:
                self.detection_history[node_id] = []
                logger.info("Detection history reset for node", node_id=node_id)
        else:
            self.detection_history.clear()
            logger.info("All detection history reset")

    def is_primary_stable(self, cluster_state: ClusterState,
                         stability_window: int = 5) -> bool:
        """
        プライマリノードが安定しているかチェック

        Args:
            cluster_state: クラスター状態
            stability_window: 安定性チェックのウィンドウサイズ

        Returns:
            bool: 安定している場合True
        """
        primary_node = cluster_state.get_primary_node()
        if not primary_node:
            return False

        node_id = primary_node.node_id
        if node_id not in self.detection_history:
            return True  # 履歴がない場合は安定とみなす

        history = self.detection_history[node_id]
        if len(history) < stability_window:
            return True  # 十分なデータがない場合は安定とみなす

        # 最近の履歴が全て安定閾値以上かチェック
        recent_history = history[-stability_window:]
        stability_threshold = 0.8

        return all(score >= stability_threshold for score in recent_history)