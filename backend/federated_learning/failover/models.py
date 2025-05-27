"""
フェイルオーバーシステム用データモデル
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
class NodeRole(Enum):
    """ノード役割定義"""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    WITNESS = "witness"
    FAILED = "failed"


class FailoverStatus(Enum):
    """フェイルオーバー状態"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING_OVER = "failing_over"
    FAILED_OVER = "failed_over"
    FAILED = "failed"


@dataclass
class FailoverEvent:
    """フェイルオーバーイベント"""
    event_id: str
    event_type: str  # "primary_failure", "automatic_failover", "manual_failover"
    source_node: str
    target_node: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: FailoverStatus = FailoverStatus.HEALTHY
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "details": self.details,
            "error_message": self.error_message
        }


@dataclass
class NodeState:
    """ノード状態"""
    node_id: str
    role: NodeRole
    status: FailoverStatus
    last_heartbeat: datetime
    health_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """ノードが健全かチェック"""
        return (
            self.status == FailoverStatus.HEALTHY and
            self.health_score > 0.7 and
            (datetime.now() - self.last_heartbeat).total_seconds() < 30
        )

    def is_primary(self) -> bool:
        """プライマリノードかチェック"""
        return self.role == NodeRole.PRIMARY

    def can_become_primary(self) -> bool:
        """プライマリになれるかチェック"""
        return (
            self.role == NodeRole.SECONDARY and
            self.is_healthy() and
            self.health_score > 0.8
        )


@dataclass
class FailoverConfiguration:
    """フェイルオーバー設定"""
    failure_detection_timeout: float = 30.0  # 障害検知タイムアウト（秒）
    failover_timeout: float = 60.0  # フェイルオーバータイムアウト（秒）
    health_check_interval: float = 5.0  # ヘルスチェック間隔（秒）
    min_secondary_nodes: int = 1  # 最小セカンダリノード数
    enable_automatic_failover: bool = True  # 自動フェイルオーバー有効
    enable_automatic_failback: bool = False  # 自動フェイルバック有効
    consistency_check_enabled: bool = True  # データ整合性チェック有効
    max_failover_attempts: int = 3  # 最大フェイルオーバー試行回数

    def validate(self) -> bool:
        """設定の妥当性チェック"""
        return (
            self.failure_detection_timeout > 0 and
            self.failover_timeout > 0 and
            self.health_check_interval > 0 and
            self.min_secondary_nodes >= 0 and
            self.max_failover_attempts > 0
        )


@dataclass
class ClusterState:
    """クラスター状態"""
    cluster_id: str
    nodes: Dict[str, NodeState] = field(default_factory=dict)
    primary_node: Optional[str] = None
    secondary_nodes: List[str] = field(default_factory=list)
    status: FailoverStatus = FailoverStatus.HEALTHY
    last_failover: Optional[datetime] = None
    failover_count: int = 0

    def get_primary_node(self) -> Optional[NodeState]:
        """プライマリノードを取得"""
        if self.primary_node and self.primary_node in self.nodes:
            return self.nodes[self.primary_node]
        return None

    def get_healthy_secondary_nodes(self) -> List[NodeState]:
        """健全なセカンダリノードのリストを取得"""
        healthy_secondaries = []
        for node_id in self.secondary_nodes:
            if node_id in self.nodes and self.nodes[node_id].is_healthy():
                healthy_secondaries.append(self.nodes[node_id])
        return healthy_secondaries

    def get_best_failover_candidate(self) -> Optional[NodeState]:
        """最適なフェイルオーバー候補を取得"""
        candidates = self.get_healthy_secondary_nodes()
        if not candidates:
            return None

        # ヘルススコアでソート
        candidates.sort(key=lambda x: x.health_score, reverse=True)
        return candidates[0]

    def is_cluster_healthy(self) -> bool:
        """クラスターが健全かチェック"""
        primary = self.get_primary_node()
        if not primary or not primary.is_healthy():
            return False

        healthy_secondaries = self.get_healthy_secondary_nodes()
        return len(healthy_secondaries) > 0

    def update_node_state(self, node_id: str, node_state: NodeState) -> None:
        """ノード状態を更新"""
        self.nodes[node_id] = node_state

        # ロール別ノードリストの更新
        if node_state.role == NodeRole.PRIMARY:
            self.primary_node = node_id
        elif node_state.role == NodeRole.SECONDARY:
            if node_id not in self.secondary_nodes:
                self.secondary_nodes.append(node_id)
        elif node_state.role == NodeRole.FAILED:
            # 失敗したノードを削除
            if self.primary_node == node_id:
                self.primary_node = None
            if node_id in self.secondary_nodes:
                self.secondary_nodes.remove(node_id)