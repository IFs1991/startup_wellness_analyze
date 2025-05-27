"""
Task 4.2: 自動フェイルオーバー機構
Kubernetes対応の高可用性フェイルオーバーシステム
"""

from .failover_coordinator import FailoverCoordinator
from .primary_failure_detector import PrimaryFailureDetector
from .auto_failover_manager import AutoFailoverManager
from .data_consistency_checker import DataConsistencyChecker
from .models import (
    FailoverEvent, FailoverStatus, NodeRole,
    NodeState, ClusterState, FailoverConfiguration
)

__all__ = [
    "FailoverCoordinator",
    "PrimaryFailureDetector",
    "AutoFailoverManager",
    "DataConsistencyChecker",
    "FailoverEvent",
    "FailoverStatus",
    "NodeRole",
    "NodeState",
    "ClusterState",
    "FailoverConfiguration"
]