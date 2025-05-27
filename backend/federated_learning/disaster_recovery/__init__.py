"""
Task 4.4: 災害復旧システム
エンタープライズグレード災害復旧・ビジネス継続性システム

このモジュールは連合学習システムの包括的な災害復旧機能を提供します：
- 自動バックアップ・復元
- 地域間レプリケーション
- RTO/RPO コンプライアンス監視
- ビジネス継続性計画
"""

from .models import (
    # Enums
    BackupType,
    BackupStrategy,
    RetentionPolicy,
    HealthStatus,
    ConsistencyLevel,
    DisasterType,
    SeverityLevel,

    # Data Models
    Contact,
    BackupMetadata,
    BackupResult,
    RestoreResult,
    BackupVerificationResult,
    ReplicationConfig,
    ReplicationStatus,
    ReplicationMetrics,
    WriteResult,
    ReadResult,
    ConsistencyCheckResult,
    FailoverResult,
    RecoveryObjective,
    DisasterEvent,
    RTOMetrics,
    RPOMetrics,
    SLAAlert,
    SLAStatus,
    ComplianceReport,
    DisasterRecoveryPlan,
    ContinuityPlan,
    ContinuityResponse,
    DetectionResult,
    ServiceRestorationVerification,
    RecoveryMetrics,
    DRValidationResult,
    NetworkConnectivity,
    StorageHealth,
    SystemHealth
)

# 実装コンポーネント（GREEN段階で作成予定）
from .backup_manager import BackupManager
from .cross_region_replication import CrossRegionReplication
from .rto_rpo_monitor import RTORPOMonitor
from .restore_manager import RestoreManager
from .continuity_planner import ContinuityPlanner
from .disaster_recovery_manager import DisasterRecoveryManager

__version__ = "1.0.0"
__author__ = "Federated Learning Team"

__all__ = [
    # Enums
    "BackupType",
    "BackupStrategy",
    "RetentionPolicy",
    "HealthStatus",
    "ConsistencyLevel",
    "DisasterType",
    "SeverityLevel",

    # Data Models
    "Contact",
    "BackupMetadata",
    "BackupResult",
    "RestoreResult",
    "BackupVerificationResult",
    "ReplicationConfig",
    "ReplicationStatus",
    "ReplicationMetrics",
    "WriteResult",
    "ReadResult",
    "ConsistencyCheckResult",
    "FailoverResult",
    "RecoveryObjective",
    "DisasterEvent",
    "RTOMetrics",
    "RPOMetrics",
    "SLAAlert",
    "SLAStatus",
    "ComplianceReport",
    "DisasterRecoveryPlan",
    "ContinuityPlan",
    "ContinuityResponse",
    "DetectionResult",
    "ServiceRestorationVerification",
    "RecoveryMetrics",
    "DRValidationResult",
    "NetworkConnectivity",
    "StorageHealth",
    "SystemHealth",

    # Core Components
    "BackupManager",
    "CrossRegionReplication",
    "RTORPOMonitor",
    "RestoreManager",
    "ContinuityPlanner",
    "DisasterRecoveryManager"
]