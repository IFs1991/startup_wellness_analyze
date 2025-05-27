"""
Task 4.4: 災害復旧システム データモデル
エンタープライズグレード災害復旧・ビジネス継続性データ構造
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4
import json


class BackupType(Enum):
    """バックアップタイプ"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    CONTINUOUS = "continuous"


class BackupStrategy(Enum):
    """バックアップ戦略"""
    DAILY_FULL = "daily_full"
    WEEKLY_FULL_DAILY_INCREMENTAL = "weekly_full_daily_incremental"
    MONTHLY_FULL_WEEKLY_DIFFERENTIAL = "monthly_full_weekly_differential"
    CONTINUOUS_CDP = "continuous_data_protection"


class RetentionPolicy(Enum):
    """保持ポリシー"""
    DAYS_7 = 7
    DAYS_30 = 30
    DAYS_90 = 90
    DAYS_365 = 365
    YEARS_7 = 2555  # 7年 (法的要件対応)


class HealthStatus(Enum):
    """健全性状態"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ConsistencyLevel(Enum):
    """整合性レベル"""
    STRONG = "strong"
    EVENTUAL = "eventual"
    WEAK = "weak"
    CAUSAL = "causal"


class DisasterType(Enum):
    """災害タイプ"""
    SYSTEM_FAILURE = "system_failure"
    NETWORK_OUTAGE = "network_outage"
    REGIONAL_OUTAGE = "regional_outage"
    DATA_CENTER_FAILURE = "data_center_failure"
    CYBER_ATTACK = "cyber_attack"
    NATURAL_DISASTER = "natural_disaster"
    HUMAN_ERROR = "human_error"
    HARDWARE_FAILURE = "hardware_failure"


class SeverityLevel(IntEnum):
    """重要度レベル"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CATASTROPHIC = 5


@dataclass
class Contact:
    """連絡先情報"""
    name: str
    role: str
    email: str
    phone: str
    escalation_level: int
    availability: str = "24/7"


@dataclass
class BackupMetadata:
    """バックアップメタデータ"""
    backup_id: str = field(default_factory=lambda: str(uuid4()))
    backup_type: BackupType = BackupType.FULL
    data_sources: List[str] = field(default_factory=list)
    size_bytes: int = 0
    compressed_size_bytes: int = 0
    creation_time: datetime = field(default_factory=datetime.utcnow)
    completion_time: Optional[datetime] = None
    checksum: str = ""
    encryption_key_id: str = ""
    retention_policy: RetentionPolicy = RetentionPolicy.DAYS_30
    storage_locations: List[str] = field(default_factory=list)
    parent_backup_id: Optional[str] = None  # 増分バックアップの場合
    verification_status: bool = False
    verification_time: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class BackupResult:
    """バックアップ実行結果"""
    success: bool
    backup_id: str
    backup_type: BackupType
    size_bytes: int
    checksum: str
    duration: timedelta
    error_message: Optional[str] = None


@dataclass
class RestoreResult:
    """復元実行結果"""
    success: bool
    backup_id: str
    restored_files: int
    failed_files: int
    total_size_bytes: int
    duration: timedelta
    restore_path: str
    error_message: Optional[str] = None


@dataclass
class BackupVerificationResult:
    """バックアップ検証結果"""
    backup_id: str
    is_valid: bool
    checksum_verified: bool
    file_count_verified: bool
    encryption_verified: bool
    calculated_checksum: str
    verification_time: datetime
    errors: List[str] = field(default_factory=list)


@dataclass
class ReplicationConfig:
    """レプリケーション設定"""
    primary_region: str
    secondary_regions: List[str]
    sync_mode: str  # "sync", "async", "semi_sync"
    conflict_resolution: str  # "timestamp", "version", "manual"
    max_sync_lag_seconds: int = 30
    compression_enabled: bool = True
    encryption_enabled: bool = True


@dataclass
class ReplicationStatus:
    """レプリケーション状態"""
    replication_id: str = field(default_factory=lambda: str(uuid4()))
    source_region: str = ""
    target_region: str = ""
    sync_lag: timedelta = timedelta(0)
    last_sync_time: datetime = field(default_factory=datetime.utcnow)
    health_status: HealthStatus = HealthStatus.UNKNOWN
    status: HealthStatus = HealthStatus.UNKNOWN  # テスト用エイリアス
    data_consistency: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    bytes_replicated: int = 0
    replication_rate_bps: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass
class ReplicationMetrics:
    """レプリケーションメトリクス"""
    regions: Any  # 動的に辞書またはリストとして使える
    total_regions: int
    healthy_regions: int
    failed_regions: int
    average_sync_lag: timedelta
    total_data_replicated: int
    replication_efficiency: float
    last_update: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WriteResult:
    """書き込み結果"""
    success: bool
    key: str
    timestamp: datetime
    error_message: Optional[str] = None


@dataclass
class ReadResult:
    """読み取り結果"""
    success: bool
    key: str
    data: Dict[str, Any]
    timestamp: datetime
    region: str
    error_message: Optional[str] = None


@dataclass
class ConsistencyCheckResult:
    """整合性チェック結果"""
    is_consistent: bool
    total_keys: int
    inconsistent_keys: int
    region_comparisons: List[Dict[str, Any]]
    check_time: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FailoverResult:
    """フェイルオーバー結果"""
    success: bool
    original_primary: str
    new_primary_region: str
    failover_time: timedelta
    data_loss_detected: bool
    services_affected: List[str]
    error_message: Optional[str] = None


@dataclass
class RecoveryObjective:
    """復旧目標"""
    rto_target: timedelta  # Recovery Time Objective
    rpo_target: timedelta  # Recovery Point Objective
    rto_warning_threshold: timedelta
    rpo_warning_threshold: timedelta
    mttr_target: timedelta = timedelta(hours=4)  # Mean Time To Repair
    mtbf_target: timedelta = timedelta(days=30)  # Mean Time Between Failures


@dataclass
class DisasterEvent:
    """災害イベント"""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    disaster_type: DisasterType = DisasterType.SYSTEM_FAILURE
    severity: SeverityLevel = SeverityLevel.MEDIUM
    affected_services: List[str] = field(default_factory=list)
    affected_regions: List[str] = field(default_factory=list)
    detection_time: datetime = field(default_factory=datetime.utcnow)
    resolution_time: Optional[datetime] = None
    root_cause: Optional[str] = None
    impact_description: str = ""
    estimated_data_loss: Optional[int] = None  # bytes
    estimated_downtime: Optional[timedelta] = None


@dataclass
class RTOMetrics:
    """RTO メトリクス"""
    target_rto: timedelta
    actual_rto: timedelta
    compliance_status: str  # "COMPLIANT", "WARNING", "VIOLATION"
    recovery_steps: List[Dict[str, Any]]
    bottlenecks: List[str]


@dataclass
class RPOMetrics:
    """RPO メトリクス"""
    target_rpo: timedelta
    data_loss_window: timedelta
    compliance_status: str
    estimated_transactions: int
    estimated_size_bytes: int
    affected_components: List[str]


@dataclass
class SLAAlert:
    """SLA アラート"""
    alert_type: str  # "RTO_WARNING", "RTO_VIOLATION", "RPO_WARNING", "RPO_VIOLATION"
    severity: str  # "INFO", "WARNING", "CRITICAL"
    message: str
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    triggered_time: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class SLAStatus:
    """SLA 状態"""
    rto_warning_triggered: bool = False
    rto_violation_triggered: bool = False
    rpo_warning_triggered: bool = False
    rpo_violation_triggered: bool = False
    warnings: List[SLAAlert] = field(default_factory=list)
    violations: List[SLAAlert] = field(default_factory=list)


@dataclass
class ComplianceReport:
    """コンプライアンス報告書"""
    period_start: datetime
    period_end: datetime
    total_incidents: int
    rto_compliance_rate: float  # 0.0 - 1.0
    rpo_compliance_rate: float  # 0.0 - 1.0
    average_rto: timedelta
    average_rpo: timedelta
    worst_rto: timedelta
    worst_rpo: timedelta
    incident_breakdown: Dict[DisasterType, int]
    recommendations: List[str]
    overall_grade: str  # A, B, C, D, F
    report_id: str = field(default_factory=lambda: str(uuid4()))
    generated_time: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DisasterRecoveryPlan:
    """災害復旧計画"""
    plan_name: str
    backup_strategy: BackupStrategy
    replication_config: ReplicationConfig
    rto_target: timedelta
    rpo_target: timedelta
    priority_services: List[str]
    escalation_contacts: List[Contact]
    disaster_scenarios: List[DisasterEvent]
    recovery_procedures: Dict[DisasterType, List[str]]
    testing_schedule: str  # cron expression
    plan_id: str = field(default_factory=lambda: str(uuid4()))
    version: str = "1.0"
    creation_time: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    last_test_date: Optional[datetime] = None
    plan_status: str = "ACTIVE"  # ACTIVE, TESTING, OUTDATED


@dataclass
class ContinuityPlan:
    """ビジネス継続性計画"""
    critical_services: List[str]
    recovery_priorities: Dict[str, int]  # service -> priority (1=highest)
    alternate_workflows: Dict[str, List[str]]
    resource_requirements: Dict[str, Any]
    communication_plan: List[Contact]
    plan_id: str = field(default_factory=lambda: str(uuid4()))
    minimum_capacity_percentage: int = 80


@dataclass
class ContinuityResponse:
    """継続性対応結果"""
    plan_executed: bool
    critical_services_maintained: bool
    capacity_maintained: float
    service_recovery_order: List[str]
    alternate_workflows_activated: List[str]
    execution_time: timedelta


@dataclass
class DetectionResult:
    """災害検知結果"""
    disaster_detected: bool
    auto_response_triggered: bool
    response_time: timedelta
    confidence_score: float
    detected_issues: List[str]


@dataclass
class ServiceRestorationVerification:
    """サービス復元検証"""
    services_restored: bool
    data_integrity_verified: bool
    performance_acceptable: bool
    user_access_verified: bool
    verification_details: Dict[str, Any]


@dataclass
class RecoveryMetrics:
    """復旧メトリクス"""
    rto_achieved: timedelta
    rpo_achieved: timedelta
    sla_compliance: bool
    recovery_efficiency: float
    cost_impact: float
    lessons_learned: List[str]


@dataclass
class DRValidationResult:
    """災害復旧計画検証結果"""
    backup_strategy_valid: bool
    replication_setup_valid: bool
    recovery_procedures_valid: bool
    rto_achievable: bool
    rpo_achievable: bool
    dependency_analysis: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    overall_score: int  # 0-100
    plan_approval_recommended: bool
    validation_errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class NetworkConnectivity:
    """ネットワーク接続性"""
    region_connectivity: Dict[str, bool]
    latency_measurements: Dict[str, float]  # milliseconds
    bandwidth_measurements: Dict[str, float]  # Mbps
    last_check: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StorageHealth:
    """ストレージ健全性"""
    total_capacity: int
    used_capacity: int
    available_capacity: int
    iops_performance: float
    throughput_performance: float
    error_rate: float
    health_score: int  # 0-100


@dataclass
class SystemHealth:
    """システム健全性"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_usage: float
    service_status: Dict[str, HealthStatus]
    overall_health: HealthStatus