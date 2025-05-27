# Phase 3 Task 3.4: 分散トランザクション管理システムの実装
# TDD GREEN段階: データモデル定義

import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timezone
from enum import Enum, auto


class TransactionStatus(Enum):
    """トランザクション状態"""
    PENDING = auto()
    PREPARING = auto()
    PREPARED = auto()
    COMMITTING = auto()
    COMMITTED = auto()
    ABORTING = auto()
    ABORTED = auto()
    COMPENSATING = auto()
    COMPENSATED = auto()
    PARTIALLY_COMMITTED = auto()
    FAILED = auto()


class StepStatus(Enum):
    """ステップ状態"""
    PENDING = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    COMPENSATING = auto()
    COMPENSATED = auto()
    SKIPPED = auto()


class IsolationLevel(Enum):
    """分離レベル"""
    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"


class ConsistencyLevel(Enum):
    """一貫性レベル"""
    STRONG = "strong"
    EVENTUAL = "eventual"
    WEAK = "weak"


@dataclass
class TransactionStep:
    """トランザクションステップ"""
    step_id: str
    operation: str
    resource_id: str
    data: Dict[str, Any]
    compensation_data: Optional[Dict[str, Any]] = None
    status: StepStatus = StepStatus.PENDING
    can_run_parallel: bool = False
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_attempts: int = 0
    max_retry_attempts: int = 3
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class CompensationAction:
    """補償アクション"""
    action_id: str
    resource_id: str
    operation: str
    data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    max_attempts: int = 3
    current_attempt: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    executed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class Transaction:
    """分散トランザクション"""
    transaction_id: str
    transaction_type: str  # "2pc", "saga", "mixed"
    steps: List[TransactionStep]
    status: TransactionStatus = TransactionStatus.PENDING
    consistency_level: str = "eventual"
    isolation_level: str = "read_committed"
    consistency_requirements: Optional[Dict[str, str]] = None
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    coordinator_id: Optional[str] = None
    error_message: Optional[str] = None
    compensation_actions: List[CompensationAction] = field(default_factory=list)


@dataclass
class SagaTransaction:
    """Sagaトランザクション"""
    saga_id: str
    steps: List[TransactionStep]
    services: Dict[str, Any]
    status: TransactionStatus = TransactionStatus.PENDING
    current_step_index: int = 0
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    compensated_steps: List[str] = field(default_factory=list)
    parallel_execution: bool = False
    max_retry_attempts: int = 3
    retry_delay: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class TwoPhaseCommitResult:
    """2相コミット結果"""
    transaction_id: str
    status: TransactionStatus
    prepared_resources: List[str] = field(default_factory=list)
    committed_resources: List[str] = field(default_factory=list)
    failed_resources: List[str] = field(default_factory=list)
    execution_time: Optional[float] = None
    prepare_time: Optional[float] = None
    commit_time: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SagaResult:
    """Saga実行結果"""
    saga_id: str
    status: TransactionStatus
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    compensated_steps: List[str] = field(default_factory=list)
    execution_time: Optional[float] = None
    total_steps: int = 0
    successful_steps: int = 0
    error_message: Optional[str] = None
    compensation_details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CompensationResult:
    """補償実行結果"""
    success: bool
    successful_compensations: List[str] = field(default_factory=list)
    failed_compensations: List[str] = field(default_factory=list)
    deadlock_detected: bool = False
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ResourceManager:
    """リソースマネージャー"""
    resource_id: str
    resource_type: str
    endpoint: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = None
    status: str = "ready"
    health_check_interval: float = 30.0
    timeout: float = 30.0
    max_connections: int = 10
    retry_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_health_check: Optional[datetime] = None


@dataclass
class DistributedLock:
    """分散ロック"""
    lock_id: str
    resource_id: str
    transaction_id: str
    owner_id: str
    lock_type: str = "exclusive"  # "exclusive", "shared"
    acquired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransactionLog:
    """トランザクションログ"""
    log_id: str
    transaction_id: str
    operation: str
    resource_id: Optional[str] = None
    status: str = "info"  # "info", "warning", "error"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ユーティリティ関数
def generate_transaction_id() -> str:
    """トランザクションID生成"""
    return f"tx_{uuid.uuid4().hex[:16]}"


def generate_step_id() -> str:
    """ステップID生成"""
    return f"step_{uuid.uuid4().hex[:12]}"


def generate_saga_id() -> str:
    """SagaID生成"""
    return f"saga_{uuid.uuid4().hex[:16]}"


def generate_compensation_id() -> str:
    """補償アクションID生成"""
    return f"comp_{uuid.uuid4().hex[:12]}"


def is_terminal_status(status: TransactionStatus) -> bool:
    """終了状態かどうかを判定"""
    terminal_statuses = {
        TransactionStatus.COMMITTED,
        TransactionStatus.ABORTED,
        TransactionStatus.COMPENSATED,
        TransactionStatus.FAILED,
        TransactionStatus.PARTIALLY_COMMITTED
    }
    return status in terminal_statuses


def can_be_compensated(status: StepStatus) -> bool:
    """補償可能かどうかを判定"""
    compensatable_statuses = {
        StepStatus.COMPLETED,
        StepStatus.FAILED
    }
    return status in compensatable_statuses


def calculate_transaction_duration(transaction: Transaction) -> Optional[float]:
    """トランザクション実行時間計算"""
    if transaction.started_at and transaction.completed_at:
        return (transaction.completed_at - transaction.started_at).total_seconds()
    return None


def get_step_by_id(transaction: Transaction, step_id: str) -> Optional[TransactionStep]:
    """ステップIDでステップを取得"""
    for step in transaction.steps:
        if step.step_id == step_id:
            return step
    return None


def get_failed_steps(transaction: Transaction) -> List[TransactionStep]:
    """失敗したステップを取得"""
    return [step for step in transaction.steps if step.status == StepStatus.FAILED]


def get_completed_steps(transaction: Transaction) -> List[TransactionStep]:
    """完了したステップを取得"""
    return [step for step in transaction.steps if step.status == StepStatus.COMPLETED]


def validate_transaction(transaction: Transaction) -> List[str]:
    """トランザクション検証"""
    errors = []

    if not transaction.transaction_id:
        errors.append("Transaction ID is required")

    if not transaction.steps:
        errors.append("At least one step is required")

    # ステップ依存関係の循環チェック
    step_ids = {step.step_id for step in transaction.steps}
    for step in transaction.steps:
        for dep in step.dependencies:
            if dep not in step_ids:
                errors.append(f"Step {step.step_id} depends on non-existent step {dep}")

    return errors