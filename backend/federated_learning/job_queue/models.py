"""
非同期ジョブキューのデータモデル
Task 4.3: 非同期ジョブキュー
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from .job_types import JobType, JobPriority, JobStatus, JobCategory


@dataclass
class JobResult:
    """ジョブ実行結果"""
    job_id: str
    job_type: JobType
    status: JobStatus
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None

    def __post_init__(self):
        if self.completed_at and self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()


@dataclass
class JobMetrics:
    """ジョブ実行メトリクス"""
    job_id: str
    job_type: JobType
    priority: JobPriority
    queue_name: str

    # タイミング情報
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 実行統計
    execution_time: float = 0.0
    queue_time: float = 0.0  # キューでの待機時間
    retry_count: int = 0
    max_retries: int = 3

    # リソース使用量
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

    # エラー情報
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # メタデータ
    worker_id: Optional[str] = None
    cluster_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_timing_metrics(self):
        """タイミングメトリクスを計算"""
        if self.started_at:
            self.queue_time = (self.started_at - self.created_at).total_seconds()

        if self.completed_at and self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()

    def is_successful(self) -> bool:
        """ジョブが成功したかチェック"""
        return self.error_type is None and self.error_message is None

    def get_total_time(self) -> float:
        """総実行時間を取得"""
        return self.queue_time + self.execution_time


@dataclass
class JobRequest:
    """ジョブリクエスト"""
    job_type: JobType
    payload: Dict[str, Any]
    priority: Optional[JobPriority] = None
    max_retries: Optional[int] = None
    timeout: Optional[int] = None
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    correlation_id: Optional[str] = None
    callback_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # ジョブIDの生成
        if not hasattr(self, 'job_id'):
            self.job_id = str(uuid.uuid4())

        # correlation_idがない場合はjob_idを使用
        if not self.correlation_id:
            self.correlation_id = self.job_id


@dataclass
class DeadLetterRecord:
    """デッドレターキューレコード"""
    job_id: str
    job_type: JobType
    original_payload: Dict[str, Any]
    failure_reason: str
    retry_count: int
    first_failed_at: datetime
    last_failed_at: datetime
    worker_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, error_message: str, worker_id: Optional[str] = None):
        """エラー情報を追加"""
        error_entry = f"{datetime.now().isoformat()}: {error_message}"
        if worker_id:
            error_entry += f" (Worker: {worker_id})"
        self.worker_errors.append(error_entry)
        self.last_failed_at = datetime.now()


@dataclass
class QueueStatistics:
    """キュー統計情報"""
    queue_name: str
    active_jobs: int = 0
    pending_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    dead_letter_jobs: int = 0

    # レート情報
    jobs_per_minute: float = 0.0
    average_execution_time: float = 0.0
    average_queue_time: float = 0.0

    # エラー率
    error_rate: float = 0.0
    retry_rate: float = 0.0

    # リソース使用量
    average_memory_usage: float = 0.0
    average_cpu_usage: float = 0.0

    # 更新時刻
    last_updated: datetime = field(default_factory=datetime.now)

    def calculate_rates(self):
        """各種レートを計算"""
        total_jobs = self.completed_jobs + self.failed_jobs

        if total_jobs > 0:
            self.error_rate = self.failed_jobs / total_jobs

        # リトライ率は別途計算が必要（リトライされたジョブ数のデータが必要）

    def get_total_jobs(self) -> int:
        """総ジョブ数を取得"""
        return (self.active_jobs + self.pending_jobs +
                self.completed_jobs + self.failed_jobs + self.dead_letter_jobs)


@dataclass
class WorkerStatus:
    """ワーカー状態情報"""
    worker_id: str
    hostname: str
    is_active: bool = True

    # 現在の状態
    current_job_id: Optional[str] = None
    current_job_type: Optional[JobType] = None
    job_started_at: Optional[datetime] = None

    # 統計情報
    total_jobs_processed: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0

    # パフォーマンス情報
    average_job_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    # 接続情報
    last_heartbeat: datetime = field(default_factory=datetime.now)
    connected_at: datetime = field(default_factory=datetime.now)

    def is_healthy(self, heartbeat_timeout: int = 60) -> bool:
        """ワーカーが健全かチェック"""
        if not self.is_active:
            return False

        time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        return time_since_heartbeat < heartbeat_timeout

    def update_heartbeat(self):
        """ハートビートを更新"""
        self.last_heartbeat = datetime.now()

    def get_success_rate(self) -> float:
        """成功率を取得"""
        if self.total_jobs_processed == 0:
            return 0.0
        return self.successful_jobs / self.total_jobs_processed