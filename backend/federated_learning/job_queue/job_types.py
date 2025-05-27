"""
ジョブタイプと優先度定義
Task 4.3: 非同期ジョブキュー
"""

from enum import Enum, IntEnum
from typing import Dict, Any


class JobType(Enum):
    """ジョブタイプ定義"""
    MODEL_TRAINING = "model_training"
    AGGREGATION = "aggregation"
    ENCRYPTION = "encryption"
    DECRYPTION = "decryption"
    HEALTH_CHECK = "health_check"
    DATA_SYNC = "data_sync"
    METRICS_COLLECTION = "metrics_collection"
    CLEANUP = "cleanup"
    NOTIFICATION = "notification"
    BACKUP = "backup"


class JobPriority(IntEnum):
    """ジョブ優先度（数値が小さいほど高優先度）"""
    CRITICAL = 1    # 緊急フェイルオーバー、セキュリティ関連
    HIGH = 2        # リアルタイム集約、重要な訓練ジョブ
    NORMAL = 3      # 通常の訓練、定期ヘルスチェック
    LOW = 4         # ログ処理、統計情報収集
    BATCH = 5       # バッチ処理、バックアップ


class JobStatus(Enum):
    """ジョブ実行状態"""
    PENDING = "pending"         # 実行待ち
    STARTED = "started"         # 実行中
    SUCCESS = "success"         # 成功完了
    FAILURE = "failure"         # 失敗
    RETRY = "retry"            # リトライ中
    REVOKED = "revoked"        # キャンセル
    REJECTED = "rejected"      # 拒否（リソース不足等）
    DEAD_LETTER = "dead_letter" # デッドレターキュー行き


class JobCategory(Enum):
    """ジョブカテゴリ（業務領域別）"""
    FEDERATED_LEARNING = "federated_learning"
    SECURITY = "security"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"
    ANALYTICS = "analytics"


# ジョブタイプとデフォルト設定のマッピング
JOB_TYPE_CONFIGS: Dict[JobType, Dict[str, Any]] = {
    JobType.MODEL_TRAINING: {
        "default_priority": JobPriority.HIGH,
        "max_retries": 3,
        "timeout": 3600,  # 1時間
        "category": JobCategory.FEDERATED_LEARNING
    },
    JobType.AGGREGATION: {
        "default_priority": JobPriority.HIGH,
        "max_retries": 5,
        "timeout": 1800,  # 30分
        "category": JobCategory.FEDERATED_LEARNING
    },
    JobType.ENCRYPTION: {
        "default_priority": JobPriority.NORMAL,
        "max_retries": 3,
        "timeout": 600,   # 10分
        "category": JobCategory.SECURITY
    },
    JobType.DECRYPTION: {
        "default_priority": JobPriority.HIGH,
        "max_retries": 3,
        "timeout": 600,   # 10分
        "category": JobCategory.SECURITY
    },
    JobType.HEALTH_CHECK: {
        "default_priority": JobPriority.NORMAL,
        "max_retries": 2,
        "timeout": 60,    # 1分
        "category": JobCategory.MONITORING
    },
    JobType.DATA_SYNC: {
        "default_priority": JobPriority.NORMAL,
        "max_retries": 5,
        "timeout": 1200,  # 20分
        "category": JobCategory.FEDERATED_LEARNING
    },
    JobType.METRICS_COLLECTION: {
        "default_priority": JobPriority.LOW,
        "max_retries": 2,
        "timeout": 300,   # 5分
        "category": JobCategory.ANALYTICS
    },
    JobType.CLEANUP: {
        "default_priority": JobPriority.LOW,
        "max_retries": 1,
        "timeout": 1800,  # 30分
        "category": JobCategory.MAINTENANCE
    },
    JobType.NOTIFICATION: {
        "default_priority": JobPriority.NORMAL,
        "max_retries": 3,
        "timeout": 30,    # 30秒
        "category": JobCategory.MONITORING
    },
    JobType.BACKUP: {
        "default_priority": JobPriority.BATCH,
        "max_retries": 2,
        "timeout": 7200,  # 2時間
        "category": JobCategory.MAINTENANCE
    }
}


def get_job_config(job_type: JobType) -> Dict[str, Any]:
    """
    ジョブタイプのデフォルト設定を取得

    Args:
        job_type: ジョブタイプ

    Returns:
        Dict[str, Any]: ジョブ設定
    """
    return JOB_TYPE_CONFIGS.get(job_type, {
        "default_priority": JobPriority.NORMAL,
        "max_retries": 3,
        "timeout": 600,
        "category": JobCategory.FEDERATED_LEARNING
    })


def get_priority_queue_name(priority: JobPriority) -> str:
    """
    優先度に基づくキュー名を取得

    Args:
        priority: ジョブ優先度

    Returns:
        str: キュー名
    """
    priority_map = {
        JobPriority.CRITICAL: "critical_queue",
        JobPriority.HIGH: "high_queue",
        JobPriority.NORMAL: "normal_queue",
        JobPriority.LOW: "low_queue",
        JobPriority.BATCH: "batch_queue"
    }
    return priority_map.get(priority, "normal_queue")