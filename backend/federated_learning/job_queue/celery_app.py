"""
Celeryアプリケーション設定
Task 4.3: 非同期ジョブキュー
"""

import os
from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from kombu import Queue, Exchange
from .job_types import JobPriority, get_priority_queue_name


# Celery設定
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'amqp://guest:guest@localhost:5672//')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Celeryアプリケーションの作成
celery_app = Celery(
    'federated_learning_jobs',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        'backend.federated_learning.job_queue.fl_jobs',
        'backend.federated_learning.job_queue.tasks'
    ]
)

# 優先度キュー設定
priority_queues = []
for priority in JobPriority:
    queue_name = get_priority_queue_name(priority)
    priority_queues.append(
        Queue(
            queue_name,
            exchange=Exchange('federated_learning', type='direct'),
            routing_key=queue_name,
            queue_arguments={
                'x-max-priority': 255,
                'x-message-ttl': 3600000,  # 1時間TTL
            }
        )
    )

# デッドレターキューの設定
dead_letter_queue = Queue(
    'dead_letter_queue',
    exchange=Exchange('federated_learning_dlx', type='direct'),
    routing_key='dead_letter',
    queue_arguments={
        'x-message-ttl': 86400000,  # 24時間TTL
        'x-max-length': 10000,      # 最大10,000メッセージ
    }
)

# Celery設定
celery_app.conf.update(
    # ブローカー設定
    broker_url=CELERY_BROKER_URL,
    result_backend=CELERY_RESULT_BACKEND,

    # タスク設定
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # ワーカー設定
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,

    # キュー設定
    task_routes={
        'backend.federated_learning.job_queue.fl_jobs.model_training_task': {
            'queue': 'high_queue',
            'priority': 8
        },
        'backend.federated_learning.job_queue.fl_jobs.aggregation_task': {
            'queue': 'high_queue',
            'priority': 9
        },
        'backend.federated_learning.job_queue.fl_jobs.encryption_task': {
            'queue': 'normal_queue',
            'priority': 5
        },
        'backend.federated_learning.job_queue.fl_jobs.health_check_task': {
            'queue': 'normal_queue',
            'priority': 3
        },
        'backend.federated_learning.job_queue.fl_jobs.data_sync_task': {
            'queue': 'normal_queue',
            'priority': 4
        },
        'backend.federated_learning.job_queue.fl_jobs.metrics_collection_task': {
            'queue': 'low_queue',
            'priority': 2
        },
        'backend.federated_learning.job_queue.fl_jobs.cleanup_task': {
            'queue': 'batch_queue',
            'priority': 1
        }
    },

    # キューの定義
    task_queues=priority_queues + [dead_letter_queue],

    # リトライ設定
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_retry_delay=60,
    task_max_retries=3,

    # 結果設定
    result_expires=3600,  # 1時間
    result_persistent=True,

    # 監視設定
    worker_send_task_events=True,
    task_send_sent_event=True,

    # デッドレター設定
    task_annotations={
        '*': {
            'dead_letter_exchange': 'federated_learning_dlx',
            'dead_letter_routing_key': 'dead_letter'
        }
    },

    # セキュリティ設定
    worker_hijack_root_logger=False,
    worker_log_color=False,

    # パフォーマンス設定
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,

    # ヘルスチェック設定
    worker_heartbeat=30,
    broker_heartbeat=120,

    # 並行性設定
    worker_concurrency=4,
    task_compression='gzip',
    result_compression='gzip'
)


@worker_process_init.connect
def worker_process_init_handler(signal, sender, **kwargs):
    """ワーカープロセス初期化ハンドラー"""
    print(f"Worker process initialized: {sender}")


@worker_process_shutdown.connect
def worker_process_shutdown_handler(signal, sender, **kwargs):
    """ワーカープロセス終了ハンドラー"""
    print(f"Worker process shutting down: {sender}")


# カスタムタスククラス
class JobTask(celery_app.Task):
    """カスタムタスククラス"""

    def on_success(self, retval, task_id, args, kwargs):
        """タスク成功時の処理"""
        print(f"Task {task_id} succeeded with result: {retval}")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """タスク失敗時の処理"""
        print(f"Task {task_id} failed with exception: {exc}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """タスクリトライ時の処理"""
        print(f"Task {task_id} retrying due to: {exc}")


# デフォルトタスククラスの設定
celery_app.Task = JobTask


def create_celery_worker():
    """Celeryワーカーインスタンスを作成"""
    return celery_app.Worker(
        loglevel='INFO',
        optimization='fair',
        pool_cls='prefork',
        concurrency=4
    )


def start_flower_monitoring(port=5555):
    """Flower監視ツールを開始"""
    import subprocess

    flower_cmd = [
        'celery',
        '-A', 'backend.federated_learning.job_queue.celery_app',
        'flower',
        f'--port={port}',
        '--basic_auth=admin:federated123'
    ]

    return subprocess.Popen(flower_cmd)


if __name__ == '__main__':
    celery_app.start()