"""
データベースバックアップスケジューラ
定期的なバックアップを管理します。
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import logging
from . import create_backup, db_manager

logger = logging.getLogger(__name__)

class BackupScheduler:
    """バックアップスケジューラクラス"""

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self._setup_schedules()

    def _setup_schedules(self):
        """バックアップスケジュールを設定"""
        # 毎日の完全バックアップ（深夜1時）
        self.scheduler.add_job(
            self._daily_backup,
            CronTrigger(hour=1),
            id='daily_backup',
            name='Daily full backup'
        )

        # 週次バックアップ（日曜日の深夜2時）
        self.scheduler.add_job(
            self._weekly_backup,
            CronTrigger(day_of_week='sun', hour=2),
            id='weekly_backup',
            name='Weekly backup with recovery point'
        )

        # 月次バックアップ（毎月1日の深夜3時）
        self.scheduler.add_job(
            self._monthly_backup,
            CronTrigger(day=1, hour=3),
            id='monthly_backup',
            name='Monthly backup with extended retention'
        )

    def start(self):
        """スケジューラを開始"""
        try:
            self.scheduler.start()
            logger.info("Backup scheduler started successfully")
        except Exception as e:
            logger.error(f"Failed to start backup scheduler: {str(e)}")
            raise

    def stop(self):
        """スケジューラを停止"""
        try:
            self.scheduler.shutdown()
            logger.info("Backup scheduler stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop backup scheduler: {str(e)}")
            raise

    def _daily_backup(self):
        """日次バックアップを実行"""
        try:
            backup_file = create_backup()
            logger.info(f"Daily backup completed: {backup_file}")
        except Exception as e:
            logger.error(f"Daily backup failed: {str(e)}")

    def _weekly_backup(self):
        """週次バックアップを実行"""
        try:
            # リカバリポイント付きのバックアップを作成
            recovery_point_name = f"weekly_{datetime.now().strftime('%Y%m%d')}"
            db_manager.create_recovery_point(recovery_point_name)
            logger.info(f"Weekly backup completed with recovery point: {recovery_point_name}")
        except Exception as e:
            logger.error(f"Weekly backup failed: {str(e)}")

    def _monthly_backup(self):
        """月次バックアップを実行"""
        try:
            # 長期保存用のバックアップを作成
            backup_file = create_backup()
            # 月次バックアップは90日間保持
            recovery_point_name = f"monthly_{datetime.now().strftime('%Y%m')}"
            db_manager.create_recovery_point(recovery_point_name)
            logger.info(f"Monthly backup completed: {backup_file}")
        except Exception as e:
            logger.error(f"Monthly backup failed: {str(e)}")

# スケジューラのインスタンスを作成
backup_scheduler = BackupScheduler()

# バックアップを作成
backup_file = db_manager.create_backup()

# リカバリポイントを作成
db_manager.create_recovery_point("before_important_update")

# リカバリポイントに復元
db_manager.restore_to_point("before_important_update")