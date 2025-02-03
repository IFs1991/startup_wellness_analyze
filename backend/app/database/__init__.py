# -*- coding: utf-8 -*-
"""
データベース初期化モジュール
SQLAlchemy 2.0 スタイルで実装
"""
import logging
import os
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

from sqlalchemy import URL
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings
from .session_manager import database_manager, get_database_manager

# ロギングの設定
logger = logging.getLogger(__name__)

# バックアップ設定
BACKUP_DIR = Path(__file__).parent.parent.parent / 'backups'
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# 開発環境の場合のみ .env を読み込み
if os.getenv("ENVIRONMENT") != "production":
    load_dotenv()

# データベースURL設定
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'startup_wellness_analyze')}"
)

# ベースモデルクラス
class Base(DeclarativeBase):
    pass

# バックアップ関連の関数
async def create_backup() -> Path:
    """データベースの完全バックアップを作成"""
    try:
        if not os.getenv('DB_PASSWORD'):
            raise ValueError("DB_PASSWORD environment variable is not set")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = BACKUP_DIR / f'backup_{timestamp}.sql'

        # pg_dumpコマンドの構築
        cmd = [
            'pg_dump',
            '-h', os.getenv('DB_HOST', 'localhost'),
            '-p', os.getenv('DB_PORT', '5432'),
            '-U', os.getenv('DB_USER', 'postgres'),
            '-F', 'c',
            '-b',
            '-v',
            '-f', str(backup_file),
            os.getenv('DB_NAME', 'startup_wellness_analyze')
        ]

        # 環境変数の設定
        env = os.environ.copy()
        env['PGPASSWORD'] = os.getenv('DB_PASSWORD')

        # バックアップの実行
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logger.info(f"バックアップが正常に作成されました: {backup_file}")
            await cleanup_old_backups(30)
            return backup_file
        else:
            raise Exception(f"バックアップ作成エラー: {stderr.decode()}")

    except Exception as e:
        logger.error(f"バックアップ作成に失敗しました: {str(e)}")
        raise

async def restore_from_backup(backup_file: Path) -> None:
    """バックアップからデータベースを復元"""
    try:
        if not os.getenv('DB_PASSWORD'):
            raise ValueError("DB_PASSWORD environment variable is not set")

        if not backup_file.exists():
            raise FileNotFoundError(f"バックアップファイルが見つかりません: {backup_file}")

        cmd = [
            'pg_restore',
            '-h', os.getenv('DB_HOST', 'localhost'),
            '-p', os.getenv('DB_PORT', '5432'),
            '-U', os.getenv('DB_USER', 'postgres'),
            '-d', os.getenv('DB_NAME', 'startup_wellness_analyze'),
            '-v',
            '--clean',
            '--if-exists',
            str(backup_file)
        ]

        env = os.environ.copy()
        env['PGPASSWORD'] = os.getenv('DB_PASSWORD')

        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logger.info(f"データベースが正常に復元されました: {backup_file}")
        else:
            raise Exception(f"復元エラー: {stderr.decode()}")

    except Exception as e:
        logger.error(f"データベース復元に失敗しました: {str(e)}")
        raise

async def cleanup_old_backups(days: int) -> None:
    """古いバックアップファイルを削除"""
    try:
        current_time = datetime.now()
        for backup_file in BACKUP_DIR.glob('backup_*.sql'):
            file_time = datetime.fromtimestamp(backup_file.stat().st_ctime)
            if (current_time - file_time).days > days:
                backup_file.unlink()
                logger.info(f"古いバックアップを削除しました: {backup_file}")

    except Exception as e:
        logger.error(f"バックアップクリーンアップに失敗しました: {str(e)}")
        raise

def get_db_config() -> Dict[str, str]:
    """データベース設定を取得"""
    if not os.getenv('DB_PASSWORD'):
        raise ValueError("DB_PASSWORD environment variable is not set")

    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "database": os.getenv("DB_NAME", "startup_wellness_analyze"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD")
    }

def init_db():
    """データベースの初期化"""
    env = os.environ.copy()
    if not os.getenv('DB_PASSWORD'):
        raise ValueError("DB_PASSWORD environment variable is not set")
    env['PGPASSWORD'] = os.getenv('DB_PASSWORD')
    # ... 残りのコード ...