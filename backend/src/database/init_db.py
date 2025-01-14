import asyncio
import logging
from typing import Optional
from .connection import DatabaseConnection
from .config import DatabaseConfig
from .models import Base, User, UserRole

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_database(config: Optional[DatabaseConfig] = None) -> None:
    """データベースを初期化する"""
    if config is None:
        config = DatabaseConfig()

    logger.info("データベースの初期化を開始します...")

    # データベース接続を作成
    db = DatabaseConnection(config.get_database_url())

    try:
        # データベース接続を確認
        if not await db.check_connection():
            logger.error("データベースに接続できません")
            return

        # テーブルを作成
        logger.info("テーブルを作成しています...")
        await db.create_database()

        # 初期データを作成
        logger.info("初期データを作成しています...")
        async with db.async_session_factory() as session:
            # 管理者ユーザーを作成
            admin_user = User(
                username="admin",
                email="admin@example.com",
                hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiAYMyzJ/eRm",  # "admin"のハッシュ値
                role=UserRole.ADMIN,
                is_active=True
            )
            session.add(admin_user)

            # テストユーザーを作成
            test_user = User(
                username="test",
                email="test@example.com",
                hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiAYMyzJ/eRm",  # "test"のハッシュ値
                role=UserRole.USER,
                is_active=True
            )
            session.add(test_user)

            await session.commit()

        logger.info("データベースの初期化が完了しました")

    except Exception as e:
        logger.error(f"データベースの初期化中にエラーが発生しました: {e}")
        raise

    finally:
        await db.cleanup()

async def drop_database(config: Optional[DatabaseConfig] = None) -> None:
    """データベースを削除する"""
    if config is None:
        config = DatabaseConfig()

    logger.info("データベースの削除を開始します...")

    # データベース接続を��成
    db = DatabaseConnection(config.get_database_url())

    try:
        # データベース接続を確認
        if not await db.check_connection():
            logger.error("データベースに接続できません")
            return

        # テーブルを削除
        logger.info("テーブルを削除しています...")
        await db.drop_database()

        logger.info("データベースの削除が完了しました")

    except Exception as e:
        logger.error(f"データベースの削除中にエラーが発生しました: {e}")
        raise

    finally:
        await db.cleanup()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="データベース初期化スクリプト")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="データベースを削除する"
    )
    args = parser.parse_args()

    if args.drop:
        asyncio.run(drop_database())
    else:
        asyncio.run(init_database())