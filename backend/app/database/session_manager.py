# -*- coding: utf-8 -*-
"""
データベースセッション管理モジュール
"""
from typing import AsyncGenerator, Optional
import logging
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy import URL, text

from app.core.config import settings

logger = logging.getLogger(__name__)

class DatabaseSessionManager:
    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None

    async def init(self, database_url: URL | str) -> None:
        """データベースエンジンとセッションメーカーを初期化"""
        if self._engine is not None:
            logger.warning("DatabaseSessionManagerは既に初期化されています")
            return

        self._engine = create_async_engine(
            database_url,
            echo=settings.DEBUG,
            pool_size=settings.SQLALCHEMY_CONFIG["pool_size"],
            max_overflow=settings.SQLALCHEMY_CONFIG["max_overflow"],
            pool_timeout=settings.SQLALCHEMY_CONFIG["pool_timeout"],
            pool_recycle=settings.SQLALCHEMY_CONFIG["pool_recycle"],
            pool_pre_ping=settings.SQLALCHEMY_CONFIG["pool_pre_ping"],
            connect_args={
                'command_timeout': 10,
                'options': f'-c application_name={settings.APP_NAME}'
            }
        )

        self._sessionmaker = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False
        )

        # 接続テスト
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
                await session.commit()
                logger.info("データベース接続が正常に確立されました")
        except Exception as e:
            logger.error(f"データベース接続テストに失敗しました: {str(e)}")
            await self.close()
            raise

    async def close(self) -> None:
        """データベース接続をクリーンアップ"""
        if self._engine is None:
            return

        await self._engine.dispose()
        self._engine = None
        self._sessionmaker = None
        logger.info("データベース接続をクローズしました")

    def get_session(self) -> AsyncSession:
        """新しいデータベースセッションを取得"""
        if self._sessionmaker is None:
            raise RuntimeError("DatabaseSessionManagerが初期化されていません")
        return self._sessionmaker()

    async def get_db(self) -> AsyncGenerator[AsyncSession, None]:
        """FastAPI依存性注入用のデータベースセッションジェネレータ"""
        if self._sessionmaker is None:
            raise RuntimeError("DatabaseSessionManagerが初期化されていません")

        async with self._sessionmaker() as session:
            try:
                yield session
            except Exception as e:
                logger.error(f"Database error: {str(e)}")
                await session.rollback()
                raise
            finally:
                await session.close()

@lru_cache()
def get_database_manager() -> DatabaseSessionManager:
    """シングルトンのDatabaseSessionManagerインスタンスを取得"""
    return DatabaseSessionManager()

# グローバルなデータベースマネージャーインスタンス
database_manager = get_database_manager()