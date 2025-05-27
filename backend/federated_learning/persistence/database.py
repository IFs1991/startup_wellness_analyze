# Phase 3 Task 3.1: データベース層の実装
# TDD GREEN段階: DatabaseManager実装

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timezone

import asyncpg
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session

# SQLAlchemyバージョン互換性対応
try:
    from sqlalchemy.ext.asyncio import async_sessionmaker
except ImportError:
    # 古いバージョンの場合はsessionmakerを使用
    from sqlalchemy.orm import sessionmaker as async_sessionmaker
from sqlalchemy.pool import NullPool
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from .models import Base

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """
    データベース接続とセッション管理を担当するクラス

    機能:
    - 非同期PostgreSQL接続管理
    - 接続プール管理
    - トランザクション管理
    - マイグレーション支援
    - ヘルスチェック
    """

    def __init__(
        self,
        database_url: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        echo: bool = False
    ):
        """
        DatabaseManagerの初期化

        Args:
            database_url: PostgreSQL接続URL
            pool_size: 接続プールサイズ
            max_overflow: 最大オーバーフロー接続数
            pool_timeout: プールタイムアウト（秒）
            echo: SQLログ出力フラグ
        """
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.echo = echo

        # 非同期エンジン
        self.async_engine = None
        self.async_session_factory = None

        # 同期エンジン（マイグレーション用）
        self.sync_engine = None
        self.sync_session_factory = None

        self.is_initialized = False
        self._connection_pool_info = {
            "active_connections": 0,
            "idle_connections": 0,
            "max_connections": pool_size + max_overflow
        }

    async def initialize(self) -> None:
        """データベース接続を初期化"""
        try:
            # データベースタイプ別のエンジン設定
            engine_kwargs = {
                "echo": self.echo,
                "pool_pre_ping": True,  # 接続確認
            }

            # SQLiteの場合はpool関連パラメータを除外
            if "sqlite" not in self.database_url:
                engine_kwargs.update({
                    "pool_size": self.pool_size,
                    "max_overflow": self.max_overflow,
                    "pool_timeout": self.pool_timeout,
                    "pool_recycle": 3600,   # 1時間で接続リサイクル
                })

            # 非同期エンジンの作成
            self.async_engine = create_async_engine(
                self.database_url,
                **engine_kwargs
            )

            # 非同期セッションファクトリ
            try:
                # 新しいバージョン
                from sqlalchemy.ext.asyncio import async_sessionmaker
                self.async_session_factory = async_sessionmaker(
                    bind=self.async_engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )
            except ImportError:
                # 古いバージョン
                self.async_session_factory = sessionmaker(
                    bind=self.async_engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )

            # 同期エンジン（マイグレーション用）
            if "postgresql+asyncpg://" in self.database_url:
                sync_url = self.database_url.replace("postgresql+asyncpg://", "postgresql://")
            elif "sqlite+aiosqlite://" in self.database_url:
                sync_url = self.database_url.replace("sqlite+aiosqlite://", "sqlite://")
            else:
                sync_url = self.database_url

            self.sync_engine = create_engine(
                sync_url,
                echo=self.echo,
                poolclass=NullPool  # マイグレーション用は接続プール不要
            )

            self.sync_session_factory = sessionmaker(
                bind=self.sync_engine,
                expire_on_commit=False
            )

            # 接続テスト
            await self._test_connection()

            self.is_initialized = True
            logger.info("Database manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _test_connection(self) -> None:
        """接続テスト（リトライ付き）"""
        async with self.async_engine.begin() as conn:
            from sqlalchemy import text
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    async def is_connected(self) -> bool:
        """データベース接続状態を確認"""
        if not self.is_initialized or not self.async_engine:
            return False

        try:
            async with self.async_engine.begin() as conn:
                from sqlalchemy import text
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.warning(f"Database connection check failed: {e}")
            return False

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """非同期セッションを取得（コンテキストマネージャー）"""
        if not self.is_initialized:
            raise RuntimeError("DatabaseManager is not initialized")

        async with self.async_session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Session error, rolling back: {e}")
                raise
            finally:
                await session.close()

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """トランザクション管理付きセッション"""
        async with self.get_session() as session:
            async with session.begin():
                try:
                    yield session
                except Exception as e:
                    logger.error(f"Transaction error: {e}")
                    raise

    def get_sync_session(self) -> Session:
        """同期セッションを取得（マイグレーション用）"""
        if not self.sync_session_factory:
            raise RuntimeError("Sync session factory is not initialized")
        return self.sync_session_factory()

    async def get_pool_info(self) -> Dict[str, Any]:
        """接続プール情報を取得"""
        if not self.async_engine:
            return self._connection_pool_info

        pool = self.async_engine.pool

        # SQLiteのStaticPoolとPostgreSQLのQueuePoolで異なるメソッドを使用
        try:
            return {
                "active_connections": pool.checkedout(),
                "idle_connections": pool.checkedin(),
                "max_connections": self.pool_size + self.max_overflow,
                "pool_size": self.pool_size,
                "overflow": pool.overflow() if hasattr(pool, 'overflow') else 0,
                "invalid_connections": pool.invalidated() if hasattr(pool, 'invalidated') else 0
            }
        except AttributeError:
            # SQLite StaticPoolの場合
            return {
                "active_connections": getattr(pool, '_creator_count', 0),
                "idle_connections": 0,
                "max_connections": 1,  # StaticPoolは通常1接続
                "pool_size": 1,
                "overflow": 0,
                "invalid_connections": 0
            }

    async def run_migrations(self) -> Dict[str, str]:
        """
        マイグレーションを実行

        注意: 実際のプロダクションでは Alembic を使用することを推奨
        ここではテスト用の簡易実装
        """
        try:
            # 開発・テスト環境でのテーブル作成
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            logger.info("Database migrations completed successfully")
            return {"status": "success", "message": "All migrations applied"}

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return {"status": "failed", "message": str(e)}

    async def create_tables(self) -> None:
        """テーブルを作成（開発・テスト用）"""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")

    async def drop_tables(self) -> None:
        """テーブルを削除（テスト用）"""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped")

    async def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        try:
            if self.async_engine:
                await self.async_engine.dispose()
                logger.info("Async engine disposed")

            if self.sync_engine:
                self.sync_engine.dispose()
                logger.info("Sync engine disposed")

            self.is_initialized = False

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック情報を取得"""
        health_info = {
            "database_connected": False,
            "pool_info": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "initialized": self.is_initialized
        }

        try:
            health_info["database_connected"] = await self.is_connected()
            health_info["pool_info"] = await self.get_pool_info()

        except Exception as e:
            health_info["error"] = str(e)

        return health_info

    async def execute_raw_query(self, query: str, params: Optional[Dict] = None) -> Any:
        """生SQLクエリを実行（管理用）"""
        async with self.get_session() as session:
            result = await session.execute(query, params or {})
            return result

    def __repr__(self):
        return f"<DatabaseManager(initialized={self.is_initialized})>"


# シングルトンパターンでグローバルインスタンスを管理
_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """グローバルDatabaseManagerインスタンスを取得"""
    global _database_manager
    if _database_manager is None:
        raise RuntimeError("DatabaseManager not initialized. Call init_database_manager() first.")
    return _database_manager


def init_database_manager(database_url: str, **kwargs) -> DatabaseManager:
    """グローバルDatabaseManagerを初期化"""
    global _database_manager
    _database_manager = DatabaseManager(database_url, **kwargs)
    return _database_manager


async def cleanup_database_manager() -> None:
    """グローバルDatabaseManagerをクリーンアップ"""
    global _database_manager
    if _database_manager:
        await _database_manager.cleanup()
        _database_manager = None