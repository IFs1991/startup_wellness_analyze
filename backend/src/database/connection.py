from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from ..database.models import Base

class DatabaseConnection:
    """データベース接続を管理するクラス"""

    def __init__(self, database_url: str):
        """データベース接続を初期化する"""
        self.database_url = database_url
        self.engine = create_async_engine(
            database_url,
            echo=False,
            pool_size=5,
            max_overflow=10
        )
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def initialize(self):
        """データベース接続を初期化する"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(lambda _: None)
            return True
        except Exception:
            return False

    async def cleanup(self):
        """データベース接続をクリーンアップする"""
        await self.engine.dispose()

    async def check_connection(self) -> bool:
        """データベース接続を確認する"""
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    @asynccontextmanager
    async def get_session(self):
        """データベースセッションを取得する"""
        session = self.async_session()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def create_database(self):
        """データベースを作成する"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_database(self):
        """データベースを削除する"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    @classmethod
    def from_config(cls, config: dict) -> 'DatabaseConnection':
        """設定からデータベース接続を作��する"""
        database_url = (
            f"postgresql+asyncpg://"
            f"{config['username']}:{config['password']}@"
            f"{config['host']}:{config['port']}/"
            f"{config['database']}"
        )
        return cls(database_url)