from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import AsyncEngine
from typing import AsyncGenerator
import os
from .models import Base

class Database:
    def __init__(self, database_url: str):
        self._engine: AsyncEngine = create_async_engine(
            database_url,
            echo=True,  # SQLログを出力
            pool_pre_ping=True,  # コネクションの生存確認
            pool_size=5,  # コネクションプールのサイズ
            max_overflow=10  # 最大オーバーフロー数
        )
        self._session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self._engine,
            class_=AsyncSession
        )

    async def create_database(self) -> None:
        """データベースとテーブルを作成する"""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_database(self) -> None:
        """データベース��テーブルを削除する"""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """セッションを取得する"""
        async with self._session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

class DatabaseSession:
    """データベースセッションのコンテキストマネージャー"""
    def __init__(self, session: AsyncSession):
        self.session = session

    async def __aenter__(self) -> AsyncSession:
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.session.rollback()
        await self.session.close()

# データベース設定
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://user:password@localhost:5432/startup_wellness"
)

# グローバルなデータベースインスタンス
database = Database(DATABASE_URL)

# 依存性注入用の関数
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPIの依存性注入で使用する���ッション取得関数"""
    async for session in database.get_session():
        yield session