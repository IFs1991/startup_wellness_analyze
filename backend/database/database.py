from typing import AsyncGenerator, List, Any, Sequence
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy import text

import sys
from pathlib import Path

# configモジュールへのパスを追加
config_path = Path(__file__).parent.parent
if str(config_path) not in sys.path:
    sys.path.append(str(config_path))

from config.config import get_settings

settings = get_settings()

# 非同期データベースエンジンの作成
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    **settings.SQLALCHEMY_CONFIG
)

# 非同期セッションの設定
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """データベースセッションの依存性注入用ジェネレータ"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# データベース操作のヘルパー関数
async def get_collection_data(session: AsyncSession, table_name: str) -> List[dict]:
    """
    指定されたテーブルのデータを取得するヘルパー関数

    Args:
        session (AsyncSession): データベースセッション
        table_name (str): 取得するテーブル名

    Returns:
        List[dict]: レコードのリスト（辞書形式）
    """
    # SQLインジェクション対策のためにパラメータバインディングを使用
    stmt = text(f"SELECT * FROM {table_name}")
    result = await session.execute(stmt)
    rows = result.fetchall()
    return [dict(row._mapping) for row in rows]

async def execute_batch_operations(session: AsyncSession, operations: List[str]):
    """
    バッチ処理を実行するヘルパー関数

    Args:
        session (AsyncSession): データベースセッション
        operations (List[str]): SQL操作のリスト
    """
    try:
        for operation in operations:
            stmt = text(operation)
            await session.execute(stmt)
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise e

# 設定ファイル (backend/config.py)