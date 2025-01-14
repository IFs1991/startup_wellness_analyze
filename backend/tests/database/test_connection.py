import pytest
import asyncio
from sqlalchemy import text
from backend.src.database.connection import DatabaseConnection
from backend.src.database.config import PostgresTestConfig
from backend.src.database.models import Base

@pytest.fixture
async def db_connection():
    """テスト用のデータベース接続を作成する"""
    config = PostgresTestConfig(
        host="localhost",
        port=5432,
        database="test_startup_wellness",
        username="postgres",
        password="postgres",
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        echo=False,
        timezone="UTC"
    )
    connection = DatabaseConnection(config.get_database_url())
    await connection.initialize()
    yield connection
    await connection.cleanup()

@pytest.fixture(autouse=True)
async def cleanup_database(db_connection):
    """テスト後にデータベースをクリーンアップする"""
    yield db_connection

@pytest.mark.asyncio
async def test_database_connection_initialization(db_connection):
    """データベース接続の初期化をテストする"""
    # 接続を確認
    is_connected = await db_connection.check_connection()
    assert is_connected is True

    # セッションを取得
    async with db_connection.get_session() as session:
        # 簡単なクエリを実行
        result = await session.execute(text("SELECT 1"))
        value = result.scalar()
        assert value == 1

@pytest.mark.asyncio
async def test_database_connection_from_config():
    """設定からのデータベース接続をテストする"""
    # テスト用の設定を取得
    config = PostgresTestConfig(
        host="localhost",
        port=5432,
        database="test_startup_wellness",
        username="postgres",
        password="postgres",
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        echo=False,
        timezone="UTC"
    )

    # 設定からデータベース接続を作成
    connection = DatabaseConnection(config.get_database_url())
    try:
        await connection.initialize()
        # 接続を確認
        is_connected = await connection.check_connection()
        assert is_connected is True

        # ���ッションを取得して簡単なクエリを実行
        async with connection.get_session() as session:
            result = await session.execute(text("SELECT 1"))
            value = result.scalar()
            assert value == 1
    finally:
        await connection.cleanup()

@pytest.mark.asyncio
async def test_database_connection_cleanup(db_connection):
    """データベース接続のクリーンアップをテストする"""
    # データベースを作成
    await db_connection.create_database()

    # データベースを削除
    await db_connection.drop_database()

    # 接続をクリーンアップ
    await db_connection.cleanup()

    # 接続が切られていることを確認
    with pytest.raises(Exception):
        async with db_connection.get_session() as session:
            await session.execute(text("SELECT 1"))

@pytest.mark.asyncio
async def test_database_connection_error_handling():
    """データベース接続のエラーンドリングをテストする"""
    # 無効なデータベースURLで接続を試みる
    invalid_url = "postgresql+asyncpg://invalid:invalid@localhost:5432/invalid_db"
    connection = DatabaseConnection(invalid_url)

    try:
        # 接続を確認
        is_connected = await connection.check_connection()
        assert is_connected is False

        # 初期化を試みる
        with pytest.raises(Exception):
            await connection.initialize()
    finally:
        await connection.cleanup()

@pytest.mark.asyncio
async def test_database_connection_pool():
    """データベース接続プールをテストする"""
    config = PostgresTestConfig(
        host="localhost",
        port=5432,
        database="test_startup_wellness",
        username="postgres",
        password="postgres",
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        echo=False,
        timezone="UTC"
    )
    connection = DatabaseConnection(config.get_database_url())
    try:
        await connection.initialize()
        # 複数のセッションを同時に作成
        sessions = []
        for _ in range(3):
            async with connection.get_session() as session:
                sessions.append(session)
                # 簡単なクエリを実行
                result = await session.execute(text("SELECT 1"))
                value = result.scalar()
                assert value == 1
    finally:
        await connection.cleanup()

@pytest.mark.asyncio
async def test_database_connection_with_config():
    """データベース設定を使用した接続をテストする"""
    # デスト用の設定を取得
    config = PostgresTestConfig(
        host="localhost",
        port=5432,
        database="test_startup_wellness",
        username="postgres",
        password="postgres",
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        echo=False,
        timezone="UTC"
    )

    # 設定からデータベース接続を作成
    connection = DatabaseConnection(config.get_database_url())
    try:
        await connection.initialize()
        # 接続を確認
        is_connected = await connection.check_connection()
        assert is_connected is True

        # セッションを取得して簡単なクエリを実行
        async with connection.get_session() as session:
            result = await session.execute(text("SELECT 1"))
            value = result.scalar()
            assert value == 1
    finally:
        await connection.cleanup()