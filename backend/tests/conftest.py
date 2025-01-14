import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from backend.src.database.models import Base
from backend.src.database.connection import DatabaseConnection
from backend.src.database.config import PostgresConfig, PostgresTestConfig

# テスト用のデータベースURL
TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/test_startup_wellness"

@pytest.fixture(scope="session")
def event_loop():
    """イベントループを作成する"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def database_connection():
    """テスト用のデータベース接続を作成する"""
    connection = DatabaseConnection(TEST_DATABASE_URL)
    await connection.initialize()
    yield connection
    await connection.cleanup()

@pytest.fixture(scope="session")
async def engine():
    """テスト用のデータベースエンジンを作成する"""
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()

@pytest.fixture
async def session(engine):
    """テスト用のセッションを作成する"""
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session

@pytest.fixture
def test_config():
    """テスト用のデータベース設定を作成する"""
    return PostgresTestConfig(
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

@pytest.fixture
async def test_database(database_connection, test_config):
    """テスト用のデータベースを初期化する"""
    # データベースを作成
    await database_connection.create_database()
    yield database_connection
    # データベースを削除
    await database_connection.drop_database()

@pytest.fixture
def test_user_data():
    """テスト用のユーザーデータを作成する"""
    return {
        "username": "test_user",
        "email": "test@example.com",
        "password": "test_password",
        "role": "user"
    }

@pytest.fixture
def test_company_data():
    """テスト用の会社データを作成する"""
    return {
        "name": "Test Company",
        "description": "Test Description",
        "industry": "Technology",
        "founded_date": "2024-01-01T00:00:00",
        "employee_count": 100,
        "website": "https://example.com",
        "location": "Tokyo, Japan"
    }

@pytest.fixture
def test_group_data():
    """テスト用のグループデータを作成する"""
    return {
        "name": "Test Group",
        "description": "Test Description",
        "is_private": False
    }

@pytest.fixture
def test_template_data():
    """テスト用のレポートテンプレートデータを作成する"""
    return {
        "name": "Test Template",
        "description": "Test Description",
        "format": "pdf",
        "template_content": "Test Content",
        "variables": {"key": "value"}
    }

@pytest.fixture
def test_report_data():
    """テスト用のレポートデータを作成する"""
    return {
        "content": "Test Report Content",
        "format": "pdf",
        "metadata": {"status": "completed"}
    }

@pytest.fixture
def test_status_data():
    """テスト用のステータスデータを作成する"""
    return {
        "type": "ACTIVE",
        "description": "Company is active"
    }

@pytest.fixture
def test_stage_data():
    """テスト用のステージデータを作成する"""
    return {
        "type": "SEED",
        "description": "Seed stage"
    }

@pytest.fixture
def test_tag_data():
    """テスト用のタグデータを作成する"""
    return {
        "name": "test_tag"
    }