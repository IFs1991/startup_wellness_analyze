import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from backend.src.database.connection import DatabaseConnection
from backend.src.database.config import DatabaseConfig

@pytest.mark.asyncio
async def test_database_connection(database_connection):
    """データベース接続のテスト"""
    assert isinstance(database_connection, DatabaseConnection)
    is_connected = await database_connection.check_connection()
    assert is_connected is True

@pytest.mark.asyncio
async def test_session(session):
    """セッションのテスト"""
    assert isinstance(session, AsyncSession)

def test_config(test_config):
    """設定のテスト"""
    assert isinstance(test_config, DatabaseConfig)
    assert test_config.host == "localhost"
    assert test_config.port == 5432
    assert test_config.database == "test_startup_wellness"
    assert test_config.username == "postgres"
    assert test_config.password == "postgres"

@pytest.mark.asyncio
async def test_database(test_database):
    """データベースのテスト"""
    assert isinstance(test_database, DatabaseConnection)
    is_connected = await test_database.check_connection()
    assert is_connected is True

def test_test_data(
    test_user_data,
    test_company_data,
    test_group_data,
    test_template_data,
    test_report_data,
    test_status_data,
    test_stage_data,
    test_tag_data
):
    """テストデータのテスト"""
    # ユーザーデータ
    assert test_user_data["username"] == "test_user"
    assert test_user_data["email"] == "test@example.com"
    assert test_user_data["password"] == "test_password"
    assert test_user_data["role"] == "user"

    # 会社データ
    assert test_company_data["name"] == "Test Company"
    assert test_company_data["industry"] == "Technology"
    assert test_company_data["employee_count"] == 100
    assert test_company_data["website"] == "https://example.com"
    assert test_company_data["location"] == "Tokyo, Japan"

    # グループデータ
    assert test_group_data["name"] == "Test Group"
    assert test_group_data["description"] == "Test Description"
    assert test_group_data["is_private"] is False

    # テンプレートデータ
    assert test_template_data["name"] == "Test Template"
    assert test_template_data["format"] == "pdf"
    assert test_template_data["template_content"] == "Test Content"
    assert test_template_data["variables"] == {"key": "value"}

    # レポートデータ
    assert test_report_data["content"] == "Test Report Content"
    assert test_report_data["format"] == "pdf"
    assert test_report_data["report_metadata"]["status"] == "completed"

    # ステータスデータ
    assert test_status_data["type"] == "ACTIVE"
    assert test_status_data["description"] == "Company is active"

    # ステージデータ
    assert test_stage_data["type"] == "SEED"
    assert test_stage_data["description"] == "Seed stage"

    # タグデータ
    assert test_tag_data["name"] == "test_tag"