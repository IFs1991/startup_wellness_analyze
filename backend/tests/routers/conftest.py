import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Generator

# メインアプリケーションのインポート
from api.main import app
from core.auth_manager import AuthManager, User, UserRole

@pytest.fixture
def test_app() -> FastAPI:
    """テスト用のFastAPIアプリケーションを提供します"""
    return app

@pytest.fixture
def client(test_app: FastAPI) -> TestClient:
    """テスト用のクライアントを提供します"""
    return TestClient(test_app)

@pytest.fixture
def mock_auth_manager() -> Generator[MagicMock, None, None]:
    """認証マネージャーのモックを提供します"""
    with patch("core.auth_manager.AuthManager") as mock:
        auth_manager = MagicMock()
        mock.return_value = auth_manager
        yield auth_manager

@pytest.fixture
def mock_user() -> User:
    """テスト用ユーザーを提供します"""
    return User(
        id="test_user_id",
        email="test@example.com",
        display_name="Test User",
        is_active=True,
        role=UserRole.USER,
        company_id="test_company_id"
    )

@pytest.fixture
def mock_admin_user() -> User:
    """テスト用管理者ユーザーを提供します"""
    return User(
        id="admin_user_id",
        email="admin@example.com",
        display_name="Admin User",
        is_active=True,
        role=UserRole.ADMIN,
        company_id="test_company_id"
    )

@pytest.fixture
def mock_analyst_user() -> User:
    """テスト用アナリストユーザーを提供します"""
    return User(
        id="analyst_user_id",
        email="analyst@example.com",
        display_name="Analyst User",
        is_active=True,
        role=UserRole.ANALYST,
        company_id="test_company_id"
    )

@pytest.fixture
def token_header() -> Dict[str, str]:
    """認証トークンヘッダーを提供します"""
    return {"Authorization": "Bearer test_token"}