import pytest
from httpx import AsyncClient
from backend.main import app
from backend.src.database.models import User
from backend.src.database.repositories.user import UserRepository

pytestmark = pytest.mark.asyncio

@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
async def test_user():
    """テスト用のユーザーデータを作成する"""
    return {
        "username": "test_user",
        "email": "test@example.com",
        "password": "test_password",
        "role": "user"
    }

async def test_register_user(async_client, test_user_data):
    """ユーザー登録のテスト"""
    response = await async_client.post("/api/users/register", json=test_user_data)
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == test_user_data["username"]
    assert data["email"] == test_user_data["email"]
    assert data["role"] == test_user_data["role"]
    assert "password" not in data

async def test_login_user(async_client, test_user_data):
    """ユーザーログインのテスト"""
    # ユーザーを登録
    await async_client.post("/api/users/register", json=test_user_data)

    # ログイン
    login_data = {
        "username": test_user_data["username"],
        "password": test_user_data["password"]
    }
    response = await async_client.post("/api/users/login", json=login_data)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

async def test_get_current_user(async_client, auth_headers):
    """現在のユーザー情報取得のテスト"""
    response = await async_client.get("/api/users/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "username" in data
    assert "email" in data
    assert "role" in data

async def test_update_user(async_client, test_user_data, auth_headers):
    """ユーザー情報更新のテスト"""
    update_data = {
        "email": "updated@example.com",
        "username": "updated_user"
    }
    response = await async_client.put("/api/users/me", json=update_data, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == update_data["email"]
    assert data["username"] == update_data["username"]

async def test_change_password(async_client, test_user_data, auth_headers):
    """パスワード変更のテスト"""
    password_data = {
        "current_password": test_user_data["password"],
        "new_password": "new_password123"
    }
    response = await async_client.post("/api/users/change-password", json=password_data, headers=auth_headers)
    assert response.status_code == 200

    # 新しいパスワードでログインできることを確認
    login_data = {
        "username": test_user_data["username"],
        "password": "new_password123"
    }
    login_response = await async_client.post("/api/users/login", json=login_data)
    assert login_response.status_code == 200

async def test_register_duplicate_user(async_client, test_user_data):
    """重複ユーザー登録のテスト"""
    # 最初のユーザーを登録
    await async_client.post("/api/users/register", json=test_user_data)

    # 同じデータで再度登録を試みる
    response = await async_client.post("/api/users/register", json=test_user_data)
    assert response.status_code == 400

async def test_login_invalid_credentials(async_client, test_user_data):
    """無効な認証情報でのログインテスト"""
    # ユーザーを登録
    await async_client.post("/api/users/register", json=test_user_data)

    # 誤ったパスワードでログインを試みる
    login_data = {
        "username": test_user_data["username"],
        "password": "wrong_password"
    }
    response = await async_client.post("/api/users/login", json=login_data)
    assert response.status_code == 401

async def test_unauthorized_access(async_client):
    """未認証アクセスのテスト"""
    response = await async_client.get("/api/users/me")
    assert response.status_code == 401

async def test_admin_operations(async_client, test_user_data):
    """管理者操作のテスト"""
    # 管理者ユーザーを作成
    admin_data = {
        "username": "admin_user",
        "email": "admin@example.com",
        "password": "admin_password",
        "role": "admin"
    }
    response = await async_client.post("/api/users/register", json=admin_data)
    assert response.status_code == 201