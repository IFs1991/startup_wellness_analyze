import pytest
from fastapi.testclient import TestClient
from backend.src.main import app
from backend.src.database.models import User
from backend.src.database.repositories.user import UserRepository

client = TestClient(app)

@pytest.fixture
def test_user():
    """テスト用のユーザーデータを作成する"""
    return {
        "username": "test_user",
        "email": "test@example.com",
        "password": "test_password",
        "role": "user"
    }

def test_register_user(test_user):
    """ユーザー登録のテスト"""
    response = client.post("/api/users/register", json=test_user)
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == test_user["username"]
    assert data["email"] == test_user["email"]
    assert data["role"] == test_user["role"]
    assert "password" not in data

def test_login_user(test_user):
    """ユーザーログインのテスト"""
    # ユーザーを登録
    client.post("/api/users/register", json=test_user)

    # ログイン
    login_data = {
        "username": test_user["username"],
        "password": test_user["password"]
    }
    response = client.post("/api/users/login", json=login_data)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_get_current_user(test_user):
    """現在のユーザー情報取得のテスト"""
    # ユーザーを登録
    client.post("/api/users/register", json=test_user)

    # ログイン
    login_data = {
        "username": test_user["username"],
        "password": test_user["password"]
    }
    login_response = client.post("/api/users/login", json=login_data)
    token = login_response.json()["access_token"]

    # 現在のユーザー情報を取得
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/api/users/me", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == test_user["username"]
    assert data["email"] == test_user["email"]
    assert data["role"] == test_user["role"]

def test_update_user(test_user):
    """ユーザー情報更新のテスト"""
    # ユーザーを登録
    client.post("/api/users/register", json=test_user)

    # ログイン
    login_data = {
        "username": test_user["username"],
        "password": test_user["password"]
    }
    login_response = client.post("/api/users/login", json=login_data)
    token = login_response.json()["access_token"]

    # ユーザー情報を更新
    update_data = {
        "email": "updated@example.com"
    }
    headers = {"Authorization": f"Bearer {token}"}
    response = client.put("/api/users/me", json=update_data, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == update_data["email"]

def test_change_password(test_user):
    """パスワード変更のテスト"""
    # ユーザーを登録
    client.post("/api/users/register", json=test_user)

    # ログイン
    login_data = {
        "username": test_user["username"],
        "password": test_user["password"]
    }
    login_response = client.post("/api/users/login", json=login_data)
    token = login_response.json()["access_token"]

    # パスワードを変更
    password_data = {
        "current_password": test_user["password"],
        "new_password": "new_password"
    }
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/api/users/change-password", json=password_data, headers=headers)
    assert response.status_code == 200

    # 新しいパスワードでログイン
    new_login_data = {
        "username": test_user["username"],
        "password": "new_password"
    }
    response = client.post("/api/users/login", json=new_login_data)
    assert response.status_code == 200

def test_register_duplicate_user(test_user):
    """重複ユーザー登録のテスト"""
    # 最初のユーザーを登録
    client.post("/api/users/register", json=test_user)

    # 同じユーザー名で登録を試みる
    response = client.post("/api/users/register", json=test_user)
    assert response.status_code == 400

def test_login_invalid_credentials(test_user):
    """無効な認証情報でのログインテスト"""
    # ユーザーを登録
    client.post("/api/users/register", json=test_user)

    # 誤ったパスワードでログイン
    login_data = {
        "username": test_user["username"],
        "password": "wrong_password"
    }
    response = client.post("/api/users/login", json=login_data)
    assert response.status_code == 401

def test_unauthorized_access():
    """未認証アクセスのテスト"""
    # 認証なしで保護されたエンドポイントにアクセス
    response = client.get("/api/users/me")
    assert response.status_code == 401

def test_admin_operations(test_user):
    """管理者操作のテスト"""
    # 管理者ユーザーを作成
    admin_user = test_user.copy()
    admin_user["role"] = "admin"
    client.post("/api/users/register", json=admin_user)

    # 管理者でログイン
    login_data = {
        "username": admin_user["username"],
        "password": admin_user["password"]
    }
    login_response = client.post("/api/users/login", json=login_data)
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 通常ユーザーを作成
    normal_user = {
        "username": "normal_user",
        "email": "normal@example.com",
        "password": "password",
        "role": "user"
    }
    client.post("/api/users/register", json=normal_user)

    # ユーザーのロールを更新
    role_data = {"role": "moderator"}
    response = client.put(f"/api/users/{normal_user['username']}/role", json=role_data, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "moderator"

    # ユーザーを非アクティブ化
    response = client.post(f"/api/users/{normal_user['username']}/deactivate", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["is_active"] is False

    # ユーザーをアクティブ化
    response = client.post(f"/api/users/{normal_user['username']}/activate", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["is_active"] is True