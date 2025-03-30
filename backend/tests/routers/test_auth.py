import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
from datetime import datetime

from core.auth_manager import User, UserRole

# 前提条件を確認するテスト
def test_router_exists():
    """認証ルーターが正しくインポートされていることを確認"""
    from api.routers import auth
    assert auth.router is not None
    assert auth.router.prefix == "/auth"
    assert "auth" in auth.router.tags

# ユーザー登録テスト
@patch("api.routers.auth.auth_manager")
def test_register_user(mock_auth_manager, client):
    """ユーザー登録エンドポイントが正しく動作することを確認"""
    # モックの設定
    mock_user = User(
        id="new_user_id",
        email="new@example.com",
        display_name="New User",
        is_active=True,
        role=UserRole.USER,
        created_at=datetime.now()
    )
    mock_auth_manager.create_user.return_value = mock_user

    # テストデータ
    user_data = {
        "email": "new@example.com",
        "password": "password123",
        "display_name": "New User",
        "role": "USER"
    }

    # リクエスト実行
    response = client.post("/auth/register", json=user_data)

    # 結果確認
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "new@example.com"
    assert data["display_name"] == "New User"
    assert "id" in data
    mock_auth_manager.create_user.assert_called_once()

# ログインテスト
@patch("api.routers.auth.auth_manager")
def test_login_for_access_token(mock_auth_manager, client):
    """ログインエンドポイントが正しく動作することを確認"""
    # モックの設定
    mock_auth_manager.authenticate_user.return_value = True
    mock_auth_manager.create_access_token.return_value = "test_token"

    # フォームデータ作成
    login_data = {
        "username": "test@example.com",
        "password": "password123"
    }

    # リクエスト実行
    response = client.post("/auth/token", data=login_data)

    # 結果確認
    assert response.status_code == 200
    data = response.json()
    assert data["access_token"] == "test_token"
    assert data["token_type"] == "bearer"
    mock_auth_manager.authenticate_user.assert_called_once()
    mock_auth_manager.create_access_token.assert_called_once()

# ログイン失敗テスト
@patch("api.routers.auth.auth_manager")
def test_login_invalid_credentials(mock_auth_manager, client):
    """無効な認証情報でログインを試みた場合のエラーハンドリングを確認"""
    # モックの設定
    mock_auth_manager.authenticate_user.return_value = False

    # フォームデータ作成
    login_data = {
        "username": "test@example.com",
        "password": "wrong_password"
    }

    # リクエスト実行
    response = client.post("/auth/token", data=login_data)

    # 結果確認
    assert response.status_code == 401
    data = response.json()
    assert "detail" in data
    assert "Invalid credentials" in data["detail"]

# 現在のユーザー情報取得テスト
@patch("api.routers.auth.get_current_active_user")
def test_read_users_me(mock_get_current_user, client, mock_user, token_header):
    """現在のユーザー情報取得エンドポイントが正しく動作することを確認"""
    # モックの設定
    mock_get_current_user.return_value = mock_user

    # リクエスト実行
    response = client.get("/auth/me", headers=token_header)

    # 結果確認
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == mock_user.id
    assert data["email"] == mock_user.email

# パスワードリセットテスト
@patch("api.routers.auth.auth_manager")
def test_reset_password(mock_auth_manager, client):
    """パスワードリセットエンドポイントが正しく動作することを確認"""
    # モックの設定
    mock_auth_manager.send_password_reset_email.return_value = True

    # テストデータ
    reset_data = {
        "email": "test@example.com"
    }

    # リクエスト実行
    response = client.post("/auth/password-reset", json=reset_data)

    # 結果確認
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "sent" in data["message"]
    mock_auth_manager.send_password_reset_email.assert_called_once_with("test@example.com")

# ユーザー一覧取得テスト（管理者権限）
@patch("api.routers.auth.get_current_admin_user")
def test_get_users(mock_get_admin_user, client, mock_admin_user, token_header):
    """ユーザー一覧取得エンドポイントが正しく動作することを確認"""
    # モックの設定
    mock_get_admin_user.return_value = mock_admin_user
    with patch("api.routers.auth.auth_manager") as mock_auth_manager:
        mock_auth_manager.get_all_users.return_value = [
            User(id="user1", email="user1@example.com", display_name="User 1", is_active=True),
            User(id="user2", email="user2@example.com", display_name="User 2", is_active=True)
        ]

        # リクエスト実行
        response = client.get("/auth/users", headers=token_header)

        # 結果確認
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["email"] == "user1@example.com"
        assert data[1]["email"] == "user2@example.com"