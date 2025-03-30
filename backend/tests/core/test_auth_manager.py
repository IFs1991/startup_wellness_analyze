import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import jwt
from datetime import datetime, timedelta
import asyncio
from firebase_admin import auth as firebase_auth

from core.auth_manager import (
    AuthManager,
    User,
    UserRole,
    get_current_user,
    get_current_active_user,
    get_current_admin_user,
    get_current_analyst_user
)
from fastapi import HTTPException, Security, SecurityScopes

# 同期関数をasyncioループで実行するヘルパー関数
def run_async(coroutine):
    """非同期コルーチンを同期的に実行するヘルパー関数"""
    return asyncio.get_event_loop().run_until_complete(coroutine)

# AuthManagerの初期化をテスト
@patch('core.auth_manager.get_app')
@patch('core.auth_manager.firestore.Client')
@patch('core.auth_manager.auth')
def test_auth_manager_initialization(mock_auth, mock_firestore_client, mock_get_app):
    """AuthManagerの初期化をテスト"""
    # モックの設定
    mock_get_app.return_value = MagicMock()
    mock_firestore_client.return_value = MagicMock()

    # インスタンス作成
    auth_manager = AuthManager()

    # 初期化が正しく行われたことを確認
    assert auth_manager is not None
    assert auth_manager._firestore_client is not None

# パスワードのハッシュ化と検証をテスト
def test_password_hashing_verification():
    """パスワードのハッシュ化と検証をテスト"""
    auth_manager = AuthManager()

    # テスト用パスワード
    password = "securePassword123"

    # パスワードをハッシュ化
    hashed_password = run_async(auth_manager.hash_password(password))

    # ハッシュ化されたパスワードを検証
    assert run_async(auth_manager.verify_password(password, hashed_password))
    assert not run_async(auth_manager.verify_password("wrongPassword", hashed_password))

# ユーザー登録をテスト
@patch.object(AuthManager, '_get_firestore_client')
@patch.object(AuthManager, '_check_initialization')
@patch('firebase_admin.auth.create_user')
async def test_register_user(mock_create_user, mock_check_init, mock_get_firestore):
    """ユーザー登録をテスト"""
    # モックの設定
    mock_check_init.return_value = None

    firestore_mock = MagicMock()
    user_collection_mock = MagicMock()
    firestore_mock.collection.return_value = user_collection_mock
    mock_get_firestore.return_value = firestore_mock

    doc_ref_mock = MagicMock()
    user_collection_mock.document.return_value = doc_ref_mock

    # Firebase Authのユーザー作成モック
    mock_create_user.return_value = MagicMock(uid="test_uid")

    # AuthManagerインスタンス取得
    auth_manager = AuthManager()

    # ユーザー登録テスト
    user_data = {
        "email": "test@example.com",
        "password": "securePassword123",
        "display_name": "Test User",
        "role": UserRole.USER,
        "company_id": "test_company"
    }

    # 登録実行
    user = await auth_manager.register_user(**user_data)

    # 結果確認
    assert user is not None
    assert user.email == "test@example.com"
    assert user.display_name == "Test User"
    assert user.role == UserRole.USER
    assert user.company_id == "test_company"
    mock_create_user.assert_called_once()

# 認証トークン作成をテスト
@patch.object(AuthManager, '_check_initialization')
@patch('core.auth_manager.jwt.encode')
async def test_create_access_token(mock_jwt_encode, mock_check_init):
    """アクセストークン作成をテスト"""
    # モックの設定
    mock_check_init.return_value = None
    mock_jwt_encode.return_value = "mock_encoded_token"

    # AuthManagerインスタンス取得
    auth_manager = AuthManager()
    auth_manager._secret_key = "test_secret_key"

    # トークンデータ
    token_data = {
        "sub": "test_user_id",
        "email": "test@example.com"
    }
    scopes = ["user"]
    expires_delta = timedelta(minutes=30)

    # トークン作成
    token = await auth_manager.create_access_token(token_data, scopes, expires_delta)

    # 結果確認
    assert token == "mock_encoded_token"
    mock_jwt_encode.assert_called_once()
    args, kwargs = mock_jwt_encode.call_args
    assert "sub" in args[0]
    assert args[0]["sub"] == "test_user_id"
    assert "scopes" in args[0]
    assert args[0]["scopes"] == scopes

# 現在のユーザー取得をテスト
@patch('core.auth_manager.get_auth_manager')
@patch('core.auth_manager.jwt.decode')
async def test_get_current_user(mock_jwt_decode, mock_get_auth_manager):
    """現在のユーザー取得をテスト"""
    # モックの設定
    auth_manager_mock = AsyncMock()
    mock_get_auth_manager.return_value = auth_manager_mock

    # JWTのデコード結果
    mock_jwt_decode.return_value = {
        "sub": "test_user_id",
        "scopes": ["user"],
        "exp": (datetime.utcnow() + timedelta(minutes=30)).timestamp()
    }

    # ユーザー情報
    mock_user = User(
        id="test_user_id",
        email="test@example.com",
        display_name="Test User",
        is_active=True,
        role=UserRole.USER
    )
    auth_manager_mock.get_user_by_id.return_value = mock_user

    # セキュリティスコープとトークン
    security_scopes = SecurityScopes(scopes=["user"])
    token = "test_token"

    # 現在のユーザーを取得
    user = await get_current_user(security_scopes, token)

    # 結果確認
    assert user is not None
    assert user.id == "test_user_id"
    assert user.email == "test@example.com"
    mock_jwt_decode.assert_called_once()
    auth_manager_mock.get_user_by_id.assert_called_once_with("test_user_id")

# 無効なトークンのテスト
@patch('core.auth_manager.get_auth_manager')
@patch('core.auth_manager.jwt.decode')
async def test_get_current_user_invalid_token(mock_jwt_decode, mock_get_auth_manager):
    """無効なトークンでのエラーをテスト"""
    # モックの設定
    auth_manager_mock = AsyncMock()
    mock_get_auth_manager.return_value = auth_manager_mock

    # JWTのデコードでエラー
    mock_jwt_decode.side_effect = jwt.PyJWTError("無効なトークン")

    # セキュリティスコープとトークン
    security_scopes = SecurityScopes(scopes=["user"])
    token = "invalid_token"

    # 例外が発生することを確認
    with pytest.raises(HTTPException) as excinfo:
        await get_current_user(security_scopes, token)

    # エラーの内容を確認
    assert excinfo.value.status_code == 401
    assert "認証情報が無効です" in excinfo.value.detail

# アクティブユーザー取得をテスト
async def test_get_current_active_user():
    """アクティブユーザー取得をテスト"""
    # アクティブなユーザー
    active_user = User(
        id="test_user_id",
        email="test@example.com",
        display_name="Test User",
        is_active=True,
        role=UserRole.USER
    )

    # 関数を実行
    result = await get_current_active_user(active_user)

    # 結果確認
    assert result == active_user

# 非アクティブユーザーのテスト
async def test_get_current_active_user_inactive():
    """非アクティブユーザーでのエラーをテスト"""
    # 非アクティブなユーザー
    inactive_user = User(
        id="test_user_id",
        email="test@example.com",
        display_name="Test User",
        is_active=False,
        role=UserRole.USER
    )

    # 例外が発生することを確認
    with pytest.raises(HTTPException) as excinfo:
        await get_current_active_user(inactive_user)

    # エラーの内容を確認
    assert excinfo.value.status_code == 400
    assert "非アクティブユーザー" in excinfo.value.detail