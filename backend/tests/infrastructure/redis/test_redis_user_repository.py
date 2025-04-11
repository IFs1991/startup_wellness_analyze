"""
RedisUserRepositoryのテスト

Redisユーザーリポジトリの機能をテストします。
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from domain.models.user import User, UserProfile, UserCredentials
from domain.repositories.user_repository import UserRepositoryInterface
from infrastructure.redis.redis_user_repository import RedisUserRepository

@pytest.fixture
def mock_redis_client():
    """Redisクライアントのモックを作成します"""
    client = AsyncMock()
    return client

@pytest.fixture
def mock_main_repository():
    """メインリポジトリのモックを作成します"""
    repo = AsyncMock(spec=UserRepositoryInterface)
    return repo

@pytest.fixture
def redis_user_repository(mock_redis_client, mock_main_repository):
    """テスト用のRedisUserRepositoryインスタンスを作成します"""
    return RedisUserRepository(
        redis_client=mock_redis_client,
        main_repository=mock_main_repository,
        ttl_seconds=3600
    )

@pytest.mark.asyncio
async def test_get_by_id_cached(redis_user_repository, mock_redis_client):
    """キャッシュからユーザーをIDで取得するテスト"""
    # 準備
    user_id = "test-user-id"
    user_data = {
        "id": user_id,
        "email": "test@example.com",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "credentials": {
            "hashed_password": "hashed_password",
            "is_active": True,
            "is_verified": True
        },
        "profile": {
            "first_name": "Test",
            "last_name": "User",
            "company_id": "company-id",
            "role": "user"
        }
    }

    # Redisからの応答をモック
    mock_redis_client.get.return_value = json.dumps(user_data)

    # 実行
    result = await redis_user_repository.get_by_id(user_id)

    # 検証
    mock_redis_client.get.assert_called_once_with(f"user:{user_id}")
    assert isinstance(result, User)
    assert result.id == user_id
    assert result.email == "test@example.com"

@pytest.mark.asyncio
async def test_get_by_id_not_cached(redis_user_repository, mock_redis_client, mock_main_repository):
    """キャッシュにないユーザーをメインリポジトリから取得するテスト"""
    # 準備
    user_id = "test-user-id"
    user = User(
        id=user_id,
        email="test@example.com",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        credentials=UserCredentials(
            hashed_password="hashed_password",
            is_active=True,
            is_verified=True
        ),
        profile=UserProfile(
            first_name="Test",
            last_name="User",
            company_id="company-id",
            role="user"
        )
    )

    # Redisからの応答をモック
    mock_redis_client.get.return_value = None

    # メインリポジトリからの応答をモック
    mock_main_repository.get_by_id.return_value = user

    # 実行
    result = await redis_user_repository.get_by_id(user_id)

    # 検証
    mock_redis_client.get.assert_called_once_with(f"user:{user_id}")
    mock_main_repository.get_by_id.assert_called_once_with(user_id)
    assert result == user

    # キャッシュが更新されたことを確認
    mock_redis_client.setex.assert_called_once()

@pytest.mark.asyncio
async def test_create_user(redis_user_repository, mock_redis_client, mock_main_repository):
    """ユーザー作成のテスト"""
    # 準備
    user = User(
        id="new-user-id",
        email="new@example.com",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        credentials=UserCredentials(
            hashed_password="hashed_password",
            is_active=True,
            is_verified=True
        ),
        profile=UserProfile(
            first_name="New",
            last_name="User",
            company_id="company-id",
            role="user"
        )
    )

    # メインリポジトリからの応答をモック
    mock_main_repository.create.return_value = user

    # 実行
    result = await redis_user_repository.create(user)

    # 検証
    mock_main_repository.create.assert_called_once_with(user)
    assert result == user

    # キャッシュが更新されたことを確認
    assert mock_redis_client.setex.call_count == 2  # userキーとemailキーの両方

@pytest.mark.asyncio
async def test_update_user(redis_user_repository, mock_redis_client, mock_main_repository):
    """ユーザー更新のテスト"""
    # 準備
    user = User(
        id="update-user-id",
        email="update@example.com",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        credentials=UserCredentials(
            hashed_password="hashed_password",
            is_active=True,
            is_verified=True
        ),
        profile=UserProfile(
            first_name="Update",
            last_name="User",
            company_id="company-id",
            role="user"
        )
    )

    # メインリポジトリからの応答をモック
    mock_main_repository.update.return_value = user

    # 実行
    result = await redis_user_repository.update(user)

    # 検証
    mock_main_repository.update.assert_called_once_with(user)
    assert result == user

    # キャッシュが更新されたことを確認
    assert mock_redis_client.setex.call_count == 2  # userキーとemailキーの両方

@pytest.mark.asyncio
async def test_delete_user(redis_user_repository, mock_redis_client, mock_main_repository):
    """ユーザー削除のテスト"""
    # 準備
    user_id = "delete-user-id"
    email = "delete@example.com"

    # 既存のユーザー情報をモック
    user_data = {
        "id": user_id,
        "email": email,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "credentials": {
            "hashed_password": "hashed_password",
            "is_active": True,
            "is_verified": True
        },
        "profile": {
            "first_name": "Delete",
            "last_name": "User",
            "company_id": "company-id",
            "role": "user"
        }
    }
    mock_redis_client.get.return_value = json.dumps(user_data)

    # 実行
    await redis_user_repository.delete(user_id)

    # 検証
    mock_main_repository.delete.assert_called_once_with(user_id)

    # キャッシュが削除されたことを確認
    mock_redis_client.delete.assert_any_call(f"user:{user_id}")
    mock_redis_client.delete.assert_any_call(f"user:email:{email}")