"""
Redisリポジトリのテスト
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel
from typing import List, Optional

from infrastructure.redis.redis_repository import (
    RedisCacheRepository,
    RedisCacheRepositoryInterface,
    get_redis_repository
)
from infrastructure.redis.redis_client import RedisClientInterface


# テスト用モデル
class TestUser(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int] = None


@pytest.fixture
def mock_redis_client():
    """Redisクライアントのモック"""
    client = AsyncMock(spec=RedisClientInterface)
    return client


@pytest.fixture
def redis_repository(mock_redis_client):
    """テスト用リポジトリ"""
    return RedisCacheRepository(
        redis_client=mock_redis_client,
        model_class=TestUser,
        prefix="test",
        default_expire=60
    )


@pytest.mark.asyncio
async def test_get_success(redis_repository, mock_redis_client):
    """get操作の成功テスト"""
    # モックのセットアップ
    user_data = {
        "id": 1,
        "name": "テストユーザー",
        "email": "test@example.com",
        "age": 30
    }
    mock_redis_client.get_json.return_value = user_data

    # テスト実行
    result = await redis_repository.get("user:1")

    # 検証
    mock_redis_client.get_json.assert_called_once_with("test:user:1")
    assert isinstance(result, TestUser)
    assert result.id == 1
    assert result.name == "テストユーザー"
    assert result.email == "test@example.com"
    assert result.age == 30


@pytest.mark.asyncio
async def test_get_not_found(redis_repository, mock_redis_client):
    """存在しないキーのgetテスト"""
    # モックのセットアップ
    mock_redis_client.get_json.return_value = None

    # テスト実行
    result = await redis_repository.get("user:999")

    # 検証
    mock_redis_client.get_json.assert_called_once_with("test:user:999")
    assert result is None


@pytest.mark.asyncio
async def test_get_exception(redis_repository, mock_redis_client):
    """get操作の例外テスト"""
    # モックのセットアップ
    mock_redis_client.get_json.side_effect = Exception("接続エラー")

    # テスト実行
    result = await redis_repository.get("user:1")

    # 検証
    mock_redis_client.get_json.assert_called_once_with("test:user:1")
    assert result is None


@pytest.mark.asyncio
async def test_set_success(redis_repository, mock_redis_client):
    """set操作の成功テスト"""
    # テストデータ
    user = TestUser(id=1, name="テストユーザー", email="test@example.com", age=30)
    mock_redis_client.set_json.return_value = True

    # テスト実行
    result = await redis_repository.set("user:1", user)

    # 検証
    mock_redis_client.set_json.assert_called_once()
    assert mock_redis_client.set_json.call_args[0][0] == "test:user:1"
    assert mock_redis_client.set_json.call_args[0][1] == user.model_dump()
    assert mock_redis_client.set_json.call_args[0][2] == 60
    assert result is True


@pytest.mark.asyncio
async def test_set_with_custom_expire(redis_repository, mock_redis_client):
    """カスタム有効期限でのset操作テスト"""
    # テストデータ
    user = TestUser(id=1, name="テストユーザー", email="test@example.com")
    mock_redis_client.set_json.return_value = True

    # テスト実行 - カスタム有効期限
    result = await redis_repository.set("user:1", user, expire=120)

    # 検証
    mock_redis_client.set_json.assert_called_once()
    assert mock_redis_client.set_json.call_args[0][0] == "test:user:1"
    assert mock_redis_client.set_json.call_args[0][1] == user.model_dump()
    assert mock_redis_client.set_json.call_args[0][2] == 120
    assert result is True


@pytest.mark.asyncio
async def test_delete_success(redis_repository, mock_redis_client):
    """delete操作の成功テスト"""
    # モックのセットアップ
    mock_redis_client.delete.return_value = True

    # テスト実行
    result = await redis_repository.delete("user:1")

    # 検証
    mock_redis_client.delete.assert_called_once_with("test:user:1")
    assert result is True


@pytest.mark.asyncio
async def test_exists_true(redis_repository, mock_redis_client):
    """exists操作でキーが存在する場合のテスト"""
    # モックのセットアップ
    mock_redis_client.exists.return_value = True

    # テスト実行
    result = await redis_repository.exists("user:1")

    # 検証
    mock_redis_client.exists.assert_called_once_with("test:user:1")
    assert result is True


@pytest.mark.asyncio
async def test_exists_false(redis_repository, mock_redis_client):
    """exists操作でキーが存在しない場合のテスト"""
    # モックのセットアップ
    mock_redis_client.exists.return_value = False

    # テスト実行
    result = await redis_repository.exists("user:999")

    # 検証
    mock_redis_client.exists.assert_called_once_with("test:user:999")
    assert result is False


@pytest.mark.asyncio
async def test_clear_all(redis_repository, mock_redis_client):
    """clear_all操作のテスト"""
    # モックのセットアップ
    mock_redis_client.flush_all.return_value = True

    # テスト実行
    result = await redis_repository.clear_all()

    # 検証
    mock_redis_client.flush_all.assert_called_once()
    assert result is True


@pytest.mark.asyncio
async def test_factory_function():
    """ファクトリ関数のテスト"""
    with patch('infrastructure.redis.redis_repository.get_redis_client') as mock_get_client:
        # モックのセットアップ
        mock_client = AsyncMock(spec=RedisClientInterface)
        mock_get_client.return_value = mock_client

        # テスト実行
        repo = get_redis_repository(
            entity_type=TestUser,
            prefix="users",
            expire=300,
            test_mode=True
        )

        # 検証
        assert isinstance(repo, RedisCacheRepository)
        mock_get_client.assert_called_once_with(test_mode=True)
        assert repo.model_class == TestUser
        assert repo.prefix == "users"
        assert repo.default_expire == 300