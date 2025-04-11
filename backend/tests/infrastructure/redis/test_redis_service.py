"""
RedisServiceのテスト

RedisServiceの各メソッドの機能をテストします。
実際のRedis接続を使わずにモックを使用してユニットテストを行います。
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, Optional

from backend.infrastructure.redis.redis_service import RedisService


@pytest.fixture
def mock_redis():
    """Redis接続のモックを作成します"""
    redis = AsyncMock()
    return redis


@pytest.fixture
def redis_service(mock_redis):
    """テスト用のRedisServiceインスタンスを作成します"""
    return RedisService(redis_client=mock_redis)


class TestRedisService:
    """RedisServiceのテストクラス"""

    @pytest.mark.asyncio
    async def test_get_value(self, redis_service, mock_redis):
        """get_valueメソッドのテスト"""
        # 準備
        mock_redis.get.return_value = b"test_value"

        # 実行
        result = await redis_service.get_value("test_key")

        # 検証
        mock_redis.get.assert_called_once_with("test_key")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_get_value_none(self, redis_service, mock_redis):
        """get_valueメソッドが存在しないキーを要求した場合のテスト"""
        # 準備
        mock_redis.get.return_value = None

        # 実行
        result = await redis_service.get_value("nonexistent_key")

        # 検証
        mock_redis.get.assert_called_once_with("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_value_error(self, redis_service, mock_redis):
        """get_valueメソッドでエラーが発生した場合のテスト"""
        # 準備
        mock_redis.get.side_effect = Exception("テストエラー")

        # 実行
        result = await redis_service.get_value("test_key")

        # 検証
        mock_redis.get.assert_called_once_with("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_value_with_ttl(self, redis_service, mock_redis):
        """TTL付きでset_valueメソッドを実行するテスト"""
        # 準備
        mock_redis.setex.return_value = True

        # 実行
        result = await redis_service.set_value("test_key", "test_value", 3600)

        # 検証
        mock_redis.setex.assert_called_once_with("test_key", 3600, "test_value")
        assert result is True

    @pytest.mark.asyncio
    async def test_set_value_without_ttl(self, redis_service, mock_redis):
        """TTLなしでset_valueメソッドを実行するテスト"""
        # 準備
        mock_redis.set.return_value = True

        # 実行
        result = await redis_service.set_value("test_key", "test_value")

        # 検証
        mock_redis.set.assert_called_once_with("test_key", "test_value")
        assert result is True

    @pytest.mark.asyncio
    async def test_set_value_error(self, redis_service, mock_redis):
        """set_valueメソッドでエラーが発生した場合のテスト"""
        # 準備
        mock_redis.set.side_effect = Exception("テストエラー")

        # 実行
        result = await redis_service.set_value("test_key", "test_value")

        # 検証
        mock_redis.set.assert_called_once_with("test_key", "test_value")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_json(self, redis_service, mock_redis):
        """get_jsonメソッドのテスト"""
        # 準備
        test_data = {"name": "テスト", "value": 123}
        mock_redis.get.return_value = json.dumps(test_data).encode("utf-8")

        # 実行
        result = await redis_service.get_json("test_key")

        # 検証
        mock_redis.get.assert_called_once_with("test_key")
        assert result == test_data

    @pytest.mark.asyncio
    async def test_get_json_none(self, redis_service, mock_redis):
        """get_jsonメソッドが存在しないキーを要求した場合のテスト"""
        # 準備
        mock_redis.get.return_value = None

        # 実行
        result = await redis_service.get_json("nonexistent_key")

        # 検証
        mock_redis.get.assert_called_once_with("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_json_invalid_json(self, redis_service, mock_redis):
        """get_jsonメソッドが無効なJSONを受け取った場合のテスト"""
        # 準備
        mock_redis.get.return_value = b"invalid json"

        # 実行
        result = await redis_service.get_json("test_key")

        # 検証
        mock_redis.get.assert_called_once_with("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_json_with_ttl(self, redis_service, mock_redis):
        """TTL付きでset_jsonメソッドを実行するテスト"""
        # 準備
        test_data = {"name": "テスト", "value": 123}
        mock_redis.setex.return_value = True

        # 実行
        result = await redis_service.set_json("test_key", test_data, 3600)

        # 検証
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]
        assert call_args[0] == "test_key"  # キー
        assert call_args[1] == 3600  # TTL
        assert json.loads(call_args[2]) == test_data  # JSON文字列
        assert result is True

    @pytest.mark.asyncio
    async def test_set_json_without_ttl(self, redis_service, mock_redis):
        """TTLなしでset_jsonメソッドを実行するテスト"""
        # 準備
        test_data = {"name": "テスト", "value": 123}
        mock_redis.set.return_value = True

        # 実行
        result = await redis_service.set_json("test_key", test_data)

        # 検証
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args[0]
        assert call_args[0] == "test_key"  # キー
        assert json.loads(call_args[1]) == test_data  # JSON文字列
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_key(self, redis_service, mock_redis):
        """delete_keyメソッドのテスト"""
        # 準備
        mock_redis.delete.return_value = 1

        # 実行
        result = await redis_service.delete_key("test_key")

        # 検証
        mock_redis.delete.assert_called_once_with("test_key")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_key_error(self, redis_service, mock_redis):
        """delete_keyメソッドでエラーが発生した場合のテスト"""
        # 準備
        mock_redis.delete.side_effect = Exception("テストエラー")

        # 実行
        result = await redis_service.delete_key("test_key")

        # 検証
        mock_redis.delete.assert_called_once_with("test_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists(self, redis_service, mock_redis):
        """existsメソッドのテスト"""
        # 準備
        mock_redis.exists.return_value = 1

        # 実行
        result = await redis_service.exists("test_key")

        # 検証
        mock_redis.exists.assert_called_once_with("test_key")
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_not_found(self, redis_service, mock_redis):
        """存在しないキーに対するexistsメソッドのテスト"""
        # 準備
        mock_redis.exists.return_value = 0

        # 実行
        result = await redis_service.exists("nonexistent_key")

        # 検証
        mock_redis.exists.assert_called_once_with("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_expire(self, redis_service, mock_redis):
        """expireメソッドのテスト"""
        # 準備
        mock_redis.expire.return_value = 1

        # 実行
        result = await redis_service.expire("test_key", 3600)

        # 検証
        mock_redis.expire.assert_called_once_with("test_key", 3600)
        assert result is True

    @pytest.mark.asyncio
    async def test_ttl(self, redis_service, mock_redis):
        """ttlメソッドのテスト"""
        # 準備
        mock_redis.ttl.return_value = 3600

        # 実行
        result = await redis_service.ttl("test_key")

        # 検証
        mock_redis.ttl.assert_called_once_with("test_key")
        assert result == 3600

    @pytest.mark.asyncio
    async def test_incr(self, redis_service, mock_redis):
        """incrメソッドのテスト"""
        # 準備
        mock_redis.incr.return_value = 42

        # 実行
        result = await redis_service.incr("counter_key")

        # 検証
        mock_redis.incr.assert_called_once_with("counter_key")
        assert result == 42

    @pytest.mark.asyncio
    async def test_rpush(self, redis_service, mock_redis):
        """rpushメソッドのテスト"""
        # 準備
        mock_redis.rpush.return_value = 3

        # 実行
        result = await redis_service.rpush("list_key", "value1", "value2", "value3")

        # 検証
        mock_redis.rpush.assert_called_once_with("list_key", "value1", "value2", "value3")
        assert result == 3

    @pytest.mark.asyncio
    async def test_lrange(self, redis_service, mock_redis):
        """lrangeメソッドのテスト"""
        # 準備
        mock_redis.lrange.return_value = [b"value1", b"value2", b"value3"]

        # 実行
        result = await redis_service.lrange("list_key", 0, 2)

        # 検証
        mock_redis.lrange.assert_called_once_with("list_key", 0, 2)
        assert result == ["value1", "value2", "value3"]

    @pytest.mark.asyncio
    async def test_hset(self, redis_service, mock_redis):
        """hsetメソッドのテスト"""
        # 準備
        mock_redis.hset.return_value = 1

        # 実行
        result = await redis_service.hset("hash_key", "field", "value")

        # 検証
        mock_redis.hset.assert_called_once_with("hash_key", "field", "value")
        assert result is True

    @pytest.mark.asyncio
    async def test_hget(self, redis_service, mock_redis):
        """hgetメソッドのテスト"""
        # 準備
        mock_redis.hget.return_value = b"value"

        # 実行
        result = await redis_service.hget("hash_key", "field")

        # 検証
        mock_redis.hget.assert_called_once_with("hash_key", "field")
        assert result == "value"

    @pytest.mark.asyncio
    async def test_hgetall(self, redis_service, mock_redis):
        """hgetallメソッドのテスト"""
        # 準備
        mock_redis.hgetall.return_value = {b"field1": b"value1", b"field2": b"value2"}

        # 実行
        result = await redis_service.hgetall("hash_key")

        # 検証
        mock_redis.hgetall.assert_called_once_with("hash_key")
        assert result == {"field1": "value1", "field2": "value2"}