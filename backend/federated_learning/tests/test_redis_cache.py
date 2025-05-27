# Phase 3 Task 3.2: Redis分散キャッシュシステムの実装
# TDD RED段階: 失敗するテストから開始（モック版）

import pytest
import pytest_asyncio
import asyncio
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Optional, List, Dict, Any
import uuid

# テスト対象となるクラス
from ..cache.redis_manager import RedisManager
from ..cache.cache_patterns import CachePattern, DistributedLock, CacheInvalidation
from ..cache.models import CacheEntry, CacheMetrics


# 共有フィクスチャ（モック版）
@pytest_asyncio.fixture
async def mock_redis_manager():
    """モック化されたRedisManagerのフィクスチャ"""
    with patch('redis.asyncio.Redis') as mock_redis_class:
        # モックRedisクライアントの設定
        mock_redis_client = AsyncMock()
        mock_redis_class.return_value = mock_redis_client

        # 基本操作のモック設定
        mock_redis_client.ping.return_value = True
        mock_redis_client.set.return_value = True
        mock_redis_client.get.return_value = None
        mock_redis_client.delete.return_value = 1
        mock_redis_client.exists.return_value = 1
        mock_redis_client.expire.return_value = True
        mock_redis_client.ttl.return_value = 300

        # Hash操作のモック
        mock_redis_client.hset.return_value = True
        mock_redis_client.hget.return_value = None
        mock_redis_client.hgetall.return_value = {}
        mock_redis_client.hdel.return_value = 1

        # 集合操作のモック
        mock_redis_client.sadd.return_value = 1
        mock_redis_client.srem.return_value = 1
        mock_redis_client.smembers.return_value = set()
        mock_redis_client.sismember.return_value = False

        # リスト操作のモック
        mock_redis_client.lpush.return_value = 1
        mock_redis_client.rpush.return_value = 1
        mock_redis_client.lpop.return_value = None
        mock_redis_client.rpop.return_value = None
        mock_redis_client.lrange.return_value = []

        # その他の操作
        mock_redis_client.keys.return_value = []
        mock_redis_client.info.return_value = {
            "used_memory": 1024,
            "used_memory_human": "1K",
            "maxmemory": 0
        }
        mock_redis_client.config_get.return_value = {"maxmemory-policy": "allkeys-lru"}
        mock_redis_client.flushdb.return_value = True
        mock_redis_client.eval.return_value = 1

        # ConnectionPoolのモック
        with patch('redis.asyncio.ConnectionPool') as mock_pool_class:
            mock_pool = Mock()
            mock_pool.created_connections = 1
            mock_pool._available_connections = []
            mock_pool._in_use_connections = []
            mock_pool_class.return_value = mock_pool

            # RedisManagerを作成
            manager = RedisManager(
                host="localhost",
                port=6379,
                db=1,
                decode_responses=True,
                max_connections=10
            )

            # モックを設定
            manager.redis_client = mock_redis_client
            manager.connection_pool = mock_pool
            manager.is_initialized = True

            yield manager


@pytest_asyncio.fixture
async def cache_pattern(mock_redis_manager):
    """CachePatternのフィクスチャ"""
    return CachePattern(mock_redis_manager)


@pytest_asyncio.fixture
async def distributed_lock(mock_redis_manager):
    """DistributedLockのフィクスチャ"""
    return DistributedLock(mock_redis_manager)


@pytest_asyncio.fixture
async def cache_invalidation(mock_redis_manager):
    """CacheInvalidationのフィクスチャ"""
    return CacheInvalidation(mock_redis_manager)


@pytest_asyncio.fixture
async def cache_metrics(mock_redis_manager):
    """CacheMetricsのフィクスチャ"""
    return CacheMetrics(mock_redis_manager)


# RedisManagerのテスト
class TestRedisManager:
    """RedisManagerクラスのテスト"""

    async def test_redis_connection(self, mock_redis_manager):
        """Redis接続テスト"""
        # 接続状態確認
        is_connected = await mock_redis_manager.is_connected()
        assert is_connected is True

        # ping確認
        mock_redis_manager.redis_client.ping.assert_called()

    async def test_redis_basic_operations(self, mock_redis_manager):
        """Redis基本操作テスト"""
        # SET操作
        mock_redis_manager.redis_client.set.return_value = True
        result = await mock_redis_manager.set("test_key", "test_value", expire=300)
        assert result is True

        # GET操作
        mock_redis_manager.redis_client.get.return_value = "test_value"
        value = await mock_redis_manager.get("test_key")
        assert value == "test_value"

        # DELETE操作
        mock_redis_manager.redis_client.delete.return_value = 1
        deleted = await mock_redis_manager.delete("test_key")
        assert deleted == 1

    async def test_redis_hash_operations(self, mock_redis_manager):
        """Redisハッシュ操作テスト"""
        # HSET操作
        mock_redis_manager.redis_client.hset.return_value = True
        result = await mock_redis_manager.hset("test_hash", "field1", "value1")
        assert result is True

        # HGET操作
        mock_redis_manager.redis_client.hget.return_value = "value1"
        value = await mock_redis_manager.hget("test_hash", "field1")
        assert value == "value1"

        # HGETALL操作
        mock_redis_manager.redis_client.hgetall.return_value = {"field1": "value1"}
        all_values = await mock_redis_manager.hgetall("test_hash")
        assert all_values == {"field1": "value1"}

    async def test_redis_connection_pool(self, mock_redis_manager):
        """Redis接続プール情報テスト"""
        pool_info = await mock_redis_manager.get_pool_info()

        assert "created_connections" in pool_info
        assert "available_connections" in pool_info
        assert "in_use_connections" in pool_info
        assert "max_connections" in pool_info

    async def test_redis_health_check(self, mock_redis_manager):
        """Redisヘルスチェックテスト"""
        health_info = await mock_redis_manager.health_check()

        assert "redis_connected" in health_info
        assert "latency_ms" in health_info
        assert "memory_usage" in health_info
        assert "timestamp" in health_info
        assert "pool_info" in health_info


# CachePatternのテスト
class TestCachePattern:
    """CachePatternクラスのテスト"""

    async def test_simple_cache_pattern(self, cache_pattern):
        """基本的なキャッシュパターンテスト"""
        # キャッシュ設定
        cache_pattern.redis.redis_client.set.return_value = True
        result = await cache_pattern.set_cache("test_key", "test_value", expire=300)
        assert result is True

        # キャッシュ取得
        cache_pattern.redis.redis_client.get.return_value = "test_value"
        value = await cache_pattern.get_cache("test_key")
        assert value == "test_value"

        # キャッシュ無効化
        cache_pattern.redis.redis_client.delete.return_value = 1
        invalidated = await cache_pattern.invalidate_cache("test_key")
        assert invalidated is True

    async def test_cache_aside_pattern(self, cache_pattern):
        """Cache-Asideパターンテスト"""
        # データ取得関数のモック
        async def mock_fetch_function(key):
            return f"data_for_{key}"

        # キャッシュミス → データ取得 → キャッシュ保存
        cache_pattern.redis.redis_client.get.return_value = None  # キャッシュミス
        cache_pattern.redis.redis_client.set.return_value = True

        data = await cache_pattern.cache_aside(
            "test_key",
            mock_fetch_function,
            "test_key",
            expire=300
        )

        assert data == "data_for_test_key"

    async def test_write_through_pattern(self, cache_pattern):
        """Write-Throughパターンテスト"""
        # 書き込み関数のモック
        async def mock_write_function(key, data):
            return True

        cache_pattern.redis.redis_client.set.return_value = True

        result = await cache_pattern.write_through(
            "test_key",
            "test_data",
            mock_write_function,
            expire=300
        )

        assert result is True

    async def test_write_behind_pattern(self, cache_pattern):
        """Write-Behindパターンテスト"""
        cache_pattern.redis.redis_client.set.return_value = True
        cache_pattern.redis.redis_client.lpush.return_value = 1

        result = await cache_pattern.write_behind(
            "test_key",
            "test_data",
            expire=300
        )

        assert result is True


# DistributedLockのテスト
class TestDistributedLock:
    """DistributedLockクラスのテスト"""

    async def test_acquire_and_release_lock(self, distributed_lock):
        """ロック取得・解放テスト"""
        lock_key = "test_lock"
        lock_value = str(uuid.uuid4())

        # ロック取得
        distributed_lock.redis.redis_client.set.return_value = True
        acquired = await distributed_lock.acquire(lock_key, lock_value, expire=30)
        assert acquired is True

        # ロック解放
        distributed_lock.redis.redis_client.eval.return_value = 1
        released = await distributed_lock.release(lock_key, lock_value)
        assert released is True

    async def test_lock_with_context_manager(self, distributed_lock):
        """コンテキストマネージャーでのロックテスト"""
        lock_key = "test_context_lock"

        distributed_lock.redis.redis_client.set.return_value = True
        distributed_lock.redis.redis_client.eval.return_value = 1

        async with distributed_lock.acquire_context(lock_key, expire=30) as acquired:
            assert acquired is True

    async def test_lock_timeout_handling(self, distributed_lock):
        """ロックタイムアウト処理テスト"""
        lock_key = "test_timeout_lock"

        # ロック状態確認
        distributed_lock.redis.redis_client.exists.return_value = 1
        is_locked = await distributed_lock.is_locked(lock_key)
        assert is_locked is True

        # ロック情報取得
        distributed_lock.redis.redis_client.get.return_value = "lock_value"
        distributed_lock.redis.redis_client.ttl.return_value = 25

        lock_info = await distributed_lock.get_lock_info(lock_key)
        assert lock_info is not None
        assert lock_info["is_locked"] is True
        assert lock_info["ttl"] == 25


# CacheInvalidationのテスト
class TestCacheInvalidation:
    """CacheInvalidationクラスのテスト"""

    async def test_tag_based_invalidation(self, cache_invalidation):
        """タグベース無効化テスト"""
        # タグ付きキャッシュ設定
        cache_invalidation.redis.redis_client.set.return_value = True
        cache_invalidation.redis.redis_client.sadd.return_value = 1
        cache_invalidation.redis.redis_client.expire.return_value = True

        result = await cache_invalidation.set_with_tags(
            "test_key",
            "test_data",
            ["tag1", "tag2"],
            expire=300
        )
        assert result is True

        # タグによる無効化
        cache_invalidation.redis.redis_client.smembers.return_value = {"test_key"}
        cache_invalidation.redis.redis_client.delete.return_value = 1

        invalidated_count = await cache_invalidation.invalidate_by_tag("tag1")
        assert invalidated_count == 1

    async def test_pattern_based_invalidation(self, cache_invalidation):
        """パターンベース無効化テスト"""
        # scan_iterのモック設定
        async def mock_scan_iter(match, count):
            if match == "session:*":
                for key in ["session:1", "session:2"]:
                    yield key

        cache_invalidation.redis.scan_iter = mock_scan_iter
        # deleteは渡されたキーの数だけ削除されたことを返す
        cache_invalidation.redis.redis_client.delete.return_value = 2

        invalidated_count = await cache_invalidation.invalidate_by_pattern("session:*")
        assert invalidated_count == 2

    async def test_time_based_invalidation(self, cache_invalidation):
        """時間ベース無効化テスト"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)

        # タイムスタンプ付きキャッシュ設定
        cache_invalidation.redis.redis_client.set.return_value = True

        result = await cache_invalidation.set_with_timestamp(
            "test_key",
            "test_data",
            timestamp=cutoff_time - timedelta(minutes=30),  # 古いタイムスタンプ
            expire=300
        )
        assert result is True

        # 古いキャッシュの無効化
        old_timestamp = (cutoff_time - timedelta(minutes=30)).isoformat()

        async def mock_scan_iter(match, count):
            if match == "ts:*":
                yield "ts:test_key"

        cache_invalidation.redis.scan_iter = mock_scan_iter
        cache_invalidation.redis.redis_client.get.return_value = old_timestamp
        cache_invalidation.redis.redis_client.delete.return_value = 1

        invalidated_count = await cache_invalidation.invalidate_older_than(cutoff_time)
        assert invalidated_count == 1


# CacheMetricsのテスト
class TestCacheMetrics:
    """CacheMetricsクラスのテスト"""

    async def test_hit_rate_tracking(self, cache_metrics):
        """ヒット率追跡テスト"""
        # ヒット記録
        cache_metrics.redis.redis_client.hincrby.return_value = 1
        cache_metrics.redis.redis_client.hset.return_value = True

        await cache_metrics.record_hit("test_cache")
        await cache_metrics.record_miss("test_cache")

        # ヒット率取得
        cache_metrics.redis.redis_client.hget.side_effect = ["5", "3"]  # hits, misses

        hit_rate = await cache_metrics.get_hit_rate("test_cache")
        assert hit_rate == 5 / (5 + 3)  # 0.625

    async def test_performance_metrics(self, cache_metrics):
        """パフォーマンスメトリクステスト"""
        # 操作時間記録
        cache_metrics.redis.redis_client.lpush.return_value = 1
        cache_metrics.redis.redis_client.ltrim.return_value = True

        await cache_metrics.record_operation_time("get", 15.5)
        await cache_metrics.record_operation_time("set", 8.2)

        # 平均時間取得
        cache_metrics.redis.redis_client.lrange.return_value = ["15.5", "8.2", "12.1"]

        avg_time = await cache_metrics.get_average_operation_time("get")
        expected_avg = (15.5 + 8.2 + 12.1) / 3
        # 浮動小数点の精度を考慮した比較
        assert abs(avg_time - expected_avg) < 0.001

    async def test_memory_usage_tracking(self, cache_metrics):
        """メモリ使用量追跡テスト"""
        cache_metrics.redis.redis_client.info.return_value = {
            "used_memory": 2048,
            "used_memory_human": "2K",
            "used_memory_peak": 4096,
            "maxmemory": 8192
        }

        memory_usage = await cache_metrics.get_memory_usage()

        assert memory_usage["used_memory"] == 2048
        assert memory_usage["used_memory_human"] == "2K"
        assert memory_usage["memory_usage_percentage"] == (2048 / 8192) * 100

    async def test_eviction_tracking(self, cache_metrics):
        """エビクション追跡テスト"""
        cache_metrics.redis.redis_client.info.return_value = {
            "evicted_keys": 10,
            "expired_keys": 25,
            "keyspace_hits": 1000,
            "keyspace_misses": 200
        }

        eviction_stats = await cache_metrics.get_eviction_stats()

        assert eviction_stats["evicted_keys"] == 10
        assert eviction_stats["expired_keys"] == 25
        assert eviction_stats["keyspace_hits"] == 1000
        assert eviction_stats["keyspace_misses"] == 200


# 統合テスト
@pytest.mark.cache
@pytest.mark.integration
class TestCacheIntegration:
    """キャッシュシステム統合テスト"""

    async def test_concurrent_cache_operations(self, mock_redis_manager):
        """並行キャッシュ操作テスト"""
        cache_pattern = CachePattern(mock_redis_manager)

        # 並行操作のシミュレーション
        mock_redis_manager.redis_client.set.return_value = True
        mock_redis_manager.redis_client.get.return_value = "test_value"

        async def cache_operation(key, value):
            await cache_pattern.set_cache(key, value)
            return await cache_pattern.get_cache(key)

        # 複数の並行操作
        tasks = [
            cache_operation(f"key_{i}", f"value_{i}")
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)
        assert len(results) == 10
        assert all(result == "test_value" for result in results)

    async def test_cache_with_lock_integration(self, mock_redis_manager):
        """キャッシュとロックの統合テスト"""
        cache_pattern = CachePattern(mock_redis_manager)
        distributed_lock = DistributedLock(mock_redis_manager)

        # ロック取得とキャッシュ操作
        mock_redis_manager.redis_client.set.return_value = True
        mock_redis_manager.redis_client.eval.return_value = 1

        lock_key = "cache_lock"
        cache_key = "protected_cache"

        async with distributed_lock.acquire_context(lock_key) as acquired:
            assert acquired is True

            # ロック保護下でのキャッシュ操作
            result = await cache_pattern.set_cache(cache_key, "protected_data")
            assert result is True

    async def test_cache_invalidation_cascade(self, mock_redis_manager):
        """カスケード無効化統合テスト"""
        cache_invalidation = CacheInvalidation(mock_redis_manager)

        # カスケード無効化のモック設定
        async def mock_scan_iter(match, count):
            if match == "session:*":
                for key in ["session:data", "session:metadata"]:
                    yield key

        cache_invalidation.redis.scan_iter = mock_scan_iter
        mock_redis_manager.redis_client.delete.return_value = 1

        # カスケード無効化実行
        invalidated_count = await cache_invalidation.invalidate_cascade(["session"])
        assert invalidated_count == 3  # session + session:data + session:metadata