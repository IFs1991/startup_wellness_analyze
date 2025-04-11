"""
Redisキャッシュクライアント

アプリケーション全体で使用されるRedisインターフェースを提供します。
効率的なキャッシュ管理とバルク操作、非同期操作をサポートします。
"""

import os
import redis
import asyncio
import aioredis
import json
import logging
import time
from typing import Optional, Any, Dict, List, Tuple, Set, Union, TypeVar, Generic, Callable
from abc import ABC, abstractmethod

from .common_logger import get_logger
from .exceptions import CacheError
from .config import get_settings
from .async_utils import retry_async

# ロギングの設定
logger = get_logger(__name__)

# 型定義
T = TypeVar('T')
CacheKey = str
CacheValue = Any
CacheMapping = Dict[CacheKey, CacheValue]
CacheTTL = int  # TTLはセカンド単位


class RedisClientInterface(ABC):
    """
    Redisクライアントインターフェース
    依存性注入のために使用される抽象クラス
    """
    @abstractmethod
    async def initialize(self) -> None:
        """Redisクライアントを初期化"""
        pass

    @abstractmethod
    async def get(self, key: CacheKey) -> Optional[CacheValue]:
        """キーに対応する値を取得する"""
        pass

    @abstractmethod
    async def mget(self, keys: List[CacheKey]) -> Dict[CacheKey, Optional[CacheValue]]:
        """複数のキーに対応する値を一括取得する"""
        pass

    @abstractmethod
    async def set(self, key: CacheKey, value: CacheValue, expiry: Optional[CacheTTL] = None) -> bool:
        """キーと値のペアを保存する"""
        pass

    @abstractmethod
    async def mset(self, mapping: CacheMapping, expiry: Optional[CacheTTL] = None) -> bool:
        """複数のキーと値のペアを一括保存する"""
        pass

    @abstractmethod
    async def delete(self, key: CacheKey) -> bool:
        """キーを削除する"""
        pass

    @abstractmethod
    async def mdelete(self, keys: List[CacheKey]) -> int:
        """複数のキーを一括削除する"""
        pass

    @abstractmethod
    async def exists(self, key: CacheKey) -> bool:
        """キーが存在するか確認する"""
        pass

    @abstractmethod
    async def has_any(self, keys: List[CacheKey]) -> bool:
        """いずれかのキーが存在するか確認する"""
        pass

    @abstractmethod
    async def has_all(self, keys: List[CacheKey]) -> bool:
        """すべてのキーが存在するか確認する"""
        pass

    @abstractmethod
    async def increment(self, key: CacheKey, amount: int = 1) -> int:
        """キーの値をインクリメントする"""
        pass

    @abstractmethod
    async def expire(self, key: CacheKey, seconds: CacheTTL) -> bool:
        """キーの有効期限を設定する"""
        pass

    @abstractmethod
    async def ttl(self, key: CacheKey) -> int:
        """キーの残り有効期限を取得する"""
        pass

    @abstractmethod
    async def flush(self) -> bool:
        """すべてのキーを削除する（注意して使用）"""
        pass


class RedisClient(RedisClientInterface):
    """
    Redisキャッシュクライアント

    アプリケーション全体で使用されるRedisインターフェースを提供します。
    同期・非同期の両方の操作をサポートします。
    """

    def __init__(self):
        """Redisクライアントの初期化"""
        self._settings = get_settings()
        self._initialized = False
        self._sync_client = None  # 同期クライアント
        self._async_client = None  # 非同期クライアント
        self._memory_cache = {}  # フォールバック用のメモリキャッシュ
        self.is_connected = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Redisクライアントを初期化"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

        try:
            # Redis接続情報
                self.host = self._settings.redis.host
                self.port = self._settings.redis.port
                self.db = self._settings.redis.db
                self.password = self._settings.redis.password
                self.use_ssl = self._settings.redis.use_ssl

                # 同期クライアント初期化
                try:
                    self._sync_client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                        ssl=self.use_ssl,
                    socket_timeout=5,
                    decode_responses=True
                )
                # 接続確認
                    self._sync_client.ping()

                    # 非同期クライアント初期化
                    if self.password:
                        redis_url = f"redis://{self.password}@{self.host}:{self.port}/{self.db}"
                    else:
                        redis_url = f"redis://{self.host}:{self.port}/{self.db}"

                    if self.use_ssl:
                        redis_url = redis_url.replace("redis://", "rediss://")

                    self._async_client = await aioredis.create_redis_pool(
                        redis_url,
                        timeout=5.0,
                        encoding="utf-8"
                    )

                logger.info(f"Redis接続成功: {self.host}:{self.port}")
                self.is_connected = True
            except redis.ConnectionError as e:
                logger.warning(f"Redis接続エラー: {e}、メモリキャッシュを使用します")
                self.is_connected = False
                    self._sync_client = None
                    self._async_client = None
                    self._memory_cache = {}

            except Exception as e:
                logger.error(f"RedisClientの初期化中にエラー発生: {e}")
                self.is_connected = False
                self._sync_client = None
                self._async_client = None
                self._memory_cache = {}

            self._initialized = True

    async def _ensure_initialized(self) -> None:
        """初期化されていることを確認"""
        if not self._initialized:
            await self.initialize()

    async def get(self, key: CacheKey) -> Optional[CacheValue]:
        """
        キーに対応する値を取得する

        Args:
            key: キャッシュキー

        Returns:
            キャッシュ値、存在しない場合はNone
        """
        await self._ensure_initialized()

        try:
            if not self.is_connected:
                return self._memory_cache.get(key)

            data = await self._async_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redisからの取得に失敗: {e}")
            # メモリキャッシュをフォールバックとして使用
            return self._memory_cache.get(key)

    async def mget(self, keys: List[CacheKey]) -> Dict[CacheKey, Optional[CacheValue]]:
        """
        複数のキーに対応する値を一括取得する

        Args:
            keys: キャッシュキーのリスト

        Returns:
            キー -> 値のマッピング、存在しないキーの値はNone
        """
        await self._ensure_initialized()

        if not keys:
            return {}

        result = {}

        try:
            if not self.is_connected:
                for key in keys:
                    result[key] = self._memory_cache.get(key)
                return result

            # 非同期で一括取得
            values = await self._async_client.mget(*keys)

            for i, key in enumerate(keys):
                value = values[i]
                if value is not None:
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        result[key] = value  # JSON以外の値はそのまま返す
                else:
                    result[key] = None

            return result
        except Exception as e:
            logger.error(f"Redisからの一括取得に失敗: {e}")
            # メモリキャッシュをフォールバックとして使用
            for key in keys:
                result[key] = self._memory_cache.get(key)
            return result

    async def set(self, key: CacheKey, value: CacheValue, expiry: Optional[CacheTTL] = None) -> bool:
        """
        キーと値のペアを保存する

        Args:
            key: キャッシュキー
            value: キャッシュ値
            expiry: 有効期限（秒）

        Returns:
            操作が成功したかどうか
        """
        await self._ensure_initialized()

        try:
            if not self.is_connected:
                self._memory_cache[key] = value
                return True

            serialized = json.dumps(value)
            if expiry:
                return await self._async_client.setex(key, expiry, serialized)
            return await self._async_client.set(key, serialized)
        except Exception as e:
            logger.error(f"Redisへの保存に失敗: {e}")
            # メモリキャッシュをフォールバックとして使用
            self._memory_cache[key] = value
            return False

    async def mset(self, mapping: CacheMapping, expiry: Optional[CacheTTL] = None) -> bool:
        """
        複数のキーと値のペアを一括保存する

        Args:
            mapping: キー -> 値のマッピング
            expiry: 有効期限（秒）

        Returns:
            操作が成功したかどうか
        """
        await self._ensure_initialized()

        if not mapping:
            return True

        try:
            if not self.is_connected:
                self._memory_cache.update(mapping)
                return True

            # 値をシリアライズ
            serialized_mapping = {}
            for key, value in mapping.items():
                serialized_mapping[key] = json.dumps(value)

            # パイプラインで一括保存
            pipeline = self._async_client.pipeline()
            pipeline.mset(serialized_mapping)

            # 有効期限を設定
            if expiry:
                for key in mapping.keys():
                    pipeline.expire(key, expiry)

            await pipeline.execute()
            return True
        except Exception as e:
            logger.error(f"Redisへの一括保存に失敗: {e}")
            # メモリキャッシュをフォールバックとして使用
            self._memory_cache.update(mapping)
            return False

    async def delete(self, key: CacheKey) -> bool:
        """
        キーを削除する

        Args:
            key: 削除するキー

        Returns:
            削除に成功したかどうか
        """
        await self._ensure_initialized()

        try:
            if not self.is_connected:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                    return True
                return False

            result = await self._async_client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Redisからの削除に失敗: {e}")
            # メモリキャッシュをフォールバックとして使用
            if key in self._memory_cache:
                del self._memory_cache[key]
                return True
            return False

    async def mdelete(self, keys: List[CacheKey]) -> int:
        """
        複数のキーを一括削除する

        Args:
            keys: 削除するキーのリスト

        Returns:
            削除されたキーの数
        """
        await self._ensure_initialized()

        if not keys:
            return 0

        try:
            if not self.is_connected:
                count = 0
                for key in keys:
                    if key in self._memory_cache:
                        del self._memory_cache[key]
                        count += 1
                return count

            result = await self._async_client.delete(*keys)
            return int(result)
        except Exception as e:
            logger.error(f"Redisからの一括削除に失敗: {e}")
            # メモリキャッシュをフォールバックとして使用
            count = 0
            for key in keys:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                    count += 1
            return count

    async def exists(self, key: CacheKey) -> bool:
        """
        キーが存在するか確認する

        Args:
            key: 確認するキー

        Returns:
            キーが存在するかどうか
        """
        await self._ensure_initialized()

        try:
            if not self.is_connected:
                return key in self._memory_cache

            result = await self._async_client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Redisのキー存在確認に失敗: {e}")
            # メモリキャッシュをフォールバックとして使用
            return key in self._memory_cache

    async def has_any(self, keys: List[CacheKey]) -> bool:
        """
        いずれかのキーが存在するか確認する

        Args:
            keys: 確認するキーのリスト

        Returns:
            いずれかのキーが存在するかどうか
        """
        await self._ensure_initialized()

        if not keys:
            return False

        try:
            if not self.is_connected:
                return any(key in self._memory_cache for key in keys)

            for key in keys:
                if await self.exists(key):
                    return True
            return False
        except Exception as e:
            logger.error(f"Redisのキー存在確認に失敗: {e}")
            # メモリキャッシュをフォールバックとして使用
            return any(key in self._memory_cache for key in keys)

    async def has_all(self, keys: List[CacheKey]) -> bool:
        """
        すべてのキーが存在するか確認する

        Args:
            keys: 確認するキーのリスト

        Returns:
            すべてのキーが存在するかどうか
        """
        await self._ensure_initialized()

        if not keys:
            return True

        try:
            if not self.is_connected:
                return all(key in self._memory_cache for key in keys)

            for key in keys:
                if not await self.exists(key):
                    return False
            return True
        except Exception as e:
            logger.error(f"Redisのキー存在確認に失敗: {e}")
            # メモリキャッシュをフォールバックとして使用
            return all(key in self._memory_cache for key in keys)

    async def increment(self, key: CacheKey, amount: int = 1) -> int:
        """
        キーの値をインクリメントする

        Args:
            key: インクリメントするキー
            amount: 増加量

        Returns:
            インクリメント後の値
        """
        await self._ensure_initialized()

        try:
            if not self.is_connected:
                if key not in self._memory_cache:
                    self._memory_cache[key] = 0
                if not isinstance(self._memory_cache[key], int):
                    self._memory_cache[key] = 0
                self._memory_cache[key] += amount
                return self._memory_cache[key]

            return await self._async_client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redisのインクリメントに失敗: {e}")
            # メモリキャッシュをフォールバックとして使用
            if key not in self._memory_cache:
                self._memory_cache[key] = 0
            if not isinstance(self._memory_cache[key], int):
                self._memory_cache[key] = 0
            self._memory_cache[key] += amount
            return self._memory_cache[key]

    async def expire(self, key: CacheKey, seconds: CacheTTL) -> bool:
        """
        キーの有効期限を設定する

        Args:
            key: 対象のキー
            seconds: 有効期限（秒）

        Returns:
            操作が成功したかどうか
        """
        await self._ensure_initialized()

        try:
            if not self.is_connected:
                # メモリキャッシュでは有効期限を完全にサポートできない
                return key in self._memory_cache

            return await self._async_client.expire(key, seconds)
        except Exception as e:
            logger.error(f"Redisの有効期限設定に失敗: {e}")
            return key in self._memory_cache

    async def ttl(self, key: CacheKey) -> int:
        """
        キーの残り有効期限を取得する

        Args:
            key: 対象のキー

        Returns:
            残り有効期限（秒）、キーが存在しない場合は-2、有効期限がない場合は-1
        """
        await self._ensure_initialized()

        try:
            if not self.is_connected:
                # メモリキャッシュでは有効期限をサポートしていない
                return -1 if key in self._memory_cache else -2

            return await self._async_client.ttl(key)
        except Exception as e:
            logger.error(f"Redisの有効期限取得に失敗: {e}")
            return -1 if key in self._memory_cache else -2

    async def flush(self) -> bool:
        """
        すべてのキーを削除する（注意して使用）

        Returns:
            操作が成功したかどうか
        """
        await self._ensure_initialized()

        try:
            if not self.is_connected:
                self._memory_cache.clear()
                return True

            await self._async_client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redisのフラッシュに失敗: {e}")
            self._memory_cache.clear()
            return False

    async def close(self) -> None:
        """
        接続を閉じる
        """
        if self._async_client is not None:
            self._async_client.close()
            await self._async_client.wait_closed()
        if self._sync_client is not None:
            self._sync_client.close()

        self._initialized = False
        self.is_connected = False


# シングルトンインスタンス
_redis_client = None


async def get_redis_client() -> RedisClientInterface:
    """
    Redisクライアントのシングルトンインスタンスを取得

    Returns:
        RedisClientInterface: Redisクライアントインスタンス
    """
    global _redis_client

    if _redis_client is None:
        _redis_client = RedisClient()
        await _redis_client.initialize()

    return _redis_client


class CacheDecorator:
    """
    関数の結果をキャッシュするデコレータ
    """

    def __init__(self, ttl: Optional[int] = None, key_prefix: str = "cache"):
        """
        初期化

        Args:
            ttl: キャッシュの有効期限（秒）
            key_prefix: キャッシュキーのプレフィックス
        """
        self.ttl = ttl
        self.key_prefix = key_prefix

    def __call__(self, func):
        """
        デコレータ本体
        """
        async def wrapper(*args, **kwargs):
            # キャッシュキーの生成
            key_parts = [self.key_prefix, func.__name__]

            # 引数をキーに含める
            for arg in args:
                key_parts.append(str(arg))

            # キーワード引数をキーに含める（ソートして一貫性を確保）
            for k in sorted(kwargs.keys()):
                key_parts.append(f"{k}_{kwargs[k]}")

            cache_key = ":".join(key_parts)

            # Redisクライアント取得
            redis_client = await get_redis_client()

            # キャッシュ確認
            cached_result = await redis_client.get(cache_key)
            if cached_result is not None:
                return cached_result

            # 関数実行
            result = await func(*args, **kwargs)

            # キャッシュに保存
            await redis_client.set(cache_key, result, self.ttl)

            return result

        return wrapper


def cached(ttl: Optional[int] = None, key_prefix: str = "cache"):
    """
    関数の結果をキャッシュするデコレータ

    Args:
        ttl: キャッシュの有効期限（秒）
        key_prefix: キャッシュキーのプレフィックス

    Returns:
        デコレータ関数
    """
    return CacheDecorator(ttl, key_prefix)