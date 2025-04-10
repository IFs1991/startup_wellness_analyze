"""
Redisキャッシュリポジトリの実装

このモジュールはアプリケーションのデータをRedisにキャッシュするための
リポジトリパターンの実装を提供します。
"""

import json
import logging
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from pydantic import BaseModel

from backend.infrastructure.redis.redis_client import RedisClientInterface

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class RedisCacheRepository(Generic[T]):
    """
    Redisを使用したデータキャッシュリポジトリ

    Pydanticモデルを使用してデータをキャッシュするための汎用リポジトリです。
    キーの名前空間を使用して異なるモデルタイプを区別します。
    """

    def __init__(self, redis_client: RedisClientInterface, model_class: Type[T], namespace: str = "cache"):
        """
        Redisキャッシュリポジトリの初期化

        Args:
            redis_client: Redisクライアントインスタンス
            model_class: キャッシュするPydanticモデルのクラス
            namespace: キーの名前空間（デフォルト: "cache"）
        """
        self.redis_client = redis_client
        self.model_class = model_class
        self.namespace = namespace

    def _get_key(self, key: str) -> str:
        """
        完全なRedisキーを生成します

        Args:
            key: ベースキー

        Returns:
            str: 名前空間を含む完全なキー
        """
        return f"{self.namespace}:{self.model_class.__name__}:{key}"

    async def get(self, key: str) -> Optional[T]:
        """
        キーに関連付けられたデータを取得します

        Args:
            key: 取得するデータのキー

        Returns:
            Optional[T]: 取得したモデルインスタンス。存在しない場合はNone
        """
        full_key = self._get_key(key)
        data = await self.redis_client.get_json(full_key)
        if data is None:
            return None

        try:
            return self.model_class.model_validate(data)
        except Exception as e:
            logger.error("モデルの検証に失敗しました: %s", str(e))
            return None

    async def set(self, key: str, value: T, expire: Optional[int] = None) -> bool:
        """
        キーにデータを設定します

        Args:
            key: 設定するデータのキー
            value: 設定するモデルインスタンス
            expire: 有効期限（秒）。Noneの場合は無期限

        Returns:
            bool: 操作が成功したかどうか
        """
        full_key = self._get_key(key)
        try:
            # Pydanticモデルを辞書に変換
            data = value.model_dump()
            return await self.redis_client.set_json(full_key, data, expire)
        except Exception as e:
            logger.error("データのキャッシュに失敗しました: %s", str(e))
            return False

    async def delete(self, key: str) -> bool:
        """
        キーに関連付けられたデータを削除します

        Args:
            key: 削除するデータのキー

        Returns:
            bool: 操作が成功したかどうか
        """
        full_key = self._get_key(key)
        return await self.redis_client.delete(full_key)

    async def exists(self, key: str) -> bool:
        """
        キーが存在するかどうかを確認します

        Args:
            key: 確認するキー

        Returns:
            bool: キーが存在する場合はTrue、存在しない場合はFalse
        """
        full_key = self._get_key(key)
        return await self.redis_client.exists(full_key)

    async def clear_all(self) -> bool:
        """
        このリポジトリに関連するすべてのデータを削除します

        注意: 現在の実装では完全なフラッシュになるため、
        同じRedisインスタンスを使用する他のデータも削除されます。

        Returns:
            bool: 操作が成功したかどうか
        """
        # TODO: 名前空間に基づいた選択的な削除の実装
        return await self.redis_client.flush_all()


def create_redis_cache_repository(
    redis_client: RedisClientInterface,
    model_class: Type[T],
    namespace: str = "cache"
) -> RedisCacheRepository[T]:
    """
    Redisキャッシュリポジトリを作成するファクトリ関数

    Args:
        redis_client: Redisクライアントインスタンス
        model_class: キャッシュするPydanticモデルのクラス
        namespace: キーの名前空間（デフォルト: "cache"）

    Returns:
        RedisCacheRepository[T]: 設定されたRedisキャッシュリポジトリインスタンス
    """
    return RedisCacheRepository(redis_client, model_class, namespace)