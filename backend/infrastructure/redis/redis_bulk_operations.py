"""
Redisバルク操作ユーティリティ

効率的なRedisのバルク操作を実装するためのユーティリティ。
非同期処理を活用して大量のデータの読み書きを最適化します。
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, TypeVar, Generic, Callable, Coroutine, Tuple

from backend.core.async_utils import TaskLimiter, gather_with_concurrency, AsyncBatchProcessor
from backend.core.common_logger import get_logger
from backend.infrastructure.redis.redis_service import RedisService

logger = get_logger(__name__)

T = TypeVar('T')


class RedisBulkOperations:
    """
    効率的なRedisバルク操作を提供するクラス

    大量のキーの読み取り、書き込み、削除などの操作を効率的に行います。
    """

    def __init__(
        self,
        redis_service: RedisService,
        max_concurrency: int = 20,
        batch_size: int = 100
    ):
        """
        初期化

        Args:
            redis_service: Redisサービスインスタンス
            max_concurrency: 最大同時実行数
            batch_size: バッチ処理のサイズ
        """
        self.redis = redis_service
        self.max_concurrency = max_concurrency
        self.batch_size = batch_size
        self.task_limiter = TaskLimiter(max_concurrency)
        self.batch_processor = AsyncBatchProcessor(batch_size, max_concurrency)

    async def get_many(self, keys: List[str]) -> List[Optional[str]]:
        """
        複数のキーの値を効率的に取得します

        Args:
            keys: 取得するキーのリスト

        Returns:
            値のリスト。存在しないキーの場合はNone
        """
        if not keys:
            return []

        logger.debug(f"{len(keys)}個のキーをバルク取得します")

        async def get_value(key: str) -> Optional[str]:
            return await self.redis.get(key)

        return await self.batch_processor.process(keys, get_value)

    async def get_many_json(self, keys: List[str]) -> List[Optional[Dict[str, Any]]]:
        """
        複数のキーのJSON値を効率的に取得します

        Args:
            keys: 取得するキーのリスト

        Returns:
            JSON値のリスト。存在しないキーの場合はNone
        """
        if not keys:
            return []

        logger.debug(f"{len(keys)}個のJSONキーをバルク取得します")

        async def get_json_value(key: str) -> Optional[Dict[str, Any]]:
            return await self.redis.get_json(key)

        return await self.batch_processor.process(keys, get_json_value)

    async def set_many(
        self,
        key_values: Dict[str, str],
        expire: Optional[int] = None
    ) -> bool:
        """
        複数のキーと値のペアを効率的に設定します

        Args:
            key_values: キーと値のペアの辞書
            expire: 有効期限（秒）

        Returns:
            すべての操作が成功した場合はTrue
        """
        if not key_values:
            return True

        logger.debug(f"{len(key_values)}個のキー・バリューペアをバルク設定します")

        async def set_value(item: Tuple[str, str]) -> bool:
            key, value = item
            return await self.redis.set(key, value, expire)

        results = await self.batch_processor.process(
            list(key_values.items()),
            set_value
        )

        return all(results)

    async def set_many_json(
        self,
        key_values: Dict[str, Dict[str, Any]],
        expire: Optional[int] = None
    ) -> bool:
        """
        複数のキーとJSON値のペアを効率的に設定します

        Args:
            key_values: キーとJSON値のペアの辞書
            expire: 有効期限（秒）

        Returns:
            すべての操作が成功した場合はTrue
        """
        if not key_values:
            return True

        logger.debug(f"{len(key_values)}個のJSON値をバルク設定します")

        async def set_json_value(item: Tuple[str, Dict[str, Any]]) -> bool:
            key, value = item
            return await self.redis.set_json(key, value, expire)

        results = await self.batch_processor.process(
            list(key_values.items()),
            set_json_value
        )

        return all(results)

    async def delete_many(self, keys: List[str]) -> int:
        """
        複数のキーを効率的に削除します

        Args:
            keys: 削除するキーのリスト

        Returns:
            削除されたキーの数
        """
        if not keys:
            return 0

        logger.debug(f"{len(keys)}個のキーをバルク削除します")

        async def delete_key(key: str) -> bool:
            return await self.redis.delete(key)

        results = await self.batch_processor.process(keys, delete_key)

        return sum(1 for r in results if r)

    async def delete_pattern(self, pattern: str) -> int:
        """
        パターンに一致するすべてのキーを効率的に削除します

        Args:
            pattern: 削除するキーのパターン

        Returns:
            削除されたキーの数
        """
        keys = await self.redis.keys(pattern)
        if not keys:
            return 0

        logger.debug(f"パターン '{pattern}' に一致する {len(keys)}個のキーを削除します")

        return await self.delete_many(keys)

    async def exists_many(self, keys: List[str]) -> Dict[str, bool]:
        """
        複数のキーの存在を効率的に確認します

        Args:
            keys: 確認するキーのリスト

        Returns:
            キーと存在状態のペアの辞書
        """
        if not keys:
            return {}

        logger.debug(f"{len(keys)}個のキーの存在をバルク確認します")

        async def check_exists(key: str) -> Tuple[str, bool]:
            exists = await self.redis.exists(key)
            return key, exists

        results = await self.batch_processor.process(keys, check_exists)

        return dict(results)

    async def cache_entities_with_ids(
        self,
        entities: List[T],
        id_getter: Callable[[T], str],
        serializer: Callable[[T], Dict[str, Any]],
        key_prefix: str,
        expire: Optional[int] = None
    ) -> bool:
        """
        エンティティのリストをIDをキーとしてキャッシュします

        Args:
            entities: キャッシュするエンティティのリスト
            id_getter: エンティティからIDを取得する関数
            serializer: エンティティをJSON辞書に変換する関数
            key_prefix: キーのプレフィックス
            expire: 有効期限（秒）

        Returns:
            すべての操作が成功した場合はTrue
        """
        if not entities:
            return True

        entity_dict = {}
        for entity in entities:
            entity_id = id_getter(entity)
            key = f"{key_prefix}{entity_id}"
            entity_dict[key] = serializer(entity)

        logger.debug(f"{len(entity_dict)}個のエンティティをバルクキャッシュします")

        return await self.set_many_json(entity_dict, expire)

    async def get_cached_entities(
        self,
        ids: List[str],
        key_prefix: str,
        deserializer: Callable[[Dict[str, Any]], T]
    ) -> Dict[str, Optional[T]]:
        """
        キャッシュされたエンティティをIDのリストで取得します

        Args:
            ids: 取得するエンティティのIDリスト
            key_prefix: キーのプレフィックス
            deserializer: JSON辞書からエンティティに変換する関数

        Returns:
            IDとエンティティのペアの辞書。キャッシュミスの場合はNone
        """
        if not ids:
            return {}

        keys = [f"{key_prefix}{id_}" for id_ in ids]
        json_values = await self.get_many_json(keys)

        result = {}
        for i, id_ in enumerate(ids):
            json_value = json_values[i]
            if json_value:
                try:
                    result[id_] = deserializer(json_value)
                except Exception as e:
                    logger.error(f"エンティティのデシリアライズに失敗しました (ID: {id_}): {e}")
                    result[id_] = None
            else:
                result[id_] = None

        return result

    async def update_entity_indices(
        self,
        entity: T,
        id_getter: Callable[[T], str],
        index_values: Dict[str, str],
        index_prefix: str,
        expire: Optional[int] = None
    ) -> bool:
        """
        エンティティのインデックスを更新します

        Args:
            entity: 更新するエンティティ
            id_getter: エンティティからIDを取得する関数
            index_values: インデックス名と値のペア
            index_prefix: インデックスキーのプレフィックス
            expire: 有効期限（秒）

        Returns:
            すべての操作が成功した場合はTrue
        """
        entity_id = id_getter(entity)
        index_dict = {}

        for index_name, index_value in index_values.items():
            if index_value:
                key = f"{index_prefix}{index_name}:{index_value}"
                index_dict[key] = entity_id

        logger.debug(f"エンティティ (ID: {entity_id}) の {len(index_dict)} 個のインデックスを更新します")

        return await self.set_many(index_dict, expire)


def create_redis_bulk_operations(
    redis_service: RedisService,
    max_concurrency: int = 20,
    batch_size: int = 100
) -> RedisBulkOperations:
    """
    Redisバルク操作ユーティリティを作成するファクトリ関数

    Args:
        redis_service: Redisサービスインスタンス
        max_concurrency: 最大同時実行数
        batch_size: バッチ処理のサイズ

    Returns:
        設定されたRedisバルク操作インスタンス
    """
    return RedisBulkOperations(redis_service, max_concurrency, batch_size)