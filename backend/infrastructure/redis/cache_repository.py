"""
Redisを利用したキャッシュリポジトリの実装

このモジュールは、Redisをバックエンドとした汎用的なキャッシュリポジトリを提供します。
各ドメインオブジェクトのキャッシュを管理するための基底クラスとインターフェースを定義します。
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Dict, Any, List, Type
import json
import time
from datetime import datetime

from backend.infrastructure.redis.redis_service import RedisService

T = TypeVar('T')


class CacheRepositoryInterface(Generic[T], ABC):
    """
    キャッシュリポジトリのインターフェース

    特定のドメインエンティティのキャッシュを管理するためのインターフェースです。
    """

    @abstractmethod
    async def get(self, key: str) -> Optional[T]:
        """
        指定されたキーに対応する値を取得します

        Args:
            key: キャッシュキー

        Returns:
            Optional[T]: キャッシュされた値、存在しない場合はNone
        """
        pass

    @abstractmethod
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """
        キーと値のペアをキャッシュに格納します

        Args:
            key: キャッシュキー
            value: 格納する値
            ttl: 有効期限（秒）、Noneの場合はデフォルト値を使用

        Returns:
            bool: 操作が成功した場合はTrue
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        指定されたキーのキャッシュを削除します

        Args:
            key: 削除するキャッシュのキー

        Returns:
            bool: 削除が成功した場合はTrue
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        指定されたキーが存在するか確認します

        Args:
            key: 確認するキャッシュのキー

        Returns:
            bool: キーが存在する場合はTrue
        """
        pass

    @abstractmethod
    async def clear_all(self) -> bool:
        """
        このリポジトリが管理するすべてのキャッシュを削除します

        Returns:
            bool: 操作が成功した場合はTrue
        """
        pass


class RedisCacheRepository(CacheRepositoryInterface[T]):
    """
    Redisを利用した汎用キャッシュリポジトリの実装

    特定のプレフィックスとシリアライズ/デシリアライズロジックを持つ
    キャッシュリポジトリを提供します。
    """

    def __init__(
        self,
        redis_service: RedisService,
        prefix: str,
        entity_class: Optional[Type[T]] = None,
        default_ttl: int = 3600
    ):
        """
        初期化

        Args:
            redis_service: Redisサービスのインスタンス
            prefix: このリポジトリが使用するキープレフィックス
            entity_class: エンティティクラス（シリアライズ/デシリアライズに使用）
            default_ttl: デフォルトのTTL（秒）
        """
        self._redis = redis_service
        self._prefix = prefix
        self._entity_class = entity_class
        self._default_ttl = default_ttl

    def _build_key(self, key: str) -> str:
        """
        プレフィックス付きのキーを生成します

        Args:
            key: ベースキー

        Returns:
            str: プレフィックス付きの完全なキー
        """
        return f"{self._prefix}:{key}"

    async def get(self, key: str) -> Optional[T]:
        """
        指定されたキーに対応する値を取得します

        Args:
            key: キャッシュキー

        Returns:
            Optional[T]: キャッシュされた値、存在しない場合はNone
        """
        full_key = self._build_key(key)
        data = await self._redis.get_json(full_key)

        if data is None:
            return None

        return self._deserialize(data)

    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """
        キーと値のペアをキャッシュに格納します

        Args:
            key: キャッシュキー
            value: 格納する値
            ttl: 有効期限（秒）、Noneの場合はデフォルト値を使用

        Returns:
            bool: 操作が成功した場合はTrue
        """
        full_key = self._build_key(key)
        serialized = self._serialize(value)
        return await self._redis.set_json(
            full_key,
            serialized,
            ttl if ttl is not None else self._default_ttl
        )

    async def delete(self, key: str) -> bool:
        """
        指定されたキーのキャッシュを削除します

        Args:
            key: 削除するキャッシュのキー

        Returns:
            bool: 削除が成功した場合はTrue
        """
        full_key = self._build_key(key)
        return await self._redis.delete_key(full_key)

    async def exists(self, key: str) -> bool:
        """
        指定されたキーが存在するか確認します

        Args:
            key: 確認するキャッシュのキー

        Returns:
            bool: キーが存在する場合はTrue
        """
        full_key = self._build_key(key)
        return await self._redis.has_key(full_key)

    async def clear_all(self) -> bool:
        """
        このリポジトリが管理するすべてのキャッシュを削除します

        Returns:
            bool: 操作が成功した場合はTrue
        """
        # このプレフィックスに一致するすべてのキーを削除
        # 注：この実装では、実際のRedisサービスにこの機能を
        # 追加する必要があります。
        pattern = f"{self._prefix}:*"
        return await self._redis.delete_pattern(pattern)

    def _serialize(self, obj: T) -> Dict[str, Any]:
        """
        オブジェクトをシリアライズします

        Args:
            obj: シリアライズするオブジェクト

        Returns:
            Dict[str, Any]: シリアライズされた辞書
        """
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()

        # 基本的な型変換
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return {'value': obj}

        if isinstance(obj, (list, tuple)):
            return {'items': [self._serialize(item) for item in obj]}

        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}

        if isinstance(obj, datetime):
            return {'timestamp': obj.timestamp()}

        # 辞書に変換可能なオブジェクトの場合
        try:
            return dict(obj)
        except (TypeError, ValueError):
            # 最後の手段として__dict__を使用
            return obj.__dict__

    def _deserialize(self, data: Dict[str, Any]) -> T:
        """
        シリアライズされたデータからオブジェクトを再構築します

        Args:
            data: デシリアライズするデータ

        Returns:
            T: 再構築されたオブジェクト
        """
        # 単純な値
        if 'value' in data and len(data) == 1:
            return data['value']

        # リスト
        if 'items' in data and len(data) == 1:
            return [self._deserialize(item) for item in data['items']]

        # 日時
        if 'timestamp' in data and len(data) == 1:
            return datetime.fromtimestamp(data['timestamp'])

        # エンティティクラスが指定されている場合
        if self._entity_class is not None:
            if hasattr(self._entity_class, 'from_dict') and callable(getattr(self._entity_class, 'from_dict')):
                return self._entity_class.from_dict(data)

            # コンストラクタに辞書を展開
            try:
                return self._entity_class(**data)
            except (TypeError, ValueError):
                pass

        # デフォルトは辞書をそのまま返す
        return data