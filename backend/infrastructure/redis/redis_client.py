"""
Redisクライアントの実装

このモジュールはRedisデータベースとの通信を担当するクライアントを提供します。
接続、キー・バリューの操作、JSONデータの取得・設定など基本的な機能を提供します。
"""

import json
import logging
from typing import Dict, Optional, Union, Any

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class RedisClientInterface:
    """
    Redisクライアントのインターフェース

    Redis操作に必要なメソッドを定義します。
    このインターフェースを実装することで、異なるRedisクライアントライブラリを使用したり、
    テスト用のモックを作成したりすることが可能になります。
    """

    async def connect(self) -> bool:
        """
        Redisサーバーに接続します。

        Returns:
            bool: 接続成功時はTrue、失敗時はFalse
        """
        raise NotImplementedError

    async def get(self, key: str) -> Optional[str]:
        """
        指定されたキーの値を取得します。

        Args:
            key: 取得するキー

        Returns:
            Optional[str]: キーに関連付けられた値。キーが存在しない場合はNone
        """
        raise NotImplementedError

    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """
        指定されたキーのJSON値を取得し、Pythonの辞書に変換します。

        Args:
            key: 取得するキー

        Returns:
            Optional[Dict[str, Any]]: 辞書形式のデータ。キーが存在しない場合はNone
        """
        raise NotImplementedError

    async def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        """
        キーと値のペアを保存します。

        Args:
            key: 保存するキー
            value: 保存する値
            expire: 有効期限（秒）。Noneの場合は無期限

        Returns:
            bool: 操作が成功したかどうか
        """
        raise NotImplementedError

    async def set_json(self, key: str, value: Dict[str, Any], expire: Optional[int] = None) -> bool:
        """
        Pythonの辞書をJSON文字列に変換して保存します。

        Args:
            key: 保存するキー
            value: 保存する辞書データ
            expire: 有効期限（秒）。Noneの場合は無期限

        Returns:
            bool: 操作が成功したかどうか
        """
        raise NotImplementedError

    async def delete(self, key: str) -> bool:
        """
        指定されたキーを削除します。

        Args:
            key: 削除するキー

        Returns:
            bool: 操作が成功したかどうか
        """
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        """
        指定されたキーが存在するか確認します。

        Args:
            key: 確認するキー

        Returns:
            bool: キーが存在する場合はTrue、存在しない場合はFalse
        """
        raise NotImplementedError

    async def flush_all(self) -> bool:
        """
        すべてのキーを削除します。

        Returns:
            bool: 操作が成功したかどうか
        """
        raise NotImplementedError


class RedisClient(RedisClientInterface):
    """
    Redisクライアントの実装クラス

    redis-pyライブラリを使用してRedisデータベースと通信します。
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0,
                 password: Optional[str] = None, ssl: bool = False):
        """
        RedisClientの初期化

        Args:
            host: Redisサーバーのホスト名
            port: Redisサーバーのポート番号
            db: データベース番号
            password: 認証パスワード（必要な場合）
            ssl: SSL接続を使用するかどうか
        """
        self.connection_params = {
            "host": host,
            "port": port,
            "db": db,
            "password": password,
            "ssl": ssl,
            "decode_responses": True,  # 文字列を自動的にデコードする
        }
        self._pool: Optional[ConnectionPool] = None
        self._redis: Optional[redis.Redis] = None

    async def connect(self) -> bool:
        """
        Redisサーバーに接続します。

        Returns:
            bool: 接続成功時はTrue、失敗時はFalse
        """
        try:
            self._pool = redis.ConnectionPool(**self.connection_params)
            self._redis = redis.Redis(connection_pool=self._pool)

            # 接続テスト
            await self._redis.ping()
            logger.info("Redisサーバーに接続しました: %s:%s", self.connection_params["host"], self.connection_params["port"])
            return True
        except RedisError as e:
            logger.error("Redisサーバーへの接続に失敗しました: %s", str(e))
            return False

    async def get(self, key: str) -> Optional[str]:
        """
        指定されたキーの値を取得します。

        Args:
            key: 取得するキー

        Returns:
            Optional[str]: キーに関連付けられた値。キーが存在しない場合はNone
        """
        try:
            if self._redis is None:
                logger.error("Redisクライアントが初期化されていません")
                return None

            return await self._redis.get(key)
        except RedisError as e:
            logger.error("キー '%s' の取得に失敗しました: %s", key, str(e))
            return None

    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """
        指定されたキーのJSON値を取得し、Pythonの辞書に変換します。

        Args:
            key: 取得するキー

        Returns:
            Optional[Dict[str, Any]]: 辞書形式のデータ。キーが存在しない場合はNone
        """
        try:
            value = await self.get(key)
            if value is None:
                return None

            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.error("キー '%s' の値をJSONとして解析できませんでした: %s", key, str(e))
            return None

    async def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        """
        キーと値のペアを保存します。

        Args:
            key: 保存するキー
            value: 保存する値
            expire: 有効期限（秒）。Noneの場合は無期限

        Returns:
            bool: 操作が成功したかどうか
        """
        try:
            if self._redis is None:
                logger.error("Redisクライアントが初期化されていません")
                return False

            await self._redis.set(key, value, ex=expire)
            return True
        except RedisError as e:
            logger.error("キー '%s' の設定に失敗しました: %s", key, str(e))
            return False

    async def set_json(self, key: str, value: Dict[str, Any], expire: Optional[int] = None) -> bool:
        """
        Pythonの辞書をJSON文字列に変換して保存します。

        Args:
            key: 保存するキー
            value: 保存する辞書データ
            expire: 有効期限（秒）。Noneの場合は無期限

        Returns:
            bool: 操作が成功したかどうか
        """
        try:
            json_value = json.dumps(value)
            return await self.set(key, json_value, expire)
        except (TypeError, ValueError) as e:
            logger.error("辞書をJSONに変換できませんでした: %s", str(e))
            return False

    async def delete(self, key: str) -> bool:
        """
        指定されたキーを削除します。

        Args:
            key: 削除するキー

        Returns:
            bool: 操作が成功したかどうか（キーが存在しない場合もTrueを返す）
        """
        try:
            if self._redis is None:
                logger.error("Redisクライアントが初期化されていません")
                return False

            await self._redis.delete(key)
            return True
        except RedisError as e:
            logger.error("キー '%s' の削除に失敗しました: %s", key, str(e))
            return False

    async def exists(self, key: str) -> bool:
        """
        指定されたキーが存在するか確認します。

        Args:
            key: 確認するキー

        Returns:
            bool: キーが存在する場合はTrue、存在しない場合はFalse
        """
        try:
            if self._redis is None:
                logger.error("Redisクライアントが初期化されていません")
                return False

            result = await self._redis.exists(key)
            return bool(result)
        except RedisError as e:
            logger.error("キー '%s' の存在確認に失敗しました: %s", key, str(e))
            return False

    async def flush_all(self) -> bool:
        """
        すべてのキーを削除します。

        Returns:
            bool: 操作が成功したかどうか
        """
        try:
            if self._redis is None:
                logger.error("Redisクライアントが初期化されていません")
                return False

            await self._redis.flushdb()
            return True
        except RedisError as e:
            logger.error("データベースのクリアに失敗しました: %s", str(e))
            return False


def create_redis_client(host: str = "localhost", port: int = 6379, db: int = 0,
                       password: Optional[str] = None, ssl: bool = False) -> RedisClient:
    """
    RedisClientのインスタンスを作成するファクトリ関数

    Args:
        host: Redisサーバーのホスト名
        port: Redisサーバーのポート番号
        db: データベース番号
        password: 認証パスワード（必要な場合）
        ssl: SSL接続を使用するかどうか

    Returns:
        RedisClient: 設定されたRedisClientインスタンス
    """
    return RedisClient(host, port, db, password, ssl)