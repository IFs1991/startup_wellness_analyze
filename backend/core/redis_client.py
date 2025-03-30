import os
import redis
import logging
import json
from typing import Optional, Any, Dict

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RedisClient:
    """
    Redisキャッシュクライアント

    アプリケーション全体で使用されるRedisインターフェースを提供します。
    """

    def __init__(self):
        """Redisクライアントの初期化"""
        try:
            # Redis接続情報
            self.host = os.environ.get("REDIS_HOST", "startup-wellness-redis")
            self.port = int(os.environ.get("REDIS_PORT", 6379))
            self.db = int(os.environ.get("REDIS_DB", 0))
            self.password = os.environ.get("REDIS_PASSWORD", None)

            # Redis接続
            try:
                self.client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    socket_timeout=5,
                    decode_responses=True
                )
                # 接続確認
                self.client.ping()
                logger.info(f"Redis接続成功: {self.host}:{self.port}")
                self.is_connected = True
            except redis.ConnectionError as e:
                logger.warning(f"Redis接続エラー: {e}、メモリキャッシュを使用します")
                self.is_connected = False
                self.client = None
                # メモリ内ディクショナリをフォールバックとして使用
                self.memory_cache = {}
            except Exception as e:
                logger.error(f"Redis初期化エラー: {e}")
                self.is_connected = False
                self.client = None
                self.memory_cache = {}
        except Exception as e:
            logger.error(f"RedisClientの初期化中にエラー発生: {e}")
            self.is_connected = False
            self.client = None
            self.memory_cache = {}

    def get(self, key: str) -> Optional[Any]:
        """キーに対応する値を取得する"""
        try:
            if not self.is_connected:
                return self.memory_cache.get(key)

            data = self.client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redisからの取得に失敗: {e}")
            # メモリキャッシュをフォールバックとして使用
            if hasattr(self, 'memory_cache'):
                return self.memory_cache.get(key)
            return None

    def set(self, key: str, value: Any, expiry: Optional[int] = None) -> bool:
        """キーと値のペアを保存する"""
        try:
            if not self.is_connected:
                self.memory_cache[key] = value
                return True

            serialized = json.dumps(value)
            if expiry:
                return self.client.setex(key, expiry, serialized)
            return self.client.set(key, serialized)
        except Exception as e:
            logger.error(f"Redisへの保存に失敗: {e}")
            # メモリキャッシュをフォールバックとして使用
            if hasattr(self, 'memory_cache'):
                self.memory_cache[key] = value
            return False

    def delete(self, key: str) -> bool:
        """
        キーを削除

        Args:
            key (str): 削除するキー

        Returns:
            bool: 操作が成功したかどうか
        """
        if self.client is None:
            return False

        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Redis delete操作エラー: {str(e)}")
            return False

    def exists(self, key: str) -> bool:
        """
        キーが存在するか確認

        Args:
            key (str): 確認するキー

        Returns:
            bool: キーが存在する場合はTrue
        """
        if self.client is None:
            return False

        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists操作エラー: {str(e)}")
            return False

def get_redis_client():
    """Redisクライアントの設定を環境に応じて調整"""
    redis_host = os.getenv("REDIS_HOST", "startup-wellness-redis")
    # Docker環境ではサービス名を使用
    if os.path.exists('/.dockerenv'):  # Dockerコンテナ内で実行されているか確認
        redis_host = "startup-wellness-redis"  # Docker Composeのサービス名

    redis_port = int(os.getenv("REDIS_PORT", 6379))
    redis_db = int(os.getenv("REDIS_DB", 0))

    try:
        client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            socket_connect_timeout=5
        )
        # 接続テスト
        client.ping()
        return client
    except redis.ConnectionError as e:
        logger.error(f"Redisクライアントの初期化中にエラーが発生しました: {e}")
        if os.getenv("ENVIRONMENT") == "development":
            logger.warning("開発環境ではRedis初期化エラーを無視します")
            return None
        raise