"""
Redisサービス

Redisデータベースへの接続とデータ操作のための抽象化レイヤー
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
import aioredis
from aioredis import Redis

logger = logging.getLogger(__name__)


class RedisService:
    """
    Redisデータベースとの通信を担当するサービスクラス。
    JSON、文字列、リストなどの様々なデータ型を操作するための抽象化レイヤーを提供します。
    """

    def __init__(self, redis_client: Redis):
        """
        RedisServiceクラスの初期化

        Args:
            redis_client: 初期化済みのRedisクライアント
        """
        self.redis = redis_client

    async def get_value(self, key: str) -> Optional[str]:
        """
        指定されたキーの文字列値を取得します

        Args:
            key: Redisキー

        Returns:
            キーに対応する文字列値。キーが存在しない場合はNone
        """
        try:
            value = await self.redis.get(key)
            return value.decode('utf-8') if value else None
        except Exception as e:
            logger.error(f"キー '{key}' の値取得中にエラーが発生しました: {str(e)}")
            return None

    async def set_value(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """
        指定されたキーに文字列値を設定します

        Args:
            key: Redisキー
            value: 設定する文字列値
            ttl: 有効期限（秒）、Noneの場合は無期限

        Returns:
            操作が成功したかどうか
        """
        try:
            if ttl:
                await self.redis.setex(key, ttl, value)
            else:
                await self.redis.set(key, value)
            return True
        except Exception as e:
            logger.error(f"キー '{key}' に値を設定中にエラーが発生しました: {str(e)}")
            return False

    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """
        指定されたキーからJSON値を取得します

        Args:
            key: Redisキー

        Returns:
            JSONをデシリアライズした辞書。キーが存在しない場合はNone
        """
        try:
            value = await self.redis.get(key)
            if not value:
                return None
            return json.loads(value.decode('utf-8'))
        except json.JSONDecodeError:
            logger.error(f"キー '{key}' の値がJSON形式ではありません")
            return None
        except Exception as e:
            logger.error(f"キー '{key}' のJSON取得中にエラーが発生しました: {str(e)}")
            return None

    async def set_json(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        指定されたキーにJSON値を設定します

        Args:
            key: Redisキー
            value: 設定するPython辞書（JSON形式に変換されます）
            ttl: 有効期限（秒）、Noneの場合は無期限

        Returns:
            操作が成功したかどうか
        """
        try:
            json_str = json.dumps(value)
            if ttl:
                await self.redis.setex(key, ttl, json_str)
            else:
                await self.redis.set(key, json_str)
            return True
        except Exception as e:
            logger.error(f"キー '{key}' にJSON値を設定中にエラーが発生しました: {str(e)}")
            return False

    async def delete_key(self, key: str) -> bool:
        """
        指定されたキーを削除します

        Args:
            key: 削除するRedisキー

        Returns:
            操作が成功したかどうか
        """
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"キー '{key}' の削除中にエラーが発生しました: {str(e)}")
            return False

    async def exists(self, key: str) -> bool:
        """
        指定されたキーが存在するか確認します

        Args:
            key: 確認するRedisキー

        Returns:
            キーが存在する場合はTrue、そうでない場合はFalse
        """
        try:
            return bool(await self.redis.exists(key))
        except Exception as e:
            logger.error(f"キー '{key}' の存在確認中にエラーが発生しました: {str(e)}")
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """
        既存のキーに有効期限を設定します

        Args:
            key: 対象のRedisキー
            ttl: 有効期限（秒）

        Returns:
            操作が成功したかどうか
        """
        try:
            return bool(await self.redis.expire(key, ttl))
        except Exception as e:
            logger.error(f"キー '{key}' の有効期限設定中にエラーが発生しました: {str(e)}")
            return False

    async def ttl(self, key: str) -> int:
        """
        キーの残り有効期限（秒）を取得します

        Args:
            key: 対象のRedisキー

        Returns:
            残り有効期限（秒）。キーが存在しない場合は-2、有効期限がない場合は-1
        """
        try:
            return await self.redis.ttl(key)
        except Exception as e:
            logger.error(f"キー '{key}' の有効期限取得中にエラーが発生しました: {str(e)}")
            return -2

    async def incr(self, key: str) -> int:
        """
        指定されたキーの値をインクリメントします

        Args:
            key: 対象のRedisキー

        Returns:
            インクリメント後の値。エラーの場合は-1
        """
        try:
            return await self.redis.incr(key)
        except Exception as e:
            logger.error(f"キー '{key}' のインクリメント中にエラーが発生しました: {str(e)}")
            return -1

    async def rpush(self, key: str, *values: str) -> int:
        """
        Redisリストの末尾に値を追加します

        Args:
            key: リストのRedisキー
            values: 追加する値

        Returns:
            操作後のリストの長さ。エラーの場合は-1
        """
        try:
            return await self.redis.rpush(key, *values)
        except Exception as e:
            logger.error(f"キー '{key}' のリストに値を追加中にエラーが発生しました: {str(e)}")
            return -1

    async def lrange(self, key: str, start: int, stop: int) -> List[str]:
        """
        Redisリストの指定範囲の要素を取得します

        Args:
            key: リストのRedisキー
            start: 開始インデックス
            stop: 終了インデックス

        Returns:
            指定範囲の要素リスト
        """
        try:
            values = await self.redis.lrange(key, start, stop)
            return [v.decode('utf-8') for v in values]
        except Exception as e:
            logger.error(f"キー '{key}' のリスト範囲取得中にエラーが発生しました: {str(e)}")
            return []

    async def hset(self, key: str, field: str, value: str) -> bool:
        """
        Redisハッシュの指定フィールドに値を設定します

        Args:
            key: ハッシュのRedisキー
            field: ハッシュフィールド
            value: 設定する値

        Returns:
            操作が成功したかどうか
        """
        try:
            await self.redis.hset(key, field, value)
            return True
        except Exception as e:
            logger.error(f"キー '{key}' のハッシュフィールド '{field}' に値を設定中にエラーが発生しました: {str(e)}")
            return False

    async def hget(self, key: str, field: str) -> Optional[str]:
        """
        Redisハッシュの指定フィールドの値を取得します

        Args:
            key: ハッシュのRedisキー
            field: ハッシュフィールド

        Returns:
            フィールドの値。存在しない場合はNone
        """
        try:
            value = await self.redis.hget(key, field)
            return value.decode('utf-8') if value else None
        except Exception as e:
            logger.error(f"キー '{key}' のハッシュフィールド '{field}' の値取得中にエラーが発生しました: {str(e)}")
            return None

    async def hgetall(self, key: str) -> Dict[str, str]:
        """
        Redisハッシュのすべてのフィールドと値を取得します

        Args:
            key: ハッシュのRedisキー

        Returns:
            フィールドと値のマッピング辞書
        """
        try:
            result = await self.redis.hgetall(key)
            return {k.decode('utf-8'): v.decode('utf-8') for k, v in result.items()}
        except Exception as e:
            logger.error(f"キー '{key}' のハッシュの全フィールド取得中にエラーが発生しました: {str(e)}")
            return {}

    async def keys(self, pattern: str) -> List[str]:
        """
        指定したパターンに一致するすべてのキーを取得します
        注: この操作は大規模なデータベースでは重いため、本番環境での使用は推奨されません

        Args:
            pattern: 検索パターン

        Returns:
            一致するキーのリスト
        """
        try:
            keys = await self.redis.keys(pattern)
            return [k.decode('utf-8') for k in keys]
        except Exception as e:
            logger.error(f"パターン '{pattern}' に一致するキーの取得中にエラーが発生しました: {str(e)}")
            return []

    async def flush_db(self) -> bool:
        """
        現在のデータベースのすべてのキーを削除します
        注: テスト環境でのみ使用するべきです

        Returns:
            操作が成功したかどうか
        """
        try:
            await self.redis.flushdb()
            return True
        except Exception as e:
            logger.error(f"データベースのフラッシュ中にエラーが発生しました: {str(e)}")
            return False

    async def close(self) -> None:
        """
        Redisクライアント接続を閉じます
        """
        try:
            await self.redis.close()
            logger.info("Redis接続を閉じました")
        except Exception as e:
            logger.error(f"Redis接続を閉じる際にエラーが発生しました: {str(e)}")


async def create_redis_service(
    host: str = 'localhost',
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    ssl: bool = False
) -> RedisService:
    """
    RedisServiceインスタンスを作成するファクトリ関数

    Args:
        host: Redisサーバーのホスト名
        port: Redisサーバーのポート
        db: 使用するデータベース番号
        password: Redis認証パスワード（必要な場合）
        ssl: SSL/TLS接続を使用するかどうか

    Returns:
        初期化されたRedisServiceインスタンス
    """
    try:
        redis_url = f"redis://{host}:{port}/{db}"
        redis_client = await aioredis.from_url(
            redis_url,
            password=password,
            ssl=ssl,
            encoding="utf-8",
            decode_responses=False  # バイト列で返す（デコードは個々のメソッドで行う）
        )

        logger.info(f"Redisサーバー（{host}:{port}、DB={db}）に接続しました")
        return RedisService(redis_client)
    except Exception as e:
        logger.error(f"Redisサーバー（{host}:{port}）への接続中にエラーが発生しました: {str(e)}")
        raise
