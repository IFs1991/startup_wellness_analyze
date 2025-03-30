# -*- coding: utf-8 -*-
"""
レート制限ミドルウェア
APIエンドポイントへのアクセス頻度を制限し、過負荷やDDoS攻撃を防止します。
"""
import time
import logging
from typing import Dict, Tuple, Optional, Callable, List, Any
from datetime import datetime, timedelta
import redis
import os
import json
import hashlib
import asyncio
from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from starlette.responses import JSONResponse

from core.auth_metrics import get_auth_metrics

# ロガーの設定
logger = logging.getLogger(__name__)

# Redis接続設定
REDIS_HOST = os.environ.get("REDIS_HOST", "startup-wellness-redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 2))  # レート制限用に別DBを使用
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)

# デフォルトのレート制限設定
DEFAULT_RATE_LIMITS = {
    "/auth/token": {"limit": 5, "period": 60},  # ログインは1分間に5回まで
    "/auth/register": {"limit": 3, "period": 300},  # 登録は5分間に3回まで
    "/auth/password-reset": {"limit": 3, "period": 3600},  # パスワードリセットは1時間に3回まで
    "/auth/mfa/verify": {"limit": 5, "period": 300},  # MFA検証は5分間に5回まで
    "*": {"limit": 60, "period": 60}  # その他のエンドポイントは1分間に60回まで
}

class RateLimiter:
    """レート制限を管理するクラス"""
    _instance = None
    _redis_client = None
    _rate_limits = DEFAULT_RATE_LIMITS
    _metrics = None
    _in_memory_cache = {}  # Redisが利用できない場合の代替

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RateLimiter, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """レート制限ミドルウェアの初期化"""
        try:
            self._initialize_redis()
            self._metrics = get_auth_metrics()
            self._load_rate_limits()
            logger.info("RateLimiterが正常に初期化されました")
        except Exception as e:
            logger.error(f"RateLimiterの初期化中にエラーが発生しました: {str(e)}")

    def _initialize_redis(self):
        """Redisクライアントの初期化"""
        try:
            self._redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True
            )
            self._redis_client.ping()  # 接続テスト
            logger.info("レート制限用Redisクライアントが初期化されました")
        except Exception as e:
            logger.error(f"レート制限用Redisクライアントの初期化中にエラーが発生しました: {str(e)}")
            self._redis_client = None

    def _load_rate_limits(self):
        """レート制限設定をロード"""
        # 環境変数からカスタム設定を取得（JSON形式）
        custom_limits = os.environ.get("RATE_LIMITS")
        if custom_limits:
            try:
                custom_limits_dict = json.loads(custom_limits)
                # デフォルト設定を上書き
                for endpoint, config in custom_limits_dict.items():
                    self._rate_limits[endpoint] = config
                logger.info("カスタムレート制限設定をロードしました")
            except json.JSONDecodeError:
                logger.error("カスタムレート制限設定の解析に失敗しました")

    def _get_limit_config(self, path: str) -> dict:
        """パスに応じたレート制限設定を取得"""
        # 正確なパスマッチを試行
        if path in self._rate_limits:
            return self._rate_limits[path]

        # プレフィックスマッチを試行
        for pattern, config in self._rate_limits.items():
            if pattern != "*" and pattern.endswith("*") and path.startswith(pattern[:-1]):
                return config

        # デフォルト設定を返す
        return self._rate_limits["*"]

    async def is_rate_limited(self, request: Request) -> Tuple[bool, Dict[str, Any]]:
        """
        リクエストがレート制限に達しているかどうかを判定

        Args:
            request: FastAPIリクエストオブジェクト

        Returns:
            Tuple[bool, Dict]: (制限超過フラグ, 制限情報)
        """
        # クライアントIPとパスを取得
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        # レート制限の設定を取得
        limit_config = self._get_limit_config(path)
        limit = limit_config["limit"]
        period = limit_config["period"]

        # キーの作成（IP + パス）
        cache_key = f"rate_limit:{client_ip}:{path}"

        # 現在のUNIXタイムスタンプ
        current_time = time.time()

        # Redisが利用可能な場合
        if self._redis_client:
            try:
                pipe = self._redis_client.pipeline()

                # 期限切れのリクエストを削除
                pipe.zremrangebyscore(cache_key, 0, current_time - period)

                # 現在のリクエスト数を取得
                pipe.zcard(cache_key)

                # 現在のリクエストを追加
                pipe.zadd(cache_key, {str(current_time): current_time})

                # キーの有効期限を設定
                pipe.expire(cache_key, period)

                # パイプラインを実行
                _, request_count, _, _ = pipe.execute()

            except Exception as e:
                logger.error(f"Redisからのレート制限情報取得中にエラーが発生しました: {str(e)}")
                # フォールバック: メモリ内キャッシュを使用
                return await self._check_in_memory_rate_limit(cache_key, period, limit, current_time)
        else:
            # Redisが利用できない場合はメモリ内キャッシュを使用
            return await self._check_in_memory_rate_limit(cache_key, period, limit, current_time)

        # レート制限情報を作成
        remaining = max(0, limit - request_count)
        reset_time = current_time + period
        is_limited = request_count >= limit

        # 制限に達した場合はメトリクスを記録
        if is_limited and self._metrics:
            self._metrics.track_rate_limit_hit(path, client_ip)
            # 怪しい活動としても記録（多数のレート制限ヒット）
            if await self._check_suspicious_activity(client_ip):
                self._metrics.track_suspicious_activity(
                    "multiple_rate_limit_hits",
                    "medium"
                )

        return is_limited, {
            "limit": limit,
            "remaining": remaining,
            "reset": reset_time,
            "path": path,
            "ip": client_ip
        }

    async def _check_in_memory_rate_limit(
        self, cache_key: str, period: int, limit: int, current_time: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """メモリ内キャッシュを使用してレート制限をチェック"""
        # キャッシュからリクエスト履歴を取得
        if cache_key not in self._in_memory_cache:
            self._in_memory_cache[cache_key] = []

        requests = self._in_memory_cache[cache_key]

        # 期限切れのリクエストを削除
        cutoff_time = current_time - period
        requests = [ts for ts in requests if ts > cutoff_time]

        # 新しいリクエストを追加
        requests.append(current_time)
        self._in_memory_cache[cache_key] = requests

        # メモリ使用量を制限（古いキーを削除）
        if len(self._in_memory_cache) > 10000:  # キャッシュが大きすぎる場合
            # 最も古いキーを削除
            oldest_key = min(
                self._in_memory_cache.keys(),
                key=lambda k: max(self._in_memory_cache[k]) if self._in_memory_cache[k] else 0
            )
            self._in_memory_cache.pop(oldest_key, None)

        # レート制限情報を作成
        request_count = len(requests)
        remaining = max(0, limit - request_count)
        reset_time = current_time + period
        is_limited = request_count > limit

        # パスとIPを抽出
        path = cache_key.split(":")[-1]
        ip = cache_key.split(":")[-2]

        return is_limited, {
            "limit": limit,
            "remaining": remaining,
            "reset": reset_time,
            "path": path,
            "ip": ip
        }

    async def _check_suspicious_activity(self, client_ip: str) -> bool:
        """
        クライアントIPの過去のレート制限ヒットを確認し、
        不審なアクティビティを検出

        Args:
            client_ip: クライアントのIPアドレス

        Returns:
            bool: 不審なアクティビティが検出された場合はTrue
        """
        suspicious_threshold = 10  # 10回以上のヒットで不審と判断
        window_seconds = 3600  # 1時間の時間枠

        # 現在のUNIXタイムスタンプ
        current_time = time.time()
        cutoff_time = current_time - window_seconds

        # レート制限ヒットカウントキー
        count_key = f"rate_limit_hits:{client_ip}"

        # Redisが利用可能な場合
        if self._redis_client:
            try:
                # 期間内のヒット回数をカウント
                hit_count = self._redis_client.zcount(count_key, cutoff_time, "+inf")

                # 新しいヒットを記録
                self._redis_client.zadd(count_key, {str(current_time): current_time})
                self._redis_client.expire(count_key, window_seconds)

                return hit_count >= suspicious_threshold
            except Exception as e:
                logger.error(f"不審なアクティビティチェック中にエラーが発生しました: {str(e)}")
                return False
        else:
            # メモリ内カウントを使用
            hits_key = f"suspicious_check:{client_ip}"
            if hits_key not in self._in_memory_cache:
                self._in_memory_cache[hits_key] = []

            hits = self._in_memory_cache[hits_key]
            hits = [ts for ts in hits if ts > cutoff_time]
            hits.append(current_time)
            self._in_memory_cache[hits_key] = hits

            return len(hits) >= suspicious_threshold

    def get_rate_limit_headers(self, limit_info: Dict[str, Any]) -> Dict[str, str]:
        """
        レート制限情報からHTTPヘッダーを生成

        Args:
            limit_info: レート制限情報の辞書

        Returns:
            Dict[str, str]: レート制限ヘッダー
        """
        return {
            "X-RateLimit-Limit": str(limit_info["limit"]),
            "X-RateLimit-Remaining": str(limit_info["remaining"]),
            "X-RateLimit-Reset": str(int(limit_info["reset"])),
        }

    def clear_rate_limits(self, client_ip: str = None, path: str = None) -> bool:
        """
        特定のIPとパスのレート制限をクリア

        Args:
            client_ip: クライアントIP（指定しない場合は全IP）
            path: パス（指定しない場合は全パス）

        Returns:
            bool: 成功した場合はTrue
        """
        pattern = "rate_limit:"
        if client_ip:
            pattern += f"{client_ip}:"
            if path:
                pattern += path

        try:
            # Redisが利用可能な場合
            if self._redis_client:
                keys = self._redis_client.keys(pattern + "*")
                if keys:
                    self._redis_client.delete(*keys)
            else:
                # メモリ内キャッシュを使用
                keys_to_delete = []
                for key in self._in_memory_cache.keys():
                    if key.startswith(pattern):
                        keys_to_delete.append(key)
                for key in keys_to_delete:
                    self._in_memory_cache.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"レート制限クリア中にエラーが発生しました: {str(e)}")
            return False

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI用のレート制限ミドルウェア
    """
    def __init__(self, app, **options):
        """ミドルウェアの初期化"""
        super().__init__(app)
        self.limiter = RateLimiter()
        # ホワイトリストのIPやパス
        self.whitelist_ips = options.get("whitelist_ips", [])
        self.whitelist_paths = options.get("whitelist_paths", ["/health", "/metrics"])
        logger.info("RateLimitMiddlewareが初期化されました")

    async def dispatch(self, request: Request, call_next) -> Response:
        """リクエスト処理のミドルウェア"""
        # ホワイトリストチェック
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        # ヘルスチェックやメトリクスエンドポイントはスキップ
        if client_ip in self.whitelist_ips or path in self.whitelist_paths:
            return await call_next(request)

        # レート制限のチェック
        is_limited, limit_info = await self.limiter.is_rate_limited(request)

        # レート制限に達した場合
        if is_limited:
            status_code = HTTP_429_TOO_MANY_REQUESTS
            headers = self.limiter.get_rate_limit_headers(limit_info)

            # レート制限超過のレスポンス
            return JSONResponse(
                status_code=status_code,
                content={
                    "detail": "リクエスト回数が制限を超えています。しばらく経ってからもう一度お試しください。",
                    "limit": limit_info["limit"],
                    "reset_seconds": int(limit_info["reset"] - time.time())
                },
                headers=headers
            )

        # 通常のリクエスト処理
        response = await call_next(request)

        # レスポンスにレート制限ヘッダーを追加
        headers = self.limiter.get_rate_limit_headers(limit_info)
        for name, value in headers.items():
            response.headers[name] = value

        return response

# シングルトンインスタンスの取得関数
def get_rate_limiter() -> RateLimiter:
    """RateLimiterのシングルトンインスタンスを返す"""
    return RateLimiter()

# レート制限デコレータ
def rate_limited(max_requests: int, window_seconds: int):
    """
    APIエンドポイントにレート制限を適用するデコレータ

    Args:
        max_requests: 許可される最大リクエスト数
        window_seconds: 制限ウィンドウの秒数

    Returns:
        デコレータ関数
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # リクエストオブジェクトを取得
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if request is None:
                # FastAPIのDependsシステムから取得を試みる
                for k, v in kwargs.items():
                    if isinstance(v, Request):
                        request = v
                        break

            # リクエストが見つからない場合、制限なしで実行
            if request is None:
                logger.warning("レート制限: リクエストオブジェクトが見つかりませんでした。制限なしで実行します。")
                return await func(*args, **kwargs)

            # カスタムレート制限設定
            path = request.url.path
            limiter = get_rate_limiter()

            # レート制限のチェック（カスタム設定で上書き）
            is_limited, limit_info = await limiter.is_rate_limited(request)

            # レート制限オーバーの場合
            if is_limited:
                headers = limiter.get_rate_limit_headers(limit_info)
                raise HTTPException(
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "message": "リクエスト回数が制限を超えています。しばらく経ってからもう一度お試しください。",
                        "limit": limit_info["limit"],
                        "reset_seconds": int(limit_info["reset"] - time.time())
                    },
                    headers=headers
                )

            # 制限内なら通常実行
            return await func(*args, **kwargs)

        return wrapper

    return decorator