# -*- coding: utf-8 -*-
"""
API ミドルウェア
--------------
FastAPIアプリケーションで使用されるミドルウェアおよびエラーハンドラを定義します。
エラー処理、ロギング、レスポンス形式の統一などの機能を提供します。
"""

import time
import uuid
import logging
import traceback
from typing import Optional, Dict, Any, Callable, List
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# 自作ロギングユーティリティをインポート
from api.logging_utils import get_logger, metrics, PROMETHEUS_AVAILABLE, ENABLE_METRICS

# ロギングの設定
logger = get_logger(__name__)

# カスタム例外クラス
class APIError(Exception):
    """
    API固有のエラー

    属性:
        status_code: HTTPステータスコード
        code: エラーコード
        message: エラーメッセージ
        details: 追加の詳細情報
    """
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)

# データベースエラー
class DatabaseError(APIError):
    """データベース操作中のエラー"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="DATABASE_ERROR",
            message=message,
            details=details
        )

# データ検証エラー
class ValidationFailedError(APIError):
    """データ検証の失敗"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="VALIDATION_FAILED",
            message=message,
            details=details
        )

# リソース未検出エラー
class ResourceNotFoundError(APIError):
    """リソースが見つからないエラー"""
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            code="RESOURCE_NOT_FOUND",
            message=f"{resource_type} with id {resource_id} not found",
            details={"resource_type": resource_type, "resource_id": resource_id}
        )

# 認証エラー
class AuthenticationError(APIError):
    """認証エラー"""
    def __init__(self, message: str = "認証に失敗しました"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            code="AUTHENTICATION_FAILED",
            message=message
        )

# 権限エラー
class PermissionDeniedError(APIError):
    """権限不足エラー"""
    def __init__(self, message: str = "この操作を実行する権限がありません"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            code="PERMISSION_DENIED",
            message=message
        )

# 拡張されたリクエスト処理タイミングミドルウェア
class TimingMiddleware(BaseHTTPMiddleware):
    """
    リクエスト処理時間を測定するミドルウェア

    このミドルウェアは各リクエストの処理時間を測定し、ログに記録します。
    処理に時間がかかる場合は警告ログも出力します。
    また、Prometheus用のメトリクスも記録します。
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # リクエストIDを生成
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # パス部分からエンドポイント名を取得
        endpoint = request.url.path
        method = request.method

        # Prometheusメトリクスのインクリメント（実行中）
        if PROMETHEUS_AVAILABLE and ENABLE_METRICS and "http_requests_in_progress" in metrics:
            metrics["http_requests_in_progress"].labels(
                method=method,
                endpoint=endpoint
            ).inc()

        # 開始時間の記録
        start_time = time.time()

        # リクエスト情報を取得
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        referer = request.headers.get("referer", "-")
        content_length = request.headers.get("content-length", "0")

        # リクエスト開始ログ
        context = {
            "method": method,
            "path": endpoint,
            "client_ip": client_host,
            "user_agent": user_agent,
            "referer": referer,
            "content_length": content_length
        }

        logger.info(
            f"リクエスト開始: {method} {endpoint}",
            extra={
                "context": context,
                "request_id": request_id
            }
        )

        status_code = 500

        try:
            # 実際のリクエスト処理
            response = await call_next(request)
            status_code = response.status_code

            # 処理時間の計算
            process_time = time.time() - start_time

            # レスポンスヘッダーの設定
            response.headers["X-Process-Time"] = f"{process_time:.6f}"
            response.headers["X-Request-ID"] = request_id

            # パフォーマンスログの記録
            log_level = logging.WARNING if process_time > 1.0 else logging.INFO

            log_context = context.copy()
            log_context.update({
                "status_code": status_code,
                "processing_time": f"{process_time:.6f}s"
            })

            logger.log(
                log_level,
                f"リクエスト完了: {method} {endpoint} - ステータス: {status_code}, 処理時間: {process_time:.6f}s",
                extra={
                    "context": log_context,
                    "request_id": request_id,
                    "duration_ms": process_time * 1000
                }
            )

            return response

        except Exception as e:
            # エラー発生時の処理
            process_time = time.time() - start_time

            # エラーの詳細をログに記録
            error_context = context.copy()
            error_context.update({
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time": f"{process_time:.6f}s"
            })

            logger.exception(
                f"リクエスト処理中にエラー発生: {method} {endpoint} - エラー: {str(e)}",
                extra={
                    "context": error_context,
                    "request_id": request_id,
                    "duration_ms": process_time * 1000
                }
            )

            raise

        finally:
            # Prometheusメトリクスの記録
            if PROMETHEUS_AVAILABLE and ENABLE_METRICS:
                # 処理中リクエストの減少
                if "http_requests_in_progress" in metrics:
                    metrics["http_requests_in_progress"].labels(
                        method=method,
                        endpoint=endpoint
                    ).dec()

                # リクエスト数のカウント
                if "http_requests_total" in metrics:
                    metrics["http_requests_total"].labels(
                        method=method,
                        endpoint=endpoint,
                        status_code=str(status_code)
                    ).inc()

                # リクエスト処理時間の記録
                if "http_request_duration_seconds" in metrics:
                    metrics["http_request_duration_seconds"].labels(
                        method=method,
                        endpoint=endpoint
                    ).observe(time.time() - start_time)

# Prometheusメトリクスミドルウェア
class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Prometheusメトリクスを公開するためのミドルウェア

    このミドルウェアはPrometheusメトリクスエクスポートを設定します。
    /metrics エンドポイントでメトリクスを公開します。
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if PROMETHEUS_AVAILABLE and ENABLE_METRICS and request.url.path == "/metrics":
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            content = generate_latest()
            return Response(content=content, media_type=CONTENT_TYPE_LATEST)

        return await call_next(request)

# API使用状況モニタリングミドルウェア
class APIUsageMonitoringMiddleware(BaseHTTPMiddleware):
    """
    API使用状況をモニタリングするミドルウェア

    このミドルウェアは、非推奨APIパスの使用を検出し、ログに記録します。
    また、Prometheusメトリクスも収集して、非推奨APIパスの使用状況を追跡します。
    """
    def __init__(self, app: FastAPI, deprecated_paths_prefix: List[str] = None):
        """
        初期化

        Args:
            app: FastAPIアプリケーションインスタンス
            deprecated_paths_prefix: 非推奨APIパスのプレフィックスリスト（デフォルトは/api/v1/）
        """
        super().__init__(app)
        self.deprecated_paths_prefix = deprecated_paths_prefix or ["/api/v1/"]

        # Prometheusメトリクスが利用可能な場合は初期化
        if PROMETHEUS_AVAILABLE and ENABLE_METRICS and "deprecated_api_usage_total" not in metrics:
            from prometheus_client import Counter
            metrics["deprecated_api_usage_total"] = Counter(
                "deprecated_api_usage_total",
                "非推奨APIパスの使用回数",
                ["method", "path", "client_ip"]
            )
            logger.info("非推奨API使用状況モニタリングメトリクスを初期化しました")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        method = request.method

        # 非推奨パスかどうかチェック
        is_deprecated_path = any(path.startswith(prefix) for prefix in self.deprecated_paths_prefix)

        if is_deprecated_path:
            client_ip = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", "unknown")
            referer = request.headers.get("referer", "-")

            # 非推奨APIパスの使用をログに記録
            logger.warning(
                f"非推奨APIパスが使用されました: {method} {path}",
                extra={
                    "context": {
                        "method": method,
                        "path": path,
                        "client_ip": client_ip,
                        "user_agent": user_agent,
                        "referer": referer,
                        "is_deprecated": True
                    }
                }
            )

            # メトリクスを記録（利用可能な場合）
            if PROMETHEUS_AVAILABLE and ENABLE_METRICS and "deprecated_api_usage_total" in metrics:
                metrics["deprecated_api_usage_total"].labels(
                    method=method,
                    path=path,
                    client_ip=client_ip
                ).inc()

        # 実際のリクエスト処理
        response = await call_next(request)

        # 非推奨パスの場合、警告ヘッダーを追加
        if is_deprecated_path:
            # 新しいパスを生成（/api/v1/ -> /api/）
            new_path = path
            for prefix in self.deprecated_paths_prefix:
                if path.startswith(prefix):
                    new_path = path.replace(prefix, "/api/")
                    break

            # 警告ヘッダーを追加
            response.headers["Warning"] = "299 - 'このAPIパスは非推奨です。新しいパスを使用してください。'"
            response.headers["X-Deprecated-API"] = "true"
            response.headers["X-New-API-Path"] = new_path

        return response

# エラーハンドラ関数
def setup_error_handlers(app: FastAPI) -> None:
    """
    FastAPIアプリケーションにエラーハンドラを設定します

    Args:
        app: FastAPIアプリケーションインスタンス
    """

    # カスタムAPIエラーのハンドラ
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))

        logger.error(
            f"APIエラー: {exc.code} - {exc.message} - {request.method} {request.url}",
            extra={
                "context": {
                    "error_code": exc.code,
                    "error_message": exc.message,
                    "status_code": exc.status_code,
                    "error_details": exc.details,
                    "method": request.method,
                    "path": str(request.url)
                },
                "request_id": request_id
            }
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                    "request_id": request_id
                }
            }
        )

    # 一般的なHTTPエラーのハンドラ
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))

        logger.error(
            f"HTTPエラー {exc.status_code}: {exc.detail} - {request.method} {request.url}",
            extra={
                "context": {
                    "status_code": exc.status_code,
                    "error_detail": exc.detail,
                    "method": request.method,
                    "path": str(request.url)
                },
                "request_id": request_id
            }
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                    "request_id": request_id
                }
            }
        )

    # バリデーションエラーのハンドラ
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        errors = exc.errors()

        logger.warning(
            f"バリデーションエラー: {request.method} {request.url}",
            extra={
                "context": {
                    "validation_errors": errors,
                    "method": request.method,
                    "path": str(request.url)
                },
                "request_id": request_id
            }
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "入力データの検証に失敗しました",
                    "details": {"errors": errors},
                    "request_id": request_id
                }
            }
        )

    # 未処理の例外のハンドラ
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))

        error_context = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "method": request.method,
            "path": str(request.url)
        }

        logger.exception(
            f"未処理の例外: {str(exc)} - {request.method} {request.url}",
            extra={
                "context": error_context,
                "request_id": request_id
            }
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "サーバー内部エラーが発生しました",
                    "request_id": request_id
                }
            }
        )

# ミドルウェアのセットアップ
def setup_middleware(app: FastAPI) -> None:
    """
    FastAPIアプリケーションにミドルウェアを設定します

    Args:
        app: FastAPIアプリケーションインスタンス
    """
    # CORSミドルウェア
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 本番環境では適切に制限すること
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # タイミングミドルウェア
    app.add_middleware(TimingMiddleware)

    # Prometheusメトリクスミドルウェア（利用可能な場合）
    if PROMETHEUS_AVAILABLE and ENABLE_METRICS:
        app.add_middleware(PrometheusMiddleware)
        logger.info("Prometheusメトリクスミドルウェアを有効化しました。/metrics エンドポイントで利用可能です。")

    # API使用状況モニタリングミドルウェア
    app.add_middleware(APIUsageMonitoringMiddleware)

# アプリケーションに依存関係を追加
def setup_app(app: FastAPI) -> None:
    """
    FastAPIアプリケーションに共通のミドルウェアとエラーハンドラをセットアップします

    この関数は、アプリケーション起動時に呼び出されるべきです。

    Args:
        app: FastAPIアプリケーションインスタンス
    """
    setup_middleware(app)
    setup_error_handlers(app)

    logger.info("アプリケーションの設定が完了しました。ミドルウェアとエラーハンドラが正常に設定されました。")