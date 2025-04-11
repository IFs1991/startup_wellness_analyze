# -*- coding: utf-8 -*-
"""
ロギングユーティリティ
------------------
アプリケーション全体で使用される統一ロギング機能とメトリクス収集を提供します。
構造化ログ、パフォーマンス測定、およびPrometheusメトリクスをサポートします。
"""

import logging
import time
import functools
import json
import uuid
import inspect
import os
from typing import Any, Dict, Optional, Callable, List, Union, TypeVar, cast
from contextlib import contextmanager
from datetime import datetime

# Prometheusメトリクスライブラリ（オプション）
try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client がインストールされていません。メトリクス機能は無効になります。")

# グローバル設定
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = os.environ.get("LOG_FORMAT", "json")  # "json" または "text"
ENABLE_METRICS = os.environ.get("ENABLE_METRICS", "true").lower() == "true"
SERVICE_NAME = os.environ.get("SERVICE_NAME", "backend-api")

# 型変数の定義
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')
Loggable = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

class StructuredLogRecord(logging.LogRecord):
    """拡張された構造化ログレコード"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.correlation_id = getattr(self, "correlation_id", None)
        self.context = getattr(self, "context", {})
        self.request_id = getattr(self, "request_id", None)
        self.user_id = getattr(self, "user_id", None)
        self.service = SERVICE_NAME
        self.duration_ms = getattr(self, "duration_ms", None)

class JsonFormatter(logging.Formatter):
    """JSON形式のログフォーマッター"""

    def format(self, record: logging.LogRecord) -> str:
        """ログレコードをJSON形式にフォーマット"""
        record = cast(StructuredLogRecord, record)
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": super().format(record),
            "service": getattr(record, "service", SERVICE_NAME),
            "function": record.funcName,
            "module": record.module,
            "line": record.lineno,
        }

        # 追加コンテキスト情報
        if hasattr(record, "correlation_id") and record.correlation_id:
            log_data["correlation_id"] = record.correlation_id
        if hasattr(record, "request_id") and record.request_id:
            log_data["request_id"] = record.request_id
        if hasattr(record, "user_id") and record.user_id:
            log_data["user_id"] = record.user_id
        if hasattr(record, "duration_ms") and record.duration_ms is not None:
            log_data["duration_ms"] = record.duration_ms

        # 追加コンテキストを含める
        if hasattr(record, "context") and record.context:
            for key, value in record.context.items():
                if key not in log_data:
                    try:
                        # 複雑なオブジェクトのシリアライズを試みる
                        json.dumps({key: value})
                        log_data[key] = value
                    except (TypeError, ValueError):
                        # シリアライズできない場合は文字列化
                        log_data[key] = str(value)

        # 例外情報を含める
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }

        try:
            return json.dumps(log_data)
        except (TypeError, ValueError):
            # JSON化できない場合はフォールバック
            error_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "ERROR",
                "message": "ログのJSON化に失敗しました",
                "original_message": str(record.msg)
            }
            return json.dumps(error_log)

# Prometheusメトリクス（有効な場合）
metrics = {}
if PROMETHEUS_AVAILABLE and ENABLE_METRICS:
    # HTTPリクエストメトリクス
    metrics["http_requests_total"] = Counter(
        "http_requests_total",
        "HTTPリクエストの総数",
        ["method", "endpoint", "status_code"]
    )
    metrics["http_request_duration_seconds"] = Histogram(
        "http_request_duration_seconds",
        "HTTPリクエスト処理時間（秒）",
        ["method", "endpoint"]
    )
    metrics["http_requests_in_progress"] = Gauge(
        "http_requests_in_progress",
        "処理中のHTTPリクエスト数",
        ["method", "endpoint"]
    )

    # 業務メトリクス
    metrics["business_operation_duration_seconds"] = Histogram(
        "business_operation_duration_seconds",
        "業務処理操作の実行時間（秒）",
        ["operation", "status"]
    )
    metrics["business_operation_total"] = Counter(
        "business_operation_total",
        "業務処理操作の実行回数",
        ["operation", "status"]
    )

    # データベースメトリクス
    metrics["db_operation_duration_seconds"] = Histogram(
        "db_operation_duration_seconds",
        "データベース操作の実行時間（秒）",
        ["operation", "status"]
    )
    metrics["db_operation_total"] = Counter(
        "db_operation_total",
        "データベース操作の実行回数",
        ["operation", "status"]
    )

def get_logger(name: str) -> logging.Logger:
    """
    指定された名前のロガーインスタンスを取得します。

    このロガーは構造化ロギングに対応し、JSON形式でログを出力します。
    また、追加コンテキスト情報をログに含めることができます。

    Args:
        name: ロガー名（通常は__name__を使用）

    Returns:
        設定済みのロガーインスタンス
    """
    logger = logging.getLogger(name)

    # ロガーが未設定の場合のみ設定
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL))

        # ハンドラーの設定
        handler = logging.StreamHandler()

        # フォーマッターの設定
        if LOG_FORMAT.lower() == "json":
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def with_context(logger: logging.Logger, **context) -> logging.Logger:
    """
    コンテキスト情報を含む新しいロガーを作成

    Args:
        logger: 基本となるロガー
        **context: 追加のコンテキスト情報

    Returns:
        コンテキスト付きロガー
    """
    # ロガーの子ロガーを作成
    child_logger = logger.getChild(str(uuid.uuid4())[:8])

    # 元のロギングメソッドを保存
    original_log = child_logger._log

    # 新しいロギングメソッドで上書き
    def _log_with_context(level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        if extra is None:
            extra = {}
        if 'context' not in extra:
            extra['context'] = {}

        # 提供されたコンテキストを追加
        for key, value in context.items():
            extra['context'][key] = value

        return original_log(level, msg, args, exc_info, extra, stack_info, stacklevel)

    # ロギングメソッドを上書き
    child_logger._log = _log_with_context

    return child_logger

def log_function_call(logger: Optional[logging.Logger] = None) -> Callable[[F], F]:
    """
    関数呼び出しをログに記録するデコレータ

    Args:
        logger: 使用するロガー（Noneの場合は関数名でロガーを作成）

    Returns:
        デコレータ関数
    """
    def decorator(func: F) -> F:
        # ロガーを設定
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 呼び出し情報を準備
            func_name = func.__name__
            module_name = func.__module__

            # 引数情報を安全に準備
            try:
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                if len(signature) > 200:
                    signature = signature[:200] + "..."
            except Exception:
                signature = "引数の文字列化中にエラーが発生しました"

            # 呼び出し開始をログ
            logger.debug(
                f"関数 {func_name} 呼び出し開始: args={signature}",
                extra={"context": {"function": func_name, "module": module_name}}
            )

            start_time = time.time()

            try:
                # 関数を実行
                result = func(*args, **kwargs)

                # 実行時間を計測
                duration_ms = (time.time() - start_time) * 1000

                # メトリクスを記録（利用可能な場合）
                if PROMETHEUS_AVAILABLE and ENABLE_METRICS and "business_operation_duration_seconds" in metrics:
                    metrics["business_operation_duration_seconds"].labels(
                        operation=f"{module_name}.{func_name}",
                        status="success"
                    ).observe(duration_ms / 1000)
                    metrics["business_operation_total"].labels(
                        operation=f"{module_name}.{func_name}",
                        status="success"
                    ).inc()

                # 完了をログ
                logger.debug(
                    f"関数 {func_name} 実行完了: 処理時間={duration_ms:.2f}ms",
                    extra={
                        "context": {
                            "function": func_name,
                            "module": module_name
                        },
                        "duration_ms": duration_ms
                    }
                )

                return result

            except Exception as e:
                # 実行時間を計測
                duration_ms = (time.time() - start_time) * 1000

                # メトリクスを記録（利用可能な場合）
                if PROMETHEUS_AVAILABLE and ENABLE_METRICS and "business_operation_duration_seconds" in metrics:
                    metrics["business_operation_duration_seconds"].labels(
                        operation=f"{module_name}.{func_name}",
                        status="error"
                    ).observe(duration_ms / 1000)
                    metrics["business_operation_total"].labels(
                        operation=f"{module_name}.{func_name}",
                        status="error"
                    ).inc()

                # エラーをログ
                logger.exception(
                    f"関数 {func_name} 実行中にエラー発生: {str(e)}, 処理時間={duration_ms:.2f}ms",
                    extra={
                        "context": {
                            "function": func_name,
                            "module": module_name,
                            "error": str(e)
                        },
                        "duration_ms": duration_ms
                    }
                )
                raise

        return cast(F, wrapper)
    return decorator

@contextmanager
def log_operation(
    logger: logging.Logger,
    operation_name: str,
    correlation_id: Optional[str] = None,
    extra_context: Optional[Dict[str, Any]] = None
):
    """
    操作の実行時間とステータスをログに記録するコンテキストマネージャ

    Args:
        logger: ロガーインスタンス
        operation_name: 操作の名前
        correlation_id: 相関ID（Noneの場合は自動生成）
        extra_context: 追加のコンテキスト情報

    Yields:
        なし（コンテキスト内で操作を実行）
    """
    # 相関IDが提供されていない場合は生成
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    # コンテキスト情報を準備
    context = {
        "operation": operation_name,
        "correlation_id": correlation_id
    }

    if extra_context:
        context.update(extra_context)

    start_time = time.time()

    # 操作開始をログ
    logger.info(
        f"操作 '{operation_name}' 開始",
        extra={"context": context, "correlation_id": correlation_id}
    )

    # メトリクスカウンター更新（操作中）
    if (
        PROMETHEUS_AVAILABLE and ENABLE_METRICS
        and "business_operation_total" in metrics
    ):
        metrics["business_operation_total"].labels(
            operation=operation_name,
            status="started"
        ).inc()

    try:
        # コンテキスト内の処理を実行
        yield

        # 処理時間を計算
        duration_ms = (time.time() - start_time) * 1000

        # メトリクスを記録（成功）
        if (
            PROMETHEUS_AVAILABLE and ENABLE_METRICS
            and "business_operation_duration_seconds" in metrics
        ):
            metrics["business_operation_duration_seconds"].labels(
                operation=operation_name,
                status="success"
            ).observe(duration_ms / 1000)
            metrics["business_operation_total"].labels(
                operation=operation_name,
                status="success"
            ).inc()

        # 操作完了をログ
        logger.info(
            f"操作 '{operation_name}' 完了: 処理時間={duration_ms:.2f}ms",
            extra={
                "context": context,
                "correlation_id": correlation_id,
                "duration_ms": duration_ms
            }
        )

    except Exception as e:
        # 処理時間を計算
        duration_ms = (time.time() - start_time) * 1000

        # メトリクスを記録（失敗）
        if (
            PROMETHEUS_AVAILABLE and ENABLE_METRICS
            and "business_operation_duration_seconds" in metrics
        ):
            metrics["business_operation_duration_seconds"].labels(
                operation=operation_name,
                status="error"
            ).observe(duration_ms / 1000)
            metrics["business_operation_total"].labels(
                operation=operation_name,
                status="error"
            ).inc()

        # エラーをログ
        error_context = context.copy()
        error_context.update({
            "error": str(e),
            "error_type": type(e).__name__
        })

        logger.exception(
            f"操作 '{operation_name}' 失敗: {str(e)}, 処理時間={duration_ms:.2f}ms",
            extra={
                "context": error_context,
                "correlation_id": correlation_id,
                "duration_ms": duration_ms
            }
        )
        raise

def trace_db_operation(operation_name: str) -> Callable[[F], F]:
    """
    データベース操作をトレースするデコレータ

    Args:
        operation_name: 操作の名前

    Returns:
        デコレータ関数
    """
    def decorator(func: F) -> F:
        logger = get_logger(func.__module__)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            correlation_id = str(uuid.uuid4())
            start_time = time.time()

            # トレース情報を準備
            logger.debug(
                f"DB操作 '{operation_name}' 開始",
                extra={
                    "context": {"db_operation": operation_name},
                    "correlation_id": correlation_id
                }
            )

            try:
                # 操作を実行
                result = await func(*args, **kwargs)

                # 処理時間を計算
                duration_ms = (time.time() - start_time) * 1000

                # メトリクスを記録
                if PROMETHEUS_AVAILABLE and ENABLE_METRICS and "db_operation_duration_seconds" in metrics:
                    metrics["db_operation_duration_seconds"].labels(
                        operation=operation_name,
                        status="success"
                    ).observe(duration_ms / 1000)
                    metrics["db_operation_total"].labels(
                        operation=operation_name,
                        status="success"
                    ).inc()

                # 完了をログ
                logger.debug(
                    f"DB操作 '{operation_name}' 完了: 処理時間={duration_ms:.2f}ms",
                    extra={
                        "context": {"db_operation": operation_name},
                        "correlation_id": correlation_id,
                        "duration_ms": duration_ms
                    }
                )

                return result

            except Exception as e:
                # 処理時間を計算
                duration_ms = (time.time() - start_time) * 1000

                # メトリクスを記録
                if PROMETHEUS_AVAILABLE and ENABLE_METRICS and "db_operation_duration_seconds" in metrics:
                    metrics["db_operation_duration_seconds"].labels(
                        operation=operation_name,
                        status="error"
                    ).observe(duration_ms / 1000)
                    metrics["db_operation_total"].labels(
                        operation=operation_name,
                        status="error"
                    ).inc()

                # エラーをログ
                logger.exception(
                    f"DB操作 '{operation_name}' 失敗: {str(e)}, 処理時間={duration_ms:.2f}ms",
                    extra={
                        "context": {
                            "db_operation": operation_name,
                            "error": str(e)
                        },
                        "correlation_id": correlation_id,
                        "duration_ms": duration_ms
                    }
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            correlation_id = str(uuid.uuid4())
            start_time = time.time()

            # トレース情報を準備
            logger.debug(
                f"DB操作 '{operation_name}' 開始",
                extra={
                    "context": {"db_operation": operation_name},
                    "correlation_id": correlation_id
                }
            )

            try:
                # 操作を実行
                result = func(*args, **kwargs)

                # 処理時間を計算
                duration_ms = (time.time() - start_time) * 1000

                # メトリクスを記録
                if PROMETHEUS_AVAILABLE and ENABLE_METRICS and "db_operation_duration_seconds" in metrics:
                    metrics["db_operation_duration_seconds"].labels(
                        operation=operation_name,
                        status="success"
                    ).observe(duration_ms / 1000)
                    metrics["db_operation_total"].labels(
                        operation=operation_name,
                        status="success"
                    ).inc()

                # 完了をログ
                logger.debug(
                    f"DB操作 '{operation_name}' 完了: 処理時間={duration_ms:.2f}ms",
                    extra={
                        "context": {"db_operation": operation_name},
                        "correlation_id": correlation_id,
                        "duration_ms": duration_ms
                    }
                )

                return result

            except Exception as e:
                # 処理時間を計算
                duration_ms = (time.time() - start_time) * 1000

                # メトリクスを記録
                if PROMETHEUS_AVAILABLE and ENABLE_METRICS and "db_operation_duration_seconds" in metrics:
                    metrics["db_operation_duration_seconds"].labels(
                        operation=operation_name,
                        status="error"
                    ).observe(duration_ms / 1000)
                    metrics["db_operation_total"].labels(
                        operation=operation_name,
                        status="error"
                    ).inc()

                # エラーをログ
                logger.exception(
                    f"DB操作 '{operation_name}' 失敗: {str(e)}, 処理時間={duration_ms:.2f}ms",
                    extra={
                        "context": {
                            "db_operation": operation_name,
                            "error": str(e)
                        },
                        "correlation_id": correlation_id,
                        "duration_ms": duration_ms
                    }
                )
                raise

        # 非同期関数かどうかで適切なラッパーを選択
        if inspect.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)

    return decorator