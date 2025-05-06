"""
可視化機能の統一エラー処理メカニズム

このモジュールでは、可視化処理中に発生する可能性のあるエラーを
統一的に処理するための機能を提供します。
"""

from fastapi import HTTPException, status
from typing import Dict, Any, Optional, Type
import logging

logger = logging.getLogger(__name__)


class VisualizationError(Exception):
    """可視化処理中に発生するエラーの基底クラス"""
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "visualization_error"
    error_message: str = "可視化処理中にエラーが発生しました"

    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message or self.error_message
        self.details = details or {}
        super().__init__(self.message)


class InvalidVisualizationTypeError(VisualizationError):
    """無効な可視化タイプが指定された場合のエラー"""
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "invalid_visualization_type"
    error_message = "指定された可視化タイプは無効です"


class InvalidAnalysisResultError(VisualizationError):
    """無効な分析結果が提供された場合のエラー"""
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "invalid_analysis_result"
    error_message = "指定された分析結果は無効または不完全です"


class VisualizationServiceError(VisualizationError):
    """可視化サービス側でエラーが発生した場合のエラー"""
    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "visualization_service_error"
    error_message = "可視化サービスでエラーが発生しました"


class ChartGenerationError(VisualizationError):
    """チャート生成時にエラーが発生した場合のエラー"""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "chart_generation_error"
    error_message = "チャート生成中にエラーが発生しました"


# エラーマッピング
ERROR_MAPPING = {
    ValueError: InvalidAnalysisResultError,
    TypeError: InvalidAnalysisResultError,
    KeyError: InvalidAnalysisResultError,
    # 他のエラータイプを追加可能
}


def handle_visualization_error(exception: Exception) -> HTTPException:
    """
    例外を適切な HTTPException に変換する

    Args:
        exception: 発生した例外

    Returns:
        適切な HTTPException
    """
    if isinstance(exception, VisualizationError):
        # すでに VisualizationError の場合はそのまま変換
        viz_error = exception
    else:
        # 適切な VisualizationError タイプに変換
        error_class = ERROR_MAPPING.get(type(exception), VisualizationError)
        viz_error = error_class(str(exception))

    # エラーログ出力
    logger.error(
        f"可視化エラー: {viz_error.error_code} - {viz_error.message}",
        extra={"details": viz_error.details, "exception": str(exception)}
    )

    # HTTPException に変換
    return HTTPException(
        status_code=viz_error.status_code,
        detail={
            "error_code": viz_error.error_code,
            "message": viz_error.message,
            "details": viz_error.details
        }
    )


def create_error_response(error_code: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    エラーレスポンスを生成する

    Args:
        error_code: エラーコード
        message: エラーメッセージ
        details: 追加の詳細情報

    Returns:
        エラーレスポンス辞書
    """
    return {
        "error_code": error_code,
        "message": message,
        "details": details or {}
    }