"""
財務分析可視化API

このモジュールは次のエンドポイントを提供します：
- POST /api/financial/visualize - 財務分析結果の可視化
- POST /api/financial/analyze-and-visualize - データ分析と可視化を一度に実行

注：このモジュールは共通可視化システムを使用するようリファクタリングされました。
"""

import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field
import json

from api.auth import get_current_user
from api.models import User
from api.dependencies import get_visualization_service
from api.middleware import APIError, ValidationFailedError
from api.core.config import get_settings, Settings
from analysis.FinancialAnalyzer import FinancialAnalyzer
from service.bigquery.client import BigQueryService

# 共通可視化システムのインポート
from api.routers.visualization import (
    UnifiedVisualizationRequest,
    UnifiedVisualizationResponse,
    visualize_analysis as unified_visualize
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/financial",
    tags=["financial"],
    responses={404: {"description": "リソースが見つかりません"}},
)

class FinancialAnalysisParams(BaseModel):
    query: str = Field(..., description="分析対象データのクエリ")
    target_variable: str = Field(..., description="分析対象の変数名")
    columns: Optional[List[str]] = Field(None, description="使用するカラム名のリスト")
    batch_size: Optional[int] = Field(None, description="バッチ処理のサイズ")

class FinancialAnalysisRequest(BaseModel):
    params: FinancialAnalysisParams
    visualization_options: Dict[str, Any] = Field(default_factory=dict, description="可視化オプション")
    dataset_id: Optional[str] = Field(None, description="結果保存先のデータセットID")
    table_id: Optional[str] = Field(None, description="結果保存先のテーブルID")

class FinancialVisualizationRequest(BaseModel):
    analysis_results: Dict[str, Any] = Field(..., description="財務分析結果")
    visualization_type: str = Field("bar", description="可視化タイプ (bar, line, pie, etc.)")
    options: Dict[str, Any] = Field(default_factory=dict, description="可視化設定オプション")

class FinancialVisualizationResponse(BaseModel):
    chart_id: str = Field(..., description="チャートID")
    url: str = Field(..., description="チャートURL")
    format: str = Field(..., description="フォーマット")
    thumbnail_url: Optional[str] = Field(None, description="サムネイルURL")
    metadata: Dict[str, Any] = Field(..., description="メタデータ")
    analysis_summary: Dict[str, Any] = Field(..., description="分析サマリー")

class FinancialAnalysisError(APIError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="FINANCIAL_ANALYSIS_ERROR",
            message=message,
            details=details
        )

class InvalidFinancialDataError(ValidationFailedError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details)

# ヘルパー関数例（実装はプロジェクト要件に応じて拡張）
def _prepare_chart_data_from_financial(
    analysis_results: Dict[str, Any],
    visualization_type: str,
    options: Dict[str, Any] = None
) -> Dict[str, Any]:
    options = options or {}
    if not analysis_results:
        raise InvalidFinancialDataError(
            message="可視化するデータが見つかりません",
            details={"reason": "財務分析結果が空です"}
        )
    # 可視化タイプごとにデータを準備（例としてbarのみ実装）
    if visualization_type == "bar":
        return {"config": {"chart_type": "bar"}, "data": analysis_results}
    else:
        raise InvalidFinancialDataError(
            message=f"サポートされていない可視化タイプ: {visualization_type}",
            details={"supported_types": ["bar"]}
        )

# エンドポイント例
@router.post("/visualize", response_model=FinancialVisualizationResponse, status_code=status.HTTP_200_OK)
async def visualize_financial_analysis(
    request: FinancialVisualizationRequest,
    raw_request: Request,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    財務分析結果の可視化

    リクエストを統一可視化エンドポイントに変換してリダイレクトします。
    """
    try:
        logger.info(f"財務分析の可視化リクエスト受信: タイプ={request.visualization_type}")

        # 統一可視化リクエストに変換
        unified_request = UnifiedVisualizationRequest(
            analysis_type="financial",
            analysis_results=request.analysis_results,
            visualization_type=request.visualization_type,
            options=request.options
        )

        # 統一可視化エンドポイントにリダイレクト
        result = await unified_visualize(
            request=unified_request,
            current_user=current_user,
            visualization_service=visualization_service,
            settings=settings
        )

        # 結果を変換して返す
        return FinancialVisualizationResponse(
            chart_id=result.chart_id,
            url=result.url,
            format=result.format,
            thumbnail_url=result.thumbnail_url,
            metadata=result.metadata,
            analysis_summary=result.analysis_summary
        )
    except Exception as e:
        logger.error(f"財務分析可視化エラー: {str(e)}")
        raise FinancialAnalysisError(message=str(e))