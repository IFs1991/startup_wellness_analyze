"""
健康投資効果指数（HIEI）可視化API

このモジュールは次のエンドポイントを提供します：
- POST /api/hiei/visualize - 健康投資効果指数分析結果の可視化
- POST /api/hiei/analyze-and-visualize - データ分析と可視化を一度に実行

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
from analysis.HealthInvestmentEffectIndexCalculator import HealthInvestmentEffectIndexCalculator
from service.bigquery.client import BigQueryService

# 共通可視化システムのインポート
from api.routers.visualization import (
    UnifiedVisualizationRequest,
    UnifiedVisualizationResponse,
    visualize_analysis as unified_visualize
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/hiei",
    tags=["hiei"],
    responses={404: {"description": "リソースが見つかりません"}},
)

class HIEIAnalysisParams(BaseModel):
    query: str = Field(..., description="分析対象データのクエリ")
    target_variable: str = Field(..., description="分析対象の変数名")
    columns: Optional[List[str]] = Field(None, description="使用するカラム名のリスト")
    batch_size: Optional[int] = Field(None, description="バッチ処理のサイズ")

class HIEIAnalysisRequest(BaseModel):
    params: HIEIAnalysisParams
    visualization_options: Dict[str, Any] = Field(default_factory=dict, description="可視化オプション")
    dataset_id: Optional[str] = Field(None, description="結果保存先のデータセットID")
    table_id: Optional[str] = Field(None, description="結果保存先のテーブルID")

class HIEIVisualizationRequest(BaseModel):
    analysis_results: Dict[str, Any] = Field(..., description="健康投資効果指数分析結果")
    visualization_type: str = Field("bar", description="可視化タイプ (bar, line, pie, etc.)")
    options: Dict[str, Any] = Field(default_factory=dict, description="可視化設定オプション")

class HIEIVisualizationResponse(BaseModel):
    chart_id: str = Field(..., description="チャートID")
    url: str = Field(..., description="チャートURL")
    format: str = Field(..., description="フォーマット")
    thumbnail_url: Optional[str] = Field(None, description="サムネイルURL")
    metadata: Dict[str, Any] = Field(..., description="メタデータ")
    analysis_summary: Dict[str, Any] = Field(..., description="分析サマリー")

class HIEIAnalysisError(APIError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="HIEI_ANALYSIS_ERROR",
            message=message,
            details=details
        )

class InvalidHIEIDataError(ValidationFailedError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details)

@router.post("/visualize", response_model=HIEIVisualizationResponse, status_code=status.HTTP_200_OK)
async def visualize_hiei_analysis(
    request: HIEIVisualizationRequest,
    raw_request: Request,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    健康投資効果指数分析結果の可視化

    リクエストを統一可視化エンドポイントに変換してリダイレクトします。
    """
    try:
        logger.info(f"健康投資効果指数分析の可視化リクエスト受信: タイプ={request.visualization_type}")

        # 統一可視化リクエストに変換
        unified_request = UnifiedVisualizationRequest(
            analysis_type="health_investment",
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
        return HIEIVisualizationResponse(
            chart_id=result.chart_id,
            url=result.url,
            format=result.format,
            thumbnail_url=result.thumbnail_url,
            metadata=result.metadata,
            analysis_summary=result.analysis_summary
        )
    except Exception as e:
        logger.error(f"健康投資効果指数可視化エラー: {str(e)}")
        raise HIEIAnalysisError(message=str(e))