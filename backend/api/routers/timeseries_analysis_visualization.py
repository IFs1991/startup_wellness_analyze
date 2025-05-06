"""
時系列分析可視化API - リダイレクト

このモジュールは、時系列分析可視化のエンドポイントを
timeseries_visualization.pyに統合するためのリダイレクトを提供します。
可視化の共通化と機能の統合のため、このモジュールは非推奨となります。
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, Optional

from api.auth import get_current_user
from api.models import User
from api.routers.timeseries_visualization import (
    TimeSeriesVisualizationRequest,
    TimeSeriesAnalysisRequest,
    TimeSeriesVisualizationResponse,
    visualize_timeseries as ts_visualize,
    analyze_and_visualize_timeseries as ts_analyze_and_visualize
)

logger = logging.getLogger(__name__)

# APIルーター定義
router = APIRouter(
    prefix="/api/timeseries-analysis",
    tags=["timeseries-analysis"],
    deprecated=True,
    responses={404: {"description": "リソースが見つかりません"}},
)

# リダイレクト関数
@router.post("/visualize", response_model=TimeSeriesVisualizationResponse, status_code=status.HTTP_200_OK)
async def visualize_timeseries(
    request: TimeSeriesVisualizationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    時系列分析結果の可視化（リダイレクト）

    このエンドポイントは非推奨です。代わりに `/api/timeseries/visualize` を使用してください。
    """
    logger.warning("非推奨のエンドポイント '/api/timeseries-analysis/visualize' が使用されました。'/api/timeseries/visualize' を使用してください。")

    # 新しいエンドポイントにリダイレクト
    return await ts_visualize(request, current_user)

@router.post("/analyze-and-visualize", response_model=TimeSeriesVisualizationResponse, status_code=status.HTTP_200_OK)
async def analyze_and_visualize_timeseries(
    request: TimeSeriesAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """
    時系列データの分析と可視化（リダイレクト）

    このエンドポイントは非推奨です。代わりに `/api/timeseries/analyze-and-visualize` を使用してください。
    """
    logger.warning("非推奨のエンドポイント '/api/timeseries-analysis/analyze-and-visualize' が使用されました。'/api/timeseries/analyze-and-visualize' を使用してください。")

    # 新しいエンドポイントにリダイレクト
    return await ts_analyze_and_visualize(request, current_user)