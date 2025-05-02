"""
統計分析可視化API

このモジュールは次のエンドポイントを提供します：
- POST /api/statistical-analysis/visualize - 統計分析結果の可視化
- POST /api/statistical-analysis/analyze-and-visualize - データ分析と可視化を一度に実行
"""

import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import json

from api.auth import get_current_user
from api.models import User
from api.dependencies import get_visualization_service
from api.middleware import APIError, ValidationFailedError
from api.core.config import get_settings, Settings
from analysis.statistical_analyzer import StatisticalAnalyzer
from service.bigquery.client import BigQueryService
from .visualization_helpers import (
    prepare_boxplot_data,
    prepare_histogram_data,
    prepare_scatter_data,
    prepare_bar_data
)

logger = logging.getLogger(__name__)

# APIルーター定義
router = APIRouter(
    prefix="/api/statistical-analysis",
    tags=["statistical-analysis"],
    responses={404: {"description": "リソースが見つかりません"}},
)

# リクエスト・レスポンスモデル定義
class StatisticalAnalysisParams(BaseModel):
    """統計分析パラメータモデル"""
    query: str = Field(..., description="分析対象データのクエリ")
    variables: List[str] = Field(..., description="分析対象の変数リスト")
    groupby_column: Optional[str] = Field(None, description="グループ化に使用する列名")
    test_type: str = Field("descriptive", description="分析タイプ (descriptive, ttest, anova, chi2)")
    alpha: float = Field(0.05, description="有意水準")

class StatisticalAnalysisRequest(BaseModel):
    """統計分析リクエストモデル"""
    params: StatisticalAnalysisParams
    visualization_options: Dict[str, Any] = Field(default_factory=dict, description="可視化オプション")
    dataset_id: Optional[str] = Field(None, description="結果保存先のデータセットID")
    table_id: Optional[str] = Field(None, description="結果保存先のテーブルID")

class StatisticalVisualizationRequest(BaseModel):
    """統計分析可視化リクエストモデル"""
    analysis_results: Dict[str, Any] = Field(..., description="統計分析結果")
    visualization_type: str = Field("boxplot", description="可視化タイプ (boxplot, histogram, bar, scatter)")
    options: Dict[str, Any] = Field(default_factory=dict, description="可視化設定オプション")

class StatisticalVisualizationResponse(BaseModel):
    """統計分析可視化レスポンスモデル"""
    chart_id: str = Field(..., description="チャートID")
    url: str = Field(..., description="チャートURL")
    format: str = Field(..., description="フォーマット")
    thumbnail_url: Optional[str] = Field(None, description="サムネイルURL")
    metadata: Dict[str, Any] = Field(..., description="メタデータ")
    analysis_summary: Dict[str, Any] = Field(..., description="分析サマリー")

# カスタム例外定義
class StatisticalAnalysisError(APIError):
    """統計分析エラー"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="STATISTICAL_ANALYSIS_ERROR",
            message=message,
            details=details
        )

class InvalidStatisticalDataError(ValidationFailedError):
    """無効な統計データエラー"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details)

# ヘルパー関数
def _prepare_chart_data_from_statistical_analysis(
    analysis_results: Dict[str, Any],
    visualization_type: str,
    options: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    統計分析結果からチャートデータを準備する

    Args:
        analysis_results: 統計分析結果
        visualization_type: 可視化タイプ
        options: 可視化オプション

    Returns:
        チャート設定とデータ
    """
    options = options or {}

    if not analysis_results:
        raise InvalidStatisticalDataError(
            message="可視化するデータが見つかりません",
            details={"reason": "統計分析結果が空です"}
        )

    # 可視化タイプに基づいてデータを準備
    if visualization_type == "boxplot":
        return prepare_boxplot_data(analysis_results, options)
    elif visualization_type == "histogram":
        return prepare_histogram_data(analysis_results, options)
    elif visualization_type == "scatter":
        return prepare_scatter_data(analysis_results, options)
    elif visualization_type == "bar":
        return prepare_bar_data(analysis_results, options)
    else:
        raise InvalidStatisticalDataError(
            message=f"サポートされていない可視化タイプ: {visualization_type}",
            details={"supported_types": ["boxplot", "histogram", "bar", "scatter"]}
        )

def _format_statistical_summary(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """統計分析結果のサマリーを生成する"""
    # メタデータの取得
    metadata = analysis_results.get("metadata", {})
    test_type = metadata.get("test_type", "descriptive")

    # 記述統計の抽出
    descriptive_stats = analysis_results.get("descriptive_stats", {})
    if isinstance(descriptive_stats, str):
        try:
            descriptive_stats = json.loads(descriptive_stats)
        except:
            descriptive_stats = {}

    # テスト結果の抽出
    test_results = analysis_results.get("test_results", {})
    if isinstance(test_results, str):
        try:
            test_results = json.loads(test_results)
        except:
            test_results = {}

    # サマリー情報の作成
    summary = {
        "analysis_type": "statistical",
        "test_type": test_type,
        "variables": metadata.get("variables", []),
        "data_points": metadata.get("data_points", 0),
        "groups": metadata.get("groups", []),
        "descriptive_stats": descriptive_stats,
    }

    # テストタイプに基づいた追加情報
    if test_type == "ttest":
        summary["test_details"] = {
            "t_statistic": test_results.get("t_statistic"),
            "p_value": test_results.get("p_value"),
            "degrees_of_freedom": test_results.get("df"),
            "alternative": test_results.get("alternative", "two-sided"),
            "significant": test_results.get("p_value", 1.0) < metadata.get("alpha", 0.05)
        }
    elif test_type == "anova":
        summary["test_details"] = {
            "f_statistic": test_results.get("f_value"),
            "p_value": test_results.get("p_value"),
            "df_between": test_results.get("df_between"),
            "df_within": test_results.get("df_within"),
            "significant": test_results.get("p_value", 1.0) < metadata.get("alpha", 0.05)
        }
    elif test_type == "chi2":
        summary["test_details"] = {
            "chi2_statistic": test_results.get("chi2"),
            "p_value": test_results.get("p_value"),
            "degrees_of_freedom": test_results.get("df"),
            "significant": test_results.get("p_value", 1.0) < metadata.get("alpha", 0.05)
        }

    return summary

# エンドポイント実装
@router.post("/visualize", response_model=StatisticalVisualizationResponse, status_code=status.HTTP_200_OK)
async def visualize_statistical_analysis(
    request: StatisticalVisualizationRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    統計分析の結果を可視化します。

    既存の分析結果から指定された可視化タイプに基づいてチャートを生成します。
    """
    try:
        logger.info(f"統計分析の可視化リクエスト受信: タイプ={request.visualization_type}")

        # 入力データの検証
        if not request.analysis_results:
            raise InvalidStatisticalDataError(
                message="無効な分析結果データです",
                details={"reason": "分析結果が空です"}
            )

        # チャートデータの準備
        chart_data = _prepare_chart_data_from_statistical_analysis(
            analysis_results=request.analysis_results,
            visualization_type=request.visualization_type,
            options=request.options
        )

        # チャート生成
        result = await visualization_service.generate_chart(
            config=chart_data["config"],
            data=chart_data["data"],
            format=request.options.get("format", "png"),
            template_id=request.options.get("template_id"),
            user_id=str(current_user.id)
        )

        # 分析サマリーを追加
        result["analysis_summary"] = _format_statistical_summary(request.analysis_results)

        return StatisticalVisualizationResponse(
            chart_id=result["chart_id"],
            url=result["url"],
            format=result["format"],
            thumbnail_url=result.get("thumbnail_url"),
            metadata=result["metadata"],
            analysis_summary=result["analysis_summary"]
        )

    except InvalidStatisticalDataError as e:
        logger.error(f"統計データ検証エラー: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"統計分析可視化中にエラー: {str(e)}")
        raise StatisticalAnalysisError(
            message=f"統計分析の可視化中にエラーが発生しました: {str(e)}"
        )

@router.post("/analyze-and-visualize", response_model=StatisticalVisualizationResponse, status_code=status.HTTP_200_OK)
async def analyze_and_visualize_statistical(
    request: StatisticalAnalysisRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    統計分析と可視化を一度のリクエストで実行します。

    データのクエリから統計分析を実行し、結果を可視化して返します。
    """
    try:
        logger.info("統計分析と可視化のリクエスト受信")

        # パラメータの検証
        if not request.params.query:
            raise InvalidStatisticalDataError(
                message="クエリが指定されていません",
                details={"reason": "分析用のSQLクエリは必須です"}
            )

        if not request.params.variables:
            raise InvalidStatisticalDataError(
                message="分析変数が指定されていません",
                details={"reason": "統計分析には少なくとも1つの変数が必要です"}
            )

        # BigQueryサービスとアナライザーの初期化
        bq_service = BigQueryService()
        analyzer = StatisticalAnalyzer(bq_service)

        try:
            # 統計分析の実行
            analysis_results = await analyzer.analyze(
                query=request.params.query,
                variables=request.params.variables,
                groupby_column=request.params.groupby_column,
                test_type=request.params.test_type,
                alpha=request.params.alpha,
                save_results=bool(request.dataset_id and request.table_id),
                dataset_id=request.dataset_id,
                table_id=request.table_id
            )

            # 可視化タイプの決定
            visualization_type = request.visualization_options.get("visualization_type", "boxplot")
            if request.params.test_type == "chi2":
                visualization_type = "bar"  # カイ二乗検定の場合はバーチャートがデフォルト

            # チャートデータの準備
            chart_data = _prepare_chart_data_from_statistical_analysis(
                analysis_results=analysis_results,
                visualization_type=visualization_type,
                options=request.visualization_options
            )

            # チャート生成
            result = await visualization_service.generate_chart(
                config=chart_data["config"],
                data=chart_data["data"],
                format=request.visualization_options.get("format", "png"),
                template_id=request.visualization_options.get("template_id"),
                user_id=str(current_user.id)
            )

            # 分析サマリーを追加
            result["analysis_summary"] = _format_statistical_summary(analysis_results)

            return StatisticalVisualizationResponse(
                chart_id=result["chart_id"],
                url=result["url"],
                format=result["format"],
                thumbnail_url=result.get("thumbnail_url"),
                metadata=result["metadata"],
                analysis_summary=result["analysis_summary"]
            )

        finally:
            # リソース解放
            analyzer.release_resources()

    except InvalidStatisticalDataError as e:
        logger.error(f"統計データ検証エラー: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"統計分析と可視化中にエラー: {str(e)}")
        raise StatisticalAnalysisError(
            message=f"統計分析と可視化の実行中にエラーが発生しました: {str(e)}"
        )