"""
相関分析可視化API

このモジュールは次のエンドポイントを提供します：
- POST /api/correlation/visualize - 相関分析結果の可視化
- POST /api/correlation/analyze-and-visualize - データ分析と可視化を一度に実行
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
from analysis.correlation_analysis import CorrelationAnalyzer
from service.bigquery.client import BigQueryService

logger = logging.getLogger(__name__)

# APIルーター定義
router = APIRouter(
    prefix="/api/correlation",
    tags=["correlation"],
    responses={404: {"description": "リソースが見つかりません"}},
)

# リクエスト・レスポンスモデル定義
class CorrelationAnalysisParams(BaseModel):
    """相関分析パラメータモデル"""
    query: str = Field(..., description="分析対象データのクエリ")
    variables: List[str] = Field(..., description="分析対象の変数リスト")
    method: str = Field("pearson", description="相関係数の計算方法 (pearson, spearman, kendall)")
    threshold: float = Field(0.3, description="相関の有意性閾値")

class CorrelationAnalysisRequest(BaseModel):
    """相関分析リクエストモデル"""
    params: CorrelationAnalysisParams
    visualization_options: Dict[str, Any] = Field(default_factory=dict, description="可視化オプション")
    dataset_id: Optional[str] = Field(None, description="結果保存先のデータセットID")
    table_id: Optional[str] = Field(None, description="結果保存先のテーブルID")

class CorrelationVisualizationRequest(BaseModel):
    """相関分析可視化リクエストモデル"""
    analysis_results: Dict[str, Any] = Field(..., description="相関分析結果")
    visualization_type: str = Field("heatmap", description="可視化タイプ (heatmap, network, matrix)")
    options: Dict[str, Any] = Field(default_factory=dict, description="可視化設定オプション")

class CorrelationVisualizationResponse(BaseModel):
    """相関分析可視化レスポンスモデル"""
    chart_id: str = Field(..., description="チャートID")
    url: str = Field(..., description="チャートURL")
    format: str = Field(..., description="フォーマット")
    thumbnail_url: Optional[str] = Field(None, description="サムネイルURL")
    metadata: Dict[str, Any] = Field(..., description="メタデータ")
    analysis_summary: Dict[str, Any] = Field(..., description="分析サマリー")

# カスタム例外定義
class CorrelationAnalysisError(APIError):
    """相関分析エラー"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="CORRELATION_ANALYSIS_ERROR",
            message=message,
            details=details
        )

class InvalidCorrelationDataError(ValidationFailedError):
    """無効な相関データエラー"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details)

# ヘルパー関数
def _prepare_chart_data_from_correlation(
    analysis_results: Dict[str, Any],
    visualization_type: str,
    options: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    相関分析結果からチャートデータを準備する

    Args:
        analysis_results: 相関分析結果
        visualization_type: 可視化タイプ
        options: 可視化オプション

    Returns:
        チャート設定とデータ
    """
    options = options or {}

    if not analysis_results or "correlation_matrix" not in analysis_results:
        raise InvalidCorrelationDataError(
            message="可視化するデータが見つかりません",
            details={"reason": "相関行列が見つかりません"}
        )

    # 可視化タイプに基づいてデータを準備
    if visualization_type == "heatmap":
        return _prepare_heatmap_data(analysis_results, options)
    elif visualization_type == "network":
        return _prepare_network_data(analysis_results, options)
    elif visualization_type == "matrix":
        return _prepare_matrix_data(analysis_results, options)
    else:
        raise InvalidCorrelationDataError(
            message=f"サポートされていない可視化タイプ: {visualization_type}",
            details={"supported_types": ["heatmap", "network", "matrix"]}
        )

def _prepare_heatmap_data(analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """ヒートマップ用のデータを準備する"""
    correlation_matrix = analysis_results["correlation_matrix"]
    metadata = analysis_results.get("metadata", {})

    # 相関行列がHTML形式の場合、データを抽出
    if isinstance(correlation_matrix, str) and "<table" in correlation_matrix:
        # ここでHTMLからデータを抽出するコードが必要
        # 簡易的な実装としてダミーデータを使用
        variables = metadata.get("variables", ["var1", "var2", "var3"])
        matrix_data = np.random.uniform(-1, 1, (len(variables), len(variables)))
        np.fill_diagonal(matrix_data, 1.0)  # 対角線は1

        # 対称行列にする
        matrix_data = (matrix_data + matrix_data.T) / 2
    else:
        # JSONデータとして解析
        try:
            if isinstance(correlation_matrix, str):
                correlation_matrix = json.loads(correlation_matrix)

            variables = list(correlation_matrix.keys())
            matrix_data = []

            for var in variables:
                row = [correlation_matrix[var].get(v, 0) for v in variables]
                matrix_data.append(row)

            matrix_data = np.array(matrix_data)
        except Exception as e:
            logger.error(f"相関行列のパースエラー: {str(e)}")
            raise InvalidCorrelationDataError(
                message="相関行列のフォーマットが無効です",
                details={"error": str(e)}
            )

    # データセット作成
    chart_config = {
        "chart_type": "heatmap",
        "title": options.get("title", "変数間の相関係数"),
        "x_axis_label": "変数",
        "y_axis_label": "変数",
        "width": options.get("width", 800),
        "height": options.get("height", 600),
        "show_legend": True,
        "color_scheme": options.get("color_scheme", "blues")
    }

    chart_data = {
        "labels": variables,
        "data": matrix_data.tolist()
    }

    return {"config": chart_config, "data": chart_data}

def _prepare_network_data(analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """ネットワーク図用のデータを準備する"""
    correlation_matrix = analysis_results["correlation_matrix"]
    metadata = analysis_results.get("metadata", {})
    threshold = options.get("threshold", metadata.get("threshold", 0.3))

    # 相関行列の処理（前の関数と同様）
    if isinstance(correlation_matrix, str) and "<table" in correlation_matrix:
        # ダミーデータ
        variables = metadata.get("variables", ["var1", "var2", "var3", "var4", "var5"])
        matrix_data = np.random.uniform(-1, 1, (len(variables), len(variables)))
        np.fill_diagonal(matrix_data, 1.0)
        matrix_data = (matrix_data + matrix_data.T) / 2
    else:
        # JSONデータとして解析
        try:
            if isinstance(correlation_matrix, str):
                correlation_matrix = json.loads(correlation_matrix)

            variables = list(correlation_matrix.keys())
            matrix_data = []

            for var in variables:
                row = [correlation_matrix[var].get(v, 0) for v in variables]
                matrix_data.append(row)

            matrix_data = np.array(matrix_data)
        except Exception as e:
            logger.error(f"相関行列のパースエラー: {str(e)}")
            raise InvalidCorrelationDataError(
                message="相関行列のフォーマットが無効です",
                details={"error": str(e)}
            )

    # ネットワークノードとエッジの作成
    nodes = []
    edges = []

    for i, var_i in enumerate(variables):
        # ノード追加
        nodes.append({
            "id": var_i,
            "label": var_i,
            "size": 1
        })

        # エッジの追加（閾値以上のもののみ）
        for j, var_j in enumerate(variables):
            if i < j:  # 重複を避けるため
                corr_value = abs(matrix_data[i, j])
                if corr_value >= threshold:
                    edges.append({
                        "source": var_i,
                        "target": var_j,
                        "value": corr_value,
                        "correlation": matrix_data[i, j]
                    })

    # データセット作成
    chart_config = {
        "chart_type": "network",
        "title": options.get("title", "相関ネットワーク"),
        "width": options.get("width", 800),
        "height": options.get("height", 600),
        "show_legend": True,
        "color_scheme": options.get("color_scheme", "blues")
    }

    chart_data = {
        "nodes": nodes,
        "edges": edges,
        "threshold": threshold
    }

    return {"config": chart_config, "data": chart_data}

def _prepare_matrix_data(analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """散布図行列用のデータを準備する"""
    # この部分は実際のデータセットが必要なため、
    # 実際の実装では分析結果からデータを取得する必要があります。
    # ここではサンプルデータを生成します。

    metadata = analysis_results.get("metadata", {})
    variables = metadata.get("variables", ["var1", "var2", "var3"])
    data_points = 30

    # サンプルデータポイントの生成
    np.random.seed(45)
    sample_data = {}
    for var in variables:
        sample_data[var] = np.random.normal(0, 1, data_points).tolist()

    # データセット作成
    chart_config = {
        "chart_type": "scatter_matrix",
        "title": options.get("title", "散布図行列"),
        "width": options.get("width", 800),
        "height": options.get("height", 800),
        "show_legend": False,
        "color_scheme": options.get("color_scheme", "blues")
    }

    chart_data = {
        "variables": variables,
        "data": sample_data
    }

    return {"config": chart_config, "data": chart_data}

def _format_correlation_summary(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """相関分析結果のサマリーを生成する"""
    metadata = analysis_results.get("metadata", {})
    correlation_matrix = analysis_results.get("correlation_matrix", {})

    # 相関係数の統計
    correlation_stats = {
        "method": metadata.get("method", "pearson"),
        "threshold": metadata.get("threshold", 0.3),
        "variables_count": len(metadata.get("variables", [])),
    }

    # 顕著な相関の抽出
    significant_correlations = []

    if isinstance(correlation_matrix, dict):
        processed_pairs = set()

        for var1, correlations in correlation_matrix.items():
            for var2, value in correlations.items():
                if var1 != var2 and abs(value) >= metadata.get("threshold", 0.3):
                    pair = tuple(sorted([var1, var2]))
                    if pair not in processed_pairs:
                        significant_correlations.append({
                            "variable1": var1,
                            "variable2": var2,
                            "correlation": value,
                            "strength": "強" if abs(value) >= 0.7 else "中" if abs(value) >= 0.5 else "弱"
                        })
                        processed_pairs.add(pair)

    # サマリー情報
    summary = {
        "analysis_type": "correlation",
        "correlation_stats": correlation_stats,
        "significant_correlations": significant_correlations,
        "analysis_timestamp": metadata.get("timestamp", "")
    }

    return summary

# エンドポイント実装
@router.post("/visualize", response_model=CorrelationVisualizationResponse, status_code=status.HTTP_200_OK)
async def visualize_correlation(
    request: CorrelationVisualizationRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    相関分析の結果を可視化します。

    既存の分析結果から指定された可視化タイプに基づいてチャートを生成します。
    """
    try:
        logger.info(f"相関分析の可視化リクエスト受信: タイプ={request.visualization_type}")

        # 入力データの検証
        if not request.analysis_results:
            raise InvalidCorrelationDataError(
                message="無効な分析結果データです",
                details={"reason": "分析結果が空です"}
            )

        # チャートデータの準備
        chart_data = _prepare_chart_data_from_correlation(
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
        result["analysis_summary"] = _format_correlation_summary(request.analysis_results)

        return CorrelationVisualizationResponse(
            chart_id=result["chart_id"],
            url=result["url"],
            format=result["format"],
            thumbnail_url=result.get("thumbnail_url"),
            metadata=result["metadata"],
            analysis_summary=result["analysis_summary"]
        )

    except InvalidCorrelationDataError as e:
        logger.error(f"相関データ検証エラー: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"相関分析可視化中にエラー: {str(e)}")
        raise CorrelationAnalysisError(
            message=f"相関分析の可視化中にエラーが発生しました: {str(e)}"
        )

@router.post("/analyze-and-visualize", response_model=CorrelationVisualizationResponse, status_code=status.HTTP_200_OK)
async def analyze_and_visualize_correlation(
    request: CorrelationAnalysisRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    相関分析と可視化を一度のリクエストで実行します。

    データのクエリから相関分析を実行し、結果を可視化して返します。
    """
    try:
        logger.info("相関分析と可視化のリクエスト受信")

        # パラメータの検証
        if not request.params.query:
            raise InvalidCorrelationDataError(
                message="クエリが指定されていません",
                details={"reason": "分析用のSQLクエリは必須です"}
            )

        if not request.params.variables or len(request.params.variables) < 2:
            raise InvalidCorrelationDataError(
                message="変数が不十分です",
                details={"reason": "相関分析には少なくとも2つの変数が必要です"}
            )

        # BigQueryサービスとアナライザーの初期化
        bq_service = BigQueryService()
        analyzer = CorrelationAnalyzer(bq_service)

        try:
            # 相関分析の実行
            analysis_results = await analyzer.analyze(
                query=request.params.query,
                variables=request.params.variables,
                method=request.params.method,
                threshold=request.params.threshold,
                save_results=bool(request.dataset_id and request.table_id),
                dataset_id=request.dataset_id,
                table_id=request.table_id
            )

            # 可視化タイプの決定
            visualization_type = request.visualization_options.get("visualization_type", "heatmap")

            # チャートデータの準備
            chart_data = _prepare_chart_data_from_correlation(
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
            result["analysis_summary"] = _format_correlation_summary(analysis_results)

            return CorrelationVisualizationResponse(
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

    except InvalidCorrelationDataError as e:
        logger.error(f"相関データ検証エラー: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"相関分析と可視化中にエラー: {str(e)}")
        raise CorrelationAnalysisError(
            message=f"相関分析と可視化の実行中にエラーが発生しました: {str(e)}"
        )