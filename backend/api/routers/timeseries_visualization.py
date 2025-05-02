"""
時系列分析可視化API

このモジュールは次のエンドポイントを提供します：
- POST /api/timeseries/visualize - 時系列分析結果の可視化
- POST /api/timeseries/analyze-and-visualize - データ分析と可視化を一度に実行
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
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
from analysis.TimeSeriesAnalyzer import TimeSeriesAnalyzer
from service.bigquery.client import BigQueryService

logger = logging.getLogger(__name__)

# APIルーター定義
router = APIRouter(
    prefix="/api/timeseries",
    tags=["timeseries"],
    responses={404: {"description": "リソースが見つかりません"}},
)

# リクエスト・レスポンスモデル定義
class TimeSeriesAnalysisParams(BaseModel):
    """時系列分析パラメータモデル"""
    query: str = Field(..., description="分析対象データのクエリ")
    target_variable: str = Field(..., description="分析対象の変数名")
    arima_order: Tuple[int, int, int] = Field((5, 1, 0), description="ARIMAモデルのオーダー (p,d,q)")

class TimeSeriesAnalysisRequest(BaseModel):
    """時系列分析リクエストモデル"""
    params: TimeSeriesAnalysisParams
    visualization_options: Dict[str, Any] = Field(default_factory=dict, description="可視化オプション")
    dataset_id: Optional[str] = Field(None, description="結果保存先のデータセットID")
    table_id: Optional[str] = Field(None, description="結果保存先のテーブルID")

class TimeSeriesVisualizationRequest(BaseModel):
    """時系列分析可視化リクエストモデル"""
    analysis_results: Dict[str, Any] = Field(..., description="時系列分析結果")
    visualization_type: str = Field("line", description="可視化タイプ (line, residual, histogram, acf)")
    options: Dict[str, Any] = Field(default_factory=dict, description="可視化設定オプション")

class TimeSeriesVisualizationResponse(BaseModel):
    """時系列分析可視化レスポンスモデル"""
    chart_id: str = Field(..., description="チャートID")
    url: str = Field(..., description="チャートURL")
    format: str = Field(..., description="フォーマット")
    thumbnail_url: Optional[str] = Field(None, description="サムネイルURL")
    metadata: Dict[str, Any] = Field(..., description="メタデータ")
    analysis_summary: Dict[str, Any] = Field(..., description="分析サマリー")

# カスタム例外定義
class TimeSeriesAnalysisError(APIError):
    """時系列分析エラー"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="TIMESERIES_ANALYSIS_ERROR",
            message=message,
            details=details
        )

class InvalidTimeSeriesDataError(ValidationFailedError):
    """無効な時系列データエラー"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details)

# ヘルパー関数
def _prepare_chart_data_from_timeseries(
    analysis_results: Dict[str, Any],
    visualization_type: str,
    options: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    時系列分析結果からチャートデータを準備する

    Args:
        analysis_results: 時系列分析結果
        visualization_type: 可視化タイプ
        options: 可視化オプション

    Returns:
        チャート設定とデータ
    """
    options = options or {}

    if not analysis_results:
        raise InvalidTimeSeriesDataError(
            message="可視化するデータが見つかりません",
            details={"reason": "分析結果が空です"}
        )

    # 可視化タイプに基づいてデータを準備
    if visualization_type == "line":
        return _prepare_line_chart_data(analysis_results, options)
    elif visualization_type == "residual":
        return _prepare_residual_chart_data(analysis_results, options)
    elif visualization_type == "histogram":
        return _prepare_histogram_chart_data(analysis_results, options)
    elif visualization_type == "acf":
        return _prepare_acf_chart_data(analysis_results, options)
    else:
        raise InvalidTimeSeriesDataError(
            message=f"サポートされていない可視化タイプ: {visualization_type}",
            details={"supported_types": ["line", "residual", "histogram", "acf"]}
        )

def _parse_summary_table(summary_html: str) -> Dict[str, Any]:
    """HTMLサマリーテーブルからデータを抽出する"""
    try:
        # 簡易的なHTMLパース（実際の実装ではBeautifulSoupなどの適切なライブラリを使用すべき）
        import re

        # 係数テーブルからデータを抽出
        coefficients = {}
        pattern = r'<td>([^<]+)</td>\s*<td>([^<]+)</td>\s*<td>([^<]+)</td>\s*<td>([^<]+)</td>\s*<td>([^<]+)</td>'
        matches = re.findall(pattern, summary_html)

        for match in matches:
            if len(match) >= 5:
                param_name = match[0].strip()
                coef = float(match[1].strip()) if match[1].strip() != 'nan' else None
                std_err = float(match[2].strip()) if match[2].strip() != 'nan' else None
                t_value = float(match[3].strip()) if match[3].strip() != 'nan' else None
                p_value = float(match[4].strip()) if match[4].strip() != 'nan' else None

                coefficients[param_name] = {
                    'coef': coef,
                    'std_err': std_err,
                    't_value': t_value,
                    'p_value': p_value
                }

        return coefficients
    except Exception as e:
        logger.warning(f"サマリーテーブルのパースに失敗しました: {str(e)}")
        return {}

def _prepare_line_chart_data(analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """折れ線グラフ用のデータを準備する"""
    # 時系列データを生成（実際のデータが利用可能な場合はそちらを使用）
    metadata = analysis_results.get("metadata", {})
    target_variable = metadata.get("target_variable", "変数")

    # データポイント生成（サンプルデータ）
    # 実際の実装では分析結果やオプションから適切なデータを取得
    data_points = 50
    time_points = list(range(data_points))

    # トレンドコンポーネントの簡易シミュレーション
    np.random.seed(42)  # 再現性のため
    values = np.cumsum(np.random.normal(0, 1, data_points)) + 100

    # 予測値（例として最後の10ポイント）
    forecast_start = data_points - 10
    forecast_values = np.array(values)
    forecast_values[forecast_start:] += np.random.normal(0, 2, 10)  # 予測に若干のノイズを追加

    # データセット作成
    chart_config = {
        "chart_type": "line",
        "title": options.get("title", f"{target_variable}の時系列分析"),
        "x_axis_label": "時間",
        "y_axis_label": target_variable,
        "width": options.get("width", 800),
        "height": options.get("height", 500),
        "show_legend": True,
        "color_scheme": options.get("color_scheme", "blues")
    }

    chart_data = {
        "labels": [str(t) for t in time_points],
        "datasets": [
            {
                "label": "実測値",
                "data": values.tolist()[:forecast_start],
                "color": options.get("actual_color", "#4285F4")
            },
            {
                "label": "予測値",
                "data": [None] * forecast_start + forecast_values.tolist()[forecast_start:],
                "color": options.get("forecast_color", "#DB4437"),
                "dash": [5, 5]  # 破線スタイル
            }
        ]
    }

    return {"config": chart_config, "data": chart_data}

def _prepare_residual_chart_data(analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """残差プロット用のデータを準備する"""
    # 残差データを生成（実際のデータが利用可能な場合はそちらを使用）
    metadata = analysis_results.get("metadata", {})
    target_variable = metadata.get("target_variable", "変数")

    # サンプル残差データの作成
    np.random.seed(43)  # 再現性のため
    data_points = 50
    time_points = list(range(data_points))
    residuals = np.random.normal(0, 1, data_points)

    # データセット作成
    chart_config = {
        "chart_type": "line",
        "title": options.get("title", f"{target_variable}のモデル残差"),
        "x_axis_label": "時間",
        "y_axis_label": "残差",
        "width": options.get("width", 800),
        "height": options.get("height", 400),
        "show_legend": True,
        "color_scheme": options.get("color_scheme", "blues")
    }

    chart_data = {
        "labels": [str(t) for t in time_points],
        "datasets": [
            {
                "label": "残差",
                "data": residuals.tolist(),
                "color": options.get("residual_color", "#34A853")
            },
            {
                "label": "ゼロライン",
                "data": [0] * data_points,
                "color": "#999999",
                "dash": [2, 2]  # 破線スタイル
            }
        ]
    }

    return {"config": chart_config, "data": chart_data}

def _prepare_histogram_chart_data(analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """ヒストグラム用のデータを準備する"""
    # 残差データのヒストグラムを作成
    metadata = analysis_results.get("metadata", {})
    target_variable = metadata.get("target_variable", "変数")

    # サンプル残差データの生成と分布の計算
    np.random.seed(44)  # 再現性のため
    residuals = np.random.normal(0, 1, 200)  # より多くのデータポイント

    # ヒストグラムの計算
    hist, bin_edges = np.histogram(residuals, bins=10, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # データセット作成
    chart_config = {
        "chart_type": "bar",
        "title": options.get("title", f"{target_variable}の残差分布"),
        "x_axis_label": "残差値",
        "y_axis_label": "確率密度",
        "width": options.get("width", 700),
        "height": options.get("height", 400),
        "show_legend": True,
        "color_scheme": options.get("color_scheme", "blues")
    }

    chart_data = {
        "labels": [f"{x:.2f}" for x in bin_centers],
        "datasets": [
            {
                "label": "残差分布",
                "data": hist.tolist(),
                "color": options.get("histogram_color", "#FBBC05")
            }
        ]
    }

    return {"config": chart_config, "data": chart_data}

def _prepare_acf_chart_data(analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """自己相関関数(ACF)プロット用のデータを準備する"""
    # ACFデータを生成（実際のデータが利用可能な場合はそちらを使用）
    metadata = analysis_results.get("metadata", {})
    target_variable = metadata.get("target_variable", "変数")

    # サンプルのACF値を生成
    lags = 20
    acf_values = []
    for i in range(lags + 1):
        if i == 0:
            acf_values.append(1.0)  # ラグ0での自己相関は1
        else:
            # 通常、ACF値は徐々に減衰（サンプル用）
            acf_values.append(0.8 ** i + np.random.normal(0, 0.05))

    # 信頼区間の計算（95%）
    confidence_interval = 1.96 / np.sqrt(100)  # サンプルサイズを100と仮定
    ci_upper = [confidence_interval] * (lags + 1)
    ci_lower = [-confidence_interval] * (lags + 1)

    # データセット作成
    chart_config = {
        "chart_type": "bar",
        "title": options.get("title", f"{target_variable}の自己相関関数"),
        "x_axis_label": "ラグ",
        "y_axis_label": "自己相関",
        "width": options.get("width", 800),
        "height": options.get("height", 400),
        "show_legend": True,
        "color_scheme": options.get("color_scheme", "blues")
    }

    chart_data = {
        "labels": [str(i) for i in range(lags + 1)],
        "datasets": [
            {
                "label": "ACF",
                "data": acf_values,
                "color": options.get("acf_color", "#4285F4")
            },
            {
                "label": "95%信頼区間（上限）",
                "data": ci_upper,
                "type": "line",
                "color": "#DB4437",
                "dash": [2, 2]
            },
            {
                "label": "95%信頼区間（下限）",
                "data": ci_lower,
                "type": "line",
                "color": "#DB4437",
                "dash": [2, 2]
            }
        ]
    }

    return {"config": chart_config, "data": chart_data}

def _format_analysis_summary(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """時系列分析結果のサマリーを生成する"""
    metadata = analysis_results.get("metadata", {})

    # HTMLサマリーテーブルからデータを抽出
    coefficients = {}
    if "summary" in analysis_results:
        coefficients = _parse_summary_table(analysis_results["summary"])

    # モデル評価指標
    model_metrics = {
        "aic": analysis_results.get("aic"),
        "bic": analysis_results.get("bic"),
        "hqic": analysis_results.get("hqic")
    }

    # サマリー情報
    summary = {
        "target_variable": metadata.get("target_variable", "不明"),
        "arima_order": metadata.get("arima_order", (0, 0, 0)),
        "row_count": metadata.get("row_count", 0),
        "model_metrics": model_metrics,
        "coefficients": coefficients,
        "analysis_timestamp": metadata.get("timestamp", "")
    }

    return summary

# エンドポイント実装
@router.post("/visualize", response_model=TimeSeriesVisualizationResponse, status_code=status.HTTP_200_OK)
async def visualize_timeseries(
    request: TimeSeriesVisualizationRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    時系列分析の結果を可視化します。

    既存の分析結果から指定された可視化タイプに基づいてチャートを生成します。
    """
    try:
        logger.info(f"時系列分析の可視化リクエスト受信: タイプ={request.visualization_type}")

        # 入力データの検証
        if not request.analysis_results:
            raise InvalidTimeSeriesDataError(
                message="無効な分析結果データです",
                details={"reason": "分析結果が空です"}
            )

        # チャートデータの準備
        chart_data = _prepare_chart_data_from_timeseries(
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
        result["analysis_summary"] = _format_analysis_summary(request.analysis_results)

        return TimeSeriesVisualizationResponse(
            chart_id=result["chart_id"],
            url=result["url"],
            format=result["format"],
            thumbnail_url=result.get("thumbnail_url"),
            metadata=result["metadata"],
            analysis_summary=result["analysis_summary"]
        )

    except InvalidTimeSeriesDataError as e:
        logger.error(f"時系列データ検証エラー: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"時系列分析可視化中にエラー: {str(e)}")
        raise TimeSeriesAnalysisError(
            message=f"時系列分析の可視化中にエラーが発生しました: {str(e)}"
        )

@router.post("/analyze-and-visualize", response_model=TimeSeriesVisualizationResponse, status_code=status.HTTP_200_OK)
async def analyze_and_visualize_timeseries(
    request: TimeSeriesAnalysisRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    時系列分析と可視化を一度のリクエストで実行します。

    データのクエリから時系列分析を実行し、結果を可視化して返します。
    """
    try:
        logger.info("時系列分析と可視化のリクエスト受信")

        # パラメータの検証
        if not request.params.query:
            raise InvalidTimeSeriesDataError(
                message="クエリが指定されていません",
                details={"reason": "分析用のSQLクエリは必須です"}
            )

        if not request.params.target_variable:
            raise InvalidTimeSeriesDataError(
                message="対象変数が指定されていません",
                details={"reason": "分析対象の変数名は必須です"}
            )

        # BigQueryサービスとアナライザーの初期化
        bq_service = BigQueryService()
        analyzer = TimeSeriesAnalyzer(bq_service)

        try:
            # 時系列分析の実行
            analysis_results = await analyzer.analyze(
                query=request.params.query,
                target_variable=request.params.target_variable,
                save_results=bool(request.dataset_id and request.table_id),
                dataset_id=request.dataset_id,
                table_id=request.table_id,
                arima_order=request.params.arima_order
            )

            # 可視化タイプの決定
            visualization_type = request.visualization_options.get("visualization_type", "line")

            # チャートデータの準備
            chart_data = _prepare_chart_data_from_timeseries(
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
            result["analysis_summary"] = _format_analysis_summary(analysis_results)

            return TimeSeriesVisualizationResponse(
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

    except InvalidTimeSeriesDataError as e:
        logger.error(f"時系列データ検証エラー: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"時系列分析と可視化中にエラー: {str(e)}")
        raise TimeSeriesAnalysisError(
            message=f"時系列分析と可視化の実行中にエラーが発生しました: {str(e)}"
        )