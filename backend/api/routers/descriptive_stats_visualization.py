"""
記述統計量可視化API

このモジュールは次のエンドポイントを提供します：
- POST /api/stats/visualize - 記述統計量分析結果の可視化
- POST /api/stats/analyze-and-visualize - データ分析と可視化を一度に実行
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from api.auth import get_current_user
from api.models import User
from api.dependencies import get_visualization_service
from api.middleware import APIError, ValidationFailedError
from api.core.config import get_settings, Settings
from analysis.calculate_descriptive_stats import DescriptiveStatsCalculator, DescriptiveStatsConfig

logger = logging.getLogger(__name__)

# APIルーター定義
router = APIRouter(
    prefix="/api/stats",
    tags=["descriptive_stats"],
    responses={404: {"description": "リソースが見つかりません"}},
)

# リクエスト・レスポンスモデル定義
class StatsQueryParams(BaseModel):
    """記述統計量クエリパラメータモデル"""
    query: str = Field(..., description="分析対象データのクエリ")
    target_variable: str = Field(..., description="分析対象の変数名")
    arima_order: Tuple[int, int, int] = Field((5, 1, 0), description="ARIMAモデルのオーダー (p,d,q)")
    columns: Optional[List[str]] = Field(None, description="使用するカラム名のリスト")
    batch_size: Optional[int] = Field(None, description="バッチ処理のサイズ")

class StatsAnalysisRequest(BaseModel):
    """記述統計量分析リクエストモデル"""
    params: StatsQueryParams = Field(..., description="分析パラメータ")
    visualization_options: Dict[str, Any] = Field(default_factory=dict, description="可視化オプション")
    dataset_id: Optional[str] = Field(None, description="結果保存先のデータセットID")
    table_id: Optional[str] = Field(None, description="結果保存先のテーブルID")

class StatsVisualizationRequest(BaseModel):
    """記述統計量可視化リクエストモデル"""
    analysis_results: Dict[str, Any] = Field(..., description="記述統計量分析結果")
    visualization_type: str = Field("bar", description="可視化タイプ (bar, line, radar, distribution)")
    options: Dict[str, Any] = Field(default_factory=dict, description="可視化設定オプション")

class JobStatusResponse(BaseModel):
    """ジョブステータスレスポンスモデル"""
    job_id: str = Field(..., description="ジョブID")
    status: str = Field(..., description="ステータス (pending, completed, failed)")
    result: Optional[Dict[str, Any]] = Field(None, description="結果データ")
    error: Optional[str] = Field(None, description="エラーメッセージ")
    created_at: str = Field(..., description="作成日時")
    completed_at: Optional[str] = Field(None, description="完了日時")

class StatsVisualizationResponse(BaseModel):
    """記述統計量可視化レスポンスモデル"""
    chart_id: str = Field(..., description="チャートID")
    url: str = Field(..., description="チャートURL")
    format: str = Field(..., description="フォーマット")
    thumbnail_url: Optional[str] = Field(None, description="サムネイルURL")
    metadata: Dict[str, Any] = Field(..., description="メタデータ")
    analysis_summary: Dict[str, Any] = Field(..., description="分析サマリー")

# カスタム例外定義
class DescriptiveStatsError(APIError):
    """記述統計量分析エラー"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="DESCRIPTIVE_STATS_ERROR",
            message=message,
            details=details
        )

class InvalidStatsDataError(ValidationFailedError):
    """無効な分析データエラー"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details)

# グローバルジョブステータス管理（本番環境では永続ストアを使用すべき）
_job_statuses: Dict[str, Dict[str, Any]] = {}

# ヘルパー関数
def _prepare_chart_data_from_stats(stats_results: Dict[str, Any],
                                visualization_type: str,
                                options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    記述統計量の分析結果からチャートデータを準備する

    Args:
        stats_results: 記述統計量分析結果
        visualization_type: 可視化タイプ
        options: 可視化オプション

    Returns:
        チャート設定とデータ
    """
    options = options or {}

    if not stats_results or "descriptive_stats" not in stats_results:
        raise InvalidStatsDataError(
            message="可視化するデータが見つかりません",
            details={"reason": "分析結果にdescriptive_statsが含まれていません"}
        )

    # 可視化タイプに基づいてデータを準備
    if visualization_type == "bar":
        return _prepare_bar_chart_data(stats_results, options)
    elif visualization_type == "line":
        return _prepare_line_chart_data(stats_results, options)
    elif visualization_type == "radar":
        return _prepare_radar_chart_data(stats_results, options)
    elif visualization_type == "distribution":
        return _prepare_distribution_chart_data(stats_results, options)
    else:
        raise InvalidStatsDataError(
            message=f"サポートされていない可視化タイプ: {visualization_type}",
            details={"supported_types": ["bar", "line", "radar", "distribution"]}
        )

def _prepare_bar_chart_data(stats_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """棒グラフ用のデータを準備する"""
    # 記述統計量データを抽出
    descriptive_stats = stats_results.get("descriptive_stats", {})

    # キーと値のリストに変換
    keys = list(descriptive_stats.keys())
    values = list(descriptive_stats.values())

    # 棒グラフ用のデータ構造
    chart_config = {
        "chart_type": "bar",
        "title": options.get("title", "記述統計量"),
        "x_axis_label": "メトリクス",
        "y_axis_label": "値",
        "width": options.get("width", 800),
        "height": options.get("height", 500),
        "show_legend": True,
        "color_scheme": options.get("color_scheme", "blues")
    }

    chart_data = {
        "labels": keys,
        "datasets": [{
            "label": stats_results.get("metadata", {}).get("target_variable", "統計量"),
            "data": values,
            "color": None
        }]
    }

    return {"config": chart_config, "data": chart_data}

def _prepare_line_chart_data(stats_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """折れ線グラフ用のデータを準備する"""
    # ARIMA関連のメトリクスを抽出
    arima_metrics = stats_results.get("arima_metrics", {})

    # エラーがある場合は除外
    if "error" in arima_metrics:
        arima_metrics.pop("error")

    # キーと値のリストに変換
    keys = list(arima_metrics.keys())
    values = [v if v is not None else 0 for v in arima_metrics.values()]

    # 折れ線グラフ用のデータ構造
    chart_config = {
        "chart_type": "line",
        "title": options.get("title", "ARIMA指標"),
        "x_axis_label": "メトリクス",
        "y_axis_label": "値",
        "width": options.get("width", 800),
        "height": options.get("height", 500),
        "show_legend": True,
        "color_scheme": options.get("color_scheme", "blues")
    }

    chart_data = {
        "labels": keys,
        "datasets": [{
            "label": stats_results.get("metadata", {}).get("target_variable", "ARIMA指標"),
            "data": values,
            "color": options.get("line_color", "#4285F4")
        }]
    }

    return {"config": chart_config, "data": chart_data}

def _prepare_radar_chart_data(stats_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """レーダーチャート用のデータを準備する"""
    # 記述統計量データを抽出
    descriptive_stats = stats_results.get("descriptive_stats", {})

    # レーダーチャートに適した指標のみ選択
    radar_metrics = ["mean", "median", "std", "skewness", "kurtosis"]
    filtered_stats = {k: descriptive_stats.get(k, 0) for k in radar_metrics if k in descriptive_stats}

    # キーと値のリストに変換
    keys = list(filtered_stats.keys())
    values = list(filtered_stats.values())

    # データのスケーリング（0-100の範囲に）
    max_val = max(abs(v) for v in values) if values else 1
    normalized_values = [min(100, abs(v) / max_val * 100) for v in values]

    # レーダーチャート用のデータ構造
    chart_config = {
        "chart_type": "radar",
        "title": options.get("title", "統計量プロファイル"),
        "width": options.get("width", 600),
        "height": options.get("height", 600),
        "show_legend": True,
        "color_scheme": options.get("color_scheme", "blues")
    }

    chart_data = {
        "labels": keys,
        "datasets": [{
            "label": stats_results.get("metadata", {}).get("target_variable", "統計量"),
            "data": normalized_values,
            "color": None
        }]
    }

    return {"config": chart_config, "data": chart_data}

def _prepare_distribution_chart_data(stats_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """分布図用のデータを準備する"""
    # 記述統計量を使用して正規分布を近似
    descriptive_stats = stats_results.get("descriptive_stats", {})
    mean = descriptive_stats.get("mean", 0)
    std = descriptive_stats.get("std", 1)

    # 分布の範囲を生成
    x_min = mean - 3 * std
    x_max = mean + 3 * std

    # 分布用のポイントを生成
    n_points = 50
    x_points = np.linspace(x_min, x_max, n_points)
    y_points = []

    for x in x_points:
        # 正規分布の確率密度関数
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        y_points.append(float(y))

    # 分布図用のデータ構造
    chart_config = {
        "chart_type": "line", # ライングラフとして描画
        "title": options.get("title", "推定分布"),
        "x_axis_label": "値",
        "y_axis_label": "確率密度",
        "width": options.get("width", 800),
        "height": options.get("height", 500),
        "show_legend": True,
        "color_scheme": options.get("color_scheme", "blues")
    }

    chart_data = {
        "labels": [f"{x:.2f}" for x in x_points],
        "datasets": [{
            "label": f"{stats_results.get('metadata', {}).get('target_variable', '変数')}の分布",
            "data": y_points,
            "fill": True,
            "color": options.get("fill_color", "rgba(66, 133, 244, 0.3)")
        }]
    }

    return {"config": chart_config, "data": chart_data}

def _format_analysis_summary(stats_results: Dict[str, Any]) -> Dict[str, Any]:
    """分析結果のサマリーを生成する"""
    descriptive_stats = stats_results.get("descriptive_stats", {})
    arima_metrics = stats_results.get("arima_metrics", {})
    metadata = stats_results.get("metadata", {})

    return {
        "target_variable": metadata.get("target_variable", "不明"),
        "data_size": metadata.get("data_size", 0),
        "arima_order": metadata.get("arima_order", (0, 0, 0)),
        "central_tendency": {
            "mean": descriptive_stats.get("mean", 0),
            "median": descriptive_stats.get("median", 0)
        },
        "dispersion": {
            "std": descriptive_stats.get("std", 0),
            "min": descriptive_stats.get("min", 0),
            "max": descriptive_stats.get("max", 0)
        },
        "distribution_shape": {
            "skewness": descriptive_stats.get("skewness", 0),
            "kurtosis": descriptive_stats.get("kurtosis", 0)
        },
        "model_quality": {
            "aic": arima_metrics.get("aic", None),
            "bic": arima_metrics.get("bic", None),
            "rmse": arima_metrics.get("rmse", None)
        },
        "analysis_timestamp": metadata.get("analysis_timestamp", "")
    }

# バックグラウンドタスク処理関数
async def _run_stats_analysis_task(job_id: str, params: Dict[str, Any], user_id: str):
    """バックグラウンドで統計分析を実行する"""
    from service.bigquery.client import BigQueryService

    try:
        _job_statuses[job_id]["status"] = "processing"

        # BigQueryサービスの初期化
        bq_service = BigQueryService()

        # 分析設定の作成
        config = DescriptiveStatsConfig(
            query=params["query"],
            target_variable=params["target_variable"],
            arima_order=tuple(params.get("arima_order", (5, 1, 0))),
            columns=params.get("columns"),
            batch_size=params.get("batch_size"),
            dataset_id=params.get("dataset_id"),
            table_id=params.get("table_id"),
            save_results=bool(params.get("dataset_id") and params.get("table_id"))
        )

        # 分析実行
        calculator = DescriptiveStatsCalculator(bq_service)
        results = await calculator.calculate(config)
        calculator.release_resources()

        # 結果を更新
        _job_statuses[job_id]["status"] = "completed"
        _job_statuses[job_id]["completed_at"] = pd.Timestamp.now().isoformat()
        _job_statuses[job_id]["result"] = results

    except Exception as e:
        logger.exception(f"記述統計量分析中にエラー: {str(e)}")
        _job_statuses[job_id]["status"] = "failed"
        _job_statuses[job_id]["completed_at"] = pd.Timestamp.now().isoformat()
        _job_statuses[job_id]["error"] = str(e)

# エンドポイント実装
@router.post("/visualize", response_model=StatsVisualizationResponse, status_code=status.HTTP_200_OK)
async def visualize_stats_analysis(
    request: StatsVisualizationRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    記述統計量分析の結果を可視化します。

    既存の分析結果から指定された可視化タイプに基づいてチャートを生成します。
    """
    try:
        logger.info(f"記述統計量分析の可視化リクエスト受信: タイプ={request.visualization_type}")

        # 入力データの検証
        if not request.analysis_results or "descriptive_stats" not in request.analysis_results:
            raise InvalidStatsDataError(
                message="無効な分析結果データです",
                details={"reason": "分析結果が空か、必要なデータが含まれていません"}
            )

        # チャートデータの準備
        chart_data = _prepare_chart_data_from_stats(
            stats_results=request.analysis_results,
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

        return StatsVisualizationResponse(
            chart_id=result["chart_id"],
            url=result["url"],
            format=result["format"],
            thumbnail_url=result.get("thumbnail_url"),
            metadata=result["metadata"],
            analysis_summary=result["analysis_summary"]
        )

    except InvalidStatsDataError as e:
        logger.error(f"記述統計量データ検証エラー: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"記述統計量の可視化中にエラー: {str(e)}")
        raise DescriptiveStatsError(
            message=f"記述統計量の可視化中にエラーが発生しました: {str(e)}"
        )

@router.post("/start-analysis", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_stats_analysis(
    request: StatsAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings)
):
    """
    記述統計量の分析をバックグラウンドで開始します。

    分析処理を非同期で実行し、ジョブIDを返します。結果は後で取得できます。
    """
    try:
        import uuid
        from datetime import datetime

        logger.info("記述統計量の分析開始リクエスト受信")

        # 入力データの検証
        if not request.params.query or not request.params.target_variable:
            raise InvalidStatsDataError(
                message="無効な分析パラメータです",
                details={"reason": "クエリと対象変数は必須です"}
            )

        # ジョブID生成
        job_id = str(uuid.uuid4())

        # ジョブステータス初期化
        _job_statuses[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "result": None,
            "error": None
        }

        # 分析パラメータの抽出
        analysis_params = {
            "query": request.params.query,
            "target_variable": request.params.target_variable,
            "arima_order": request.params.arima_order,
            "columns": request.params.columns,
            "batch_size": request.params.batch_size,
            "dataset_id": request.dataset_id,
            "table_id": request.table_id
        }

        # バックグラウンドタスク追加
        background_tasks.add_task(
            _run_stats_analysis_task,
            job_id=job_id,
            params=analysis_params,
            user_id=str(current_user.id)
        )

        return JobStatusResponse(
            job_id=job_id,
            status="pending",
            created_at=_job_statuses[job_id]["created_at"]
        )

    except InvalidStatsDataError as e:
        logger.error(f"記述統計量データ検証エラー: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"記述統計量分析の開始中にエラー: {str(e)}")
        raise DescriptiveStatsError(
            message=f"記述統計量分析の開始中にエラーが発生しました: {str(e)}"
        )

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_stats_analysis_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    記述統計量分析ジョブのステータスを確認します。

    Args:
        job_id: ジョブID

    Returns:
        ジョブのステータス情報
    """
    if job_id not in _job_statuses:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ジョブID '{job_id}' が見つかりません"
        )

    status_data = _job_statuses[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=status_data["status"],
        result=status_data.get("result"),
        error=status_data.get("error"),
        created_at=status_data["created_at"],
        completed_at=status_data.get("completed_at")
    )

@router.post("/analyze-and-visualize", response_model=StatsVisualizationResponse, status_code=status.HTTP_200_OK)
async def analyze_and_visualize_stats(
    request: StatsAnalysisRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    記述統計量の分析と可視化を一度のリクエストで実行します。

    分析を即時実行し、結果を可視化して返します。
    """
    try:
        from service.bigquery.client import BigQueryService

        logger.info("記述統計量の分析と可視化リクエスト受信")

        # 入力データの検証
        if not request.params.query or not request.params.target_variable:
            raise InvalidStatsDataError(
                message="無効な分析パラメータです",
                details={"reason": "クエリと対象変数は必須です"}
            )

        # BigQueryサービスの初期化
        bq_service = BigQueryService()

        # 分析設定の作成
        config = DescriptiveStatsConfig(
            query=request.params.query,
            target_variable=request.params.target_variable,
            arima_order=request.params.arima_order,
            columns=request.params.columns,
            batch_size=request.params.batch_size,
            dataset_id=request.dataset_id,
            table_id=request.table_id,
            save_results=bool(request.dataset_id and request.table_id)
        )

        # 分析実行
        calculator = DescriptiveStatsCalculator(bq_service)
        analysis_results = await calculator.calculate(config)
        calculator.release_resources()

        # 可視化タイプの決定
        visualization_type = request.visualization_options.get("visualization_type", "bar")

        # チャートデータの準備
        chart_data = _prepare_chart_data_from_stats(
            stats_results=analysis_results,
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

        return StatsVisualizationResponse(
            chart_id=result["chart_id"],
            url=result["url"],
            format=result["format"],
            thumbnail_url=result.get("thumbnail_url"),
            metadata=result["metadata"],
            analysis_summary=result["analysis_summary"]
        )

    except InvalidStatsDataError as e:
        logger.error(f"記述統計量データ検証エラー: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"記述統計量の分析と可視化中にエラー: {str(e)}")
        raise DescriptiveStatsError(
            message=f"記述統計量の分析と可視化中にエラーが発生しました: {str(e)}"
        )