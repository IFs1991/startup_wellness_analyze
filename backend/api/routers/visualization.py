"""
可視化API

このモジュールは次のエンドポイントを提供します：
- POST /api/visualizations/chart - 単一チャートの生成
- POST /api/visualizations/charts - 複数チャートの生成
- POST /api/visualizations/dashboard - ダッシュボードの生成
- POST /api/visualizations/chart/background - バックグラウンドでのチャート生成
- GET /api/visualizations/status/{job_id} - チャート生成ジョブのステータス確認
- POST /api/visualizations/visualize - 統一された可視化プロセッサを使用したチャート生成（新機能）
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from api.auth import get_current_user
from api.models import User
from api.utils.cache import cache_response
from api.dependencies import get_visualization_service
from api.middleware import (
    APIError,
    ResourceNotFoundError,
    ValidationFailedError
)
from api.core.config import get_settings, Settings
from api.visualization.factory import VisualizationProcessorFactory
from api.visualization.errors import handle_visualization_error
# キャッシュユーティリティをインポート
from api.utils.caching import async_cache, _generate_cache_key, get_from_cache, add_to_cache

logger = logging.getLogger(__name__)

# APIルーター定義
router = APIRouter(
    prefix="/api/visualizations",
    tags=["visualizations"],
    responses={404: {"description": "リソースが見つかりません"}},
)

# リクエスト・レスポンスモデル定義
class ChartConfig(BaseModel):
    """チャート設定モデル"""
    chart_type: str = Field(..., description="チャートの種類 (bar, line, pie, scatter)")
    title: Optional[str] = Field(None, description="チャートのタイトル")
    x_axis_label: Optional[str] = Field(None, description="X軸のラベル")
    y_axis_label: Optional[str] = Field(None, description="Y軸のラベル")
    color_scheme: Optional[str] = Field(None, description="カラースキーム")
    show_legend: Optional[bool] = Field(True, description="凡例を表示するかどうか")
    width: Optional[int] = Field(800, description="チャートの幅")
    height: Optional[int] = Field(500, description="チャートの高さ")

class ChartDataset(BaseModel):
    """チャートデータセットモデル"""
    label: str = Field(..., description="データセットのラベル")
    data: List[float] = Field(..., description="データ値のリスト")
    color: Optional[str] = Field(None, description="データセットの色")

class ChartData(BaseModel):
    """チャートデータモデル"""
    labels: List[str] = Field(..., description="データラベルのリスト")
    datasets: List[ChartDataset] = Field(..., description="データセットのリスト")

class ChartRequest(BaseModel):
    """チャート生成リクエストモデル"""
    config: ChartConfig = Field(..., description="チャート設定")
    data: ChartData = Field(..., description="チャートデータ")
    format: Optional[str] = Field("png", description="出力フォーマット (png, svg, pdf)")
    template_id: Optional[str] = Field(None, description="テンプレートID")

class MultipleChartRequest(BaseModel):
    """複数チャート生成リクエストモデル"""
    charts: List[ChartRequest] = Field(..., description="生成するチャートのリスト")

class DashboardSection(BaseModel):
    """ダッシュボードセクションモデル"""
    title: str = Field(..., description="セクションのタイトル")
    charts: List[int] = Field(..., description="セクションに含めるチャートのインデックス")

class DashboardRequest(BaseModel):
    """ダッシュボード生成リクエストモデル"""
    title: str = Field(..., description="ダッシュボードのタイトル")
    description: Optional[str] = Field(None, description="ダッシュボードの説明")
    sections: List[DashboardSection] = Field(..., description="ダッシュボードのセクション")
    chart_ids: List[str] = Field(..., description="使用するチャートIDのリスト")
    theme: Optional[str] = Field("light", description="テーマ (light, dark, blue)")
    format: Optional[str] = Field("pdf", description="出力フォーマット (pdf, html)")

class ChartResponse(BaseModel):
    """チャート生成レスポンスモデル"""
    chart_id: str = Field(..., description="チャートID")
    url: str = Field(..., description="チャートURL")
    format: str = Field(..., description="フォーマット")
    thumbnail_url: Optional[str] = Field(None, description="サムネイルURL")
    metadata: Dict[str, Any] = Field(..., description="メタデータ")

class DashboardResponse(BaseModel):
    """ダッシュボード生成レスポンスモデル"""
    dashboard_id: str = Field(..., description="ダッシュボードID")
    url: str = Field(..., description="ダッシュボードURL")
    format: str = Field(..., description="フォーマット")
    chart_ids: List[str] = Field(..., description="使用したチャートID")
    metadata: Dict[str, Any] = Field(..., description="メタデータ")

class JobStatusResponse(BaseModel):
    """ジョブステータスレスポンス"""
    job_id: str = Field(..., description="ジョブID")
    status: str = Field(..., description="ステータス (pending, completed, failed)")
    result: Optional[Dict[str, Any]] = Field(None, description="結果データ")
    error: Optional[str] = Field(None, description="エラーメッセージ")
    created_at: str = Field(..., description="作成日時")
    completed_at: Optional[str] = Field(None, description="完了日時")

# グローバルジョブステータス管理（実際の実装では永続ストアを使用すべき）
_job_statuses: Dict[str, Dict[str, Any]] = {}

# カスタム例外定義
class ChartGenerationError(APIError):
    """チャート生成エラー"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="CHART_GENERATION_ERROR",
            message=message,
            details=details
        )

class ChartNotFoundError(ResourceNotFoundError):
    """チャートが見つからないエラー"""
    def __init__(self, chart_id: str):
        super().__init__(resource_type="Chart", resource_id=chart_id)

class InvalidChartDataError(ValidationFailedError):
    """無効なチャートデータエラー"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details)

# ヘルパー関数
async def _generate_chart_task(
    job_id: str,
    request: ChartRequest,
    user: User,
    service: Any
):
    """バックグラウンドでチャートを生成するタスク"""
    try:
        _job_statuses[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "result": None,
            "error": None
        }

        # 処理時間シミュレーション（実際の実装では削除）
        await asyncio.sleep(2)
        _job_statuses[job_id]["status"] = "completed"
        _job_statuses[job_id]["completed_at"] = datetime.now().isoformat()

        # チャート生成
        chart_result = await service.generate_chart(
            config=request.config.dict(),
            data=request.data.dict(),
            format=request.format,
            template_id=request.template_id,
            user_id=str(user.id)
        )

        # 成功
        _job_statuses[job_id]["result"] = chart_result
    except Exception as e:
        logger.exception(f"チャート生成エラー: {str(e)}")
        _job_statuses[job_id]["status"] = "failed"
        _job_statuses[job_id]["completed_at"] = datetime.now().isoformat()
        _job_statuses[job_id]["error"] = str(e)

# エンドポイント実装
@router.post("/chart", response_model=ChartResponse, status_code=status.HTTP_201_CREATED)
@cache_response(expire=3600)
async def generate_chart(
    request: ChartRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    チャートを生成します。

    指定されたデータと設定に基づいてチャートを作成し、画像を返します。
    """
    try:
        logger.info("チャート生成リクエスト受信")

        # データバリデーション
        if not request.data.labels or not request.data.datasets:
            raise InvalidChartDataError(
                message="チャートデータが不完全です",
                details={"reason": "labels または datasets が空です"}
            )

        # チャート生成サービスの呼び出し
        result = await visualization_service.generate_chart(
            config=request.config.dict(),
            data=request.data.dict(),
            format=request.format,
            template_id=request.template_id,
            user_id=str(current_user.id)
        )

        # 成功レスポンスの返却
        return {
            "chart_id": result["chart_id"],
            "url": result["url"],
            "format": result["format"],
            "thumbnail_url": result.get("thumbnail_url"),
            "metadata": result["metadata"]
        }

    except InvalidChartDataError as e:
        logger.error(f"チャートデータ検証エラー: {str(e)}")
        raise
    except ChartGenerationError as e:
        logger.error(f"チャート生成エラー: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"予期せぬエラー: {str(e)}")
        raise ChartGenerationError(message=f"チャート生成中にエラーが発生しました: {str(e)}")

@router.post("/charts", response_model=List[ChartResponse], status_code=status.HTTP_201_CREATED)
async def generate_multiple_charts(
    request: MultipleChartRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    複数のチャートを生成します。

    複数のチャート設定を一度に提出し、すべてのチャートを並行して生成します。
    """
    try:
        logger.info(f"複数チャート生成リクエスト受信: {len(request.charts)}個のチャート")

        # データバリデーション
        if not request.charts:
            raise InvalidChartDataError(
                "チャートデータが無効です",
                {"detail": "生成するチャートが指定されていません"}
            )

        # 並行処理でチャートを生成
        tasks = []
        for chart_request in request.charts:
            tasks.append(
                visualization_service.generate_chart(
                    config=chart_request.config.dict(),
                    data=chart_request.data.dict(),
                    format=chart_request.format,
                    template_id=chart_request.template_id,
                    user_id=str(current_user.id)
                )
            )

        # すべてのタスクを待機
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # エラーチェック
        responses = []
        errors = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"チャート {i+1}: {str(result)}")
            else:
                responses.append(ChartResponse(**result))

        # エラーがあった場合は部分的な結果と一緒にエラーを返す
        if errors:
            return JSONResponse(
                status_code=status.HTTP_207_MULTI_STATUS,
                content={
                    "success": True,
                    "data": [r.dict() for r in responses],
                    "errors": errors,
                    "message": f"{len(responses)}/{len(request.charts)}個のチャート生成に成功、{len(errors)}個が失敗"
                }
            )

        return responses

    except Exception as e:
        # 新しいエラー処理ミドルウェアがこれらの例外を捕捉してフォーマットします
        if isinstance(e, APIError):
            raise
        logger.exception(f"予期せぬエラー: {str(e)}")
        raise ChartGenerationError(f"複数チャート生成中にエラーが発生しました: {str(e)}")

@router.post("/dashboard", response_model=DashboardResponse, status_code=status.HTTP_201_CREATED)
async def generate_dashboard(
    request: DashboardRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    ダッシュボードを生成します。

    既存のチャートを組み合わせてダッシュボードを作成します。
    """
    try:
        logger.info(f"ダッシュボード生成リクエスト受信: {request.title}")

        # データバリデーション
        if not request.chart_ids:
            raise InvalidChartDataError(
                "ダッシュボードデータが無効です",
                {"detail": "ダッシュボードに表示するチャートが指定されていません"}
            )

        # チャートの存在確認
        for chart_id in request.chart_ids:
            chart_exists = await visualization_service.check_chart_exists(chart_id, str(current_user.id))
            if not chart_exists:
                raise ChartNotFoundError(chart_id)

        # ダッシュボード生成
        dashboard_result = await visualization_service.generate_dashboard(
            title=request.title,
            description=request.description,
            sections=[section.dict() for section in request.sections],
            chart_ids=request.chart_ids,
            theme=request.theme,
            format=request.format,
            user_id=str(current_user.id)
        )

        return DashboardResponse(**dashboard_result)

    except Exception as e:
        # 新しいエラー処理ミドルウェアがこれらの例外を捕捉してフォーマットします
        if isinstance(e, APIError):
            raise
        logger.exception(f"予期せぬエラー: {str(e)}")
        raise ChartGenerationError(f"ダッシュボード生成中にエラーが発生しました: {str(e)}")

@router.post("/chart/background", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_chart_background(
    request: ChartRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    バックグラウンドでチャートを生成します。

    負荷の高いチャート生成処理をバックグラウンドで実行し、ジョブIDを返します。
    ジョブのステータスは `/api/visualizations/status/{job_id}` で確認できます。
    """
    try:
        logger.info("バックグラウンドチャート生成リクエスト受信")

        # データバリデーション
        if not request.data.labels or not request.data.datasets:
            raise InvalidChartDataError(
                "チャートデータが無効です",
                {"detail": "labels と datasets は空にできません"}
            )

        # ジョブID生成
        job_id = str(uuid.uuid4())

        # バックグラウンドタスク追加
        _job_statuses[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "result": None,
            "error": None
        }

        background_tasks.add_task(
            _generate_chart_task,
            job_id=job_id,
            request=request,
            user=current_user,
            service=visualization_service
        )

        return JobStatusResponse(
            job_id=job_id,
            status="pending",
            created_at=_job_statuses[job_id]["created_at"]
        )

    except Exception as e:
        # 新しいエラー処理ミドルウェアがこれらの例外を捕捉してフォーマットします
        if isinstance(e, APIError):
            raise
        logger.exception(f"予期せぬエラー: {str(e)}")
        raise ChartGenerationError(f"バックグラウンドチャート処理の開始中にエラーが発生しました: {str(e)}")

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_chart_status(job_id: str):
    """
    チャート生成ジョブのステータスを確認します。

    Args:
        job_id: ジョブID

    Returns:
        ジョブステータス情報
    """
    if job_id not in _job_statuses:
        raise ChartNotFoundError(chart_id=job_id)

    status_data = _job_statuses[job_id]
    return {
        "job_id": job_id,
        "status": status_data["status"],
        "result": status_data.get("result"),
        "error": status_data.get("error"),
        "created_at": status_data["created_at"],
        "completed_at": status_data.get("completed_at")
    }

@router.get("/download/{chart_id}")
async def download_chart(
    chart_id: str,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    生成されたチャートをダウンロードします。

    Args:
        chart_id: チャートID

    Returns:
        チャートファイル
    """
    try:
        # チャートファイルの取得
        file_path = await visualization_service.get_chart_file_path(
            chart_id=chart_id,
            user_id=str(current_user.id)
        )

        if not file_path or not os.path.exists(file_path):
            raise ChartNotFoundError(chart_id=chart_id)

        # ファイル応答
        return FileResponse(
            path=file_path,
            filename=os.path.basename(file_path),
            media_type=f"image/{os.path.splitext(file_path)[1][1:]}"
        )
    except ChartNotFoundError:
        raise
    except Exception as e:
        logger.exception(f"チャートダウンロード中にエラーが発生しました: {str(e)}")
        raise ChartGenerationError(message=f"チャートダウンロード中にエラーが発生しました: {str(e)}")

class AnalyzerVisualizationRequest(BaseModel):
    """分析クラスの可視化リクエストモデル"""
    analyzer_type: str = Field(..., description="分析クラスの種類 (financial, vc_roi, health_investment, knowledge_transfer)")
    analysis_results: Dict[str, Any] = Field(..., description="分析結果データ")
    visualization_type: str = Field("bar", description="可視化タイプ (bar, line, pie, scatter, heatmap, network など)")
    options: Dict[str, Any] = Field(default_factory=dict, description="可視化設定オプション")

class AnalyzerVisualizationResponse(BaseModel):
    chart_id: str = Field(..., description="チャートID")
    url: str = Field(..., description="チャートURL")
    format: str = Field(..., description="フォーマット")
    thumbnail_url: Optional[str] = Field(None, description="サムネイルURL")
    metadata: Dict[str, Any] = Field(..., description="メタデータ")
    analysis_summary: Dict[str, Any] = Field(..., description="分析サマリー")


def _prepare_chart_data_from_analyzer(analyzer_type: str, analysis_results: Dict[str, Any], visualization_type: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    各分析クラスの結果からチャートデータを準備する
    """
    options = options or {}
    if analyzer_type == "financial":
        # 財務分析用のチャートデータ変換（例: 売上推移や利益率など）
        labels = analysis_results.get("labels") or []
        datasets = analysis_results.get("datasets") or []
        chart_config = {
            "chart_type": visualization_type,
            "title": options.get("title", "財務分析チャート"),
            "x_axis_label": options.get("x_axis_label", "期間"),
            "y_axis_label": options.get("y_axis_label", "値"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "show_legend": True,
            "color_scheme": options.get("color_scheme", "default")
        }
        chart_data = {
            "labels": labels,
            "datasets": datasets
        }
        return {"config": chart_config, "data": chart_data}
    elif analyzer_type == "vc_roi":
        # VC ROI計算結果のチャートデータ変換
        labels = analysis_results.get("labels") or []
        datasets = analysis_results.get("datasets") or []
        chart_config = {
            "chart_type": visualization_type,
            "title": options.get("title", "VC ROI分析チャート"),
            "x_axis_label": options.get("x_axis_label", "指標"),
            "y_axis_label": options.get("y_axis_label", "ROI (%)"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "show_legend": True,
            "color_scheme": options.get("color_scheme", "greens")
        }
        chart_data = {
            "labels": labels,
            "datasets": datasets
        }
        return {"config": chart_config, "data": chart_data}
    elif analyzer_type == "health_investment":
        # 健康投資効果指数のチャートデータ変換
        labels = analysis_results.get("labels") or []
        datasets = analysis_results.get("datasets") or []
        chart_config = {
            "chart_type": visualization_type,
            "title": options.get("title", "健康投資効果指数チャート"),
            "x_axis_label": options.get("x_axis_label", "期間"),
            "y_axis_label": options.get("y_axis_label", "指数値"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "show_legend": True,
            "color_scheme": options.get("color_scheme", "blues")
        }
        chart_data = {
            "labels": labels,
            "datasets": datasets
        }
        return {"config": chart_config, "data": chart_data}
    elif analyzer_type == "knowledge_transfer":
        # 知識移転指数のチャートデータ変換
        labels = analysis_results.get("labels") or []
        datasets = analysis_results.get("datasets") or []
        chart_config = {
            "chart_type": visualization_type,
            "title": options.get("title", "知識移転指数チャート"),
            "x_axis_label": options.get("x_axis_label", "期間"),
            "y_axis_label": options.get("y_axis_label", "KTI"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "show_legend": True,
            "color_scheme": options.get("color_scheme", "purples")
        }
        chart_data = {
            "labels": labels,
            "datasets": datasets
        }
        return {"config": chart_config, "data": chart_data}
    else:
        raise InvalidChartDataError(
            message=f"サポートされていないanalyzer_type: {analyzer_type}",
            details={"supported_types": ["financial", "vc_roi", "health_investment", "knowledge_transfer"]}
        )


@router.post("/analyze-and-visualize", response_model=AnalyzerVisualizationResponse, status_code=status.HTTP_200_OK)
async def analyze_and_visualize(
    request: AnalyzerVisualizationRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    分析クラスの結果を可視化します。
    FinancialAnalyzer, VCROICalculator, HealthInvestmentEffectIndexCalculator, KnowledgeTransferIndexCalculatorの分析結果を受け取り、チャートを生成します。
    """
    try:
        logger.info(f"分析可視化リクエスト受信: analyzer_type={request.analyzer_type}, type={request.visualization_type}")
        if not request.analysis_results:
            raise InvalidChartDataError(
                message="無効な分析結果データです",
                details={"reason": "分析結果が空です"}
            )
        chart_data = _prepare_chart_data_from_analyzer(
            analyzer_type=request.analyzer_type,
            analysis_results=request.analysis_results,
            visualization_type=request.visualization_type,
            options=request.options
        )
        result = await visualization_service.generate_chart(
            config=chart_data["config"],
            data=chart_data["data"],
            format=request.options.get("format", "png"),
            template_id=request.options.get("template_id"),
            user_id=str(current_user.id)
        )
        result["analysis_summary"] = {"summary": "分析サマリーは各analyzerで拡張可能"}
        return AnalyzerVisualizationResponse(
            chart_id=result["chart_id"],
            url=result["url"],
            format=result["format"],
            thumbnail_url=result.get("thumbnail_url"),
            metadata=result["metadata"],
            analysis_summary=result["analysis_summary"]
        )
    except InvalidChartDataError as e:
        logger.error(f"分析データ検証エラー: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"分析可視化中にエラー: {str(e)}")
        raise ChartGenerationError(
            message=f"分析可視化の実行中にエラーが発生しました: {str(e)}"
        )

# 新しい統一可視化用のモデル定義
class UnifiedVisualizationRequest(BaseModel):
    """統一された可視化リクエストモデル"""
    analysis_type: str = Field(..., description="分析タイプ (association, correlation, descriptive_stats, predictive_model, survival_analysis, timeseries, cluster, pca など)")
    analysis_results: Dict[str, Any] = Field(..., description="分析結果データ")
    visualization_type: str = Field(..., description="可視化タイプ (分析タイプに依存)")
    options: Dict[str, Any] = Field(default_factory=dict, description="可視化設定オプション")

class UnifiedVisualizationResponse(BaseModel):
    """統一された可視化レスポンスモデル"""
    chart_id: str = Field(..., description="チャートID")
    url: str = Field(..., description="チャートURL")
    format: str = Field(..., description="フォーマット")
    thumbnail_url: Optional[str] = Field(None, description="サムネイルURL")
    metadata: Dict[str, Any] = Field(..., description="メタデータ")
    analysis_summary: Dict[str, Any] = Field(..., description="分析サマリー")

# 統一された可視化エンドポイント（キャッシュ機能を追加）
@router.post("/visualize", response_model=UnifiedVisualizationResponse, status_code=status.HTTP_200_OK)
async def visualize_analysis(
    request: UnifiedVisualizationRequest,
    raw_request: Request = None,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    分析結果を可視化します（統一されたプロセッサを使用）

    分析タイプ、可視化タイプ、オプションに基づいて適切な可視化プロセッサを選択し、
    チャートを生成します。各分析タイプは固有の可視化方法をサポートしています。
    """
    try:
        logger.info(f"可視化リクエスト受信: 分析タイプ={request.analysis_type}, 可視化タイプ={request.visualization_type}")

        # キャッシュキーを生成
        cache_key = _generate_cache_key(
            request.analysis_type,
            request.analysis_results,
            request.visualization_type,
            request.options
        )

        # キャッシュから結果を取得
        cached_result = get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"キャッシュから結果を取得: {cache_key}")
            return UnifiedVisualizationResponse(**cached_result)

        # プロセッサの取得
        processor = VisualizationProcessorFactory.get_processor(request.analysis_type)
        if not processor:
            return HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"未対応の分析タイプです: {request.analysis_type}"
            )

        # チャートデータの準備（メモリ効率を考慮）
        chart_data = await _prepare_chart_data_optimized(
            processor,
            request.analysis_results,
            request.visualization_type,
            request.options
        )

        # 分析サマリーの生成
        analysis_summary = processor.format_summary(request.analysis_results)

        # フォーマット設定
        format = request.options.get("format", "png")
        template_id = request.options.get("template_id")

        # チャート生成サービスの呼び出し
        result = await visualization_service.generate_chart(
            config=chart_data["config"],
            data=chart_data["data"],
            format=format,
            template_id=template_id,
            user_id=str(current_user.id)
        )

        # サマリー情報の追加
        result["analysis_summary"] = analysis_summary

        # レスポンスの生成
        response = UnifiedVisualizationResponse(
            chart_id=result["chart_id"],
            url=result["url"],
            format=result["format"],
            thumbnail_url=result.get("thumbnail_url"),
            metadata=result["metadata"],
            analysis_summary=result["analysis_summary"]
        )

        # 結果をキャッシュに保存（有効期限：1時間）
        add_to_cache(cache_key, response.dict(), 3600)

        return response

    except Exception as e:
        logger.exception(f"可視化処理中にエラーが発生しました: {str(e)}")
        raise handle_visualization_error(e)

# メモリ効率のために最適化されたチャートデータ準備関数
async def _prepare_chart_data_optimized(
    processor,
    analysis_results: Dict[str, Any],
    visualization_type: str,
    options: Dict[str, Any]
) -> Dict[str, Any]:
    """
    チャートデータをメモリ効率よく準備します。
    大きなデータセットの場合、非同期で処理することで他のリクエスト処理を妨げないようにします。

    Args:
        processor: 可視化プロセッサ
        analysis_results: 分析結果
        visualization_type: 可視化タイプ
        options: オプション

    Returns:
        準備されたチャートデータ
    """
    # 大規模なデータセットの場合
    is_large_dataset = _is_large_dataset(analysis_results)

    if is_large_dataset:
        # 大規模データセットの場合、非同期で処理（I/Oバウンドな処理を想定）
        # 実際のデータ処理をコルーチンとして実行
        return await asyncio.to_thread(
            processor.prepare_chart_data,
            analysis_results,
            visualization_type,
            options
        )
    else:
        # 小規模データセットの場合は通常通り処理
        return processor.prepare_chart_data(
            analysis_results,
            visualization_type,
            options
        )

def _is_large_dataset(analysis_results: Dict[str, Any]) -> bool:
    """
    データセットが大きいかどうかを判定します。

    Args:
        analysis_results: 分析結果

    Returns:
        大きいデータセットの場合はTrue
    """
    # データサイズの判定基準
    if "data" in analysis_results:
        if isinstance(analysis_results["data"], list) and len(analysis_results["data"]) > 1000:
            return True
        if isinstance(analysis_results["data"], dict) and str(analysis_results["data"]).__sizeof__() > 100000:
            return True

    # 結果全体のサイズで判定
    return str(analysis_results).__sizeof__() > 500000