"""
可視化APIのroutersバージョン

このモジュールは次のエンドポイントを提供します：
- POST /api/visualizations/chart - 単一チャートの生成
- POST /api/visualizations/charts - 複数チャートの生成
- POST /api/visualizations/dashboard - ダッシュボードの生成
- POST /api/visualizations/chart/background - バックグラウンドでのチャート生成
- GET /api/visualizations/status/{job_id} - チャート生成ジョブのステータス確認
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
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