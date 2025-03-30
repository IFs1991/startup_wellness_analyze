import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from backend.core.visualization.gemini_chart_generator import GeminiChartGenerator
from backend.core.config import Settings, get_settings

# .envファイルを読み込む
load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/visualizations", tags=["visualizations"])

# チャート生成リクエスト用モデル
class ChartRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="グラフ化するデータ")
    chart_type: str = Field(..., description="グラフの種類（bar, line, scatter, pie, heatmapなど）")
    title: str = Field(..., description="グラフのタイトル")
    description: Optional[str] = Field(None, description="グラフの説明（オプション）")
    width: int = Field(800, description="画像の幅")
    height: int = Field(500, description="画像の高さ")
    theme: str = Field("professional", description="テーマ（professional, dark, light, modernなど）")
    use_cache: bool = Field(True, description="キャッシュを使用するかどうか")

# 複数チャート生成リクエスト用モデル
class MultiplChartRequest(BaseModel):
    chart_configs: List[Dict[str, Any]] = Field(..., description="複数チャートの設定")
    use_cache: bool = Field(True, description="キャッシュを使用するかどうか")

# ダッシュボード生成リクエスト用モデル
class DashboardRequest(BaseModel):
    dashboard_data: Dict[str, Any] = Field(..., description="ダッシュボード用データ")
    title: str = Field(..., description="ダッシュボードのタイトル")
    layout: Optional[List[Dict[str, Any]]] = Field(None, description="チャートのレイアウト設定")
    width: int = Field(1200, description="ダッシュボードの幅")
    height: int = Field(800, description="ダッシュボードの高さ")
    theme: str = Field("professional", description="テーマ")

# チャート結果レスポンスモデル
class ChartResponse(BaseModel):
    success: bool = Field(..., description="成功したかどうか")
    image_data: Optional[str] = Field(None, description="Base64エンコードされた画像データ")
    format: Optional[str] = Field(None, description="画像フォーマット")
    width: Optional[int] = Field(None, description="画像の幅")
    height: Optional[int] = Field(None, description="画像の高さ")
    cached: Optional[bool] = Field(None, description="キャッシュから取得したかどうか")
    error: Optional[str] = Field(None, description="エラーメッセージ（失敗時）")

# ダッシュボードレスポンスモデル
class DashboardResponse(BaseModel):
    success: bool = Field(..., description="成功したかどうか")
    html: Optional[str] = Field(None, description="ダッシュボードのHTML")
    width: Optional[int] = Field(None, description="ダッシュボードの幅")
    height: Optional[int] = Field(None, description="ダッシュボードの高さ")
    error: Optional[str] = Field(None, description="エラーメッセージ（失敗時）")

# Gemini Chart Generatorのインスタンスを取得する依存関数
def get_chart_generator(settings: Settings = Depends(get_settings)):
    """設定から可視化ジェネレーターインスタンスを取得"""
    try:
        # 設定からGemini APIキーを取得
        api_key = settings.gemini_api_key
        if not api_key:
            logger.warning("Gemini API key not found in settings, trying environment variables")

        # GeminiChartGeneratorインスタンスを作成
        # ここでは明示的にAPIキーを渡し、クラス内で.envや環境変数からも読み込まれる
        return GeminiChartGenerator(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize chart generator: {e}")
        raise HTTPException(status_code=500, detail="可視化サービスの初期化に失敗しました")

@router.post("/chart", response_model=ChartResponse)
async def generate_chart(
    request: ChartRequest,
    chart_generator: GeminiChartGenerator = Depends(get_chart_generator)
):
    """
    データからチャートを生成するエンドポイント
    """
    try:
        result = await chart_generator.generate_chart(
            data=request.data,
            chart_type=request.chart_type,
            title=request.title,
            description=request.description,
            width=request.width,
            height=request.height,
            theme=request.theme,
            use_cache=request.use_cache
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "チャート生成に失敗しました"))

        return result
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"チャート生成中にエラーが発生しました: {str(e)}")

@router.post("/multiple-charts", response_model=List[ChartResponse])
async def generate_multiple_charts(
    request: MultiplChartRequest,
    chart_generator: GeminiChartGenerator = Depends(get_chart_generator)
):
    """
    複数のチャートを一度に生成するエンドポイント
    """
    try:
        results = await chart_generator.generate_multiple_charts(
            chart_configs=request.chart_configs,
            use_cache=request.use_cache
        )
        return results
    except Exception as e:
        logger.error(f"Multiple charts generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"複数チャート生成中にエラーが発生しました: {str(e)}")

@router.post("/dashboard", response_model=DashboardResponse)
async def generate_dashboard(
    request: DashboardRequest,
    chart_generator: GeminiChartGenerator = Depends(get_chart_generator)
):
    """
    データからダッシュボードを生成するエンドポイント
    """
    try:
        result = await chart_generator.generate_dashboard(
            dashboard_data=request.dashboard_data,
            title=request.title,
            layout=request.layout,
            width=request.width,
            height=request.height,
            theme=request.theme
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "ダッシュボード生成に失敗しました"))

        return result
    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"ダッシュボード生成中にエラーが発生しました: {str(e)}")

@router.post("/chart/background", response_model=Dict[str, Any])
async def generate_chart_background(
    request: ChartRequest,
    background_tasks: BackgroundTasks,
    chart_generator: GeminiChartGenerator = Depends(get_chart_generator)
):
    """
    バックグラウンドでチャートを生成するエンドポイント
    結果はキャッシュに保存され、後でIDを使って取得できる
    """
    try:
        # キャッシュキーを生成（実際のチャート生成と同じロジック）
        cache_key = chart_generator._generate_cache_key(
            data=request.data,
            chart_type=request.chart_type,
            title=request.title,
            width=request.width,
            height=request.height,
            theme=request.theme
        )

        # バックグラウンドタスクとしてチャート生成を追加
        background_tasks.add_task(
            chart_generator.generate_chart,
            data=request.data,
            chart_type=request.chart_type,
            title=request.title,
            description=request.description,
            width=request.width,
            height=request.height,
            theme=request.theme,
            use_cache=request.use_cache
        )

        return {
            "success": True,
            "message": "チャート生成をバックグラウンドで開始しました",
            "cache_key": cache_key
        }
    except Exception as e:
        logger.error(f"Background chart generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"バックグラウンドチャート生成中にエラーが発生しました: {str(e)}")

@router.get("/chart/status/{cache_key}", response_model=ChartResponse)
async def get_chart_status(
    cache_key: str,
    chart_generator: GeminiChartGenerator = Depends(get_chart_generator)
):
    """
    バックグラウンドで生成されたチャートのステータスを確認
    キャッシュにあれば結果を返す
    """
    try:
        # キャッシュファイルの存在確認
        cache_path = chart_generator.cache_dir / f"{cache_key}.png"

        if not cache_path.exists():
            return {
                "success": False,
                "error": "チャートはまだ生成中か、生成に失敗しました"
            }

        # キャッシュから画像を読み込み
        with open(cache_path, "rb") as f:
            image_data = f.read()

        import base64
        return {
            "success": True,
            "image_data": base64.b64encode(image_data).decode("utf-8"),
            "format": "png",
            "cached": True
        }
    except Exception as e:
        logger.error(f"Chart status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"チャートステータス確認中にエラーが発生しました: {str(e)}")