"""
Geminiを使用した可視化 API

Google Gemini AIを使用してチャートやグラフを動的に生成するためのエンドポイントを提供します。
"""
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import json
import base64
import os
import uuid
from pathlib import Path

# 可視化コアモジュール
from core.visualization.gemini_chart_generator import GeminiChartGenerator
from core.config import Settings, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/visualization/gemini", tags=["visualization"])

# チャート生成リクエスト用モデル
class ChartRequest(BaseModel):
    """チャート生成リクエスト"""
    data: Dict[str, Any] = Field(..., description="グラフ化するデータ")
    chart_type: str = Field(..., description="グラフの種類（bar, line, scatter, pie, heatmapなど）")
    title: str = Field(..., description="グラフのタイトル")
    description: Optional[str] = Field(None, description="グラフの説明（オプション）")
    width: int = Field(800, description="画像の幅")
    height: int = Field(500, description="画像の高さ")
    theme: str = Field("professional", description="テーマ（professional, dark, light, modernなど）")
    use_cache: bool = Field(True, description="キャッシュを使用するかどうか")

# 複数チャート生成リクエスト用モデル
class MultipleChartRequest(BaseModel):
    """複数チャート生成リクエスト"""
    chart_configs: List[Dict[str, Any]] = Field(..., description="複数チャートの設定")
    use_cache: bool = Field(True, description="キャッシュを使用するかどうか")

# ダッシュボード生成リクエスト用モデル
class DashboardRequest(BaseModel):
    """ダッシュボード生成リクエスト"""
    dashboard_data: Dict[str, Any] = Field(..., description="ダッシュボード用データ")
    title: str = Field(..., description="ダッシュボードのタイトル")
    layout: Optional[List[Dict[str, Any]]] = Field(None, description="チャートのレイアウト設定")
    width: int = Field(1200, description="ダッシュボードの幅")
    height: int = Field(800, description="ダッシュボードの高さ")
    theme: str = Field("professional", description="テーマ")

# チャート結果レスポンスモデル
class ChartResponse(BaseModel):
    """チャート生成レスポンス"""
    success: bool = Field(..., description="成功したかどうか")
    image_data: Optional[str] = Field(None, description="Base64エンコードされた画像データ")
    format: Optional[str] = Field(None, description="画像フォーマット")
    width: Optional[int] = Field(None, description="画像の幅")
    height: Optional[int] = Field(None, description="画像の高さ")
    cached: Optional[bool] = Field(None, description="キャッシュから取得したかどうか")
    error: Optional[str] = Field(None, description="エラーメッセージ（失敗時）")

# ダッシュボードレスポンスモデル
class DashboardResponse(BaseModel):
    """ダッシュボード生成レスポンス"""
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
            api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("Gemini API キーが設定されていません")

        # GeminiChartGeneratorインスタンスを作成
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

    Args:
        request: チャート生成リクエスト
        chart_generator: チャート生成ジェネレーター

    Returns:
        ChartResponse: 生成されたチャート情報
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
    request: MultipleChartRequest,
    chart_generator: GeminiChartGenerator = Depends(get_chart_generator)
):
    """
    複数のチャートを一度に生成するエンドポイント

    Args:
        request: 複数チャート生成リクエスト
        chart_generator: チャート生成ジェネレーター

    Returns:
        List[ChartResponse]: 生成された複数チャートの情報
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

    Args:
        request: ダッシュボード生成リクエスト
        chart_generator: チャート生成ジェネレーター

    Returns:
        DashboardResponse: 生成されたダッシュボード情報
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

    Args:
        request: チャート生成リクエスト
        background_tasks: バックグラウンドタスク
        chart_generator: チャート生成ジェネレーター

    Returns:
        Dict[str, Any]: バックグラウンド処理情報
    """
    try:
        # キャッシュキーを生成
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

    Args:
        cache_key: キャッシュキー
        chart_generator: チャート生成ジェネレーター

    Returns:
        ChartResponse: チャート情報
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

        return {
            "success": True,
            "image_data": base64.b64encode(image_data).decode("utf-8"),
            "format": "png",
            "cached": True
        }
    except Exception as e:
        logger.error(f"Chart status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"チャートステータス確認中にエラーが発生しました: {str(e)}")

# --- 既存コードの後ろに追記 ---

# --- 分析結果→可視化データ変換ヘルパー ---
def convert_text_mining_result_to_chart_data(result: dict) -> dict:
    """
    TextMinerの分析結果をGemini可視化用データ形式に変換
    """
    # 例: 感情分析スコアのヒストグラムや時系列
    if "sentiment_scores" in result:
        return {
            "labels": [str(i) for i in range(len(result["sentiment_scores"]))],
            "values": result["sentiment_scores"]
        }
    # その他のケースも適宜拡張
    return {"labels": [], "values": []}

def convert_causal_inference_result_to_chart_data(result: dict) -> dict:
    """
    CausalInferenceAnalyzerの結果をGemini可視化用データ形式に変換
    """
    # 例: 効果の時系列や信頼区間
    if "effect_series" in result:
        effect_series = result["effect_series"]
        if hasattr(effect_series, "to_dict"):
            effect_series = effect_series.to_dict()
        return {
            "labels": list(effect_series.keys()),
            "values": list(effect_series.values())
        }
    return {"labels": [], "values": []}

def convert_portfolio_network_result_to_chart_data(result: dict) -> dict:
    """
    PortfolioNetworkAnalyzerの結果をGemini可視化用データ形式に変換
    """
    # 例: ノード間のエッジ数や中心性指標など
    if "centrality" in result:
        return {
            "labels": list(result["centrality"].keys()),
            "values": list(result["centrality"].values())
        }
    return {"labels": [], "values": []}

# --- 新しい可視化タイプ対応エンドポイント ---
from fastapi import Body

@router.post("/analyze-visualize", response_model=ChartResponse)
async def analyze_and_visualize(
    analysis_type: str = Body(..., description="分析タイプ: text_mining, causal_inference, portfolio_network"),
    analysis_result: dict = Body(..., description="分析結果データ"),
    chart_type: str = Body("bar", description="グラフの種類"),
    title: str = Body("分析可視化", description="グラフタイトル"),
    description: str = Body("", description="グラフ説明"),
    width: int = Body(800),
    height: int = Body(500),
    theme: str = Body("professional"),
    use_cache: bool = Body(True),
    chart_generator: GeminiChartGenerator = Depends(get_chart_generator)
):
    """
    分析結果をGeminiで可視化する統合エンドポイント
    """
    # 分析タイプごとにデータ変換
    if analysis_type == "text_mining":
        chart_data = convert_text_mining_result_to_chart_data(analysis_result)
    elif analysis_type == "causal_inference":
        chart_data = convert_causal_inference_result_to_chart_data(analysis_result)
    elif analysis_type == "portfolio_network":
        chart_data = convert_portfolio_network_result_to_chart_data(analysis_result)
    else:
        raise HTTPException(status_code=400, detail=f"未対応の分析タイプ: {analysis_type}")

    result = await chart_generator.generate_chart(
        data=chart_data,
        chart_type=chart_type,
        title=title,
        description=description,
        width=width,
        height=height,
        theme=theme,
        use_cache=use_cache
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "可視化に失敗しました"))
    return result