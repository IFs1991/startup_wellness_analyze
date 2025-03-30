# -*- coding: utf-8 -*-
"""
市場分析 API ルーター
スタートアップの市場ポジションと競争環境を分析するエンドポイントを提供します。
"""
from fastapi import APIRouter, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import pandas as pd
from pydantic import BaseModel, Field

# 分析モジュールのインポート
from analysis.MarketAnalyzer import MarketAnalyzer as AnalysisMarketEngine
# 非同期操作とFirestore接続のためのコアモジュール
from core.market_analyzer import MarketAnalyzer as CoreMarketAnalyzer

# 認証関連のインポート
from core.auth_manager import User, get_current_active_user, get_current_analyst_user

# リクエストモデルの定義
class MarketSizeRequest(BaseModel):
    """市場規模推定リクエストモデル"""
    company_id: str = Field(..., description="分析対象企業ID")
    market_data: Dict[str, Any] = Field(..., description="市場データ")
    growth_factors: Optional[Dict[str, float]] = Field(None, description="成長係数")
    projection_years: int = Field(5, description="予測年数")
    metadata: Optional[Dict[str, Any]] = Field(None, description="追加のメタデータ")

class CompetitiveMapRequest(BaseModel):
    """競合マッピングリクエストモデル"""
    company_id: str = Field(..., description="分析対象企業ID")
    competitor_data: Dict[str, Any] = Field(..., description="競合企業データ")
    dimensions: List[str] = Field(..., description="分析軸")
    focal_company: Optional[str] = Field(None, description="中心企業（自社）")
    metadata: Optional[Dict[str, Any]] = Field(None, description="追加のメタデータ")

class MarketTrendsRequest(BaseModel):
    """市場トレンド分析リクエストモデル"""
    company_id: str = Field(..., description="分析対象企業ID")
    keyword_list: List[str] = Field(..., description="分析対象キーワードリスト")
    date_range: Optional[Dict[str, str]] = Field(None, description="分析期間")
    metadata: Optional[Dict[str, Any]] = Field(None, description="追加のメタデータ")

# レスポンスモデルの定義
class MarketAnalysisResponse(BaseModel):
    """市場分析レスポンスモデル"""
    status: str = "success"
    data: Dict[str, Any]
    analyzed_at: datetime = Field(default_factory=datetime.now)
    message: Optional[str] = None

# ロガーの設定
logger = logging.getLogger(__name__)

# ルーターの定義
router = APIRouter(
    prefix="/market",
    tags=["market"],
    responses={404: {"description": "Not found"}}
)

# 分析エンジンの初期化
_analysis_engine = AnalysisMarketEngine()

# ユーザーアクセス権限チェック
async def _check_market_access(user: User, company_id: str):
    """
    ユーザーの市場データアクセス権限をチェック

    Args:
        user: ユーザー情報
        company_id: 分析対象企業ID

    Raises:
        HTTPException: アクセス権限がない場合
    """
    # 管理者とアナリストはすべての企業にアクセス可能
    if user.role in ["admin", "analyst"]:
        return

    # 一般ユーザーは自分の企業のデータのみアクセス可能
    if user.company_id != company_id:
        raise HTTPException(
            status_code=403,
            detail="指定された企業の市場データへのアクセス権限がありません"
        )

# APIエンドポイント定義
@router.post("/market-size", response_model=MarketAnalysisResponse)
async def estimate_market_size(
    request: MarketSizeRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    市場規模推定を実行する

    TAM（全体市場規模）・SAM（実行可能市場規模）・SOM（獲得可能市場規模）の推定と予測を行います。
    """
    try:
        # アクセス権限のチェック
        await _check_market_access(current_user, request.company_id)

        # 分析エンジンを使用して市場規模推定（分析ロジックのみ使用）
        market_size_results = _analysis_engine.estimate_market_size(
            request.market_data,
            growth_factors=request.growth_factors,
            projection_years=request.projection_years
        )

        # Firestoreへの保存（コアモジュールを使用）
        core_analyzer = CoreMarketAnalyzer()
        analysis_id = await core_analyzer.save_market_size_analysis(
            request.company_id,
            market_size_results,
            request.metadata
        )

        # レスポンスの作成
        return MarketAnalysisResponse(
            status="success",
            data={
                "analysis_id": analysis_id,
                "market_size": market_size_results,
                "company_id": request.company_id
            },
            message="市場規模推定が完了しました"
        )

    except ValueError as e:
        logger.error(f"市場規模推定の入力エラー: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"市場規模推定中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail="市場規模推定の実行中にエラーが発生しました")

@router.post("/competitive-map", response_model=MarketAnalysisResponse)
async def create_competitive_map(
    request: CompetitiveMapRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    競合マッピングを実行する

    競合企業をポジショニングマップ上にマッピングし、競争環境を可視化します。
    """
    try:
        # アクセス権限のチェック
        await _check_market_access(current_user, request.company_id)

        # 競合データをDataFrameに変換
        competitor_df = pd.DataFrame(request.competitor_data)

        # 分析エンジンを使用して競合マッピング（分析ロジックのみ使用）
        competitive_map_results = _analysis_engine.create_competitive_map(
            competitor_df,
            dimensions=request.dimensions,
            focal_company=request.focal_company
        )

        # Firestoreへの保存（コアモジュールを使用）
        core_analyzer = CoreMarketAnalyzer()
        analysis_id = await core_analyzer.save_competitive_map_analysis(
            request.company_id,
            competitive_map_results,
            request.metadata
        )

        # レスポンスの作成
        return MarketAnalysisResponse(
            status="success",
            data={
                "analysis_id": analysis_id,
                "competitive_map": competitive_map_results,
                "company_id": request.company_id
            },
            message="競合マッピングが完了しました"
        )

    except ValueError as e:
        logger.error(f"競合マッピングの入力エラー: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"競合マッピング中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail="競合マッピングの実行中にエラーが発生しました")

@router.post("/market-trends", response_model=MarketAnalysisResponse)
async def analyze_market_trends(
    request: MarketTrendsRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    市場トレンド分析を実行する

    特定のキーワードの時系列トレンドを分析し、市場動向を把握します。
    """
    try:
        # アクセス権限のチェック
        await _check_market_access(current_user, request.company_id)

        # 分析エンジンの使用可能なメソッドを確認
        # analysis/MarketAnalyzerにはtrack_market_trendsに相当するメソッドとして
        # analyze_market_trendsがある場合があります。ここでは両方に対応できるようにしています
        if hasattr(_analysis_engine, "track_market_trends"):
            trends_method = _analysis_engine.track_market_trends
        elif hasattr(_analysis_engine, "analyze_market_trends"):
            trends_method = _analysis_engine.analyze_market_trends
        else:
            raise ValueError("市場トレンド分析メソッドが見つかりません")

        # 市場トレンド分析を実行
        keyword_data = {"keywords": request.keyword_list}
        market_trends_results = trends_method(
            keyword_data,
            date_range=request.date_range
        )

        # Firestoreへの保存（コアモジュールを使用）
        core_analyzer = CoreMarketAnalyzer()
        analysis_id = await core_analyzer.save_market_trends_analysis(
            request.company_id,
            market_trends_results,
            request.metadata
        )

        # レスポンスの作成
        return MarketAnalysisResponse(
            status="success",
            data={
                "analysis_id": analysis_id,
                "market_trends": market_trends_results,
                "company_id": request.company_id
            },
            message="市場トレンド分析が完了しました"
        )

    except ValueError as e:
        logger.error(f"市場トレンド分析の入力エラー: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"市場トレンド分析中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail="市場トレンド分析の実行中にエラーが発生しました")

@router.get("/analysis/{analysis_id}", response_model=MarketAnalysisResponse)
async def get_market_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    保存された市場分析結果を取得する

    保存された市場分析の結果をIDで検索します。
    """
    try:
        # コアモジュールを使用して保存された分析結果を取得
        core_analyzer = CoreMarketAnalyzer()
        analysis_result = await core_analyzer.get_market_analysis(analysis_id)

        if not analysis_result:
            raise HTTPException(status_code=404, detail="指定されたIDの分析結果が見つかりません")

        # 分析結果の企業IDを取得
        company_id = analysis_result.get("company_id")

        # アクセス権限のチェック
        await _check_market_access(current_user, company_id)

        # レスポンスの作成
        return MarketAnalysisResponse(
            status="success",
            data=analysis_result,
            analyzed_at=analysis_result.get("created_at", datetime.now()),
            message="市場分析結果の取得が完了しました"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"市場分析結果の取得中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail="市場分析結果の取得中にエラーが発生しました")