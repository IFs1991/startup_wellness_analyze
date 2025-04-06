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
from analysis.MarketAnalyzer import MarketAnalyzer
# 削除: from core.market_analyzer import MarketAnalyzer as CoreMarketAnalyzer
# 保存に必要なコア機能があればインポート
# from service.firestore.client import FirestoreService

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
# _analysis_engine = MarketAnalyzer() # Instantiate here or use Depends

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
    # market_analyzer: MarketAnalyzer = Depends(...)
):
    """
    市場規模推定を実行する

    TAM（全体市場規模）・SAM（実行可能市場規模）・SOM（獲得可能市場規模）の推定と予測を行います。
    """
    try:
        await _check_market_access(current_user, request.company_id)

        market_analyzer = MarketAnalyzer() # Instantiate here for now

        # 分析エンジンを使用して市場規模推定
        market_size_results = await market_analyzer.estimate_market_size( # Assuming async
            request.market_data,
            growth_factors=request.growth_factors,
            projection_years=request.projection_years
        )

        # 結果の保存 (MarketAnalyzerが担うと仮定)
        analysis_id = await market_analyzer.save_analysis_result(
            company_id=request.company_id,
            analysis_type="market_size",
            result_data=market_size_results,
            metadata=request.metadata
        )

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
        logger.error(f"市場規模推定の入力エラー: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"市場規模推定中にエラーが発生しました: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="市場規模推定の実行中にエラーが発生しました")

@router.post("/competitive-map", response_model=MarketAnalysisResponse)
async def create_competitive_map(
    request: CompetitiveMapRequest,
    current_user: User = Depends(get_current_active_user)
    # market_analyzer: MarketAnalyzer = Depends(...)
):
    """
    競合マッピングを実行する

    競合企業をポジショニングマップ上にマッピングし、競争環境を可視化します。
    """
    try:
        await _check_market_access(current_user, request.company_id)

        market_analyzer = MarketAnalyzer() # Instantiate here for now

        competitor_df = pd.DataFrame(request.competitor_data)

        # 分析エンジンを使用して競合マッピング
        competitive_map_results = await market_analyzer.create_competitive_map( # Assuming async
            competitor_df,
            dimensions=request.dimensions,
            focal_company=request.focal_company
        )

        # 結果の保存 (MarketAnalyzerが担うと仮定)
        analysis_id = await market_analyzer.save_analysis_result(
            company_id=request.company_id,
            analysis_type="competitive_map",
            result_data=competitive_map_results,
            metadata=request.metadata
        )

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
        logger.error(f"競合マッピングの入力エラー: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"競合マッピング中にエラーが発生しました: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="競合マッピングの実行中にエラーが発生しました")

@router.post("/market-trends", response_model=MarketAnalysisResponse)
async def analyze_market_trends(
    request: MarketTrendsRequest,
    current_user: User = Depends(get_current_active_user)
    # market_analyzer: MarketAnalyzer = Depends(...)
):
    """
    市場トレンド分析を実行する

    特定のキーワードの時系列トレンドを分析し、市場動向を把握します。
    """
    try:
        await _check_market_access(current_user, request.company_id)

        market_analyzer = MarketAnalyzer() # Instantiate here for now

        # Check for appropriate method in the analyzer
        # Assuming analyze_market_trends is the primary method now
        if not hasattr(market_analyzer, "analyze_market_trends"):
             raise ValueError("市場トレンド分析メソッド (analyze_market_trends) が見つかりません")

        trends_method = market_analyzer.analyze_market_trends # Assuming async

        # 市場トレンド分析を実行
        market_trends_results = await trends_method(
            keyword_list=request.keyword_list, # Assuming method takes list directly
            date_range=request.date_range
        )

        # 結果の保存 (MarketAnalyzerが担うと仮定)
        analysis_id = await market_analyzer.save_analysis_result(
            company_id=request.company_id,
            analysis_type="market_trends",
            result_data=market_trends_results,
            metadata=request.metadata
        )

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
        logger.error(f"市場トレンド分析の入力エラー: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"市場トレンド分析中にエラーが発生しました: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="市場トレンド分析の実行中にエラーが発生しました")

@router.get("/analysis/{analysis_id}", response_model=MarketAnalysisResponse)
async def get_market_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_active_user)
    # market_analyzer: MarketAnalyzer = Depends(...)
):
    """
    保存された市場分析結果を取得する

    保存された市場分析の結果をIDで検索します。
    """
    try:
        market_analyzer = MarketAnalyzer() # Instantiate here for now

        # 保存された結果を取得するメソッドが必要 (MarketAnalyzerが担うと仮定)
        analysis_result = await market_analyzer.get_analysis_result(analysis_id)

        if not analysis_result:
            raise HTTPException(status_code=404, detail="指定された分析結果が見つかりません")

        # アクセス権限チェック (取得した結果の company_id を使用)
        await _check_market_access(current_user, analysis_result.get("company_id"))

        return MarketAnalysisResponse(
             status="success",
             data=analysis_result # analysis_result が適切な Dict 形式であると仮定
        )
    except HTTPException as e:
        raise e # Re-raise HTTPException
    except Exception as e:
        logger.error(f"市場分析結果取得エラー (ID: {analysis_id}): {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="分析結果の取得中にエラーが発生しました")