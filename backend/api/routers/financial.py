# -*- coding: utf-8 -*-
"""
財務分析 API ルーター
スタートアップの財務状況と成長性を分析するエンドポイントを提供します。
"""
from fastapi import APIRouter, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import pandas as pd
from pydantic import BaseModel, Field

# 分析モジュールのインポート
from analysis.FinancialAnalyzer import FinancialAnalyzer as AnalysisFinancialAnalyzer
# 非同期操作とFirestore接続のためのコアモジュール
from core.financial_analyzer import FinancialAnalyzer as CoreFinancialAnalyzer

# 認証関連のインポート
from core.auth_manager import User, get_current_active_user, get_current_analyst_user

# リクエストモデルの定義
class FinancialAnalysisRequest(BaseModel):
    """財務分析リクエストモデル"""
    company_id: str = Field(..., description="分析対象企業ID")
    financial_data: Dict[str, Any] = Field(..., description="財務データ")
    period: str = Field("monthly", description="分析期間（monthly/quarterly/yearly）")
    metadata: Optional[Dict[str, Any]] = Field(None, description="追加のメタデータ")

class UnitEconomicsRequest(BaseModel):
    """ユニットエコノミクス分析リクエストモデル"""
    company_id: str = Field(..., description="分析対象企業ID")
    revenue_data: Dict[str, Any] = Field(..., description="収益データ")
    customer_data: Dict[str, Any] = Field(..., description="顧客データ")
    cost_data: Dict[str, Any] = Field(..., description="コストデータ")
    metadata: Optional[Dict[str, Any]] = Field(None, description="追加のメタデータ")

class GrowthMetricsRequest(BaseModel):
    """成長指標分析リクエストモデル"""
    company_id: str = Field(..., description="分析対象企業ID")
    financial_data: Dict[str, Any] = Field(..., description="財務データ")
    benchmark_data: Optional[Dict[str, Any]] = Field(None, description="ベンチマークデータ")
    metadata: Optional[Dict[str, Any]] = Field(None, description="追加のメタデータ")

# レスポンスモデルの定義
class FinancialAnalysisResponse(BaseModel):
    """財務分析レスポンスモデル"""
    status: str = "success"
    data: Dict[str, Any]
    analyzed_at: datetime = Field(default_factory=datetime.now)
    message: Optional[str] = None

# ロガーの設定
logger = logging.getLogger(__name__)

# ルーターの定義
router = APIRouter(
    prefix="/financial",
    tags=["financial"],
    responses={404: {"description": "Not found"}}
)

# 分析エンジンの初期化
_analysis_engine = AnalysisFinancialAnalyzer()

# ユーザーアクセス権限チェック
async def _check_financial_access(user: User, company_id: str):
    """
    ユーザーの財務データアクセス権限をチェック

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
            detail="指定された企業の財務データへのアクセス権限がありません"
        )

# APIエンドポイント定義
@router.post("/burn-rate", response_model=FinancialAnalysisResponse)
async def analyze_burn_rate(
    request: FinancialAnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    バーンレート分析を実行する

    スタートアップのキャッシュバーン率とランウェイ（資金枯渇までの期間）を計算します。
    """
    try:
        # アクセス権限のチェック
        await _check_financial_access(current_user, request.company_id)

        # 財務データをDataFrameに変換
        financial_df = pd.DataFrame(request.financial_data)

        # 分析エンジンを使用してバーンレート計算（分析ロジックのみ使用）
        burn_rate_results = _analysis_engine.calculate_burn_rate(
            financial_df,
            period=request.period
        )

        # Firestoreへの保存（コアモジュールを使用）
        core_analyzer = CoreFinancialAnalyzer()
        analysis_id = await core_analyzer.save_burn_rate_analysis(
            request.company_id,
            burn_rate_results,
            request.metadata
        )

        # レスポンスの作成
        return FinancialAnalysisResponse(
            status="success",
            data={
                "analysis_id": analysis_id,
                "burn_rate": burn_rate_results,
                "company_id": request.company_id
            },
            message="バーンレート分析が完了しました"
        )

    except ValueError as e:
        logger.error(f"バーンレート分析の入力エラー: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"バーンレート分析中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail="バーンレート分析の実行中にエラーが発生しました")

@router.post("/unit-economics", response_model=FinancialAnalysisResponse)
async def analyze_unit_economics(
    request: UnitEconomicsRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    ユニットエコノミクス分析を実行する

    顧客獲得コスト（CAC）、顧客生涯価値（LTV）などの指標を計算します。
    """
    try:
        # アクセス権限のチェック
        await _check_financial_access(current_user, request.company_id)

        # データをDataFrameに変換
        revenue_df = pd.DataFrame(request.revenue_data)
        customer_df = pd.DataFrame(request.customer_data)
        cost_df = pd.DataFrame(request.cost_data)

        # 分析エンジンを使用してユニットエコノミクス分析（分析ロジックのみ使用）
        unit_economics_results = _analysis_engine.analyze_unit_economics(
            revenue_df,
            customer_df,
            cost_df
        )

        # Firestoreへの保存（コアモジュールを使用）
        core_analyzer = CoreFinancialAnalyzer()
        analysis_id = await core_analyzer.save_unit_economics_analysis(
            request.company_id,
            unit_economics_results,
            request.metadata
        )

        # レスポンスの作成
        return FinancialAnalysisResponse(
            status="success",
            data={
                "analysis_id": analysis_id,
                "unit_economics": unit_economics_results,
                "company_id": request.company_id
            },
            message="ユニットエコノミクス分析が完了しました"
        )

    except ValueError as e:
        logger.error(f"ユニットエコノミクス分析の入力エラー: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"ユニットエコノミクス分析中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail="ユニットエコノミクス分析の実行中にエラーが発生しました")

@router.post("/growth-metrics", response_model=FinancialAnalysisResponse)
async def evaluate_growth_metrics(
    request: GrowthMetricsRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    成長指標評価を実行する

    売上成長率、利益成長率などの成長指標を評価します。
    """
    try:
        # アクセス権限のチェック
        await _check_financial_access(current_user, request.company_id)

        # 財務データをDataFrameに変換
        financial_df = pd.DataFrame(request.financial_data)
        benchmark_df = pd.DataFrame(request.benchmark_data) if request.benchmark_data else None

        # 分析エンジンを使用して成長指標評価（分析ロジックのみ使用）
        growth_metrics_results = _analysis_engine.evaluate_growth_metrics(
            financial_df,
            benchmark_df
        )

        # Firestoreへの保存（コアモジュールを使用）
        core_analyzer = CoreFinancialAnalyzer()
        analysis_id = await core_analyzer.save_growth_metrics_analysis(
            request.company_id,
            growth_metrics_results,
            request.metadata
        )

        # レスポンスの作成
        return FinancialAnalysisResponse(
            status="success",
            data={
                "analysis_id": analysis_id,
                "growth_metrics": growth_metrics_results,
                "company_id": request.company_id
            },
            message="成長指標評価が完了しました"
        )

    except ValueError as e:
        logger.error(f"成長指標評価の入力エラー: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"成長指標評価中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail="成長指標評価の実行中にエラーが発生しました")

@router.get("/analysis/{analysis_id}", response_model=FinancialAnalysisResponse)
async def get_financial_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    保存された財務分析結果を取得する

    保存された財務分析の結果をIDで検索します。
    """
    try:
        # コアモジュールを使用して保存された分析結果を取得
        core_analyzer = CoreFinancialAnalyzer()
        analysis_result = await core_analyzer.get_financial_analysis(analysis_id)

        if not analysis_result:
            raise HTTPException(status_code=404, detail="指定されたIDの分析結果が見つかりません")

        # 分析結果の企業IDを取得
        company_id = analysis_result.get("company_id")

        # アクセス権限のチェック
        await _check_financial_access(current_user, company_id)

        # レスポンスの作成
        return FinancialAnalysisResponse(
            status="success",
            data=analysis_result,
            analyzed_at=analysis_result.get("created_at", datetime.now()),
            message="財務分析結果の取得が完了しました"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"財務分析結果の取得中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail="財務分析結果の取得中にエラーが発生しました")