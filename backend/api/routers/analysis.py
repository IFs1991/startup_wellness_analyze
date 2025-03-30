# -*- coding: utf-8 -*-
"""
分析 API ルーター
スタートアップの健康状態と財務データの関連分析を提供します。
"""
from fastapi import APIRouter, HTTPException, Depends, Request, Security
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
from pydantic import BaseModel, Field

# コアモジュールのインポート
from core.correlation_analyzer import CorrelationAnalyzer
from core.cluster_analyzer import ClusterAnalyzer
from core.pca_analyzer import PCAAnalyzer
from core.survival_analyzer import SurvivalAnalyzer
from core.text_miner import TextMiner
from core.time_series_analyzer import TimeSeriesAnalyzer
from core.association_analyzer import AssociationAnalyzer

# 認証関連のインポート
from core.auth_manager import User, get_current_active_user, get_current_analyst_user

# 型定義
AnalysisResult = Dict[str, Any]

# リクエストモデルの定義
class AnalysisCondition(BaseModel):
    """分析条件モデル"""
    field: str
    operator: str
    value: Any

class AnalysisRequest(BaseModel):
    """分析リクエストモデル"""
    collection_name: str = Field(..., description="分析対象のコレクション名")
    conditions: Optional[List[Dict[str, Any]]] = Field(None, description="クエリ条件")
    target_column: Optional[str] = Field(None, description="対象カラム")
    n_clusters: int = Field(3, description="クラスター数")
    n_components: int = Field(2, description="主成分数")
    duration_col: Optional[str] = Field(None, description="期間カラム")
    event_col: Optional[str] = Field(None, description="イベントカラム")
    min_support: float = Field(0.1, description="最小サポート値")
    periods: int = Field(12, description="予測期間")
    company_id: Optional[str] = Field(None, description="分析対象企業ID")

# レスポンスモデルの定義
class AnalysisResponse(BaseModel):
    """分析レスポンスモデル"""
    status: str = "success"
    data: Dict[str, Any]
    analyzed_at: datetime = Field(default_factory=datetime.now)
    message: Optional[str] = None

# ログ設定
logger = logging.getLogger(__name__)

# ルーターの定義
router = APIRouter(
    prefix="/analysis",
    tags=["analysis"],
    responses={404: {"description": "Not found"}}
)

# 分析共通処理
async def _check_analysis_access(user: User, company_id: Optional[str] = None):
    """
    ユーザーの分析アクセス権限をチェック

    Args:
        user: ユーザー情報
        company_id: 分析対象企業ID

    Raises:
        HTTPException: アクセス権限がない場合
    """
    # 管理者とアナリストはすべての企業にアクセス可能
    if user.role in ["admin", "analyst"]:
        return

    # 企業ユーザーは自社データのみアクセス可能
    if company_id and company_id != user.company_id:
        logger.warning(f"異なる企業のデータへのアクセス試行: user_id={user.id}, requested_company={company_id}")
        raise HTTPException(
            status_code=403,
            detail="この企業のデータにアクセスする権限がありません"
        )

    # データアクセス制限チェック
    if user.data_access and company_id not in user.data_access:
        logger.warning(f"アクセス制限データへのアクセス試行: user_id={user.id}, company_id={company_id}")
        raise HTTPException(
            status_code=403,
            detail="このデータにアクセスする権限がありません"
        )

# 各分析用エンドポイント
@router.post("/correlation", response_model=AnalysisResponse)
async def perform_correlation_analysis(
    request_data: AnalysisRequest,
    current_user: User = Security(get_current_analyst_user)
):
    """相関分析を実行するエンドポイント"""
    try:
        # アクセス権限チェック
        await _check_analysis_access(current_user, request_data.company_id)

        analyzer = CorrelationAnalyzer()
        # データの取得と前処理は、coreモジュールの責務
        data = await analyzer.get_data(
            collection_name=request_data.collection_name,
            conditions=request_data.conditions
        )

        # 相関分析の実行
        result = await analyzer.analyze(
            data=data,
            variables=data.columns.tolist() if request_data.target_column is None else [request_data.target_column],
            user_id=current_user.id
        )

        return AnalysisResponse(
            data=result.to_dict() if hasattr(result, 'to_dict') else {"correlation_matrix": result}
        )
    except Exception as e:
        logger.error(f"相関分析エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clustering", response_model=AnalysisResponse)
async def perform_cluster_analysis(
    request_data: AnalysisRequest,
    current_user: User = Security(get_current_analyst_user)
):
    """クラスタリング分析を実行するエンドポイント"""
    try:
        # アクセス権限チェック
        await _check_analysis_access(current_user, request_data.company_id)

        analyzer = ClusterAnalyzer()
        data = await analyzer.get_data(
            collection_name=request_data.collection_name,
            conditions=request_data.conditions
        )

        result = await analyzer.analyze(
            data=data,
            n_clusters=request_data.n_clusters,
            user_id=current_user.id
        )

        return AnalysisResponse(data=result)
    except Exception as e:
        logger.error(f"クラスタリング分析エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pca", response_model=AnalysisResponse)
async def perform_pca_analysis(
    request_data: AnalysisRequest,
    current_user: User = Security(get_current_analyst_user)
):
    """主成分分析を実行するエンドポイント"""
    try:
        # アクセス権限チェック
        await _check_analysis_access(current_user, request_data.company_id)

        analyzer = PCAAnalyzer()
        data = await analyzer.get_data(
            collection_name=request_data.collection_name,
            conditions=request_data.conditions
        )

        result = await analyzer.analyze(
            data=data,
            n_components=request_data.n_components,
            user_id=current_user.id
        )

        return AnalysisResponse(data=result)
    except Exception as e:
        logger.error(f"主成分分析エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 他の分析エンドポイントも同様のパターンで実装
@router.post("/survival", response_model=AnalysisResponse)
async def perform_survival_analysis(
    request_data: AnalysisRequest,
    current_user: User = Security(get_current_analyst_user)
):
    """生存時間分析を実行するエンドポイント"""
    try:
        # アクセス権限チェック
        await _check_analysis_access(current_user, request_data.company_id)

        analyzer = SurvivalAnalyzer()
        data = await analyzer.get_data(
            collection_name=request_data.collection_name,
            conditions=request_data.conditions
        )

        result = await analyzer.analyze(
            data=data,
            duration_col=request_data.duration_col,
            event_col=request_data.event_col,
            user_id=current_user.id
        )

        return AnalysisResponse(data=result)
    except Exception as e:
        logger.error(f"生存時間分析エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/text_mining", response_model=AnalysisResponse)
async def perform_text_mining(
    request_data: AnalysisRequest,
    current_user: User = Security(get_current_analyst_user)
):
    """テキストマイニングを実行するエンドポイント"""
    try:
        # アクセス権限チェック
        await _check_analysis_access(current_user, request_data.company_id)

        analyzer = TextMiner()
        data = await analyzer.get_data(
            collection_name=request_data.collection_name,
            conditions=request_data.conditions
        )

        result = await analyzer.analyze(
            data=data,
            target_column=request_data.target_column,
            user_id=current_user.id
        )

        return AnalysisResponse(data=result)
    except Exception as e:
        logger.error(f"テキストマイニングエラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/time_series", response_model=AnalysisResponse)
async def perform_time_series_analysis(
    request_data: AnalysisRequest,
    current_user: User = Security(get_current_analyst_user)
):
    """時系列分析を実行するエンドポイント"""
    try:
        # アクセス権限チェック
        await _check_analysis_access(current_user, request_data.company_id)

        analyzer = TimeSeriesAnalyzer()
        data = await analyzer.get_data(
            collection_name=request_data.collection_name,
            conditions=request_data.conditions
        )

        result = await analyzer.analyze(
            data=data,
            target_column=request_data.target_column,
            periods=request_data.periods,
            user_id=current_user.id
        )

        return AnalysisResponse(data=result)
    except Exception as e:
        logger.error(f"時系列分析エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/association", response_model=AnalysisResponse)
async def perform_association_analysis(
    request_data: AnalysisRequest,
    current_user: User = Security(get_current_analyst_user)
):
    """アソシエーション分析を実行するエンドポイント"""
    try:
        # アクセス権限チェック
        await _check_analysis_access(current_user, request_data.company_id)

        analyzer = AssociationAnalyzer()
        data = await analyzer.get_data(
            collection_name=request_data.collection_name,
            conditions=request_data.conditions
        )

        result = await analyzer.analyze(
            data=data,
            min_support=request_data.min_support,
            user_id=current_user.id
        )

        return AnalysisResponse(data=result)
    except Exception as e:
        logger.error(f"アソシエーション分析エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))