# -*- coding: utf-8 -*-
"""
チーム分析 API ルーター
スタートアップの創業チーム、組織成長、企業文化を分析するエンドポイントを提供します。
"""
from fastapi import APIRouter, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import pandas as pd
from pydantic import BaseModel, Field

# 分析モジュールのインポート
from analysis.Team_Analyzer import TeamAnalyzer
# 削除: from core.team_analyzer import TeamAnalyzer as CoreTeamAnalyzer
# Firestore Service など、保存に必要なコア機能があればインポート
# from service.firestore.client import FirestoreService # Example, adjust if needed

# 認証関連のインポート
from core.auth_manager import User, get_current_active_user, get_current_analyst_user

# リクエストモデルの定義
class FoundingTeamRequest(BaseModel):
    """創業チーム評価リクエストモデル"""
    company_id: str = Field(..., description="分析対象企業ID")
    founder_profiles: List[Dict[str, Any]] = Field(..., description="創業者プロフィールデータ")
    company_stage: str = Field(..., description="企業のステージ")
    industry: str = Field("software", description="業界")
    metadata: Optional[Dict[str, Any]] = Field(None, description="追加のメタデータ")

class OrgGrowthRequest(BaseModel):
    """組織成長分析リクエストモデル"""
    company_id: str = Field(..., description="分析対象企業ID")
    employee_data: Dict[str, Any] = Field(..., description="従業員データ")
    timeline: str = Field("1y", description="分析期間")
    company_stage: str = Field("series_a", description="企業のステージ")
    industry: str = Field("software", description="業界")
    metadata: Optional[Dict[str, Any]] = Field(None, description="追加のメタデータ")

class CultureStrengthRequest(BaseModel):
    """企業文化強度分析リクエストモデル"""
    company_id: str = Field(..., description="分析対象企業ID")
    engagement_data: Dict[str, Any] = Field(..., description="エンゲージメントデータ")
    survey_results: Optional[Dict[str, Any]] = Field(None, description="サーベイ結果データ")
    metadata: Optional[Dict[str, Any]] = Field(None, description="追加のメタデータ")

# レスポンスモデルの定義
class TeamAnalysisResponse(BaseModel):
    """チーム分析レスポンスモデル"""
    status: str = "success"
    data: Dict[str, Any]
    analyzed_at: datetime = Field(default_factory=datetime.now)
    message: Optional[str] = None

# ロガーの設定
logger = logging.getLogger(__name__)

# ルーターの定義
router = APIRouter(
    prefix="/team",
    tags=["team"],
    responses={404: {"description": "Not found"}}
)

# 分析エンジンの初期化
# _analysis_engine = TeamAnalyzer() # Instantiate here or use Depends

# ユーザーアクセス権限チェック
async def _check_team_access(user: User, company_id: str):
    """
    ユーザーのチームデータアクセス権限をチェック

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
            detail="指定された企業のチームデータへのアクセス権限がありません"
        )

# APIエンドポイント定義
@router.post("/founding-team", response_model=TeamAnalysisResponse)
async def evaluate_founding_team(
    request: FoundingTeamRequest,
    current_user: User = Depends(get_current_active_user)
    # team_analyzer: TeamAnalyzer = Depends(...) # Inject via Depends recommended
):
    """
    創業チームの評価を実行する

    創業チームの経験、スキルセット、ドメイン知識などを評価し、チーム全体のスコアを算出します。
    """
    try:
        await _check_team_access(current_user, request.company_id)

        team_analyzer = TeamAnalyzer() # Instantiate here for now
        # FirestoreService などの依存関係をコンストラクタに渡す必要がある場合がある
        # team_analyzer = TeamAnalyzer(db=firestore_service) # Example

        # 分析エンジンを使用して創業チーム評価
        founding_team_results = await team_analyzer.evaluate_founding_team( # Assuming async method
            request.founder_profiles,
            company_stage=request.company_stage,
            industry=request.industry
        )

        # 結果の保存 (TeamAnalyzerが担うと仮定)
        analysis_id = await team_analyzer.save_analysis_result(
            company_id=request.company_id,
            analysis_type="founding_team",
            result_data=founding_team_results,
            metadata=request.metadata
        )

        return TeamAnalysisResponse(
            status="success",
            data={
                "analysis_id": analysis_id,
                "founding_team": founding_team_results,
                "company_id": request.company_id
            },
            message="創業チーム評価が完了しました"
        )

    except ValueError as e:
        logger.error(f"創業チーム評価の入力エラー: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"創業チーム評価中にエラーが発生しました: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="創業チーム評価の実行中にエラーが発生しました")

@router.post("/org-growth", response_model=TeamAnalysisResponse)
async def analyze_org_growth(
    request: OrgGrowthRequest,
    current_user: User = Depends(get_current_active_user)
    # team_analyzer: TeamAnalyzer = Depends(...)
):
    """
    組織成長分析を実行する

    従業員の成長率、組織構造の変化、部門間のバランスなどを分析します。
    """
    try:
        await _check_team_access(current_user, request.company_id)

        team_analyzer = TeamAnalyzer() # Instantiate here for now

        employee_df = pd.DataFrame(request.employee_data)

        # 分析エンジンを使用して組織成長分析
        org_growth_results = await team_analyzer.analyze_org_growth( # Assuming async
            employee_df,
            timeline=request.timeline,
            company_stage=request.company_stage,
            industry=request.industry
        )

        # 結果の保存 (TeamAnalyzerが担うと仮定)
        analysis_id = await team_analyzer.save_analysis_result(
             company_id=request.company_id,
             analysis_type="org_growth",
             result_data=org_growth_results,
             metadata=request.metadata
        )

        return TeamAnalysisResponse(
            status="success",
            data={
                "analysis_id": analysis_id,
                "org_growth": org_growth_results,
                "company_id": request.company_id
            },
            message="組織成長分析が完了しました"
        )

    except ValueError as e:
        logger.error(f"組織成長分析の入力エラー: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"組織成長分析中にエラーが発生しました: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="組織成長分析の実行中にエラーが発生しました")

@router.post("/culture-strength", response_model=TeamAnalysisResponse)
async def measure_culture_strength(
    request: CultureStrengthRequest,
    current_user: User = Depends(get_current_active_user)
    # team_analyzer: TeamAnalyzer = Depends(...)
):
    """
    企業文化の強さを分析する

    エンゲージメントデータと従業員サーベイから企業文化の強さと特性を評価します。
    """
    try:
        await _check_team_access(current_user, request.company_id)

        team_analyzer = TeamAnalyzer() # Instantiate here for now

        engagement_df = pd.DataFrame(request.engagement_data)
        survey_df = pd.DataFrame(request.survey_results) if request.survey_results else None

        # 分析エンジンを使用して企業文化強度分析
        culture_results = await team_analyzer.measure_culture_strength( # Assuming async
            engagement_df,
            survey_results=survey_df
        )

        # 結果の保存 (TeamAnalyzerが担うと仮定)
        analysis_id = await team_analyzer.save_analysis_result(
             company_id=request.company_id,
             analysis_type="culture_strength",
             result_data=culture_results,
             metadata=request.metadata
        )

        return TeamAnalysisResponse(
            status="success",
            data={
                "analysis_id": analysis_id,
                "culture_strength": culture_results,
                "company_id": request.company_id
            },
            message="企業文化分析が完了しました"
        )

    except ValueError as e:
        logger.error(f"企業文化分析の入力エラー: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"企業文化分析中にエラーが発生しました: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="企業文化分析の実行中にエラーが発生しました")

@router.get("/analysis/{analysis_id}", response_model=TeamAnalysisResponse)
async def get_team_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_active_user)
    # team_analyzer: TeamAnalyzer = Depends(...)
):
    """
    保存されたチーム分析結果を取得する

    保存されたチーム分析の結果をIDで検索します。
    """
    try:
        team_analyzer = TeamAnalyzer() # Instantiate here for now
        # 保存された結果を取得するメソッドが必要 (TeamAnalyzerが担うと仮定)
        analysis_result = await team_analyzer.get_analysis_result(analysis_id)

        if not analysis_result:
            raise HTTPException(status_code=404, detail="指定された分析結果が見つかりません")

        # アクセス権限チェック (取得した結果の company_id を使用)
        await _check_team_access(current_user, analysis_result.get("company_id"))

        return TeamAnalysisResponse(
             status="success",
             data=analysis_result # analysis_result が適切な Dict 形式であると仮定
        )
    except HTTPException as e:
        raise e # Re-raise HTTPException (e.g., 403, 404)
    except Exception as e:
        logger.error(f"チーム分析結果取得エラー (ID: {analysis_id}): {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="分析結果の取得中にエラーが発生しました")