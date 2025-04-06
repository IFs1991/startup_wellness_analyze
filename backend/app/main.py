# -*- coding: utf-8 -*-

"""
Startup Wellness データ分析システム バックエンド API

要件定義書と requirements.txt を元に作成された FastAPI アプリケーションです。
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from io import BytesIO
from datetime import datetime
import os
import uuid
import json

# アプリケーションのバージョン定義
VERSION = "1.0.0"

import uvicorn
import pandas as pd
import numpy as np
from fastapi import (
    FastAPI, Depends, HTTPException, status, UploadFile, File, Request,
    BackgroundTasks, Body, Header, WebSocket, WebSocketDisconnect
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.security import OAuth2PasswordBearer # OAuth2 スキームのインポート
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, initialize_app, auth
from google.cloud import firestore

# Coreモジュールのインポート (分析関連以外)
from core.wellness_score_calculator import WellnessScoreCalculator, create_wellness_score_calculator # これは分析実行というよりスコア計算なのでcoreに残す判断もあり
from core.data_preprocessor import DataPreprocessor
from core.data_quality_checker import DataQualityChecker
from core.model_evaluator import ModelEvaluator
from core.generative_ai_manager import GenerativeAIManager
from core.custom_report_builder import CustomReportBuilder
from core.pdf_report_generator import PDFReportGenerator
from core.security import SecurityManager
from core.rate_limiter import RateLimitMiddleware, get_rate_limiter
from core.auth_metrics import get_auth_metrics
from core.dashboard_creator import DashboardCreator
from core.compliance_manager import ComplianceManager, get_compliance_manager
from core.redis_client import RedisClient
from core.auth_manager import AuthManager, get_auth_manager
from core.anonymization import AnonymizationService
from core.config import get_settings
from core.utils import PlotUtility, StatisticsUtility
from core.scalability import FirestoreScalabilityService, get_scalability_service
from core.security_config import get_secret_key
from core.data_input import GoogleFormsConnector, ExternalDataFetcher, DataInputError # 共通モジュールからインポート
from core.subscription_manager import SubscriptionManager, get_subscription_manager, SubscriptionManagerError
from core.feature_engineer import FeatureEngineer # 分析の前処理/特徴量生成に近いのでcoreに残す判断もあり

# Analysisモジュールのインポート
from analysis.correlation_analysis import CorrelationAnalyzer
from analysis.ClusterAnalyzer import ClusterAnalyzer
from analysis.TimeSeriesAnalyzer import TimeSeriesAnalyzer
from analysis.SurvivalAnalyzer import SurvivalAnalyzer
from analysis.PredictiveModelAnalyzer import PredictiveModelAnalyzer # core.performance_predictor に相当と仮定
from analysis.StartupSurvivabilityAnalyzer import StartupSurvivabilityAnalyzer
from analysis.PCAAnalyzer import PCAAnalyzer
from analysis.TextMiner import TextMiner
from analysis.AssociationAnalyzer import AssociationAnalyzer
from analysis.calculate_descriptive_stats import DescriptiveStatsCalculator # core.descriptive_stats_calculator に相当
from analysis.FinancialAnalyzer import FinancialAnalyzer
from analysis.MarketAnalyzer import MarketAnalyzer
from analysis.Team_Analyzer import TeamAnalyzer

# APIルーターのインポート
from api.routers import all_routers

# Import routers and dependencies
from service.firestore.client import FirestoreService, StorageError, ValidationError
from database.connection import get_db

# Neo4jの初期化をインポート
from database.neo4j import init_neo4j, Neo4jService

# Models
class DashboardConfigModel(BaseModel):
    """ダッシュボード設定モデル"""
    title: str
    description: Optional[str] = None
    widgets: List[Dict]
    layout: Dict

class GraphConfig(BaseModel):
    """グラフ設定モデル"""
    type: str
    title: str
    data_source: str
    settings: Dict
    filters: Optional[List[Dict]] = None

class VisualizationResponse(BaseModel):
    """可視化レスポンスモデル"""
    id: str
    created_at: datetime
    updated_at: datetime
    config: Dict
    data: Dict
    created_by: str

class ReportBase(BaseModel):
    """レポートベースモデル"""
    title: str
    description: Optional[str] = None
    report_type: str
    parameters: Optional[Dict] = None

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK
try:
    firebase_app = firebase_admin.get_app()
    logger.info("Firebase SDKはすでに初期化されています")
except ValueError:
    try:
        # 環境変数GOOGLE_APPLICATION_CREDENTIALSが設定されているか確認
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and os.path.exists(creds_path):
            logger.info(f"GOOGLE_APPLICATION_CREDENTIALSから認証情報を使用します: {creds_path}")
            cred = credentials.Certificate(creds_path)
            firebase_app = firebase_admin.initialize_app(cred)
        else:
            # 環境変数が設定されていないか、ファイルが存在しない場合
            logger.warning(f"認証情報ファイルが見つかりません: {creds_path}")
            # credentialsディレクトリ内の利用可能なJSONファイルを探す
            cred_dir = os.path.join(os.path.dirname(__file__), "credentials")
            if os.path.exists(cred_dir):
                cred_files = [f for f in os.listdir(cred_dir) if f.endswith('.json')]
                if cred_files:
                    cred_path = os.path.join(cred_dir, cred_files[0])
                    logger.info(f"認証情報ファイルを自動検出しました: {cred_path}")
                    cred = credentials.Certificate(cred_path)
                    firebase_app = firebase_admin.initialize_app(cred)
                else:
                    logger.warning("認証情報ファイルが見つかりません。デフォルト認証情報を試みます。")
                    firebase_app = firebase_admin.initialize_app()
            else:
                logger.warning("認証情報ディレクトリが見つかりません。デフォルト認証情報を試みます。")
                firebase_app = firebase_admin.initialize_app()
    except Exception as e:
        logger.error(f"Firebase初期化エラー: {str(e)}")
        logger.warning("Firebase機能は無効になります。アプリは制限された機能で動作します。")

# Initialize services
firestore_service = FirestoreService()

# Core/Analysisサービスの初期化
try:
    # Core Services
    auth_manager = get_auth_manager(firestore_service=firestore_service)
    data_preprocessor = DataPreprocessor()
    data_quality_checker = DataQualityChecker(firestore_service=firestore_service, generative_ai_manager=None) # Placeholder for AI manager
    compliance_manager = get_compliance_manager(firestore_service=firestore_service, generative_ai_manager=None) # Placeholder
    subscription_manager = get_subscription_manager(firestore_service=firestore_service, auth_manager=auth_manager)
    external_data_fetcher = ExternalDataFetcher()
    graph_generator = GraphGenerator()
    interactive_visualizer = InteractiveVisualizer(db=firestore_service)
    redis_client = RedisClient()
    dashboard_creator = DashboardCreator(config=DashboardConfigModel(title="Dummy", widgets=[], layout={}), db=firestore_service) # Dummy config
    feature_engineer = FeatureEngineer(db=firestore_service)
    model_evaluator = ModelEvaluator()
    security_manager = SecurityManager()
    auth_metrics = get_auth_metrics()
    rate_limiter = get_rate_limiter()
    google_forms_connector = GoogleFormsConnector() # Initialization might need adjustments
    plot_utility = PlotUtility()
    statistics_utility = StatisticsUtility()
    scalability_service = get_scalability_service()

    # Generative AI Manager (Optional)
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key:
        try:
            generative_ai_manager = GenerativeAIManager(api_key=gemini_api_key)
            # Inject into services that use it
            data_quality_checker.generative_ai_manager = generative_ai_manager
            compliance_manager.generative_ai_manager = generative_ai_manager
        except Exception as e:
            logger.error(f"GenerativeAIManager初期化エラー: {e}")
            generative_ai_manager = None
    else:
        generative_ai_manager = None
        logger.warning("環境変数 GEMINI_API_KEY が設定されていません。GenerativeAIManagerは利用できません。")

    # Reporting Services (Optional)
    try:
        custom_report_builder = CustomReportBuilder()
    except Exception as e:
        logger.warning(f"CustomReportBuilder初期化エラー: {e}. 機能は利用できません。")
        custom_report_builder = None
    try:
        pdf_report_generator = PDFReportGenerator()
    except Exception as e:
        logger.warning(f"PDFReportGenerator初期化エラー: {e}. 機能は利用できません。")
        pdf_report_generator = None

    # Analysis Services (using analysis directory)
    # BigQueryServiceのインスタンスを作成（必要に応じて）
    try:
        from service.bigquery.client import BigQueryService
        bq_service = BigQueryService()
        # CorrelationAnalyzer は BigQueryService を必要とする
        correlation_analyzer = CorrelationAnalyzer(bq_service=bq_service)
    except ImportError:
        logger.warning("BigQueryServiceをインポートできませんでした。一部の分析機能が制限されます。")
        correlation_analyzer = None
    except Exception as e:
        logger.error(f"CorrelationAnalyzer初期化エラー: {e}")
        correlation_analyzer = None

    # 他のアナライザーの初期化
    try:
        cluster_analyzer = ClusterAnalyzer() # 引数が必要な場合は適宜追加
    except Exception as e:
        logger.error(f"ClusterAnalyzer初期化エラー: {e}")
        cluster_analyzer = None

    try:
        time_series_analyzer = TimeSeriesAnalyzer(db=firestore_service)
    except Exception as e:
        logger.error(f"TimeSeriesAnalyzer初期化エラー: {e}")
        time_series_analyzer = None

    try:
        survival_analyzer = SurvivalAnalyzer()
    except Exception as e:
        logger.error(f"SurvivalAnalyzer初期化エラー: {e}")
        survival_analyzer = None

    try:
        performance_predictor = PredictiveModelAnalyzer()
    except Exception as e:
        logger.error(f"PredictiveModelAnalyzer初期化エラー: {e}")
        performance_predictor = None

    try:
        startup_survival_analyzer = StartupSurvivabilityAnalyzer()
    except Exception as e:
        logger.error(f"StartupSurvivabilityAnalyzer初期化エラー: {e}")
        startup_survival_analyzer = None

    try:
        pca_analyzer = PCAAnalyzer()
    except Exception as e:
        logger.error(f"PCAAnalyzer初期化エラー: {e}")
        pca_analyzer = None

    try:
        association_analyzer = AssociationAnalyzer()
    except Exception as e:
        logger.error(f"AssociationAnalyzer初期化エラー: {e}")
        association_analyzer = None

    try:
        descriptive_stats_calculator = DescriptiveStatsCalculator(db=firestore_service)
    except Exception as e:
        logger.error(f"DescriptiveStatsCalculator初期化エラー: {e}")
        descriptive_stats_calculator = None

    # TextMiner depends on API key
    if gemini_api_key:
        text_miner = TextMiner(db=firestore_service, gemini_api_key=gemini_api_key) # From analysis
    else:
        text_miner = None
        logger.warning("TextMinerはAPIキーがないため初期化されませんでした。")

    # Other specific analyzers (assuming default init or pass db)
    try:
        financial_analyzer = FinancialAnalyzer(db=firestore_service)
        financial_analyzer_available = True
    except Exception as e:
        logger.error(f"FinancialAnalyzer初期化エラー (analysis): {e}")
        financial_analyzer = None
        financial_analyzer_available = False # Keep track if it failed

    try:
        market_analyzer = MarketAnalyzer(db=firestore_service)
        market_analyzer_available = True
    except Exception as e:
        logger.error(f"MarketAnalyzer初期化エラー (analysis): {e}")
        market_analyzer = None
        market_analyzer_available = False

    try:
        team_analyzer = TeamAnalyzer(db=firestore_service)
        team_analyzer_available = True
    except Exception as e:
        logger.error(f"TeamAnalyzer初期化エラー (analysis): {e}")
        team_analyzer = None
        team_analyzer_available = False

    # Wellness Score Calculator needs specific analyzers, ensure they are initialized first
    # Check if required analyzers exist before initializing
    if correlation_analyzer and time_series_analyzer:
        wellness_calculator = WellnessScoreCalculator(
            correlation_analyzer=correlation_analyzer,
            time_series_analyzer=time_series_analyzer,
            firestore_client=firestore_service
        )
    else:
        wellness_calculator = None
        logger.error("WellnessScoreCalculator の依存関係（CorrelationAnalyzer または TimeSeriesAnalyzer）が初期化に失敗したため、初期化できません。")


    # Settings and Security Key (mostly unchanged)
    settings = get_settings()
    secret_key = get_secret_key()
    if secret_key == "your-secret-key-for-development-only":
         logger.warning("開発用のデフォルトシークレットキーが使用されています。本番環境では変更してください。")

    # Neo4j Initialization (unchanged)
    init_neo4j()
    logger.info("Neo4jデータベースの初期化が完了しました")

except Exception as e:
    logger.critical(f"サービス初期化中に致命的なエラー: {str(e)}", exc_info=True)
    raise RuntimeError(f"アプリケーションの初期化に失敗しました: {e}") from e

# Error handling middleware
class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except ValidationError as e:
            logger.error(f"バリデーションエラー: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"detail": str(e)}
            )
        except StorageError as e:
            logger.error(f"ストレージエラー: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": str(e)}
            )
        except Exception as e:
            logger.error(f"予期せぬエラーが発生: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error occurred"}
            )

# FastAPI application
app = FastAPI(
    title="Startup Wellness API",
    description="スタートアップウェルネス分析システム用バックエンドAPI",
    version="1.0.0"
)

# Add middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# レート制限ミドルウェアを追加
app.add_middleware(
    RateLimitMiddleware,
    whitelist_paths=["/health", "/metrics", "/webhooks/payjp"],
    whitelist_ips=[]  # 必要に応じて信頼できるIPを追加
)

# Background tasks
async def process_analysis_data(
    analysis_id: str,
    data: Dict,
    analysis_type: str,
    user_id: str,
    background_tasks: BackgroundTasks
) -> None:
    """バックグラウンドで分析データを処理"""
    # Note: Ensure analyzers are available before calling analyze
    try:
        processed_data = await data_preprocessor.preprocess(data)
        quality_report = await data_quality_checker.check(processed_data)

        analysis_results = {}
        # Use analyzers from 'analysis' package
        if analysis_type == "wellness_score" and wellness_calculator:
            analysis_results = await wellness_calculator.calculate(processed_data) # Stays as is, depends on others
        elif analysis_type == "correlation" and correlation_analyzer:
            analysis_results = await correlation_analyzer.analyze(processed_data)
        elif analysis_type == "cluster" and cluster_analyzer:
            analysis_results = await cluster_analyzer.analyze(processed_data)
        elif analysis_type == "time_series" and time_series_analyzer:
            analysis_results = await time_series_analyzer.analyze(processed_data)
        elif analysis_type == "survival" and survival_analyzer:
            analysis_results = await survival_analyzer.analyze(processed_data)
        elif analysis_type == "startup_survival" and startup_survival_analyzer:
            analysis_results = await startup_survival_analyzer.analyze(processed_data)
        elif analysis_type == "pca" and pca_analyzer:
            analysis_results = await pca_analyzer.analyze(processed_data)
        elif analysis_type == "association" and association_analyzer: # Added association
             analysis_results = await association_analyzer.find_rules(processed_data) # Assuming find_rules method
        elif analysis_type == "text_mining" and text_miner: # Added text mining
             analysis_results = await text_miner.analyze_dataframe(processed_data) # Assuming analyze_dataframe method
        # Add other analysis types if needed (financial, market, team)

        result_data = {
            'id': analysis_id,
            'data': {
                'source': data,
                'processed': processed_data,
                'quality_report': quality_report,
                'analysis': analysis_results,
                'last_updated': datetime.now()
            },
            'status': 'completed'
        }

        if model_evaluator:
            background_tasks.add_task(
                model_evaluator.evaluate,
                analysis_results,
                analysis_type
            )

        return result_data

    except Exception as e:
        logger.error(f"分析処理エラー (ID: {analysis_id}, Type: {analysis_type}): {str(e)}", exc_info=True)
        # Return error status, DO NOT save partial/error results to DB here
        # The status should be updated where the task was initiated or monitored
        # This function should perhaps raise the exception to be handled by the caller/task runner
        raise e # Re-raise the exception to be caught by the task runner

# API Router設定 - すべてのルーターを一括登録
for router in all_routers:
    app.include_router(router)

# 廃止されたルーターへのリダイレクト警告
@app.get("/api/v1/visualization", include_in_schema=False)
async def deprecated_visualization_redirect():
    """
    廃止されたvisualizationエンドポイントへのリダイレクト
    """
    logger.warning("廃止されたvisualizationエンドポイントにアクセスがありました。新しいAPIエンドポイント /api/visualization を使用してください。")
    raise HTTPException(
        status_code=status.HTTP_301_MOVED_PERMANENTLY,
        detail="このエンドポイントは廃止されました。新しいAPIエンドポイント /api/visualization を使用してください。"
    )

@app.get("/api/v1/reports", include_in_schema=False)
async def deprecated_reports_redirect():
    """
    廃止されたreportsエンドポイントへのリダイレクト
    """
    logger.warning("廃止されたreportsエンドポイントにアクセスがありました。新しいAPIエンドポイントへ更新を推奨します。")
    raise HTTPException(
        status_code=status.HTTP_301_MOVED_PERMANENTLY,
        detail="このエンドポイントは廃止されました。新しいAPIエンドポイント /api/reports を使用してください。"
    )

# OAuth2スキームの定義
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # tokenUrlは実際のトークン取得エンドポイントに合わせる

# Firebase IDトークンを検証し、ユーザー情報を返す依存性注入関数
async def get_current_firebase_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Authorization: Bearer <Firebase ID Token> ヘッダーからトークンを検証し、
    Firebaseユーザー情報を返す。
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="認証トークンが提供されていません",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        # Firebase Admin SDKを使用してトークンを検証
        decoded_token = auth.verify_id_token(token)
        # decoded_token には uid, email, name などが含まれる
        logger.debug(f"Firebaseトークン検証成功: UID={decoded_token.get('uid')}")
        return decoded_token
    except auth.ExpiredIdTokenError:
        logger.warning("Firebase IDトークンが期限切れです")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="トークンが期限切れです",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except auth.InvalidIdTokenError as e:
        logger.error(f"無効なFirebase IDトークンです: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"無効なトークンです: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"トークン検証中に予期せぬエラー: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="トークン検証中にエラーが発生しました",
            headers={"WWW-Authenticate": "Bearer"},
        )

# APIルート定義
@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    return {"message": f"スタートアップウェルネス分析 API - バージョン {VERSION}"}

@app.get("/api/subscription/plans")
async def get_available_plans(current_user: Dict[str, Any] = Depends(get_current_firebase_user)):
    """利用可能なサブスクリプションプラン一覧を取得"""
    user_id = current_user.get("uid") # uidを取得
    if not user_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ユーザーIDがトークンに含まれていません")
    try:
        plans_raw = await subscription_manager.list_plans()
        # フロントエンドが必要とする形式に整形
        plans = [
            {
                "id": plan.id,
                "name": plan.metadata.get("name_ja", plan.nickname or plan.id), # 日本語名があれば優先
                "price": plan.amount,
                "currency": plan.currency,
                "interval": plan.interval,
                "features": plan.metadata.get("features", "").split(","), # メタデータから取得
                # 必要に応じて他のメタデータを追加
                 "maxCompanies": int(plan.metadata.get("max_companies", 1)),
                 "maxEmployees": int(plan.metadata.get("max_employees", 100)),
            }
            for plan in plans_raw
        ]
        return {"plans": plans}
    except SubscriptionManagerError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"プラン取得エラー(User: {user_id}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="プランの取得に失敗しました")

@app.get("/api/subscription/status")
async def get_user_subscription_status(current_user: Dict[str, Any] = Depends(get_current_firebase_user)):
    """ユーザーの現在のサブスクリプション状態を取得"""
    user_id = current_user.get("uid")
    if not user_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ユーザーIDがトークンに含まれていません")
    try:
        status_details = await subscription_manager.get_subscription_details(user_id)
        if status_details:
            return {"status": status_details}
        else:
            # ユーザーが見つからない場合やエラーの場合
            return {"status": {"status": "inactive", "source": "not_found"}} # デフォルトの非アクティブ状態
    except SubscriptionManagerError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"サブスクリプション状態取得エラー(User: {user_id}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="状態の取得に失敗しました")

class ChangePlanRequest(BaseModel):
    plan_id: str
    payment_method_id: Optional[str] = None # 新規登録や支払い方法変更時に必要

@app.post("/api/subscription/change")
async def change_user_plan(request_body: ChangePlanRequest, current_user: Dict[str, Any] = Depends(get_current_firebase_user)):
    """ユーザーのサブスクリプションプランを変更（または新規作成）"""
    user_id = current_user.get("uid")
    user_email = current_user.get("email") # 顧客作成用にEmailも取得
    if not user_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ユーザーIDがトークンに含まれていません")
    if not user_email:
         logger.warning(f"ユーザー {user_id} のEmailがトークンに含まれていません。顧客作成に失敗する可能性があります。")
         # Emailがない場合の代替処理が必要か検討 (例: Firestoreから取得)

    try:
        # 顧客取得/作成時にEmailを渡すように修正が必要かも
        # subscription_manager.get_or_create_customer(user_id, user_email) を内部で呼ぶ想定

        subscription = await subscription_manager.create_or_update_subscription(
            user_id=user_id,
            plan_id=request_body.plan_id,
            payment_method_id=request_body.payment_method_id
        )
        updated_status = await subscription_manager.get_subscription_details(user_id)
        return {"message": "プランが正常に変更されました", "status": updated_status}
    except SubscriptionManagerError as e:
        if "支払い方法IDが必要" in str(e):
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        elif "Emailが見つかりません" in str(e): # Email不足エラーを追加
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ユーザーEmail情報が必要です。")
        else:
             raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"プラン変更エラー(User: {user_id}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="プランの変更に失敗しました")

class CancelSubscriptionRequest(BaseModel):
    at_period_end: bool = True # デフォルトは期間終了時キャンセル

@app.post("/api/subscription/cancel")
async def cancel_user_subscription(request_body: CancelSubscriptionRequest, current_user: Dict[str, Any] = Depends(get_current_firebase_user)):
    """ユーザーのサブスクリプションをキャンセル"""
    user_id = current_user.get("uid")
    if not user_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ユーザーIDがトークンに含まれていません")
    try:
        subscription = await subscription_manager.cancel_subscription(
            user_id=user_id,
            at_period_end=request_body.at_period_end
        )
        updated_status = await subscription_manager.get_subscription_details(user_id)
        return {"message": "サブスクリプションのキャンセルを受け付けました", "status": updated_status}
    except SubscriptionManagerError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"サブスクリプションキャンセルエラー(User: {user_id}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="キャンセル処理に失敗しました")

# --- Pay.jp Webhook Endpoint ---
@app.post("/webhooks/payjp", include_in_schema=False) # スキーマには含めない
async def payjp_webhook_handler(request: Request, payjp_signature: Optional[str] = Header(None)):
    """Pay.jpからのWebhookイベントを受信して処理する"""
    payload = await request.body()
    if not payjp_signature:
        logger.error("WebhookリクエストにPayjp-Signatureヘッダーがありません")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Signature header missing")

    logger.info(f"Pay.jp Webhook受信開始 (Signature: {payjp_signature[:10]}...)")
    try:
        success = await subscription_manager.handle_webhook(payload, payjp_signature)
        if success:
            logger.info("Pay.jp Webhook処理成功")
            return Response(status_code=status.HTTP_200_OK)
        else:
            logger.warning("Pay.jp Webhook処理失敗 (検証エラーまたは内部エラー)")
            # Pay.jpは2xx以外で再送を試みる。意図しない再送ループを避けるため、
            # 検証エラーなど再送しても無駄な場合は200を返すことも検討。
            # ここでは処理失敗を一律500とする。
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Webhook processing failed")
    except Exception as e:
        # SubscriptionManager内でキャッチされなかった予期せぬエラー
        logger.error(f"Webhookハンドラーで予期せぬエラー: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during webhook handling")

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {} # user_id: websocket
        self.user_info: Dict[str, Dict[str, Any]] = {} # user_id: firebase_user_info

    async def connect(self, websocket: WebSocket, user_id: str, user_info: Dict[str, Any]):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_info[user_id] = user_info
        logger.info(f"WebSocket connected: User {user_id}")

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            del self.user_info[user_id]
            logger.info(f"WebSocket disconnected: User {user_id}")

    async def send_personal_message(self, message: Dict, user_id: str):
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to user {user_id}: {e}")
                # Optionally disconnect on send error
                # self.disconnect(user_id)

    async def broadcast(self, message: str): # 必要であれば実装
        for user_id in self.active_connections:
            await self.send_personal_message({"type": "broadcast", "message": message}, user_id)

    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self.user_info.get(user_id)

manager = ConnectionManager()

# --- WebSocket Message Handler ---
async def handle_websocket_message(websocket: WebSocket, user_id: str, data: Dict):
    """WebSocketメッセージを処理する"""
    message_type = data.get("type")
    payload = data.get("payload", {})
    logger.info(f"Received message type: {message_type} from user: {user_id}")

    try:
        if message_type == "get_dashboard_data":
            # dashboard_creatorからデータを取得 (非同期を想定)
            # 実際のデータ取得ロジックに置き換える
            dashboard_data = await dashboard_creator.get_dashboard_data(user_id, payload.get("period"))
            await manager.send_personal_message({"type": "dashboard_data", "data": dashboard_data}, user_id)

        elif message_type == "get_company_data":
            company_id = payload.get("company_id")
            if company_id:
                # 企業データを取得するロジック (非同期を想定)
                # 例: firestore_serviceから取得
                company_data = await firestore_service.get_document("companies", company_id) # 仮
                # 分析データも取得して組み合わせる必要がある
                # employee_metrics = await ...
                # cluster_data = await cluster_analyzer.analyze_for_company(company_id) # 仮
                # ... 他の分析データも同様に取得 ...

                response_data = {
                    "companyInfo": company_data, # 仮
                    # "employeeMetrics": employee_metrics,
                    # "clusterData": cluster_data,
                    # ...
                }
                await manager.send_personal_message({"type": "company_data", "data": response_data}, user_id)
            else:
                 await manager.send_personal_message({"type": "error", "message": "Company ID is required"}, user_id)

        elif message_type == "get_compliance_report":
            company_id = payload.get("company_id")
            # ComplianceManagerからレポート取得 (非同期を想定)
            report = await compliance_manager.get_compliance_report(user_id, company_id)
            await manager.send_personal_message({"type": "compliance_report", "report": report}, user_id)

        elif message_type == "get_compliance_config":
            company_id = payload.get("company_id")
            config = await compliance_manager.get_compliance_config(user_id, company_id)
            await manager.send_personal_message({"type": "compliance_config", "config": config}, user_id)

        elif message_type == "update_compliance_config":
            company_id = payload.get("company_id")
            config_data = payload.get("config")
            await compliance_manager.update_compliance_config(user_id, company_id, config_data)
            await manager.send_personal_message({"type": "config_updated"}, user_id)
            # 更新後の設定を再送してもよい
            # updated_config = await compliance_manager.get_compliance_config(user_id, company_id)
            # await manager.send_personal_message({"type": "compliance_config", "config": updated_config}, user_id)

        elif message_type == "run_compliance_check":
            company_id = payload.get("company_id")
            result_report = await compliance_manager.run_check(user_id, company_id)
            await manager.send_personal_message({"type": "compliance_report", "report": result_report}, user_id)

        elif message_type == "remediate_violation":
            # payload から必要な情報を取得
            # await compliance_manager.remediate_violation(...)
            # 更新後のレポートを送信
            pass # 実装

        elif message_type == "get_quality_report":
            company_id = payload.get("company_id")
            report = await data_quality_checker.get_quality_report(company_id) # user_idも必要？
            await manager.send_personal_message({"type": "quality_report", "report": report}, user_id)

        elif message_type == "get_quality_config":
            company_id = payload.get("company_id")
            config = await data_quality_checker.get_quality_config(company_id)
            await manager.send_personal_message({"type": "quality_config", "config": config}, user_id)

        elif message_type == "update_quality_config":
            company_id = payload.get("company_id")
            config_data = payload.get("config")
            await data_quality_checker.update_quality_config(company_id, config_data)
            await manager.send_personal_message({"type": "config_updated"}, user_id)

        elif message_type == "run_quality_check":
            company_id = payload.get("company_id")
            # データが必要かもしれない
            # data = await firestore_service.get_data_for_company(company_id) # 仮
            # report = await data_quality_checker.check(data)
            report = {} # 仮
            await manager.send_personal_message({"type": "quality_report", "report": report}, user_id)

        elif message_type == "run_auto_fix":
            company_id = payload.get("company_id")
            # 修正処理を実行し、結果のレポートを返す
            # report = await data_quality_checker.auto_fix(company_id) # 仮
            report = {} # 仮
            await manager.send_personal_message({"type": "quality_report", "report": report}, user_id)


        elif message_type == "generate_report":
            report_id = payload.get("report_id")
            company_id = payload.get("company_id")
            config = payload.get("config")
            # レポート生成をバックグラウンドタスクで開始
            # background_tasks.add_task(generate_report_task, report_id, user_id, company_id, config)
            await manager.send_personal_message({"type": "report_status", "id": report_id, "status": "processing"}, user_id)
            # 実際の生成処理を実装

        elif message_type == "check_report_status":
            report_id = payload.get("report_id")
            # report_id に紐づく状態を取得 (DBなどから)
            status_data = {"id": report_id, "status": "completed", "url": "/path/to/report.pdf"} # 仮
            await manager.send_personal_message({"type": "report_status", **status_data}, user_id)

        elif message_type == "cancel_report":
            report_id = payload.get("report_id")
            # report_id の生成タスクをキャンセル
            logger.info(f"Report cancellation requested for {report_id} (implementation needed)")
            await manager.send_personal_message({"type": "report_cancelled", "id": report_id}, user_id)

        elif message_type == "get_user_settings":
            # user_id に紐づく設定を取得
            settings = await auth_manager.get_user_settings(user_id) # AuthManagerに関数を追加想定
            await manager.send_personal_message({"type": "user_settings", "settings": settings}, user_id)

        elif message_type == "get_system_settings":
            # システム全体の設定を取得 (管理者権限が必要かも)
            # settings = await auth_manager.get_system_settings() # AuthManagerに関数を追加想定
            settings = {"version": VERSION, "maintenance": False} # 仮
            await manager.send_personal_message({"type": "system_settings", "settings": settings}, user_id)

        elif message_type == "update_user_settings":
            settings_data = payload.get("settings")
            await auth_manager.update_user_settings(user_id, settings_data) # AuthManagerに関数を追加想定
            await manager.send_personal_message({"type": "settings_updated"}, user_id)

        elif message_type == "reset_user_settings":
            await auth_manager.reset_user_settings(user_id) # AuthManagerに関数を追加想定
            # 更新後の設定を送信
            settings = await auth_manager.get_user_settings(user_id)
            await manager.send_personal_message({"type": "user_settings", "settings": settings}, user_id)

        elif message_type == "get_text_mining_data":
            company_id = payload.get("company_id")
            text_source = payload.get("text_source", "all")
            filters = payload.get("filters", {})
            if text_miner:
                # text_miner からデータを取得 (非同期を想定)
                # data = await text_miner.analyze(company_id, text_source, filters)
                data = {"sentiment": {"positive": 0.7, "neutral": 0.2, "negative": 0.1}, "keywords": []} # Placeholder
                await manager.send_personal_message({"type": "text_mining_data", "data": data}, user_id)
            else:
                await manager.send_personal_message({"type": "error", "message": "Text mining service not available"}, user_id)


        elif message_type == "analyze_custom_text":
            text = payload.get("text")
            request_id = payload.get("request_id")
            if text_miner and text:
                 # analysis_result = await text_miner.analyze_single_text(text)
                 analysis_result = {"sentiment": {"positive": 0.9, "neutral": 0.1, "negative": 0.0}} # 仮
                 await manager.send_personal_message({"type": "text_analysis_result", "request_id": request_id, "data": analysis_result}, user_id)
            elif not text_miner:
                 await manager.send_personal_message({"type": "error", "message": "Text mining service not available", "request_id": request_id}, user_id)
            else:
                 await manager.send_personal_message({"type": "error", "message": "Text is required", "request_id": request_id}, user_id)

        elif message_type == "get_analysis_data":
            company_id = payload.get("company_id")
            filters = payload.get("filters", {})
            # 各アナライザーを呼び出してデータを取得・集約 (非同期)
            # correlation = await correlation_analyzer.analyze(...)
            # cluster = await cluster_analyzer.analyze(...)
            # ...
            analysis_data = {
                # "correlationData": correlation,
                # "clusterData": cluster,
                # ...
            }
            await manager.send_personal_message({"type": "analysis_data", "data": analysis_data}, user_id)

        else:
            logger.warning(f"Unknown message type received: {message_type} from user: {user_id}")
            await manager.send_personal_message({"type": "error", "message": f"Unknown message type: {message_type}"}, user_id)

    except Exception as e:
        logger.error(f"Error handling message type {message_type} for user {user_id}: {e}", exc_info=True)
        await manager.send_personal_message({"type": "error", "message": f"Error processing message: {message_type}"}, user_id)


# --- WebSocket Endpoint ---
@app.websocket("/ws/{path:path}")
async def websocket_endpoint(websocket: WebSocket, path: str):
    """
    WebSocket接続エンドポイント
    - 接続時に認証トークンを要求
    - メッセージタイプに応じて処理を振り分け
    """
    user_id: Optional[str] = None
    try:
        # 最初のメッセージは認証トークンを期待
        init_data = await websocket.receive_json()
        if init_data.get("type") == "init" and "token" in init_data:
            token = init_data["token"]
            try:
                # HTTP用の関数を流用してトークン検証
                # Note: Depends(oauth2_scheme) の代わりにトークンを直接渡す
                decoded_token = await get_current_firebase_user(token=token)
                user_id = decoded_token.get("uid")
                if user_id:
                    await manager.connect(websocket, user_id, decoded_token)
                    # 接続成功メッセージを返すなど（任意）
                    await manager.send_personal_message({"type": "connected", "user_id": user_id}, user_id)

                    # 認証後のメッセージループ
                    while True:
                        data = await websocket.receive_json()
                        await handle_websocket_message(websocket, user_id, data)
                else:
                    logger.warning(f"WebSocket connection attempt failed: UID not found in token for path {path}")
                    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                    return

            except HTTPException as e: # トークン検証エラー
                logger.warning(f"WebSocket connection failed due to auth error: {e.detail} for path {path}")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=e.detail)
                return
            except Exception as e: # その他の認証エラー
                logger.error(f"Unexpected error during WebSocket authentication for path {path}: {e}", exc_info=True)
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Authentication error")
                return

        else:
            logger.warning(f"WebSocket connection attempt failed: Invalid init message for path {path}")
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            return

    except WebSocketDisconnect:
        if user_id:
            manager.disconnect(user_id)
        else:
            logger.info(f"WebSocket disconnected before authentication for path {path}")
    except Exception as e:
        logger.error(f"Unexpected WebSocket error for path {path} (User: {user_id}): {e}", exc_info=True)
        if user_id:
            manager.disconnect(user_id)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    reload = os.environ.get("RELOAD", "false").lower() == "true"

    logger.info(f"サーバーを起動します - Host: {host}, Port: {port}, Reload: {reload}")
    try:
        uvicorn.run("main:app", host=host, port=port, reload=reload, ws_max_size=1024 * 1024 * 16) # WebSocketメッセージサイズ上限を増やす例
    except Exception as e:
        logger.critical(f"サーバー起動エラー: {str(e)}", exc_info=True)