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
from analysis.BayesianInferenceAnalyzer import BayesianInferenceAnalyzer

# APIルーターとミドルウェアのインポート
from api.routers import all_routers
from api.middleware import setup_app
from api.dependencies import ServiceProvider, service_provider

# Import routers and dependencies
from service.firestore.client import FirestoreService, StorageError, ValidationError
from database.connection import get_db, init_db, get_neo4j_driver

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

    # BayesianInferenceAnalyzerの初期化
    try:
        bayesian_inference_analyzer = BayesianInferenceAnalyzer(firestore_client=firestore_service)
        bayesian_inference_analyzer_available = True
    except Exception as e:
        logger.error(f"BayesianInferenceAnalyzer初期化エラー: {e}")
        bayesian_inference_analyzer = None
        bayesian_inference_analyzer_available = False

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

    # 作成したサービスを依存性注入システムに登録
    # ServiceProviderクラスを介してサービスを登録
    service_provider.register_service("firestore_service", FirestoreService)
    service_provider.register_service("auth_manager", auth_manager)
    service_provider.register_service("data_preprocessor", data_preprocessor)
    service_provider.register_service("data_quality_checker", data_quality_checker)
    service_provider.register_service("generative_ai_manager", generative_ai_manager)
    service_provider.register_service("custom_report_builder", custom_report_builder)
    service_provider.register_service("pdf_report_generator", pdf_report_generator)
    service_provider.register_service("compliance_manager", compliance_manager)

    # 必要なアナライザーを登録
    if correlation_analyzer:
        service_provider.register_service("correlation_analyzer", correlation_analyzer)
    if cluster_analyzer:
        service_provider.register_service("cluster_analyzer", cluster_analyzer)
    if time_series_analyzer:
        service_provider.register_service("time_series_analyzer", time_series_analyzer)
    if survival_analyzer:
        service_provider.register_service("survival_analyzer", survival_analyzer)

    # 初期化時のエラーをロギング
except Exception as e:
    logger.error(f"サービス初期化中にエラーが発生しました: {str(e)}")

# Initialize the FastAPI application
app = FastAPI(
    title="Startup Wellness Analysis API",
    description="Startup企業の健全性を分析するためのAPI",
    version=VERSION
)

# CORSの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 実際の環境では適切に制限すること
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ミドルウェアのセットアップ
setup_app(app)  # api.middleware.setup_appを使用

# routersディレクトリからルーターを登録
for router in all_routers:
    app.include_router(router)

# 後方互換性のためにroutesディレクトリからルーターを登録
try:
    from api.routes.visualization import router as visualization_router
    from api.routes.reports import router as reports_router

    app.include_router(visualization_router)
    app.include_router(reports_router)
    logger.info("後方互換性レイヤー（routes/）が正常に登録されました")
except ImportError as e:
    logger.warning(f"後方互換性レイヤーの登録中にエラーが発生しました: {e}")

# サービスの登録
# 可視化サービス
try:
    from service.visualization.chart_service import VisualizationService
    service_provider.register_service("visualization_service", VisualizationService)
except ImportError:
    logger.warning("VisualizationServiceをインポートできませんでした。可視化機能が制限されます。")

# レポートサービス
try:
    from service.reports.report_service import ReportService
    service_provider.register_service("report_service", ReportService)
except ImportError:
    logger.warning("ReportServiceをインポートできませんでした。レポート機能が制限されます。")

# 会社分析サービス
try:
    from service.analysis.company_analysis_service import CompanyAnalysisService
    service_provider.register_service("company_analysis_service", CompanyAnalysisService)
except ImportError:
    logger.warning("CompanyAnalysisServiceをインポートできませんでした。会社分析機能が制限されます。")

# 起動時のNeo4j初期化処理を更新
@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時に実行される処理"""
    # データベース初期化
    try:
        init_db()
        logger.info("データベースを初期化しました")
    except Exception as e:
        logger.error(f"データベース初期化中にエラーが発生しました: {str(e)}")

# アプリケーションの起動（直接実行された場合のみ）
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)