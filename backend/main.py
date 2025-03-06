"""
Startup Wellness データ分析システム バックエンド API

このファイルはアプリケーションのエントリーポイントとして機能し、
アプリケーションの起動と初期設定を担当します。
"""

import os
import sys
from pathlib import Path

# プロジェクトルートを追加（より安全な方法）
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Dockerコンテナでの追加パス設定
if os.path.exists('/app/backend'):
    sys.path.insert(0, '/app/backend')
if os.path.exists('/app/backend/database/models'):
    sys.path.insert(0, '/app/backend/database/models')

# 標準ライブラリのインポート
import logging
from datetime import datetime

# サードパーティライブラリのインポート
import uvicorn
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware

# Firestore クライアントのインポート
from src.database.firestore.client import FirestoreClient

# Stripe設定のインポート
from src.payment.stripe_client import StripeClient

# ローカルモジュールのインポート
from routers import (
    auth_router,
    company_router,
    analysis_router,
    report_router,
    visualization_router,
    subscription_router,  # 追加: サブスクリプションルーター
    pricing_router        # 追加: 価格設定ルーター
)

# ミドルウェアのインポート
from middleware.auth_middleware import verify_token, get_current_user
from middleware.subscription_middleware import feature_access_required, subscription_required

# ログディレクトリの設定
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, initialize_app

try:
    firebase_app = firebase_admin.get_app()
except ValueError:
    try:
        cred = credentials.Certificate(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
        firebase_app = initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {str(e)}")
        raise

# Stripe APIキーの設定
stripe_api_key = os.getenv('STRIPE_API_KEY')
stripe_webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
if not stripe_api_key:
    logger.warning("STRIPE_API_KEY environment variable not set")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    アプリケーションの起動時と終了時の処理を管理
    """
    try:
        # Firestoreクライアントの初期化確認
        firestore_client = FirestoreClient()

        # Stripeクライアントの初期化
        if stripe_api_key:
            stripe_client = StripeClient(api_key=stripe_api_key)
            logger.info("Stripeクライアントが初期化されました")

        logger.info("アプリケーションの起動処理が完了しました")
        yield
    except Exception as e:
        logger.error(f"アプリケーションの起動中にエラーが発生しました: {str(e)}")
        raise

# FastAPIアプリケーションの初期化
app = FastAPI(
    title="Startup Wellness Analysis API",
    description="スタートアップのウェルネス分析APIサービス",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    lifespan=lifespan
)

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 認証ミドルウェアの設定
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """
    認証情報を検証するミドルウェア
    """
    # 認証が不要なパスのリスト
    public_paths = [
        "/",
        "/health",
        "/api/auth/login",
        "/api/auth/register",
        "/api/auth/reset-password",
        "/api/v1/docs",
        "/api/v1/redoc",
        "/api/v1/openapi.json",
        "/api/webhook/stripe"  # Stripe Webhookは認証不要
    ]

    # 認証チェックをスキップするパスかどうか
    path = request.url.path
    if any(path.startswith(public_path) for public_path in public_paths):
        return await call_next(request)

    # Firebase認証トークンの検証
    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            raise ValueError("認証トークンがありません")

        # トークンを検証し、ユーザー情報を取得
        user = await verify_token(token)

        # リクエストステートにユーザー情報を設定
        request.state.user = user

        return await call_next(request)
    except Exception as e:
        # 認証エラーをログに記録
        logger.warning(f"認証エラー: {str(e)}")

        # 認証エラーを返す代わりに次のミドルウェアへ通過させる
        # 各ルーターのエンドポイントでDependsを使用して認証を強制することができる
        request.state.user = None
        return await call_next(request)

# Firestoreクライアントの初期化
firestore_client = FirestoreClient()

# Stripeクライアントの初期化
if stripe_api_key:
    stripe_client = StripeClient(api_key=stripe_api_key, webhook_secret=stripe_webhook_secret)

# ルーターの登録
app.include_router(auth_router.router, prefix="/api/auth", tags=["認証"])
app.include_router(company_router.router, prefix="/api/companies", tags=["企業"])
app.include_router(analysis_router.router, prefix="/api/analysis", tags=["分析"])
app.include_router(report_router.router, prefix="/api/reports", tags=["レポート"])
app.include_router(visualization_router.router, prefix="/api/visualizations", tags=["可視化"])

# 追加: 課金関連のルーターを登録
app.include_router(subscription_router.router, prefix="/api/subscriptions", tags=["サブスクリプション"])
app.include_router(pricing_router.router, prefix="/api/pricing", tags=["価格設定"])

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "Welcome to Startup Wellness API",
        "version": "1.0.0",
    }

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# 認証が必要なテストエンドポイント
@app.get("/api/v1/me")
async def get_me(current_user=Depends(get_current_user)):
    """
    現在のユーザー情報を取得するテストエンドポイント
    認証ミドルウェアのテストに使用
    """
    return {"user": current_user}

# 機能アクセス制限テストエンドポイント
@app.get("/api/v1/test-access")
async def test_feature_access(
    feature_access: bool = Depends(feature_access_required(feature="reports"))
):
    """
    機能アクセス制限のテストエンドポイント
    """
    return {"message": "この機能へのアクセスが許可されています"}

# プラン制限テストエンドポイント
@app.get("/api/v1/test-subscription")
async def test_subscription(
    subscription: bool = Depends(subscription_required(allowed_plans=["business", "enterprise"]))
):
    """
    サブスクリプション制限のテストエンドポイント
    """
    return {"message": "このプランではこの機能を利用できます"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )