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

# 標準ライブラリのインポート
import logging
from datetime import datetime

# サードパーティライブラリのインポート
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Firestore クライアントのインポート
from backend.src.database.firestore.client import FirestoreClient

# ローカルモジュールのインポート
from backend.routers import (
    auth_router,
    company_router,
    analysis_router,
    report_router,
    visualization_router
)

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

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    アプリケーションの起動時と終了時の処理を管理
    """
    try:
        # Firestoreクライアントの初期化確認
        firestore_client = FirestoreClient()
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

# Firestoreクライアントの初期化
firestore_client = FirestoreClient()

# ルーターの登録
app.include_router(auth_router.router, prefix="/api/auth", tags=["認証"])
app.include_router(company_router.router, prefix="/api/companies", tags=["企業"])
app.include_router(analysis_router.router, prefix="/api/analysis", tags=["分析"])
app.include_router(report_router.router, prefix="/api/reports", tags=["レポート"])
app.include_router(visualization_router.router, prefix="/api/visualizations", tags=["可視化"])

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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )