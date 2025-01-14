"""
Startup Wellness データ分析システム バックエンド API

このファイルはアプリケーションのエントリーポイントとして機能し、
アプリケーションの起動と初期設定を担当します。
"""

import logging
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import configuration
from backend.app.core.config import settings

# Import routers individually to ensure proper loading
from backend.api.routers.auth import router as auth_router
from backend.api.routers.visualization import router as visualization_router
from backend.api.routers.analysis import router as analysis_router
from backend.api.routers.data_input import router as data_input_router
from backend.api.routers.data_processing import router as data_processing_router
from backend.api.routers.prediction import router as prediction_router
from backend.api.routers.report_generation import router as report_generation_router

# Initialize logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, initialize_app

try:
    firebase_app = firebase_admin.get_app()
except ValueError:
    try:
        cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
        firebase_app = initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {str(e)}")
        raise

# FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="データ分析システム用バックエンドAPI",
    version=settings.VERSION,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with proper error handling
try:
    # 各ルーターを個別に登録
    app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
    app.include_router(visualization_router, prefix="/api/visualization", tags=["visualization"])
    app.include_router(analysis_router, prefix="/api/analysis", tags=["analysis"])
    app.include_router(data_input_router, prefix="/api/data_input", tags=["data_input"])
    app.include_router(data_processing_router, prefix="/api/data_processing", tags=["data_processing"])
    app.include_router(prediction_router, prefix="/api/prediction", tags=["prediction"])
    app.include_router(report_generation_router, prefix="/api/report_generation", tags=["report_generation"])
    logger.info("All routers initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize routers: {str(e)}")
    raise

# Health check endpoint
@app.get("/health")
async def health_check():
    """APIの稼働状態を確認"""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)