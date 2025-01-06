"""
Startup Wellness データ分析システム バックエンド API

要件定義書と requirements.txt を元に作成された FastAPI アプリケーションです。
"""

import logging
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import configuration
from backend.app.core.config import settings

# Import routers from the unified location
from backend.api.routers import (
    auth,
    data_input,
    analysis,
    visualization,
    data_processing,
    prediction,
    report_generation
)

# Initialize logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT
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
    version=settings.VERSION
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
    app.include_router(auth(), prefix="/api/auth", tags=["auth"])
    app.include_router(visualization(), prefix="/api/visualization", tags=["visualization"])
    app.include_router(analysis(), prefix="/api/analysis", tags=["analysis"])
    app.include_router(data_input(), prefix="/api/data_input", tags=["data_input"])
    app.include_router(data_processing(), prefix="/api/data_processing", tags=["data_processing"])
    app.include_router(prediction(), prefix="/api/prediction", tags=["prediction"])
    app.include_router(report_generation(), prefix="/api/report_generation", tags=["report_generation"])
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