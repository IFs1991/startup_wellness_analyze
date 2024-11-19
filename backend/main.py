# -*- coding: utf-8 -*-

"""
Startup Wellness データ分析システム バックエンド API

要件定義書と requirements.txt を元に作成された FastAPI アプリケーションです。
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from io import BytesIO
from datetime import datetime

import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Request, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, initialize_app, auth
from google.cloud import firestore

# Import routers and dependencies
from api.routers import (
    auth, data_input, analysis, visualization,
    data_processing, prediction, report_generation
)
from service.firestore.client import FirestoreService, StorageError, ValidationError
from database.database import get_db

# Models
class DashboardConfig(BaseModel):
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
except ValueError:
    cred = credentials.Certificate("path/to/your/serviceAccount.json")
    firebase_app = initialize_app(cred)

# Initialize services
firestore_service = FirestoreService()

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
    description="データ分析システム用バックエンドAPI",
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

# Background tasks
async def process_visualization_data(
    visualization_id: str,
    config: Dict,
    user_id: str
) -> None:
    """バックグラウンドで可視化データを処理"""
    try:
        # データソースからデータを取得
        data_source_conditions = []
        if config.get('filters'):
            data_source_conditions = [
                {'field': f['field'], 'operator': f['operator'], 'value': f['value']}
                for f in config['filters']
            ]

        source_data = await firestore_service.fetch_documents(
            collection_name=config['data_source'],
            conditions=data_source_conditions
        )

        # データ処理と更新
        processed_data = {
            'id': visualization_id,
            'data': {
                'source': source_data,
                'processed': {},  # 実際のデータ処理ロジックを実装
                'last_updated': datetime.now()
            },
            'status': 'completed'
        }

        await firestore_service.save_results(
            results=[processed_data],
            collection_name='visualizations'
        )

    except Exception as e:
        logger.error(f"Error processing visualization data: {str(e)}")
        error_data = {
            'id': visualization_id,
            'status': 'error',
            'error': str(e),
            'last_updated': datetime.now()
        }
        await firestore_service.save_results(
            results=[error_data],
            collection_name='visualizations'
        )

# Include routers with prefix and tags
app.include_router(auth.router)
app.include_router(
    analysis.router,
    prefix="/api/analysis",
    tags=["analysis"],
)
app.include_router(
    data_input.router,
    prefix="/data_input",
    tags=["data_input"],
)
app.include_router(
    data_processing.router,
    prefix="/data_processing",
    tags=["data_processing"],
)
app.include_router(
    prediction.router,
    prefix="/prediction",
    tags=["prediction"],
)
app.include_router(
    report_generation.router,
    prefix="/report_generation",
    tags=["report_generation"],
)
app.include_router(
    visualization.router,
    prefix="/visualization",
    tags=["visualization"],
)

# API Endpoints
@app.post("/dashboard/create", response_model=VisualizationResponse)
async def create_dashboard(
    config: DashboardConfig,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(auth.get_current_user)
) -> VisualizationResponse:
    """新しいダッシュボードを作成"""
    try:
        dashboard_data = {
            'id': f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'type': 'dashboard',
            'config': config.dict(),
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'created_by': current_user['uid'],
            'data': {},
            'status': 'processing'
        }

        # Firestoreにダッシュボード設定を保存
        await firestore_service.save_results(
            results=[dashboard_data],
            collection_name='visualizations'
        )

        # バックグラウンドでデータ処理を実行
        background_tasks.add_task(
            process_visualization_data,
            dashboard_data['id'],
            config.dict(),
            current_user['uid']
        )

        return VisualizationResponse(**dashboard_data)

    except Exception as e:
        logger.error(f"Error creating dashboard: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/graph/create", response_model=VisualizationResponse)
async def create_graph(
    config: GraphConfig,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(auth.get_current_user)
) -> VisualizationResponse:
    """新しいグラフを作成"""
    try:
        graph_data = {
            'id': f"graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'type': 'graph',
            'config': config.dict(),
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'created_by': current_user['uid'],
            'data': {},
            'status': 'processing'
        }

        # Firestoreにグラフ設定を保存
        await firestore_service.save_results(
            results=[graph_data],
            collection_name='visualizations'
        )

        # バックグラウンドでデータ処理を実行
        background_tasks.add_task(
            process_visualization_data,
            graph_data['id'],
            config.dict(),
            current_user['uid']
        )

        return VisualizationResponse(**graph_data)

    except Exception as e:
        logger.error(f"Error creating graph: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/visualizations/user", response_model=List[VisualizationResponse])
async def get_user_visualizations(
    current_user: dict = Depends(auth.get_current_user)
) -> List[VisualizationResponse]:
    """ユーザーの可視化一覧を取得"""
    try:
        conditions = [
            {'field': 'created_by', 'operator': '==', 'value': current_user['uid']}
        ]

        visualizations = await firestore_service.fetch_documents(
            collection_name='visualizations',
            conditions=conditions
        )

        return [VisualizationResponse(**v) for v in visualizations]

    except Exception as e:
        logger.error(f"Error fetching visualizations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """APIの稼働状態を確認"""
    return {
        "status": "healthy",
        "version": app.version,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)