"""
Startup Wellness データ分析システム バックエンド API

要件定義書と requirements.txt を元に作成された FastAPI アプリケーションです。
"""

import logging
import os
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import List
import uuid
from sqlalchemy.orm import Session

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, initialize_app

# Import routers and dependencies
from api.routers import auth, data_input, analysis, visualization, data_processing, prediction, report_generation
from service.firestore.client import FirestoreService, StorageError, ValidationError
from service.tasks import process_visualization_data
from schemas import DashboardConfig, GraphConfig, VisualizationResponse, CompanyCreate, Company
from database.models import Company as CompanyModel
from database import get_db, Base, engine

# Initialize database
Base.metadata.create_all(bind=engine)

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
    cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not cred_path or not os.path.exists(cred_path):
        logger.error(f"Firebase credentials file not found at {cred_path}")
        raise FileNotFoundError(f"Firebase credentials file not found at {cred_path}")

    try:
        cred = credentials.Certificate(cred_path)
        firebase_app = initialize_app(cred)
        logger.info("Firebase initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {str(e)}")
        raise

# Initialize services
try:
    firestore_service = FirestoreService()
    logger.info("Firestore service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Firestore service: {str(e)}")
    raise

# Error handling middleware
class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
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

# Include routers with prefix and tags
app.include_router(auth.router)
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(data_input.router, prefix="/data_input", tags=["data_input"])
app.include_router(data_processing.router, prefix="/data_processing", tags=["data_processing"])
app.include_router(prediction.router, prefix="/prediction", tags=["prediction"])
app.include_router(report_generation.router, prefix="/report_generation", tags=["report_generation"])
app.include_router(visualization.router, prefix="/visualization", tags=["visualization"])

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

        await firestore_service.save_results(
            results=[dashboard_data],
            collection_name='visualizations'
        )

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

        await firestore_service.save_results(
            results=[graph_data],
            collection_name='visualizations'
        )

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

router = APIRouter()

@router.post("/companies", response_model=Company)
async def create_company(company: CompanyCreate, db: Session = Depends(get_db)):
    db_company = CompanyModel(
        id=str(uuid.uuid4()),
        **company.dict()
    )
    try:
        db.add(db_company)
        db.commit()
        db.refresh(db_company)
        return db_company
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/companies", response_model=List[Company])
async def list_companies(db: Session = Depends(get_db)):
    companies = db.query(CompanyModel).all()
    return companies

@router.get("/companies/{company_id}", response_model=Company)
async def get_company(company_id: str, db: Session = Depends(get_db)):
    company = db.query(CompanyModel).filter(CompanyModel.id == company_id).first()
    if company is None:
        raise HTTPException(status_code=404, detail="Company not found")
    return company

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)