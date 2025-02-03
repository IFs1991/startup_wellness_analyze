from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks

from backend.app.schemas import DashboardConfig, GraphConfig, VisualizationResponse
from backend.routers.auth_router import verify_token as get_current_user
from backend.src.database.firestore.client import FirestoreClient as FirestoreService
from backend.app.services.visualization import BaseVisualizationService

router = APIRouter()


def get_visualization_service() -> BaseVisualizationService:
    """依存性注入: 可視化サービスの取得"""
    return BaseVisualizationService(FirestoreService())


@router.post("/dashboard/create", response_model=VisualizationResponse)
async def create_dashboard(
    config: DashboardConfig,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    visualization_service: BaseVisualizationService = Depends(get_visualization_service)
) -> VisualizationResponse:
    """新しいダッシュボードを作成"""
    try:
        return await visualization_service.create_visualization(
            config=config.model_dump(),
            visualization_type="dashboard",
            user_id=current_user['uid']
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/graph/create", response_model=VisualizationResponse)
async def create_graph(
    config: GraphConfig,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    visualization_service: BaseVisualizationService = Depends(get_visualization_service)
) -> VisualizationResponse:
    """新しいグラフを作成"""
    try:
        return await visualization_service.create_visualization(
            config=config.model_dump(),
            visualization_type="graph",
            user_id=current_user['uid']
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/user", response_model=List[VisualizationResponse])
async def get_user_visualizations(
    current_user: dict = Depends(get_current_user),
    visualization_service: BaseVisualizationService = Depends(get_visualization_service)
) -> List[VisualizationResponse]:
    """ユーザーの可視化一覧を取得"""
    try:
        return await visualization_service.get_user_visualizations(
            user_id=current_user['uid']
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )