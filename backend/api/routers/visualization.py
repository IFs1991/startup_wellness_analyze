# -*- coding: utf-8 -*-
"""
可視化 API ルーター
Firestoreを使用したダッシュボード生成、グラフ作成、
インタラクティブな可視化機能を提供します。
"""
# 1. APIRouterのインポート
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime
from service.firestore.client import FirestoreService
from firebase_admin import auth
from .auth import get_current_user

# 2. routerオブジェクトの定義
router = APIRouter(
    prefix="/visualization",
    tags=["visualization"],
    responses={404: {"description": "Not found"}}
)

# 3. サービスの初期化
firestore_service = FirestoreService()

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

@router.post("/dashboard/", response_model=VisualizationResponse)
async def create_dashboard(
    config: DashboardConfig,
    current_user: dict = Depends(get_current_user)
):
    """ダッシュボードを作成します"""
    try:
        dashboard_data = {
            'type': 'dashboard',
            'config': config.dict(),
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'created_by': current_user['uid'],
            'data': {}  # 実際のデータはクエリ時に動的に取得
        }

        # Firestoreにダッシュボード設定を保存
        result = await firestore_service.save_results(
            results=[dashboard_data],
            collection_name='visualizations'
        )

        return VisualizationResponse(**dashboard_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/graph/", response_model=VisualizationResponse)
async def generate_graph(
    config: GraphConfig,
    current_user: dict = Depends(get_current_user)
):
    """グラフを生成します"""
    try:
        # データソースからデータを取得
        data_source_conditions = []
        if config.filters:
            data_source_conditions = [
                {'field': f['field'], 'operator': f['operator'], 'value': f['value']}
                for f in config.filters
            ]

        source_data = await firestore_service.fetch_documents(
            collection_name=config.data_source,
            conditions=data_source_conditions
        )

        graph_data = {
            'type': 'graph',
            'config': config.dict(),
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'created_by': current_user['uid'],
            'data': {
                'source': source_data,
                'processed': {}  # データ処理ロジックをここに実装
            }
        }

        # Firestoreにグラフ設定とデータを保存
        result = await firestore_service.save_results(
            results=[graph_data],
            collection_name='visualizations'
        )

        return VisualizationResponse(**graph_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/visualizations/", response_model=List[VisualizationResponse])
async def get_visualizations(current_user: dict = Depends(get_current_user)):
    """ユーザーの可視化一覧を取得します"""
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/visualization/{visualization_id}", response_model=VisualizationResponse)
async def get_visualization(
    visualization_id: str,
    current_user: dict = Depends(get_current_user)
):
    """特定の可視化を取得します"""
    try:
        conditions = [
            {'field': 'id', 'operator': '==', 'value': visualization_id},
            {'field': 'created_by', 'operator': '==', 'value': current_user['uid']}
        ]

        visualization = await firestore_service.fetch_documents(
            collection_name='visualizations',
            conditions=conditions,
            limit=1
        )

        if not visualization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Visualization not found"
            )

        return VisualizationResponse(**visualization[0])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.put("/visualization/{visualization_id}", response_model=VisualizationResponse)
async def update_visualization(
    visualization_id: str,
    update_data: Dict,
    current_user: dict = Depends(get_current_user)
):
    """可視化を更新します"""
    try:
        conditions = [
            {'field': 'id', 'operator': '==', 'value': visualization_id},
            {'field': 'created_by', 'operator': '==', 'value': current_user['uid']}
        ]

        # 既存の可視化を取得
        existing_visualization = await firestore_service.fetch_documents(
            collection_name='visualizations',
            conditions=conditions,
            limit=1
        )

        if not existing_visualization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Visualization not found"
            )

        # 更新データを準備
        update_data['updated_at'] = datetime.now()

        # Firestoreで更新を実行
        await firestore_service.save_results(
            results=[update_data],
            collection_name='visualizations'
        )

        # 更新後のデータを取得
        updated_visualization = await firestore_service.fetch_documents(
            collection_name='visualizations',
            conditions=conditions,
            limit=1
        )

        return VisualizationResponse(**updated_visualization[0])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.delete("/visualization/{visualization_id}")
async def delete_visualization(
    visualization_id: str,
    current_user: dict = Depends(get_current_user)
):
    """可視化を削除します"""
    try:
        conditions = [
            {'field': 'id', 'operator': '==', 'value': visualization_id},
            {'field': 'created_by', 'operator': '==', 'value': current_user['uid']}
        ]

        # 可視化の存在確認
        visualization = await firestore_service.fetch_documents(
            collection_name='visualizations',
            conditions=conditions,
            limit=1
        )

        if not visualization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Visualization not found"
            )

        # Firestoreからドキュメントを削除
        await firestore_service.delete_document(
            collection_name='visualizations',
            document_id=visualization_id
        )

        return {"message": "Visualization deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )