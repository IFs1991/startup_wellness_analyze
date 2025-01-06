# -*- coding: utf-8 -*-
"""
レポート生成 API ルーター
PDF レポート自動生成、カスタマイズされたレポート作成、
インタラクティブなレポーティング、レポートアーカイブ機能を提供します。
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional, Dict
from pydantic import BaseModel
from datetime import datetime
from ...service.firestore.client import FirestoreService
from firebase_admin import auth

# APIRouterの初期化
router = APIRouter(
    prefix="/report_generation",
    tags=["report_generation"],
    responses={404: {"description": "Not found"}}
)

# サービスの初期化
firestore_service = FirestoreService()

# ベースモデルの定義
class ReportBase(BaseModel):
    title: str
    description: Optional[str] = None
    report_type: str
    parameters: Optional[Dict] = None

class ReportResponse(ReportBase):
    id: str
    user_id: str
    created_at: datetime
    status: str
    download_url: Optional[str] = None

async def get_current_user(token: str = Depends(auth.verify_id_token)) -> dict:
    """Firebaseトークンを検証してユーザー情報を取得"""
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/pdf/", response_model=ReportResponse)
async def generate_pdf_report(
    report_params: ReportBase,
    current_user: dict = Depends(get_current_user)
):
    """PDF レポートを自動生成します。"""
    try:
        # レポートデータを準備
        report_data = {
            'title': report_params.title,
            'description': report_params.description,
            'report_type': 'pdf',
            'parameters': report_params.parameters,
            'user_id': current_user['uid'],
            'created_at': datetime.now(),
            'status': 'processing',
            'download_url': None
        }

        # FirestoreServiceを使用してレポートデータを保存
        await firestore_service.save_results(
            results=[report_data],
            collection_name='reports'
        )

        # ここでCloud Functionsをトリガーしてバックグラウンドでレポート生成
        # TODO: Cloud Pub/SubまたはCloud Functionsの呼び出しを実装

        return ReportResponse(**report_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/custom/", response_model=ReportResponse)
async def create_custom_report(
    report_params: ReportBase,
    current_user: dict = Depends(get_current_user)
):
    """カスタマイズされたレポートを作成します。"""
    try:
        # レポートデータを準備
        report_data = {
            'title': report_params.title,
            'description': report_params.description,
            'report_type': 'custom',
            'parameters': report_params.parameters,
            'user_id': current_user['uid'],
            'created_at': datetime.now(),
            'status': 'processing',
            'download_url': None
        }

        # FirestoreServiceを使用してレポートデータを保存
        await firestore_service.save_results(
            results=[report_data],
            collection_name='reports'
        )

        return ReportResponse(**report_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/reports/", response_model=List[ReportResponse])
async def get_user_reports(
    current_user: dict = Depends(get_current_user),
    limit: int = 10,
    offset: int = 0
):
    """ユーザーのレポート一覧を取得します。"""
    try:
        conditions = [
            {'field': 'user_id', 'operator': '==', 'value': current_user['uid']}
        ]

        # FirestoreServiceを使用してレポート一覧を取得
        reports = await firestore_service.fetch_documents(
            collection_name='reports',
            conditions=conditions,
            limit=limit,
            offset=offset
        )

        return [ReportResponse(**report) for report in reports]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/reports/{report_id}", response_model=ReportResponse)
async def get_report_status(
    report_id: str,
    current_user: dict = Depends(get_current_user)
):
    """特定のレポートの状態を取得します。"""
    try:
        conditions = [
            {'field': 'id', 'operator': '==', 'value': report_id},
            {'field': 'user_id', 'operator': '==', 'value': current_user['uid']}
        ]

        # FirestoreServiceを使用してレポート情報を取得
        report = await firestore_service.fetch_documents(
            collection_name='reports',
            conditions=conditions,
            limit=1
        )

        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Report not found"
            )

        return ReportResponse(**report[0])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )