# -*- coding: utf-8 -*-
"""
データ入力 API ルーター
Google Forms、CSV、外部データソースからのデータ入力、
およびファイルアップロード機能を提供します。
"""
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime
from ..dependencies import get_current_user
from ...service.firestore.client import FirestoreService, StorageError, ValidationError

router = APIRouter(
    prefix="/data_input",
    tags=["data_input"],
    responses={404: {"description": "Not found"}}
)

# データモデル定義
class DataInputBase(BaseModel):
    source: str
    data_type: str
    timestamp: datetime = datetime.now()
    metadata: Optional[Dict] = None

class FormResponse(DataInputBase):
    form_id: str
    responses: Dict
    respondent_email: Optional[str]

class CSVData(DataInputBase):
    filename: str
    row_count: int
    processed_data: List[Dict]
    storage_path: str

class ExternalData(DataInputBase):
    source_name: str
    data: Dict
    integration_status: str

# FirestoreServiceのインスタンスを作成
firestore_service = FirestoreService()

# Google Forms 連携 API
@router.get("/google_forms/{form_id}", response_model=FormResponse)
async def get_google_forms_data(
    form_id: str,
    current_user: dict = Depends(get_current_user)
):
    """指定された Google Forms からアンケート結果を取得します。"""
    try:
        # Google Forms APIとの連携処理は別のサービスで実装
        form_responses = {
            "question1": "answer1",
            "question2": "answer2"
        }  # 実際のレスポンスデータに置き換え

        doc_id = await firestore_service.save_form_response(
            form_id=form_id,
            responses=form_responses,
            user_id=current_user['uid'],
            metadata={"source": "google_forms_api"}
        )

        # 保存したデータを取得
        saved_data = await firestore_service.fetch_documents(
            collection_name='form_responses',
            conditions=[{'field': 'id', 'operator': '==', 'value': doc_id}]
        )

        if not saved_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Saved form response not found"
            )

        return FormResponse(**saved_data[0])

    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

# CSV インポート API
@router.post("/csv/", response_model=CSVData)
async def import_csv_data(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """CSV ファイルからデータをインポートします。"""
    try:
        doc_id, storage_path = await firestore_service.save_csv_data(
            file=file,
            user_id=current_user['uid'],
            metadata={"source": "csv_upload"}
        )

        # 保存したデータを取得
        saved_data = await firestore_service.fetch_documents(
            collection_name='csv_imports',
            conditions=[{'field': 'id', 'operator': '==', 'value': doc_id}]
        )

        if not saved_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Saved CSV data not found"
            )

        return CSVData(**saved_data[0])

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

# 外部データ統合 API
@router.get("/external_data/{source_name}", response_model=ExternalData)
async def get_external_data(
    source_name: str,
    current_user: dict = Depends(get_current_user)
):
    """指定された外部データソースからデータを取得します。"""
    try:
        # 外部データソースとの連携処理は別のサービスで実装
        external_data = {
            'source': 'external_api',
            'data_type': 'integration',
            'source_name': source_name,
            'data': {},  # 実際の外部データに置き換え
            'integration_status': 'completed',
            'user_id': current_user['uid'],
            'timestamp': datetime.now()
        }

        doc_ids = await firestore_service.save_results(
            results=[external_data],
            collection_name='external_data'
        )

        if not doc_ids:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save external data"
            )

        # 保存したデータを取得
        saved_data = await firestore_service.fetch_documents(
            collection_name='external_data',
            conditions=[{'field': 'id', 'operator': '==', 'value': doc_ids[0]}]
        )

        if not saved_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Saved external data not found"
            )

        return ExternalData(**saved_data[0])

    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

# ファイルアップロード API
@router.post("/upload/", response_model=List[Dict])
async def upload_user_files(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    施術後のアンケートデータ、問診票、損益計算書をアップロードします。
    """
    try:
        upload_results = []

        for file in files:
            doc_id, storage_path = await firestore_service.save_file_upload(
                file=file,
                user_id=current_user['uid'],
                metadata={
                    "source": "file_upload",
                    "upload_type": "user_document"
                }
            )

            # 保存したファイルのメタデータを取得
            saved_data = await firestore_service.fetch_documents(
                collection_name='file_uploads',
                conditions=[{'field': 'id', 'operator': '==', 'value': doc_id}]
            )

            if saved_data:
                upload_results.append(saved_data[0])

        if not upload_results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save uploaded files"
            )

        return upload_results

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )