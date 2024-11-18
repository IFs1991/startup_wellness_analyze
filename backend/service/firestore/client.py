# -*- coding: utf-8 -*-
"""
Firestoreサービス
データの永続化、取得、更新、削除の機能を提供します。
Google Cloud Storageとのインテグレーションもサポートしています。
"""
from firebase_admin import firestore
import firebase_admin
from google.cloud import storage
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime
import logging
import pandas as pd
import io
from fastapi import UploadFile, HTTPException
from pydantic import BaseModel

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 定数定義
MAX_BATCH_SIZE = 500
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_CONTENT_TYPES = {
    'text/csv',
    'application/pdf',
    'image/jpeg',
    'image/png',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
}

class StorageError(Exception):
    """ストレージ操作に関するエラー"""
    pass

class ValidationError(Exception):
    """データバリデーションに関するエラー"""
    pass

class FirestoreService:
    def __init__(self):
        """
        Firestoreクライアントを初期化します
        """
        try:
            # firebase-adminの初期化
            if not firebase_admin._apps:
                firebase_admin.initialize_app()
            self.db = firestore.client()
            self.storage_client = storage.Client()
            self.bucket_name = 'your-bucket-name'  # 環境変数から取得することを推奨
            logger.info("Firestore and Storage clients initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize clients: {str(e)}")
            raise

    async def fetch_documents(
        self,
        collection_name: str,
        conditions: Optional[List[Dict[str, Any]]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        order_by: str = 'created_at',
        direction: str = 'desc'
    ) -> List[Dict[str, Any]]:
        """
        条件に基づいてドキュメントを取得します
        """
        try:
            logger.info(f"Fetching documents from collection: {collection_name}")
            collection_ref = self.db.collection(collection_name)
            query = collection_ref

            if conditions:
                for condition in conditions:
                    field = condition.get('field')
                    operator = condition.get('operator', '==')
                    value = condition.get('value')

                    if all([field, operator, value is not None]):
                        query = query.where(field, operator, value)
                        logger.debug(f"Applied query condition: {field} {operator} {value}")

            query = query.order_by(order_by, direction=direction)

            if offset > 0:
                query = query.offset(offset)

            if limit:
                query = query.limit(limit)

            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, query.get)

            results = []
            for doc in docs:
                data = doc.to_dict()
                if data is not None:
                    data['id'] = doc.id
                    results.append(data)

            logger.info(f"Successfully fetched {len(results)} documents")
            return results

        except Exception as e:
            error_msg = f"Error fetching documents from Firestore: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    async def save_results(
        self,
        results: List[Dict[str, Any]],
        collection_name: str,
        batch_size: int = MAX_BATCH_SIZE
    ) -> List[str]:
        """
        結果をバッチ処理でFirestoreに保存します
        """
        try:
            logger.info(f"Starting to save {len(results)} results to collection: {collection_name}")
            loop = asyncio.get_event_loop()
            doc_ids = []

            for i in range(0, len(results), batch_size):
                batch = self.db.batch()
                batch_results = results[i:i + batch_size]

                logger.debug(f"Processing batch {i//batch_size + 1} with {len(batch_results)} items")

                for result in batch_results:
                    if 'created_at' not in result:
                        result['created_at'] = datetime.now()
                    doc_ref = self.db.collection(collection_name).document()
                    doc_ids.append(doc_ref.id)
                    batch.set(doc_ref, result)

                await loop.run_in_executor(None, batch.commit)
                logger.debug(f"Committed batch {i//batch_size + 1}")

            logger.info("Successfully saved all results to Firestore")
            return doc_ids

        except Exception as e:
            error_msg = f"Error saving results to Firestore: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    async def save_form_response(
        self,
        form_id: str,
        responses: Dict,
        user_id: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Google Formsのレスポンスを保存します
        """
        try:
            form_data = {
                'source': 'google_forms',
                'data_type': 'survey',
                'form_id': form_id,
                'responses': responses,
                'user_id': user_id,
                'timestamp': datetime.now(),
                'metadata': metadata
            }

            doc_ids = await self.save_results(
                results=[form_data],
                collection_name='form_responses'
            )
            return doc_ids[0]

        except Exception as e:
            error_msg = f"Error saving form response: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    async def save_csv_data(
        self,
        file: UploadFile,
        user_id: str,
        metadata: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """
        CSVデータを保存しCloud Storageにアップロードします
        """
        try:
            # ファイルバリデーション
            await self._validate_file(file)

            # CSVファイルの読み込みと処理
            contents = await file.read()
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            processed_data = df.to_dict('records')

            # Cloud Storageへの保存
            blob_name = f"csv_uploads/{user_id}/{datetime.now().isoformat()}_{file.filename}"
            await self._upload_to_storage(contents, blob_name)

            # Firestoreへのメタデータ保存
            csv_data = {
                'source': 'csv_upload',
                'data_type': 'csv',
                'filename': file.filename,
                'storage_path': blob_name,
                'row_count': len(processed_data),
                'processed_data': processed_data,
                'user_id': user_id,
                'timestamp': datetime.now(),
                'metadata': metadata
            }

            doc_ids = await self.save_results(
                results=[csv_data],
                collection_name='csv_imports'
            )
            return doc_ids[0], blob_name

        except Exception as e:
            error_msg = f"Error saving CSV data: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    async def save_file_upload(
        self,
        file: UploadFile,
        user_id: str,
        metadata: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """
        ファイルをアップロードし、メタデータを保存します
        """
        try:
            # ファイルバリデーション
            await self._validate_file(file)

            contents = await file.read()

            # Cloud Storageへの保存
            blob_name = f"uploads/{user_id}/{datetime.now().isoformat()}_{file.filename}"
            await self._upload_to_storage(contents, blob_name)

            # Firestoreにメタデータを保存
            file_metadata = {
                'filename': file.filename,
                'storage_path': blob_name,
                'content_type': file.content_type,
                'size': len(contents),
                'user_id': user_id,
                'upload_timestamp': datetime.now(),
                'metadata': metadata
            }

            doc_ids = await self.save_results(
                results=[file_metadata],
                collection_name='file_uploads'
            )
            return doc_ids[0], blob_name

        except Exception as e:
            error_msg = f"Error uploading file: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    async def update_document(
        self,
        collection_name: str,
        document_id: str,
        data: Dict[str, Any]
    ) -> None:
        """
        ドキュメントを更新します
        """
        try:
            logger.info(f"Updating document {document_id} in collection {collection_name}")
            doc_ref = self.db.collection(collection_name).document(document_id)

            # 更新日時を自動追加
            data['updated_at'] = datetime.now()

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: doc_ref.update(data))

            logger.info(f"Successfully updated document {document_id}")

        except Exception as e:
            error_msg = f"Error updating document in Firestore: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    async def delete_document(
        self,
        collection_name: str,
        document_id: str
    ) -> None:
        """
        ドキュメントを削除します
        """
        try:
            logger.info(f"Deleting document {document_id} from collection {collection_name}")
            doc_ref = self.db.collection(collection_name).document(document_id)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, doc_ref.delete)

            logger.info(f"Successfully deleted document {document_id}")

        except Exception as e:
            error_msg = f"Error deleting document from Firestore: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    async def _validate_file(self, file: UploadFile) -> None:
        """
        ファイルのバリデーションを行います
        """
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise ValidationError(f"Unsupported file type: {file.content_type}")

        # ファイルサイズのチェックは実際のコンテンツを読む必要がある
        contents = await file.read()
        await file.seek(0)  # ファイルポインタを先頭に戻す

        if len(contents) > MAX_FILE_SIZE:
            raise ValidationError(f"File size exceeds maximum allowed size of {MAX_FILE_SIZE} bytes")

    async def _upload_to_storage(self, contents: bytes, blob_name: str) -> None:
        """
        Cloud Storageにファイルをアップロードします
        """
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: blob.upload_from_string(contents))

            logger.info(f"Successfully uploaded file to {blob_name}")

        except Exception as e:
            error_msg = f"Error uploading to Cloud Storage: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    async def close(self) -> None:
        """
        クライアント接続を閉じます（firebase-adminの場合は不要）
        """
        try:
            logger.info("Firestore client connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing Firestore client connection: {str(e)}")
            raise

def get_firestore_client() -> firestore.firestore.Client:
    """
    Firestoreクライアントのインスタンスを取得します
    """
    service = FirestoreService()
    return service.db