"""
Firestoreサービス
データの永続化、取得、更新、削除の機能を提供します。
Google Cloud Storageとのインテグレーションもサポートしています。
"""
from typing import List, Dict, Any, Optional, Tuple
from google.cloud import firestore, storage
from google.cloud.firestore import AsyncClient
from firebase_admin import credentials, initialize_app
import firebase_admin
from datetime import datetime
import logging
import os
from fastapi import UploadFile
import asyncio
from contextlib import contextmanager

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

class FirestoreClient:
    """Firestoreクライアントクラス"""
    _instance = None

    def __new__(cls, project_id: str = None, emulator_host: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(FirestoreClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, project_id: str = None, emulator_host: Optional[str] = None):
        """
        Args:
            project_id (str): プロジェクトID
            emulator_host (Optional[str]): エミュレータのホスト（開発環境用）
        """
        if self._initialized:
            return

        try:
            if emulator_host:
                os.environ["FIRESTORE_EMULATOR_HOST"] = emulator_host

            # Firebase Adminの初期化（まだ初期化されていない場合）
            if not firebase_admin._apps:
                if os.getenv("ENVIRONMENT") == "development":
                    # 開発環境ではエミュレータを使用
                    cred = credentials.ApplicationDefault()
                else:
                    # 本番環境では認証情報ファイルを使用
                    cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                    if not cred_path or not os.path.exists(cred_path):
                        raise FileNotFoundError("Firebase credentials file not found")
                    cred = credentials.Certificate(cred_path)

                initialize_app(cred)

            self.client = firestore.AsyncClient(project=project_id)
            self.storage_client = storage.Client()
            self.bucket_name = os.getenv('FIREBASE_STORAGE_BUCKET')
            self._initialized = True
            logger.info("Firestore and Storage clients initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize clients: {str(e)}")
            raise

    async def close(self):
        """クライアントを閉じる"""
        await self.client.close()

    @property
    def native_client(self) -> AsyncClient:
        """ネイティブのFirestoreクライアントを取得"""
        return self.client

    @contextmanager
    def get_transaction(self):
        """トランザクションを取得"""
        transaction = self.client.transaction()
        try:
            yield transaction
        except Exception as e:
            logger.error(f"Transaction error: {str(e)}")
            raise

    async def get_document(self, collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """ドキュメントを取得"""
        doc_ref = self.client.collection(collection).document(doc_id)
        doc = await doc_ref.get()
        return doc.to_dict() if doc.exists else None

    async def create_document(self, collection: str, doc_id: str, data: Dict[str, Any]) -> None:
        """ドキュメントを作成"""
        data['created_at'] = datetime.utcnow()
        data['updated_at'] = datetime.utcnow()
        await self.client.collection(collection).document(doc_id).set(data)

    async def update_document(self, collection: str, doc_id: str, data: Dict[str, Any]) -> None:
        """ドキュメントを更新"""
        data['updated_at'] = datetime.utcnow()
        await self.client.collection(collection).document(doc_id).update(data)

    async def delete_document(self, collection: str, doc_id: str) -> None:
        """ドキュメントを削除"""
        await self.client.collection(collection).document(doc_id).delete()

    async def query_documents(
        self,
        collection: str,
        filters: Optional[List[Dict[str, Any]]] = None,
        order_by: Optional[Tuple[str, str]] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """ドキュメントをクエリ"""
        query = self.client.collection(collection)

        if filters:
            for filter_condition in filters:
                field = filter_condition.get('field')
                operator = filter_condition.get('operator', '==')
                value = filter_condition.get('value')
                if all([field, operator, value is not None]):
                    query = query.where(field, operator, value)

        if order_by:
            field, direction = order_by
            query = query.order_by(field, direction=direction)

        if offset > 0:
            query = query.offset(offset)

        if limit:
            query = query.limit(limit)

        docs = await query.get()
        return [doc.to_dict() for doc in docs]

    async def batch_write(self, operations: List[Dict[str, Any]]) -> None:
        """バッチ書き込みを実行"""
        for i in range(0, len(operations), MAX_BATCH_SIZE):
            batch = self.client.batch()
            batch_operations = operations[i:i + MAX_BATCH_SIZE]

            try:
                for op in batch_operations:
                    doc_ref = self.client.collection(op['collection']).document(
                        op.get('doc_id', self.client.collection(op['collection']).document().id)
                    )

                    if op['type'] == 'create':
                        op['data']['created_at'] = datetime.utcnow()
                        op['data']['updated_at'] = datetime.utcnow()
                        batch.set(doc_ref, op['data'])
                    elif op['type'] == 'update':
                        op['data']['updated_at'] = datetime.utcnow()
                        batch.update(doc_ref, op['data'])
                    elif op['type'] == 'delete':
                        batch.delete(doc_ref)

                await batch.commit()
            except Exception as e:
                logger.error(f"Batch operation failed: {str(e)}")
                raise

    async def save_file(self, file: UploadFile, user_id: str, metadata: Optional[Dict] = None) -> Tuple[str, str]:
        """ファイルを保存"""
        try:
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

            doc_ref = self.client.collection('file_uploads').document()
            await doc_ref.set(file_metadata)
            return doc_ref.id, blob_name

        except Exception as e:
            error_msg = f"Error uploading file: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    async def _validate_file(self, file: UploadFile) -> None:
        """ファイルのバリデーション"""
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise ValidationError(f"Unsupported file type: {file.content_type}")

        contents = await file.read()
        await file.seek(0)

        if len(contents) > MAX_FILE_SIZE:
            raise ValidationError(f"File size exceeds maximum allowed size of {MAX_FILE_SIZE} bytes")

    async def _upload_to_storage(self, contents: bytes, blob_name: str) -> None:
        """Cloud Storageにファイルをアップロード"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: blob.upload_from_string(contents)
            )
            logger.info(f"Successfully uploaded file to {blob_name}")
        except Exception as e:
            error_msg = f"Error uploading to Cloud Storage: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

def get_firestore_client(
    project_id: Optional[str] = None,
    emulator_host: Optional[str] = None
) -> FirestoreClient:
    """Firestoreクライアントのインスタンスを取得

    Args:
        project_id (Optional[str]): プロジェクトID（未指定の場合は環境変数から取得）
        emulator_host (Optional[str]): エミュレータのホスト（開発環境用）

    Returns:
        FirestoreClient: Firestoreクライアントのインスタンス
    """
    if not project_id:
        project_id = os.getenv("FIREBASE_PROJECT_ID")
        if not project_id:
            raise ValueError("FIREBASE_PROJECT_ID must be set")

    if not emulator_host and os.getenv("ENVIRONMENT") == "development":
        emulator_host = os.getenv("FIRESTORE_EMULATOR_HOST")

    return FirestoreClient(project_id, emulator_host)