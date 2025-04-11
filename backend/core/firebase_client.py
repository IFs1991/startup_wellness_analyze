"""
Firebase/Firestoreクライアントモジュール
シングルトンパターンとインターフェースを使用した一貫したFirebase接続管理
"""
import os
import asyncio
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic, Protocol, Callable, Tuple
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, storage
from abc import ABC, abstractmethod
import threading
from functools import wraps
import time
from .common_logger import get_logger
from .exceptions import FirestoreError, DatabaseError
from .config import get_settings
from .async_utils import TaskLimiter, async_timed

# ロギングの設定
logger = get_logger(__name__)

# 型定義
T = TypeVar('T')
DocumentDict = Dict[str, Any]
QueryFilter = Dict[str, Any]
BatchOperation = Dict[str, Any]  # 'type', 'collection', 'doc_id', 'data' を含む


class FirebaseClientInterface(ABC):
    """
    Firebaseクライアントインターフェース
    依存性注入のために使用される抽象クラス
    """
    @abstractmethod
    async def initialize(self) -> None:
        """Firebaseを初期化"""
        pass

    @abstractmethod
    async def get_document(self, collection: str, document_id: str) -> Optional[DocumentDict]:
        """ドキュメントを取得"""
        pass

    @abstractmethod
    async def get_documents(self, collection: str, document_ids: List[str]) -> Dict[str, Optional[DocumentDict]]:
        """複数のドキュメントを一括取得"""
        pass

    @abstractmethod
    async def add_document(self, collection: str, data: DocumentDict) -> str:
        """ドキュメントを追加"""
        pass

    @abstractmethod
    async def add_documents(self, collection: str, data_list: List[DocumentDict]) -> List[str]:
        """複数のドキュメントを一括追加"""
        pass

    @abstractmethod
    async def set_document(self, collection: str, document_id: str, data: DocumentDict, merge: bool = False) -> None:
        """ドキュメントを設定"""
        pass

    @abstractmethod
    async def set_documents(self, collection: str, documents: Dict[str, DocumentDict], merge: bool = False) -> None:
        """複数のドキュメントを一括設定"""
        pass

    @abstractmethod
    async def update_document(self, collection: str, document_id: str, data: DocumentDict) -> None:
        """ドキュメントを更新"""
        pass

    @abstractmethod
    async def update_documents(self, collection: str, updates: Dict[str, DocumentDict]) -> None:
        """複数のドキュメントを一括更新"""
        pass

    @abstractmethod
    async def delete_document(self, collection: str, document_id: str) -> None:
        """ドキュメントを削除"""
        pass

    @abstractmethod
    async def delete_documents(self, collection: str, document_ids: List[str]) -> None:
        """複数のドキュメントを一括削除"""
        pass

    @abstractmethod
    async def query_documents(
        self,
        collection: str,
        filters: Optional[List[QueryFilter]] = None,
        order_by: Optional[List[Dict[str, str]]] = None,
        limit: Optional[int] = None,
        start_after: Optional[DocumentDict] = None
    ) -> List[DocumentDict]:
        """ドキュメントをクエリ"""
        pass

    @abstractmethod
    async def paginated_query(
        self,
        collection: str,
        page_size: int = 100,
        filters: Optional[List[QueryFilter]] = None,
        order_by: Optional[List[Dict[str, str]]] = None,
        max_pages: Optional[int] = None
    ) -> List[List[DocumentDict]]:
        """ページネーション付きクエリ"""
        pass

    @abstractmethod
    async def batch_update(self, operations: List[Dict[str, Any]]) -> None:
        """バッチ更新を実行"""
        pass

    @abstractmethod
    async def batch_operations(self, operations: List[BatchOperation]) -> None:
        """異なる種類の操作を含むバッチ処理"""
        pass

    @abstractmethod
    async def transaction(self, transaction_callable: Callable[[Any], T]) -> T:
        """トランザクションを実行"""
        pass

    @abstractmethod
    async def collection_group_query(
        self,
        collection_id: str,
        filters: Optional[List[QueryFilter]] = None,
        order_by: Optional[List[Dict[str, str]]] = None,
        limit: Optional[int] = None
    ) -> List[DocumentDict]:
        """コレクショングループクエリを実行"""
        pass


class FirebaseClient(FirebaseClientInterface):
    """
    Firebaseクライアントの実装
    シングルトンパターンによる一貫したインスタンス管理
    """
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    _db = None
    _bucket = None
    _settings = get_settings()
    _task_limiter = TaskLimiter(20)  # 同時実行するFirebase操作の制限

    def __new__(cls):
        """シングルトンパターンによるインスタンス生成"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FirebaseClient, cls).__new__(cls)
            return cls._instance

    async def initialize(self) -> None:
        """Firebaseを初期化"""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            try:
                # 設定から必要な情報を取得
                project_id = self._settings.firebase.project_id
                credentials_path = self._settings.firebase.credentials_path
                storage_bucket = self._settings.firebase.storage_bucket
                use_emulator = self._settings.firebase.use_emulator

                # 本番環境では環境変数やサービスアカウントを使用
                if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or credentials_path:
                    try:
                        # 明示的にパスが指定されている場合
                        if credentials_path and os.path.exists(credentials_path):
                            cred = credentials.Certificate(credentials_path)
                            firebase_admin.initialize_app(cred, {
                                'projectId': project_id,
                                'storageBucket': storage_bucket
                            })
                        # 環境変数が設定されている場合
                        else:
                            firebase_admin.initialize_app()
                    except ValueError:
                        # すでに初期化されている場合
                        pass
                # 開発環境ではエミュレータを使用
                elif use_emulator:
                    os.environ["FIRESTORE_EMULATOR_HOST"] = f"{self._settings.firebase.emulator_host}:{self._settings.firebase.emulator_port}"
                    firebase_admin.initialize_app()
                # デフォルト認証情報を使用
                else:
                    try:
                        cred = credentials.ApplicationDefault()
                        firebase_admin.initialize_app(cred, {
                            'projectId': project_id
                        })
                    except ValueError:
                        # すでに初期化されている場合
                        pass

                self._db = firestore.client()

                # Storage Bucketの初期化（設定されている場合）
                if storage_bucket:
                    self._bucket = storage.bucket()

                logger.info("Firebase initialized successfully")
                self._initialized = True

            except Exception as e:
                error_msg = f"Failed to initialize Firebase: {str(e)}"
                logger.error(error_msg)
                raise FirestoreError(error_msg) from e

    def _ensure_initialized(self):
        """初期化を確認するデコレータ"""
        def decorator(func):
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                if not self._initialized:
                    await self.initialize()
                return await func(self, *args, **kwargs)
            return wrapper
        return decorator

    @_ensure_initialized
    async def get_document(self, collection: str, document_id: str) -> Optional[DocumentDict]:
        """
        ドキュメントを取得

        Args:
            collection: コレクション名
            document_id: ドキュメントID

        Returns:
            取得したドキュメント

        Raises:
            FirestoreError: Firestore操作エラー
        """
        try:
            doc_ref = self._db.collection(collection).document(document_id)
            doc = doc_ref.get()
            if doc.exists:
                return {**doc.to_dict(), "id": doc.id}
            return None
        except Exception as e:
            error_msg = f"Failed to get document {document_id} from {collection}: {str(e)}"
            logger.error(error_msg)
            raise FirestoreError(error_msg) from e

    @_ensure_initialized
    @async_timed
    async def get_documents(self, collection: str, document_ids: List[str]) -> Dict[str, Optional[DocumentDict]]:
        """
        複数のドキュメントを一括取得

        Args:
            collection: コレクション名
            document_ids: ドキュメントIDのリスト

        Returns:
            ドキュメントID -> ドキュメントデータのマップ

        Raises:
            FirestoreError: Firestore操作エラー
        """
        if not document_ids:
            return {}

        result = {}

        # 同時に取得する数を制限
        async def get_doc(doc_id: str) -> Tuple[str, Optional[DocumentDict]]:
            try:
                doc = await self.get_document(collection, doc_id)
                return (doc_id, doc)
            except Exception as e:
                logger.error(f"Error fetching document {doc_id}: {str(e)}")
                return (doc_id, None)

        # 並列処理で取得
        tasks = [self._task_limiter.run(get_doc(doc_id)) for doc_id in document_ids]
        results = await asyncio.gather(*tasks)

        for doc_id, doc in results:
            result[doc_id] = doc

        return result

    @_ensure_initialized
    async def add_document(self, collection: str, data: DocumentDict) -> str:
        """
        ドキュメントを追加

        Args:
            collection: コレクション名
            data: 追加するデータ

        Returns:
            生成されたドキュメントID

        Raises:
            FirestoreError: Firestore操作エラー
        """
        try:
            # タイムスタンプの自動追加
            doc_data = {
                **data,
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP
            }

            doc_ref = self._db.collection(collection).document()
            doc_ref.set(doc_data)
            return doc_ref.id
        except Exception as e:
            error_msg = f"Failed to add document to {collection}: {str(e)}"
            logger.error(error_msg)
            raise FirestoreError(error_msg) from e

    @_ensure_initialized
    @async_timed
    async def add_documents(self, collection: str, data_list: List[DocumentDict]) -> List[str]:
        """
        複数のドキュメントを一括追加

        Args:
            collection: コレクション名
            data_list: 追加するデータのリスト

        Returns:
            生成されたドキュメントIDのリスト

        Raises:
            FirestoreError: Firestore操作エラー
        """
        if not data_list:
            return []

        # 5000件ごとにバッチ処理（Firestoreの制限）
        MAX_BATCH_SIZE = 500
        doc_ids = []

        try:
            for i in range(0, len(data_list), MAX_BATCH_SIZE):
                batch_data = data_list[i:i+MAX_BATCH_SIZE]
                batch = self._db.batch()

                batch_ids = []
                for data in batch_data:
                    # タイムスタンプの自動追加
                    doc_data = {
                        **data,
                        'created_at': firestore.SERVER_TIMESTAMP,
                        'updated_at': firestore.SERVER_TIMESTAMP
                    }

                    doc_ref = self._db.collection(collection).document()
                    batch_ids.append(doc_ref.id)
                    batch.set(doc_ref, doc_data)

                # バッチコミット
                batch.commit()
                doc_ids.extend(batch_ids)

            return doc_ids
        except Exception as e:
            error_msg = f"Failed to add documents to {collection}: {str(e)}"
            logger.error(error_msg)
            raise FirestoreError(error_msg) from e

    @_ensure_initialized
    async def set_document(self, collection: str, document_id: str, data: DocumentDict, merge: bool = False) -> None:
        """
        ドキュメントを設定

        Args:
            collection: コレクション名
            document_id: ドキュメントID
            data: 設定するデータ
            merge: マージするかどうか

        Raises:
            FirestoreError: Firestore操作エラー
        """
        try:
            # タイムスタンプの自動追加
            if merge:
                doc_data = {
                    **data,
                    'updated_at': firestore.SERVER_TIMESTAMP
                }
                if not await self.get_document(collection, document_id):
                    doc_data['created_at'] = firestore.SERVER_TIMESTAMP
            else:
                doc_data = {
                    **data,
                    'created_at': firestore.SERVER_TIMESTAMP,
                    'updated_at': firestore.SERVER_TIMESTAMP
                }

            doc_ref = self._db.collection(collection).document(document_id)
            doc_ref.set(doc_data, merge=merge)
        except Exception as e:
            error_msg = f"Failed to set document {document_id} in {collection}: {str(e)}"
            logger.error(error_msg)
            raise FirestoreError(error_msg) from e

    @_ensure_initialized
    @async_timed
    async def set_documents(self, collection: str, documents: Dict[str, DocumentDict], merge: bool = False) -> None:
        """
        複数のドキュメントを一括設定

        Args:
            collection: コレクション名
            documents: ドキュメントID -> データのマップ
            merge: マージするかどうか

        Raises:
            FirestoreError: Firestore操作エラー
        """
        if not documents:
            return

        # 5000件ごとにバッチ処理（Firestoreの制限）
        MAX_BATCH_SIZE = 500
        doc_ids = list(documents.keys())

        try:
            for i in range(0, len(doc_ids), MAX_BATCH_SIZE):
                batch_ids = doc_ids[i:i+MAX_BATCH_SIZE]
                batch = self._db.batch()

                for doc_id in batch_ids:
                    data = documents[doc_id]

                    # タイムスタンプの自動追加
                    if merge:
                        doc_data = {
                            **data,
                            'updated_at': firestore.SERVER_TIMESTAMP
                        }
                    else:
                        doc_data = {
                            **data,
                            'created_at': firestore.SERVER_TIMESTAMP,
                            'updated_at': firestore.SERVER_TIMESTAMP
                        }

                    doc_ref = self._db.collection(collection).document(doc_id)
                    batch.set(doc_ref, doc_data, merge=merge)

                # バッチコミット
                batch.commit()
        except Exception as e:
            error_msg = f"Failed to set documents in {collection}: {str(e)}"
            logger.error(error_msg)
            raise FirestoreError(error_msg) from e

    @_ensure_initialized
    async def update_document(self, collection: str, document_id: str, data: DocumentDict) -> None:
        """
        ドキュメントを更新

        Args:
            collection: コレクション名
            document_id: ドキュメントID
            data: 更新するデータ

        Raises:
            FirestoreError: Firestore操作エラー
        """
        try:
            # タイムスタンプの自動追加
            doc_data = {
                **data,
                'updated_at': firestore.SERVER_TIMESTAMP
            }

            doc_ref = self._db.collection(collection).document(document_id)

            # ドキュメントが存在するか確認
            doc = doc_ref.get()
            if not doc.exists:
                raise FirestoreError(f"Document {document_id} does not exist in {collection}")

            doc_ref.update(doc_data)
        except FirestoreError as e:
            # 自作例外はそのまま再送出
            raise
        except Exception as e:
            error_msg = f"Failed to update document {document_id} in {collection}: {str(e)}"
            logger.error(error_msg)
            raise FirestoreError(error_msg) from e

    @_ensure_initialized
    @async_timed
    async def update_documents(self, collection: str, updates: Dict[str, DocumentDict]) -> None:
        """
        複数のドキュメントを一括更新

        Args:
            collection: コレクション名
            updates: ドキュメントID -> 更新データのマップ

        Raises:
            FirestoreError: Firestore操作エラー
        """
        if not updates:
            return

        # 5000件ごとにバッチ処理（Firestoreの制限）
        MAX_BATCH_SIZE = 500
        doc_ids = list(updates.keys())

        try:
            for i in range(0, len(doc_ids), MAX_BATCH_SIZE):
                batch_ids = doc_ids[i:i+MAX_BATCH_SIZE]
                batch = self._db.batch()

                for doc_id in batch_ids:
                    data = updates[doc_id]

                    # タイムスタンプの自動追加
                    doc_data = {
                        **data,
                        'updated_at': firestore.SERVER_TIMESTAMP
                    }

                    doc_ref = self._db.collection(collection).document(doc_id)
                    batch.update(doc_ref, doc_data)

                # バッチコミット
                batch.commit()
        except Exception as e:
            error_msg = f"Failed to update documents in {collection}: {str(e)}"
            logger.error(error_msg)
            raise FirestoreError(error_msg) from e

    @_ensure_initialized
    async def delete_document(self, collection: str, document_id: str) -> None:
        """
        ドキュメントを削除

        Args:
            collection: コレクション名
            document_id: ドキュメントID

        Raises:
            FirestoreError: Firestore操作エラー
        """
        try:
            doc_ref = self._db.collection(collection).document(document_id)
            doc_ref.delete()
        except Exception as e:
            error_msg = f"Failed to delete document {document_id} from {collection}: {str(e)}"
            logger.error(error_msg)
            raise FirestoreError(error_msg) from e

    @_ensure_initialized
    @async_timed
    async def delete_documents(self, collection: str, document_ids: List[str]) -> None:
        """
        複数のドキュメントを一括削除

        Args:
            collection: コレクション名
            document_ids: 削除するドキュメントIDのリスト

        Raises:
            FirestoreError: Firestore操作エラー
        """
        if not document_ids:
            return

        # 5000件ごとにバッチ処理（Firestoreの制限）
        MAX_BATCH_SIZE = 500

        try:
            for i in range(0, len(document_ids), MAX_BATCH_SIZE):
                batch_ids = document_ids[i:i+MAX_BATCH_SIZE]
                batch = self._db.batch()

                for doc_id in batch_ids:
                    doc_ref = self._db.collection(collection).document(doc_id)
                    batch.delete(doc_ref)

                # バッチコミット
                batch.commit()
        except Exception as e:
            error_msg = f"Failed to delete documents from {collection}: {str(e)}"
            logger.error(error_msg)
            raise FirestoreError(error_msg) from e

    @_ensure_initialized
    async def query_documents(
        self,
        collection: str,
        filters: Optional[List[QueryFilter]] = None,
        order_by: Optional[List[Dict[str, str]]] = None,
        limit: Optional[int] = None,
        start_after: Optional[DocumentDict] = None
    ) -> List[DocumentDict]:
        """
        ドキュメントをクエリ

        Args:
            collection: コレクション名
            filters: フィルター条件のリスト [{'field': 'field_name', 'op': '==', 'value': value}]
            order_by: ソート条件のリスト [{'field': 'field_name', 'direction': 'asc/desc'}]
            limit: 取得件数の上限
            start_after: このドキュメントの後から取得

        Returns:
            クエリ結果のドキュメントリスト

        Raises:
            FirestoreError: Firestore操作エラー
        """
        try:
            query = self._db.collection(collection)

            # フィルター適用
            if filters:
                for filter_condition in filters:
                    field = filter_condition.get('field')
                    op = filter_condition.get('op')
                    value = filter_condition.get('value')
                    query = query.where(field, op, value)

            # ソート適用
            if order_by:
                for sort_condition in order_by:
                    field = sort_condition.get('field')
                    direction = sort_condition.get('direction', 'asc')
                    query = query.order_by(
                        field,
                        direction=firestore.Query.DESCENDING if direction == 'desc' else firestore.Query.ASCENDING
                    )

            # カーソルページネーション
            if start_after:
                if order_by:
                    # ソート条件があれば、その順序に基づいてstart_afterを適用
                    start_doc_values = [start_after.get(sort_condition.get('field')) for sort_condition in order_by]
                    query = query.start_after(*start_doc_values)
                else:
                    # ソート条件がなければ、ドキュメント参照を使用
                    start_doc_ref = self._db.collection(collection).document(start_after.get('id'))
                    query = query.start_after(start_doc_ref)

            # リミット適用
            if limit:
                query = query.limit(limit)

            # クエリ実行
            results = []
            for doc in query.stream():
                results.append({**doc.to_dict(), "id": doc.id})

            return results
        except Exception as e:
            error_msg = f"Failed to query documents from {collection}: {str(e)}"
            logger.error(error_msg)
            raise FirestoreError(error_msg) from e

    @_ensure_initialized
    @async_timed
    async def paginated_query(
        self,
        collection: str,
        page_size: int = 100,
        filters: Optional[List[QueryFilter]] = None,
        order_by: Optional[List[Dict[str, str]]] = None,
        max_pages: Optional[int] = None
    ) -> List[List[DocumentDict]]:
        """
        ページネーション付きクエリ

        Args:
            collection: コレクション名
            page_size: 1ページあたりの取得件数
            filters: フィルター条件のリスト
            order_by: ソート条件のリスト
            max_pages: 取得する最大ページ数

        Returns:
            ページごとのドキュメントリストのリスト

        Raises:
            FirestoreError: Firestore操作エラー
        """
        try:
            pages = []
            last_doc = None
            page_count = 0

            while True:
                # ページ取得
                page = await self.query_documents(
                    collection,
                    filters=filters,
                    order_by=order_by,
                    limit=page_size,
                    start_after=last_doc
                )

                if not page:
                    break

                pages.append(page)
                page_count += 1

                # 最大ページ数に達したら終了
                if max_pages and page_count >= max_pages:
                    break

                # 次のページのスタート位置を設定
                last_doc = page[-1]

            return pages
        except Exception as e:
            error_msg = f"Failed to execute paginated query on {collection}: {str(e)}"
            logger.error(error_msg)
            raise FirestoreError(error_msg) from e

    @_ensure_initialized
    async def batch_update(self, operations: List[Dict[str, Any]]) -> None:
        """
        バッチ更新を実行

        Args:
            operations: 操作のリスト
                [
                    {'collection': 'coll_name', 'doc_id': 'doc_id', 'data': {}, 'merge': True},
                    ...
                ]

        Raises:
            FirestoreError: Firestore操作エラー
        """
        if not operations:
            return

        try:
            # 500件ごとにバッチ処理（Firestoreの制限）
            MAX_BATCH_SIZE = 500

            for i in range(0, len(operations), MAX_BATCH_SIZE):
                batch_ops = operations[i:i+MAX_BATCH_SIZE]
                batch = self._db.batch()

                for op in batch_ops:
                    collection = op.get('collection')
                    doc_id = op.get('doc_id')
                    data = op.get('data', {})
                    merge = op.get('merge', False)

                    # タイムスタンプの自動追加
                    doc_data = {
                        **data,
                        'updated_at': firestore.SERVER_TIMESTAMP
                    }

                    if not merge or not self._db.collection(collection).document(doc_id).get().exists:
                        doc_data['created_at'] = firestore.SERVER_TIMESTAMP

                    doc_ref = self._db.collection(collection).document(doc_id)
                    batch.set(doc_ref, doc_data, merge=merge)

                # バッチコミット
                batch.commit()

        except Exception as e:
            error_msg = f"Failed to execute batch update: {str(e)}"
            logger.error(error_msg)
            raise FirestoreError(error_msg) from e

    @_ensure_initialized
    @async_timed
    async def batch_operations(self, operations: List[BatchOperation]) -> None:
        """
        異なる種類の操作を含むバッチ処理

        Args:
            operations: 操作のリスト
                [
                    {'type': 'set', 'collection': 'coll_name', 'doc_id': 'doc_id', 'data': {}, 'merge': True},
                    {'type': 'update', 'collection': 'coll_name', 'doc_id': 'doc_id', 'data': {}},
                    {'type': 'delete', 'collection': 'coll_name', 'doc_id': 'doc_id'},
                    ...
                ]

        Raises:
            FirestoreError: Firestore操作エラー
        """
        if not operations:
            return

        try:
            # 500件ごとにバッチ処理（Firestoreの制限）
            MAX_BATCH_SIZE = 500

            for i in range(0, len(operations), MAX_BATCH_SIZE):
                batch_ops = operations[i:i+MAX_BATCH_SIZE]
                batch = self._db.batch()

                for op in batch_ops:
                    op_type = op.get('type')
                    collection = op.get('collection')
                    doc_id = op.get('doc_id')
                    doc_ref = self._db.collection(collection).document(doc_id)

                    if op_type == 'set':
                        data = op.get('data', {})
                        merge = op.get('merge', False)

                        # タイムスタンプの自動追加
                        doc_data = {
                            **data,
                            'updated_at': firestore.SERVER_TIMESTAMP
                        }

                        if not merge:
                            doc_data['created_at'] = firestore.SERVER_TIMESTAMP

                        batch.set(doc_ref, doc_data, merge=merge)

                    elif op_type == 'update':
                        data = op.get('data', {})

                        # タイムスタンプの自動追加
                        doc_data = {
                            **data,
                            'updated_at': firestore.SERVER_TIMESTAMP
                        }

                        batch.update(doc_ref, doc_data)

                    elif op_type == 'delete':
                        batch.delete(doc_ref)

                # バッチコミット
                batch.commit()

        except Exception as e:
            error_msg = f"Failed to execute batch operations: {str(e)}"
            logger.error(error_msg)
            raise FirestoreError(error_msg) from e

    @_ensure_initialized
    async def transaction(self, transaction_callable: Callable[[Any], T]) -> T:
        """
        トランザクションを実行

        Args:
            transaction_callable: トランザクション内で実行するコールバック関数

        Returns:
            コールバック関数の戻り値

        Raises:
            FirestoreError: Firestore操作エラー
        """
        try:
            transaction = self._db.transaction()
            return transaction.get(transaction_callable)
        except Exception as e:
            error_msg = f"Failed to execute transaction: {str(e)}"
            logger.error(error_msg)
            raise FirestoreError(error_msg) from e

    @_ensure_initialized
    @async_timed
    async def collection_group_query(
        self,
        collection_id: str,
        filters: Optional[List[QueryFilter]] = None,
        order_by: Optional[List[Dict[str, str]]] = None,
        limit: Optional[int] = None
    ) -> List[DocumentDict]:
        """
        コレクショングループクエリを実行

        Args:
            collection_id: コレクションID
            filters: フィルター条件のリスト
            order_by: ソート条件のリスト
            limit: 取得件数の上限

        Returns:
            クエリ結果のドキュメントリスト

        Raises:
            FirestoreError: Firestore操作エラー
        """
        try:
            query = self._db.collection_group(collection_id)

            # フィルター適用
            if filters:
                for filter_condition in filters:
                    field = filter_condition.get('field')
                    op = filter_condition.get('op')
                    value = filter_condition.get('value')
                    query = query.where(field, op, value)

            # ソート適用
            if order_by:
                for sort_condition in order_by:
                    field = sort_condition.get('field')
                    direction = sort_condition.get('direction', 'asc')
                    query = query.order_by(
                        field,
                        direction=firestore.Query.DESCENDING if direction == 'desc' else firestore.Query.ASCENDING
                    )

            # リミット適用
            if limit:
                query = query.limit(limit)

            # クエリ実行
            results = []
            for doc in query.stream():
                results.append({**doc.to_dict(), "id": doc.id, "path": doc.reference.path})

            return results
        except Exception as e:
            error_msg = f"Failed to execute collection group query on {collection_id}: {str(e)}"
            logger.error(error_msg)
            raise FirestoreError(error_msg) from e


class MockFirebaseClient(FirebaseClientInterface):
    """
    モックFirebaseクライアント
    テストや開発環境で使用するためのインメモリ実装
    """
    def __init__(self):
        self._collections = {}
        self._initialized = True
        logger.info("Mock Firebase Client initialized")

    async def initialize(self) -> None:
        """初期化（モックでは何もしない）"""
        pass

    async def get_document(self, collection: str, document_id: str) -> Optional[DocumentDict]:
        """ドキュメントを取得"""
        if collection not in self._collections or document_id not in self._collections[collection]:
            return None
        return {**self._collections[collection][document_id], "id": document_id}

    async def add_document(self, collection: str, data: DocumentDict) -> str:
        """ドキュメントを追加"""
        if collection not in self._collections:
            self._collections[collection] = {}

        # IDを生成
        import uuid
        document_id = str(uuid.uuid4())

        # タイムスタンプを追加
        data = {
            **data,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        self._collections[collection][document_id] = data
        return document_id

    async def set_document(self, collection: str, document_id: str, data: DocumentDict, merge: bool = False) -> None:
        """ドキュメントを設定"""
        if collection not in self._collections:
            self._collections[collection] = {}

        # タイムスタンプを更新
        current_time = datetime.now().isoformat()
        if merge and document_id in self._collections[collection]:
            # マージの場合は既存データに新しいデータをマージ
            merged_data = {**self._collections[collection][document_id], **data, "updated_at": current_time}
            self._collections[collection][document_id] = merged_data
        else:
            # 新規作成または上書きの場合
            data = {**data, "created_at": current_time, "updated_at": current_time}
            self._collections[collection][document_id] = data

    async def update_document(self, collection: str, document_id: str, data: DocumentDict) -> None:
        """ドキュメントを更新"""
        if collection not in self._collections or document_id not in self._collections[collection]:
            raise FirestoreError(f"Document {document_id} not found in {collection}")

        # タイムスタンプを更新
        data = {**data, "updated_at": datetime.now().isoformat()}

        # 既存データを更新
        self._collections[collection][document_id] = {
            **self._collections[collection][document_id],
            **data
        }

    async def delete_document(self, collection: str, document_id: str) -> None:
        """ドキュメントを削除"""
        if collection in self._collections and document_id in self._collections[collection]:
            del self._collections[collection][document_id]

    async def query_documents(
        self,
        collection: str,
        filters: Optional[List[QueryFilter]] = None,
        order_by: Optional[List[Dict[str, str]]] = None,
        limit: Optional[int] = None,
        start_after: Optional[DocumentDict] = None
    ) -> List[DocumentDict]:
        """ドキュメントをクエリ"""
        if collection not in self._collections:
            return []

        # コレクションからドキュメントを取得
        documents = []
        for doc_id, doc_data in self._collections[collection].items():
            documents.append({**doc_data, "id": doc_id})

        # フィルタを適用
        if filters:
            filtered_docs = []
            for doc in documents:
                match = True
                for filter_dict in filters:
                    field = filter_dict.get("field")
                    op = filter_dict.get("op")
                    value = filter_dict.get("value")

                    if field not in doc:
                        match = False
                        break

                    if op == "==":
                        if doc[field] != value:
                            match = False
                            break
                    elif op == "!=":
                        if doc[field] == value:
                            match = False
                            break
                    elif op == ">":
                        if doc[field] <= value:
                            match = False
                            break
                    elif op == ">=":
                        if doc[field] < value:
                            match = False
                            break
                    elif op == "<":
                        if doc[field] >= value:
                            match = False
                            break
                    elif op == "<=":
                        if doc[field] > value:
                            match = False
                            break

                if match:
                    filtered_docs.append(doc)

            documents = filtered_docs

        # ソート順を適用
        if order_by:
            for order_dict in reversed(order_by):
                field = order_dict.get("field")
                direction = order_dict.get("direction", "asc")
                reverse = (direction != "asc")
                documents = sorted(documents, key=lambda doc: doc.get(field), reverse=reverse)

        # ページネーションを適用
        if start_after:
            start_index = 0
            for i, doc in enumerate(documents):
                if doc.get("id") == start_after.get("id"):
                    start_index = i + 1
                    break
            documents = documents[start_index:]

        # 件数制限を適用
        if limit and limit > 0:
            documents = documents[:limit]

        return documents

    async def batch_update(self, operations: List[Dict[str, Any]]) -> None:
        """バッチ更新を実行"""
        for op in operations:
            operation = op.get("operation")
            collection = op.get("collection")
            document_id = op.get("document_id")
            data = op.get("data", {})
            merge = op.get("merge", False)

            if operation == "set":
                await self.set_document(collection, document_id, data, merge)
            elif operation == "update":
                await self.update_document(collection, document_id, data)
            elif operation == "delete":
                await self.delete_document(collection, document_id)

    async def transaction(self, transaction_callable: Callable[[Any], T]) -> T:
        """トランザクションを実行（モックでは単純に関数を実行）"""
        # モックトランザクションオブジェクト
        class MockTransaction:
            def get(self, doc_ref):
                collection = doc_ref.collection
                document_id = doc_ref.document_id
                if collection not in self._collections or document_id not in self._collections[collection]:
                    return None
                return {**self._collections[collection][document_id], "id": document_id}

            def set(self, doc_ref, data, merge=False):
                pass

            def update(self, doc_ref, data):
                pass

            def delete(self, doc_ref):
                pass

        mock_tx = MockTransaction()
        return transaction_callable(mock_tx)


def get_firebase_client(use_mock: bool = False) -> FirebaseClientInterface:
    """
    Firebaseクライアントのファクトリ関数

    Args:
        use_mock: モッククライアントを使用するかどうか

    Returns:
        FirebaseClientInterface
    """
    if use_mock:
        return MockFirebaseClient()
    else:
        return FirebaseClient()


# バックワード互換性のための関数
def get_firestore_client(use_mock: bool = False) -> FirebaseClientInterface:
    """
    レガシーコード互換のためのFirestoreクライアント取得関数

    Args:
        use_mock: モッククライアントを使用するかどうか

    Returns:
        FirebaseClientInterface
    """
    return get_firebase_client(use_mock)