# -*- coding: utf-8 -*-
"""
Firestoreクライアント
GoogleのクラウドドキュメントデータベースであるFirestoreへのアクセスを提供します。
"""

import os
import logging
from google.cloud import firestore
from functools import lru_cache
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import firebase_admin

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# エラークラスの定義
class StorageError(Exception):
    """ストレージ操作に関連するエラー"""
    pass

class ValidationError(Exception):
    """データバリデーションに関連するエラー"""
    pass

@lru_cache(maxsize=1)
def get_firestore_client() -> firestore.Client:
    """
    Firestoreクライアントのシングルトンインスタンスを取得します。

    Returns:
        firestore.Client: 初期化されたFirestoreクライアント

    Raises:
        EnvironmentError: 認証情報が見つからない場合
    """
    try:
        # 環境変数からキーファイルのパスを取得
        key_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

        # 環境変数が設定されていない場合はデフォルトパスを使用
        if not key_path:
            key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.keys', 'japanese-teacher-salary-b684e2053f7c.json')

        # キーファイルが存在するか確認
        if os.path.exists(key_path):
            logger.info(f"Firestoreクライアントを初期化します: {key_path}")
            firebase_admin.initialize_app(firebase_admin.credentials.Certificate(key_path))
        else:
            # キーファイルが見つからない場合はデフォルト認証情報を使用
            logger.warning(f"キーファイルが見つかりません: {key_path}。デフォルト認証情報を使用します。")
            try:
                firebase_admin.initialize_app()
            except ValueError as e:
                # すでに初期化されている場合はエラーを無視
                if 'The default Firebase app already exists.' in str(e):
                    logger.info("デフォルトのFirebaseアプリはすでに初期化されています。")
                else:
                    raise

        # Firestoreクライアントの初期化
        client = firestore.Client()
        logger.info("Firestoreクライアントが正常に初期化されました")
        return client

    except Exception as e:
        logger.error(f"Firestoreクライアントの初期化中にエラーが発生しました: {str(e)}")
        raise

# FirestoreServiceクラスの追加
class FirestoreService:
    """
    Firestoreサービスクラス
    FirestoreClientをラップし、アプリケーション向けのサービスインターフェースを提供します。
    """
    def __init__(self):
        try:
            self.client = get_firestore_client()
            logger.info("FirestoreServiceが正常に初期化されました")
        except Exception as e:
            logger.error(f"FirestoreService初期化エラー: {str(e)}")
            raise StorageError(f"Firestoreサービスの初期化に失敗しました: {str(e)}")

    async def save_document(self, collection_name: str, document_id: str, data: Dict[str, Any], merge: bool = False) -> str:
        """
        ドキュメントを保存します

        Args:
            collection_name (str): コレクション名
            document_id (str): ドキュメントID
            data (Dict[str, Any]): 保存するデータ
            merge (bool): 既存データとマージするかどうか

        Returns:
            str: ドキュメントID

        Raises:
            StorageError: ストレージ操作に失敗した場合
        """
        try:
            doc_ref = self.client.collection(collection_name).document(document_id)
            doc_ref.set(data, merge=merge)
            logger.info(f"ドキュメントが保存されました: {collection_name}/{document_id}")
            return document_id
        except Exception as e:
            logger.error(f"ドキュメント保存エラー: {str(e)}")
            raise StorageError(f"ドキュメントの保存に失敗しました: {str(e)}")

    async def get_document(self, collection_name: str, document_id: str) -> Optional[Dict[str, Any]]:
        """
        ドキュメントを取得します

        Args:
            collection_name (str): コレクション名
            document_id (str): ドキュメントID

        Returns:
            Optional[Dict[str, Any]]: ドキュメントデータ（存在しない場合はNone）

        Raises:
            StorageError: ストレージ操作に失敗した場合
        """
        try:
            doc_ref = self.client.collection(collection_name).document(document_id)
            doc = doc_ref.get()
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            logger.error(f"ドキュメント取得エラー: {str(e)}")
            raise StorageError(f"ドキュメントの取得に失敗しました: {str(e)}")

    async def query_documents(self, collection_name: str, filters: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        コレクションに対してクエリを実行します

        Args:
            collection_name (str): コレクション名
            filters (Optional[List[Dict[str, Any]]]): フィルター条件のリスト

        Returns:
            List[Dict[str, Any]]: 検索結果のリスト

        Raises:
            StorageError: ストレージ操作に失敗した場合
        """
        try:
            collection_ref = self.client.collection(collection_name)
            query = collection_ref

            if filters:
                for filter_item in filters:
                    field = filter_item.get('field')
                    op = filter_item.get('operator', '==')
                    value = filter_item.get('value')
                    query = query.where(field, op, value)

            docs = query.stream()
            results = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                results.append(data)

            return results
        except Exception as e:
            logger.error(f"クエリ実行エラー: {str(e)}")
            raise StorageError(f"クエリの実行に失敗しました: {str(e)}")

    async def update_document(self, collection_name: str, document_id: str, data: Dict[str, Any]) -> str:
        """
        ドキュメントを更新します

        Args:
            collection_name (str): コレクション名
            document_id (str): ドキュメントID
            data (Dict[str, Any]): 更新するデータ

        Returns:
            str: ドキュメントID

        Raises:
            StorageError: ストレージ操作に失敗した場合
        """
        try:
            doc_ref = self.client.collection(collection_name).document(document_id)
            doc_ref.update(data)
            logger.info(f"ドキュメントが更新されました: {collection_name}/{document_id}")
            return document_id
        except Exception as e:
            logger.error(f"ドキュメント更新エラー: {str(e)}")
            raise StorageError(f"ドキュメントの更新に失敗しました: {str(e)}")

    async def delete_document(self, collection_name: str, document_id: str) -> None:
        """
        ドキュメントを削除します

        Args:
            collection_name (str): コレクション名
            document_id (str): ドキュメントID

        Raises:
            StorageError: ストレージ操作に失敗した場合
        """
        try:
            doc_ref = self.client.collection(collection_name).document(document_id)
            doc_ref.delete()
            logger.info(f"ドキュメントが削除されました: {collection_name}/{document_id}")
        except Exception as e:
            logger.error(f"ドキュメント削除エラー: {str(e)}")
            raise StorageError(f"ドキュメントの削除に失敗しました: {str(e)}")

class FirestoreClient:
    """
    Firestoreクライアントクラス
    firebase_adminのFirestoreクライアントをラップし、データベースへの
    基本的なアクセスを提供します。
    """
    _instance = None
    _client = None

    def __new__(cls):
        """シングルトンパターンの実装"""
        if cls._instance is None:
            cls._instance = super(FirestoreClient, cls).__new__(cls)
            try:
                # firebase_adminからFirestoreクライアントを取得
                cls._client = firestore.client()
                logger.info("FirestoreClientが正常に初期化されました")
            except Exception as e:
                logger.error(f"FirestoreClient初期化エラー: {str(e)}")
                raise
        return cls._instance

    def collection(self, collection_name: str):
        """
        コレクションへの参照を取得します

        Args:
            collection_name (str): コレクション名

        Returns:
            Collection: Firestoreコレクション参照
        """
        return self._client.collection(collection_name)

    def document(self, collection_name: str, document_id: str):
        """
        ドキュメントへの参照を取得します

        Args:
            collection_name (str): コレクション名
            document_id (str): ドキュメントID

        Returns:
            DocumentReference: Firestoreドキュメント参照
        """
        return self._client.collection(collection_name).document(document_id)

    def batch(self):
        """
        書き込みバッチを作成します

        Returns:
            WriteBatch: Firestore書き込みバッチ
        """
        return self._client.batch()

    def transaction(self):
        """
        トランザクションを作成します

        Returns:
            Transaction: Firestoreトランザクション
        """
        return self._client.transaction()

    async def get_document(self, collection_name: str, document_id: str) -> Optional[Dict[str, Any]]:
        """
        ドキュメントを取得します

        Args:
            collection_name (str): コレクション名
            document_id (str): ドキュメントID

        Returns:
            Optional[Dict[str, Any]]: ドキュメントデータ（存在しない場合はNone）
        """
        try:
            doc_ref = self.document(collection_name, document_id)
            doc = doc_ref.get()
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            logger.error(f"ドキュメント取得エラー: {str(e)}")
            raise

    async def query_documents(
        self,
        collection_name: str,
        filters: Optional[List[Dict[str, Any]]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        start_after: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        コレクションに対してクエリを実行します

        Args:
            collection_name (str): コレクション名
            filters (Optional[List[Dict[str, Any]]]): フィルター条件のリスト
            order_by (Optional[str]): ソートフィールド
            limit (Optional[int]): 取得件数の上限
            start_after (Optional[Any]): 開始位置

        Returns:
            List[Dict[str, Any]]: 検索結果のリスト
        """
        try:
            query = self.collection(collection_name)

            # フィルターの適用
            if filters:
                for filter_condition in filters:
                    field = filter_condition.get('field')
                    op = filter_condition.get('operator', '==')
                    value = filter_condition.get('value')

                    if all(x is not None for x in [field, op, value]):
                        query = query.where(field, op, value)

            # ソート順の適用
            if order_by:
                direction = firestore.Query.DESCENDING if order_by.startswith('-') else firestore.Query.ASCENDING
                field_name = order_by[1:] if order_by.startswith('-') else order_by
                query = query.order_by(field_name, direction=direction)

            # 開始位置の適用
            if start_after:
                query = query.start_after(start_after)

            # 取得件数の制限
            if limit:
                query = query.limit(limit)

            # クエリの実行
            docs = query.stream()

            # 結果の変換
            results = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                results.append(data)

            return results
        except Exception as e:
            logger.error(f"クエリ実行エラー: {str(e)}")
            raise

# グローバルインスタンス
_firestore_client_instance = None

def get_firestore_client() -> FirestoreClient:
    """
    FirestoreClientのシングルトンインスタンスを取得します

    Returns:
        FirestoreClient: 初期化済みのFirestoreClientインスタンス
    """
    global _firestore_client_instance
    if _firestore_client_instance is None:
        try:
            _firestore_client_instance = FirestoreClient()
        except Exception as e:
            logger.error(f"FirestoreClient初期化中にエラーが発生しました: {str(e)}")
            logger.warning("モックFirestoreClientを使用します")
            from unittest.mock import MagicMock
            _firestore_client_instance = MagicMock()
            # モックにcollectionメソッドを追加
            _firestore_client_instance.collection = MagicMock(return_value=MagicMock())
    return _firestore_client_instance


class MockFirestoreClient:
    """
    Firestoreクライアントのモック実装
    テストや初期化失敗時のフォールバックとして使用します。
    """
    def __init__(self):
        logger.info("モックFirestoreClientが初期化されました")

    def collection(self, collection_name: str):
        """モックコレクション参照を返します"""
        logger.debug(f"モックコレクション参照: {collection_name}")
        return MockCollectionReference(collection_name)

    def document(self, collection_name: str, document_id: str):
        """モックドキュメント参照を返します"""
        logger.debug(f"モックドキュメント参照: {collection_name}/{document_id}")
        return MockDocumentReference(collection_name, document_id)

    def batch(self):
        """モックバッチを返します"""
        return MockBatch()


class MockCollectionReference:
    """モックコレクション参照"""
    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def document(self, document_id: str = None):
        """モックドキュメント参照を返します"""
        return MockDocumentReference(self.collection_name, document_id)

    def where(self, field: str, op: str, value: Any):
        """モッククエリを返します"""
        return self

    def order_by(self, field: str, direction: str = 'ASCENDING'):
        """モッククエリを返します"""
        return self

    def limit(self, count: int):
        """モッククエリを返します"""
        return self

    def stream(self):
        """空のドキュメントリストを返します"""
        return []


class MockDocumentReference:
    """モックドキュメント参照"""
    def __init__(self, collection_name: str, document_id: str = None):
        self.collection_name = collection_name
        self.id = document_id or "mock-doc-id"

    def get(self):
        """モックドキュメントスナップショットを返します"""
        return MockDocumentSnapshot(self.id)

    def set(self, data: Dict[str, Any], merge: bool = False):
        """操作をログに記録するだけです"""
        logger.debug(f"モックドキュメント設定: {self.collection_name}/{self.id}")
        return None

    def update(self, data: Dict[str, Any]):
        """操作をログに記録するだけです"""
        logger.debug(f"モックドキュメント更新: {self.collection_name}/{self.id}")
        return None

    def delete(self):
        """操作をログに記録するだけです"""
        logger.debug(f"モックドキュメント削除: {self.collection_name}/{self.id}")
        return None


class MockDocumentSnapshot:
    """モックドキュメントスナップショット"""
    def __init__(self, document_id: str):
        self.id = document_id
        self.exists = False

    def to_dict(self):
        """空の辞書を返します"""
        return {}


class MockBatch:
    """モックバッチ"""
    def __init__(self):
        self.operations = []

    def set(self, document_ref, data, merge=False):
        """操作をログに記録するだけです"""
        self.operations.append(("set", document_ref.id, data))
        return self

    def update(self, document_ref, data):
        """操作をログに記録するだけです"""
        self.operations.append(("update", document_ref.id, data))
        return self

    def delete(self, document_ref):
        """操作をログに記録するだけです"""
        self.operations.append(("delete", document_ref.id))
        return self

    def commit(self):
        """操作をログに記録するだけです"""
        logger.debug(f"モックバッチコミット: {len(self.operations)} 操作")
        return None