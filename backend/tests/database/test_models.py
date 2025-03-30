import pytest
from unittest.mock import patch, MagicMock
import uuid
from datetime import datetime
import json

from backend.database.models import FirestoreModel, User, FirestoreService

class TestFirestoreModel:
    """FirestoreModelの基本機能テスト"""

    def test_base_model_initialization(self):
        """FirestoreModelの初期化が正しく動作することを確認"""
        # 基本クラスからサブクラスを作成
        class TestModel(FirestoreModel):
            name: str
            value: int

            class Config:
                collection_name = "test_collection"

        # インスタンス作成
        model = TestModel(name="test", value=123)

        # 結果確認
        assert model.id is not None
        assert model.created_at is not None
        assert model.updated_at is not None
        assert model.name == "test"
        assert model.value == 123

        # コレクション名の取得確認
        assert TestModel.get_collection_name() == "test_collection"

    def test_to_dict(self):
        """to_dict メソッドが正しく動作することを確認"""
        # テスト用モデル
        class TestModel(FirestoreModel):
            name: str
            value: int
            optional_value: str = None

        # インスタンス作成とID設定
        model_id = str(uuid.uuid4())
        now = datetime.utcnow()
        model = TestModel(
            id=model_id,
            name="test",
            value=123,
            created_at=now,
            updated_at=now
        )

        # 辞書変換
        result = model.to_dict()

        # 結果確認
        assert isinstance(result, dict)
        assert result["id"] == model_id
        assert result["name"] == "test"
        assert result["value"] == 123
        assert "optional_value" not in result  # Noneの値は除外される

    def test_from_dict(self):
        """from_dict メソッドが正しく動作することを確認"""
        # テスト用モデル
        class TestModel(FirestoreModel):
            name: str
            value: int
            optional_value: str = None

        # テスト用データ
        model_id = str(uuid.uuid4())
        now = datetime.utcnow()
        data = {
            "id": model_id,
            "name": "test",
            "value": 123,
            "created_at": now,
            "updated_at": now
        }

        # dictからモデル生成
        model = TestModel.from_dict(data)

        # 結果確認
        assert model.id == model_id
        assert model.name == "test"
        assert model.value == 123
        assert model.created_at == now
        assert model.updated_at == now
        assert model.optional_value is None


class TestFirestoreService:
    """FirestoreServiceのテスト"""

    @patch('backend.database.models.firestore.Client')
    def test_init(self, mock_client):
        """FirestoreServiceの初期化が正しく動作することを確認"""
        # モックの設定
        mock_db = MagicMock()
        mock_client.return_value = mock_db

        # サービスのインスタンス化
        service = FirestoreService()

        # 結果確認
        assert service.db is mock_db
        mock_client.assert_called_once()

    @patch('backend.database.models.firestore.Client')
    async def test_create_document(self, mock_client):
        """create_document メソッドが正しく動作することを確認"""
        # モックの設定
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_doc_ref = MagicMock()

        mock_client.return_value = mock_db
        mock_db.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_doc_ref
        mock_doc_ref.id = "new_doc_id"

        # テスト用モデル
        class TestModel(FirestoreModel):
            name: str

            class Config:
                collection_name = "test_collection"

        # モデルインスタンス
        model = TestModel(name="test_doc")

        # サービスインスタンス
        service = FirestoreService()

        # メソッド呼び出し
        doc_id = await service.create_document(model)

        # 結果確認
        assert doc_id == "new_doc_id"
        mock_db.collection.assert_called_once_with("test_collection")
        mock_collection.document.assert_called_once()
        mock_doc_ref.set.assert_called_once()

    @patch('backend.database.models.firestore.Client')
    async def test_get_document(self, mock_client):
        """get_document メソッドが正しく動作することを確認"""
        # モックの設定
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_doc_ref = MagicMock()
        mock_doc_snapshot = MagicMock()

        mock_client.return_value = mock_db
        mock_db.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_doc_ref
        mock_doc_ref.get.return_value = mock_doc_snapshot
        mock_doc_snapshot.exists = True
        mock_doc_snapshot.to_dict.return_value = {"id": "doc_id", "name": "test_doc"}

        # サービスインスタンス
        service = FirestoreService()

        # メソッド呼び出し
        result = await service.get_document("test_collection", "doc_id")

        # 結果確認
        assert result is not None
        assert result["id"] == "doc_id"
        assert result["name"] == "test_doc"
        mock_db.collection.assert_called_once_with("test_collection")
        mock_collection.document.assert_called_once_with("doc_id")
        mock_doc_ref.get.assert_called_once()

    @patch('backend.database.models.firestore.Client')
    async def test_get_document_not_exists(self, mock_client):
        """存在しないドキュメントの取得テスト"""
        # モックの設定
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_doc_ref = MagicMock()
        mock_doc_snapshot = MagicMock()

        mock_client.return_value = mock_db
        mock_db.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_doc_ref
        mock_doc_ref.get.return_value = mock_doc_snapshot
        mock_doc_snapshot.exists = False

        # サービスインスタンス
        service = FirestoreService()

        # メソッド呼び出し
        result = await service.get_document("test_collection", "nonexistent_id")

        # 結果確認
        assert result is None
        mock_db.collection.assert_called_once_with("test_collection")
        mock_collection.document.assert_called_once_with("nonexistent_id")

    @patch('backend.database.models.firestore.Client')
    async def test_query_documents(self, mock_client):
        """query_documents メソッドが正しく動作することを確認"""
        # モックの設定
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_doc1 = MagicMock()
        mock_doc2 = MagicMock()

        mock_client.return_value = mock_db
        mock_db.collection.return_value = mock_collection
        mock_collection.where.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.stream.return_value = [mock_doc1, mock_doc2]

        mock_doc1.id = "doc1"
        mock_doc1.to_dict.return_value = {"name": "Doc 1"}
        mock_doc2.id = "doc2"
        mock_doc2.to_dict.return_value = {"name": "Doc 2"}

        # サービスインスタンス
        service = FirestoreService()

        # メソッド呼び出し
        results = await service.query_documents(
            "test_collection",
            filters=[("field", "==", "value")],
            limit=10
        )

        # 結果確認
        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["name"] == "Doc 1"
        assert results[1]["id"] == "doc2"
        assert results[1]["name"] == "Doc 2"

        mock_db.collection.assert_called_once_with("test_collection")
        mock_collection.where.assert_called_once_with("field", "==", "value")
        mock_query.limit.assert_called_once_with(10)