import pytest
from unittest.mock import patch, MagicMock
import uuid
from datetime import datetime

from backend.database import crud
from backend.database.database import DataCategory

class TestUserCrud:
    """ユーザー関連のCRUD操作のテスト"""

    @patch('backend.database.crud.db')
    def test_get_user(self, mock_db):
        """ユーザー取得関数のテスト"""
        # モックの設定
        mock_doc = MagicMock()
        mock_db.collection.return_value.document.return_value.get.return_value = mock_doc
        mock_doc.exists = True
        mock_doc.to_dict.return_value = {
            "id": "test_user_id",
            "username": "testuser",
            "email": "test@example.com",
            "is_active": True
        }

        # 関数を呼び出し
        user = crud.get_user("test_user_id")

        # 結果を確認
        assert user is not None
        assert user["id"] == "test_user_id"
        assert user["username"] == "testuser"
        mock_db.collection.assert_called_once()
        mock_db.collection.return_value.document.assert_called_once_with("test_user_id")

    @patch('backend.database.crud.db')
    def test_get_user_not_found(self, mock_db):
        """存在しないユーザーの取得テスト"""
        # モックの設定
        mock_doc = MagicMock()
        mock_db.collection.return_value.document.return_value.get.return_value = mock_doc
        mock_doc.exists = False

        # 関数を呼び出し
        user = crud.get_user("nonexistent_id")

        # 結果を確認
        assert user is None
        mock_db.collection.assert_called_once()

    @patch('backend.database.crud.db')
    def test_get_user_by_username(self, mock_db):
        """ユーザー名によるユーザー取得関数のテスト"""
        # モックの設定
        mock_query = MagicMock()
        mock_doc = MagicMock()
        mock_db.collection.return_value.where.return_value.limit.return_value = mock_query
        mock_query.stream.return_value = [mock_doc]
        mock_doc.to_dict.return_value = {
            "id": "test_user_id",
            "username": "testuser",
            "email": "test@example.com"
        }

        # 関数を呼び出し
        user = crud.get_user_by_username("testuser")

        # 結果を確認
        assert user is not None
        assert user["username"] == "testuser"
        mock_db.collection.assert_called_once()
        mock_db.collection.return_value.where.assert_called_once_with('username', '==', 'testuser')

    @patch('backend.database.crud.db')
    def test_get_users(self, mock_db):
        """ユーザー一覧取得関数のテスト"""
        # モックの設定
        mock_query = MagicMock()
        mock_doc1 = MagicMock()
        mock_doc2 = MagicMock()
        mock_db.collection.return_value.limit.return_value.offset.return_value = mock_query
        mock_query.stream.return_value = [mock_doc1, mock_doc2]
        mock_doc1.to_dict.return_value = {"id": "user1", "username": "user1"}
        mock_doc2.to_dict.return_value = {"id": "user2", "username": "user2"}

        # 関数を呼び出し
        users = crud.get_users(skip=10, limit=20)

        # 結果を確認
        assert len(users) == 2
        assert users[0]["id"] == "user1"
        assert users[1]["id"] == "user2"
        mock_db.collection.assert_called_once_with('users')
        mock_db.collection.return_value.limit.assert_called_once_with(20)
        mock_db.collection.return_value.limit.return_value.offset.assert_called_once_with(10)

    @patch('backend.database.crud.db')
    def test_create_user(self, mock_db):
        """ユーザー作成関数のテスト"""
        # モックの設定
        mock_doc_ref = MagicMock()
        mock_db.collection.return_value.document.return_value = mock_doc_ref
        mock_doc_ref.id = "new_user_id"

        # テストデータ
        user_data = {
            "username": "newuser",
            "email": "new@example.com",
            "hashed_password": "hashedpwd"
        }

        # 関数を呼び出し
        new_user = crud.create_user(user_data)

        # 結果を確認
        assert new_user is not None
        assert new_user["id"] == "new_user_id"
        assert new_user["username"] == "newuser"
        mock_db.collection.assert_called_once_with('users')
        mock_db.collection.return_value.document.assert_called_once()
        mock_doc_ref.set.assert_called_once()


class TestStartupCrud:
    """スタートアップ関連のCRUD操作のテスト"""

    @patch('backend.database.crud.db')
    def test_get_startup(self, mock_db):
        """スタートアップ取得関数のテスト"""
        # モックの設定
        mock_doc = MagicMock()
        mock_db.collection.return_value.document.return_value.get.return_value = mock_doc
        mock_doc.exists = True
        mock_doc.to_dict.return_value = {
            "id": "test_startup_id",
            "name": "Test Startup",
            "industry": "Tech"
        }

        # 関数を呼び出し
        startup = crud.get_startup("test_startup_id")

        # 結果を確認
        assert startup is not None
        assert startup["id"] == "test_startup_id"
        assert startup["name"] == "Test Startup"
        mock_db.collection.assert_called_once()
        mock_db.collection.return_value.document.assert_called_once_with("test_startup_id")

    @patch('backend.database.crud.db')
    def test_get_startups(self, mock_db):
        """スタートアップ一覧取得関数のテスト"""
        # モックの設定
        mock_query = MagicMock()
        mock_doc1 = MagicMock()
        mock_doc2 = MagicMock()
        mock_db.collection.return_value.limit.return_value.offset.return_value = mock_query
        mock_query.stream.return_value = [mock_doc1, mock_doc2]
        mock_doc1.to_dict.return_value = {"id": "startup1", "name": "Startup 1"}
        mock_doc2.to_dict.return_value = {"id": "startup2", "name": "Startup 2"}

        # 関数を呼び出し
        startups = crud.get_startups(skip=5, limit=10)

        # 結果を確認
        assert len(startups) == 2
        assert startups[0]["id"] == "startup1"
        assert startups[1]["id"] == "startup2"
        mock_db.collection.assert_called_once_with('startups')
        mock_db.collection.return_value.limit.assert_called_once_with(10)
        mock_db.collection.return_value.limit.return_value.offset.assert_called_once_with(5)

    @patch('backend.database.crud.db')
    def test_create_startup(self, mock_db):
        """スタートアップ作成関数のテスト"""
        # モックの設定
        mock_doc_ref = MagicMock()
        mock_db.collection.return_value.document.return_value = mock_doc_ref
        mock_doc_ref.id = "new_startup_id"

        # テストデータ
        startup_data = {
            "name": "New Startup",
            "industry": "FinTech",
            "owner_id": "user_id"
        }

        # 関数を呼び出し
        new_startup = crud.create_startup(startup_data)

        # 結果を確認
        assert new_startup is not None
        assert new_startup["id"] == "new_startup_id"
        assert new_startup["name"] == "New Startup"
        mock_db.collection.assert_called_once_with('startups')
        mock_db.collection.return_value.document.assert_called_once()
        mock_doc_ref.set.assert_called_once()


class TestVASCrud:
    """VASデータ関連のCRUD操作のテスト"""

    @patch('backend.database.crud.db')
    def test_get_vas_data(self, mock_db):
        """VASデータ取得関数のテスト"""
        # モックの設定
        mock_doc = MagicMock()
        mock_db.collection.return_value.document.return_value.get.return_value = mock_doc
        mock_doc.exists = True
        mock_doc.to_dict.return_value = {
            "id": "test_vas_id",
            "startup_id": "test_startup_id",
            "total_score": 8.5
        }

        # 関数を呼び出し
        vas_data = crud.get_vas_data("test_vas_id")

        # 結果を確認
        assert vas_data is not None
        assert vas_data["id"] == "test_vas_id"
        assert vas_data["total_score"] == 8.5
        mock_db.collection.assert_called_once()

    @patch('backend.database.crud.db')
    def test_get_vas_datas(self, mock_db):
        """スタートアップごとのVASデータ一覧取得関数のテスト"""
        # モックの設定
        mock_query = MagicMock()
        mock_collection = MagicMock()
        mock_where = MagicMock()
        mock_doc1 = MagicMock()
        mock_doc2 = MagicMock()

        mock_db.collection.return_value = mock_collection
        mock_collection.where.return_value = mock_where
        mock_where.limit.return_value.offset.return_value = mock_query
        mock_query.stream.return_value = [mock_doc1, mock_doc2]

        mock_doc1.to_dict.return_value = {"id": "vas1", "startup_id": "startup1", "total_score": 8.0}
        mock_doc2.to_dict.return_value = {"id": "vas2", "startup_id": "startup1", "total_score": 8.5}

        # 関数を呼び出し
        vas_datas = crud.get_vas_datas("startup1", skip=0, limit=10)

        # 結果を確認
        assert len(vas_datas) == 2
        assert vas_datas[0]["id"] == "vas1"
        assert vas_datas[1]["id"] == "vas2"
        mock_db.collection.assert_called_once()
        mock_collection.where.assert_called_once_with('startup_id', '==', 'startup1')

    @patch('backend.database.crud.db')
    def test_create_vas_data(self, mock_db):
        """VASデータ作成関数のテスト"""
        # モックの設定
        mock_doc_ref = MagicMock()
        mock_db.collection.return_value.document.return_value = mock_doc_ref
        mock_doc_ref.id = "new_vas_id"

        # テストデータ
        vas_data = {
            "startup_id": "startup1",
            "total_score": 8.7,
            "product_score": 9.0,
            "team_score": 8.5
        }

        # 関数を呼び出し
        new_vas = crud.create_vas_data(vas_data)

        # 結果を確認
        assert new_vas is not None
        assert new_vas["id"] == "new_vas_id"
        assert new_vas["total_score"] == 8.7
        mock_db.collection.assert_called_once()
        mock_doc_ref.set.assert_called_once()