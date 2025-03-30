import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
from sqlalchemy.orm import Session

from backend.database.database import (
    DatabaseManager,
    DatabaseType,
    DataCategory,
    get_firestore_client,
    get_db_session,
    get_db_for_category,
    init_db
)

class TestDatabaseManager:
    """DatabaseManagerのテストクラス"""

    def test_singleton_instance(self):
        """DatabaseManagerがシングルトンパターンで正しく動作することを確認"""
        # 2つのインスタンスが同じオブジェクトを参照することを確認
        instance1 = DatabaseManager()
        instance2 = DatabaseManager()
        assert instance1 is instance2

    @patch('backend.database.database.initialize_app')
    @patch('backend.database.database.firestore.Client')
    @patch('backend.database.database.create_engine')
    def test_initialize_databases(self, mock_create_engine, mock_firestore_client, mock_initialize_app):
        """データベースの初期化が正しく行われることを確認"""
        # テスト用のインスタンス取得
        db_manager = DatabaseManager()

        # _initialize_databasesを呼び出す
        db_manager._initialize_databases()

        # Firestoreが初期化されたことを確認
        mock_initialize_app.assert_called_once()
        mock_firestore_client.assert_called_once()

        # SQLエンジンが初期化されたことを確認
        mock_create_engine.assert_called_once()

    @patch.object(DatabaseManager, '_initialize_databases')
    @patch.object(DatabaseManager, '_firestore_client', None)
    def test_get_firestore_client(self, mock_initialize):
        """Firestoreクライアントの取得が正しく行われることを確認"""
        # モックの設定
        mock_client = MagicMock()
        DatabaseManager._firestore_client = mock_client

        # Firestoreクライアントを取得
        client = DatabaseManager.get_firestore_client()

        # 結果を確認
        assert client is mock_client
        mock_initialize.assert_not_called()  # すでに初期化されているので呼ばれないはず

    @patch.object(DatabaseManager, '_initialize_databases')
    @patch.object(DatabaseManager, '_sql_engine', None)
    @patch.object(DatabaseManager, '_sql_session', None)
    def test_get_sql_session(self, mock_initialize):
        """SQLセッションの取得が正しく行われることを確認"""
        # モックの設定
        mock_engine = MagicMock()
        mock_session = MagicMock()
        mock_session_instance = MagicMock()

        DatabaseManager._sql_engine = mock_engine
        DatabaseManager._sql_session = mock_session
        mock_session.return_value = mock_session_instance

        # セッションマネージャーでセッションを取得
        with DatabaseManager.get_sql_session() as session:
            assert session is mock_session_instance

        # 結果を確認
        mock_session.assert_called_once()
        mock_session_instance.close.assert_called_once()
        mock_initialize.assert_not_called()  # すでに初期化されているので呼ばれないはず

    @patch.object(DatabaseManager, 'get_firestore_client')
    @patch.object(DatabaseManager, 'get_sql_session')
    def test_get_db_by_type(self, mock_get_sql_session, mock_get_firestore_client):
        """タイプ指定でのデータベース取得が正しく行われることを確認"""
        # モックの設定
        mock_firestore = MagicMock()
        mock_sql = MagicMock()

        mock_get_firestore_client.return_value = mock_firestore
        mock_get_sql_session.return_value = mock_sql

        # NoSQLデータベースを取得
        db1 = DatabaseManager.get_db_by_type(DatabaseType.NOSQL)
        assert db1 is mock_firestore

        # SQLデータベースを取得
        db2 = DatabaseManager.get_db_by_type(DatabaseType.SQL)
        assert mock_get_sql_session.return_value.__enter__.called

        # 無効な値の場合
        with pytest.raises(ValueError):
            DatabaseManager.get_db_by_type("invalid_type")

    @patch.object(DatabaseManager, 'get_db_by_type')
    def test_get_db_for_data_category(self, mock_get_db_by_type):
        """データカテゴリ指定でのデータベース取得が正しく行われることを確認"""
        # SQL用カテゴリのテスト
        DatabaseManager.get_db_for_data_category(DataCategory.STRUCTURED)
        mock_get_db_by_type.assert_called_with(DatabaseType.SQL)

        mock_get_db_by_type.reset_mock()

        # NoSQL用カテゴリのテスト
        DatabaseManager.get_db_for_data_category(DataCategory.REALTIME)
        mock_get_db_by_type.assert_called_with(DatabaseType.NOSQL)

    def test_get_collection_name(self):
        """コレクション名の取得が正しく行われることを確認"""
        # USER_MASTERカテゴリのテスト
        collection_name = DatabaseManager.get_collection_name(DataCategory.USER_MASTER)
        assert collection_name == "users"

        # CHATカテゴリのテスト
        collection_name = DatabaseManager.get_collection_name(DataCategory.CHAT)
        assert collection_name == "chat_messages"

class TestDatabaseFunctions:
    """データベース関連のユーティリティ関数のテストクラス"""

    @patch('backend.database.database.DatabaseManager.get_firestore_client')
    def test_get_firestore_client(self, mock_get_firestore):
        """グローバルのget_firestore_client関数が正しく動作することを確認"""
        # モックの設定
        mock_client = MagicMock()
        mock_get_firestore.return_value = mock_client

        # 関数を呼び出し
        client = get_firestore_client()

        # 結果を確認
        assert client is mock_client
        mock_get_firestore.assert_called_once()

    @patch('backend.database.database.DatabaseManager.get_sql_session')
    def test_get_db_session(self, mock_get_sql_session):
        """グローバルのget_db_session関数が正しく動作することを確認"""
        # モックの設定
        mock_session = MagicMock()
        mock_get_sql_session.return_value = mock_session

        # 関数を呼び出し
        with get_db_session() as session:
            pass

        # 結果を確認
        mock_get_sql_session.assert_called_once()

    @patch('backend.database.database.DatabaseManager.get_db_for_data_category')
    def test_get_db_for_category(self, mock_get_db):
        """グローバルのget_db_for_category関数が正しく動作することを確認"""
        # モックの設定
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        # 関数を呼び出し（文字列指定）
        db1 = get_db_for_category("user_master")

        # 結果を確認
        assert db1 is mock_db
        mock_get_db.assert_called_once()

        mock_get_db.reset_mock()

        # 関数を呼び出し（列挙型指定）
        db2 = get_db_for_category(DataCategory.USER_MASTER)

        # 結果を確認
        assert db2 is mock_db
        mock_get_db.assert_called_once()

    @patch('backend.database.database.get_db_session')
    @patch('backend.database.database.models_sql.Base.metadata.create_all')
    def test_init_db(self, mock_create_all, mock_get_db_session):
        """init_db関数が正しく動作することを確認"""
        # モックの設定
        mock_session = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        # 関数を呼び出し
        init_db()

        # 結果を確認
        mock_create_all.assert_called_once()
        mock_get_db_session.assert_called_once()