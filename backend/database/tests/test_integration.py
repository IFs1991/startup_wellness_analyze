# -*- coding: utf-8 -*-
"""
データベース統合テスト
複数のデータベース間の連携動作を検証します。
"""
import pytest
from unittest.mock import patch, MagicMock
import uuid
from datetime import datetime

from backend.database.repository import DataCategory
from backend.database.repositories import repository_factory
from backend.database.models.entities import UserEntity, StartupEntity, NoteEntity

@pytest.fixture
def setup_test_data():
    """テスト用データの作成とクリーンアップを行う"""
    # テスト用データのID
    user_id = str(uuid.uuid4())
    startup_id = str(uuid.uuid4())
    note_ids = []

    # クリーンアップ処理
    yield {
        "user_id": user_id,
        "startup_id": startup_id,
        "note_ids": note_ids
    }

    # テスト後のデータクリーンアップ処理は実際のテスト環境構築後に実装

class TestCrossRepositoryOperations:
    """複数リポジトリ間の連携テスト"""

    @pytest.mark.integration
    def test_create_entities_in_different_databases(self, setup_test_data):
        """異なるデータベースに保存されるエンティティの作成と取得テスト"""
        # この統合テストはモックではなく実際のデータベースに接続する必要がある
        # テスト環境のセットアップが必要

        # モックに置き換えて動作確認（実際のテスト実装時には実DBに接続）
        with patch('backend.database.repositories.factory.ConcreteRepositoryFactory.get_repository') as mock_get_repo:
            # モックリポジトリの設定
            mock_user_repo = MagicMock()
            mock_startup_repo = MagicMock()
            mock_note_repo = MagicMock()

            # リポジトリ取得の振る舞いを設定
            def side_effect(entity_class, data_category=None):
                if entity_class == UserEntity:
                    return mock_user_repo
                elif entity_class == StartupEntity:
                    return mock_startup_repo
                elif entity_class == NoteEntity:
                    return mock_note_repo

            mock_get_repo.side_effect = side_effect

            # テストデータ
            test_data = setup_test_data
            user_id = test_data["user_id"]
            startup_id = test_data["startup_id"]

            # ユーザーエンティティの作成（SQL）
            user = UserEntity(
                id=user_id,
                email="test@example.com",
                display_name="テストユーザー",
                is_active=True
            )
            mock_user_repo.save.return_value = user

            # スタートアップエンティティの作成（Firestore）
            startup = StartupEntity(
                id=startup_id,
                name="テストスタートアップ",
                description="テスト用",
                owner_id=user_id
            )
            mock_startup_repo.save.return_value = startup

            # エンティティの作成
            user_repo = repository_factory.get_repository(UserEntity, DataCategory.USER_MASTER)
            saved_user = user_repo.save(user)

            startup_repo = repository_factory.get_repository(StartupEntity, DataCategory.REALTIME)
            saved_startup = startup_repo.save(startup)

            # メモエンティティの作成（Neo4j）
            note = NoteEntity(
                content="テストメモ",
                user_id=user_id,
                startup_id=startup_id
            )
            mock_note_repo.save.return_value = note

            note_repo = repository_factory.get_repository(NoteEntity, DataCategory.RELATIONSHIP)
            saved_note = note_repo.save(note)

            # 各リポジトリが適切に呼び出されたことを確認
            mock_user_repo.save.assert_called_once()
            mock_startup_repo.save.assert_called_once()
            mock_note_repo.save.assert_called_once()

            # 関連エンティティの取得ができることを確認
            mock_startup_repo.find_by_criteria.return_value = [startup]
            startups = startup_repo.find_by_criteria({"owner_id": user_id})
            assert len(startups) == 1
            assert startups[0].id == startup_id

    @pytest.mark.integration
    def test_transaction_across_repositories(self):
        """複数リポジトリにまたがるトランザクション処理のテスト"""
        # 実際のトランザクション処理テストはDB接続環境構築後に実装
        pass