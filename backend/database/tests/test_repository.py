# -*- coding: utf-8 -*-
"""
リポジトリパターンの単体テスト
"""
import pytest
from unittest.mock import MagicMock, patch
import uuid
from datetime import datetime

from backend.database.repository import Repository, DataCategory
from backend.database.repositories import (
    FirestoreRepository,
    SQLRepository,
    Neo4jRepository,
    repository_factory
)
from backend.database.models.base import BaseEntity, ModelType
from backend.database.models.entities import UserEntity, StartupEntity

class TestRepositoryPattern:
    """リポジトリパターンの基本機能テスト"""

    def test_repository_factory(self):
        """リポジトリファクトリが適切なリポジトリを返すかテスト"""
        # モックの設定
        with patch('backend.database.repositories.factory.ConcreteRepositoryFactory.get_firestore_client') as mock_firestore, \
             patch('backend.database.repositories.factory.ConcreteRepositoryFactory.get_sql_session') as mock_sql, \
             patch('backend.database.repositories.factory.ConcreteRepositoryFactory.get_neo4j_driver') as mock_neo4j:

            # モックオブジェクトの作成
            mock_firestore.return_value = MagicMock()
            mock_sql.return_value = MagicMock()
            mock_neo4j.return_value = MagicMock()

            # テスト対象のメソッド呼び出し
            repo1 = repository_factory.get_repository(UserEntity, DataCategory.USER_MASTER)
            repo2 = repository_factory.get_repository(StartupEntity, DataCategory.REALTIME)
            repo3 = repository_factory.get_repository(UserEntity, DataCategory.RELATIONSHIP)

            # 検証
            assert isinstance(repo1, SQLRepository)
            assert isinstance(repo2, FirestoreRepository)
            assert isinstance(repo3, Neo4jRepository)

    def test_repository_interface_compliance(self):
        """各リポジトリ実装がインターフェースを適切に実装しているかテスト"""
        # モックの設定
        mock_firestore = MagicMock()
        mock_session = MagicMock()
        mock_driver = MagicMock()

        # リポジトリインスタンスの作成
        firestore_repo = FirestoreRepository(mock_firestore, UserEntity)
        sql_repo = SQLRepository(mock_session, MagicMock(), UserEntity)
        neo4j_repo = Neo4jRepository(mock_driver, UserEntity)

        # 各リポジトリがRepository抽象クラスのインスタンスであることを確認
        assert isinstance(firestore_repo, Repository)
        assert isinstance(sql_repo, Repository)
        assert isinstance(neo4j_repo, Repository)

        # 各リポジトリが必要なメソッドを持っていることを確認
        for repo in [firestore_repo, sql_repo, neo4j_repo]:
            assert hasattr(repo, 'find_by_id')
            assert hasattr(repo, 'find_all')
            assert hasattr(repo, 'find_by_criteria')
            assert hasattr(repo, 'save')
            assert hasattr(repo, 'update')
            assert hasattr(repo, 'delete')
            assert hasattr(repo, 'count')