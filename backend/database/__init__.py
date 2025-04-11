# -*- coding: utf-8 -*-
"""
データベースモジュール
Firestore、PostgreSQL、Neo4jのハイブリッド運用を実現するデータベース接続とモデル定義を提供します。

このモジュールはデータベース抽象化レイヤーを提供し、アプリケーションがデータベースの実装詳細から独立できるようにします。
"""

# 設定管理
from .config import get_database_config

# データベース接続管理
from .connection import (
    Database,
    DatabaseType,
    get_firestore_client,
    get_db_session,
    get_neo4j_driver,
    get_repository,
    init_db,
    get_db,
    Base
)

# リポジトリ抽象化
from .repository import (
    Repository,
    RepositoryFactory,
    DataCategory,
    RepositoryException,
    EntityNotFoundException,
    ValidationException
)

# リポジトリの具象実装
from .repositories import (
    FirestoreRepository,
    SQLRepository,
    Neo4jRepository,
    repository_factory
)

# モデル定義
from .models.base import BaseEntity, ModelType
from . import models
from . import models_sql

# 移行ヘルパー
from .migration_helper import MigrationHelper

# マイグレーション
from . import migration

__all__ = [
    # 設定管理
    'get_database_config',

    # データベース管理
    'Database',
    'DatabaseType',
    'get_firestore_client',
    'get_db_session',
    'get_neo4j_driver',
    'get_repository',
    'init_db',
    'get_db',
    'Base',

    # リポジトリ抽象化
    'Repository',
    'RepositoryFactory',
    'DataCategory',
    'RepositoryException',
    'EntityNotFoundException',
    'ValidationException',

    # リポジトリ実装
    'FirestoreRepository',
    'SQLRepository',
    'Neo4jRepository',
    'repository_factory',

    # モデル
    'BaseEntity',
    'ModelType',
    'models',
    'models_sql',

    # 移行ヘルパー
    'MigrationHelper',

    # マイグレーション
    'migration'
]
