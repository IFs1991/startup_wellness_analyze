"""
データベースモジュール
Firestore、PostgreSQL、Neo4jのハイブリッド運用を実現するデータベース接続とモデル定義を提供します。

このモジュールは低レベルなデータベース接続管理、基本的なCRUD操作、データモデル定義を担当します。
アプリケーションのデータ永続化レイヤーとして機能し、サービス層（backend.service）から利用されます。
"""

# データベース接続管理
from .database import (
    DatabaseManager,
    DatabaseType,
    DataCategory,
    get_firestore_client,
    get_db_session,
    get_db_for_category,
    init_db,
    Base
)

# モデル定義
from . import models
from . import models_sql

# Firestore CRUD操作
from . import crud

# PostgreSQL CRUD操作
from . import crud_sql

# Neo4jデータベース操作
from .neo4j import (
    Neo4jDatabaseManager,
    Neo4jService,
    get_neo4j_driver,
    get_neo4j_session,
    init_neo4j
)

# マイグレーション
from . import migration

# PostgreSQL直接アクセス（下位互換性のため）
from .postgres import get_db as get_postgres_db

__all__ = [
    # データベース管理
    'DatabaseManager',
    'DatabaseType',
    'DataCategory',
    'get_firestore_client',
    'get_db_session',
    'get_db_for_category',
    'get_postgres_db',  # 下位互換性のため
    'init_db',
    'Base',

    # モデル定義
    'models',
    'models_sql',

    # CRUD操作
    'crud',
    'crud_sql',

    # Neo4j関連
    'Neo4jDatabaseManager',
    'Neo4jService',
    'get_neo4j_driver',
    'get_neo4j_session',
    'init_neo4j',

    # マイグレーション
    'migration'
]
