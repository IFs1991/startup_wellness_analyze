# -*- coding: utf-8 -*-
"""
データベース接続管理モジュール
複数のデータベースバックエンドを統一的に扱うための機能を提供します。
"""
import os
import glob
from enum import Enum
from typing import Any, Dict, Optional, Union, Generator, List, Type, TypeVar, Callable
from contextlib import contextmanager
import logging
# Firebase/Firestore
from firebase_admin import credentials, firestore, initialize_app
import firebase_admin
# SQLAlchemy
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from neo4j import GraphDatabase, Driver, Session as Neo4jSession

from .repository import DataCategory
from .models.base import BaseEntity, ModelType
from .config import (
    DatabaseType,
    get_db_config,
    get_db_type_for_category,
    DataCategory as ConfigDataCategory
)

# ロガーの設定
logger = logging.getLogger(__name__)

# SQLAlchemyのベースクラス
Base = declarative_base()

# 型変数
T = TypeVar('T', bound=BaseEntity)

class Database:
    """
    統合データベース管理クラス

    複数のデータベースバックエンドを統一的に扱うための機能を提供します。
    """
    _instance = None
    _firestore_client = None
    _sql_engine = None
    _sql_session_factory = None
    _neo4j_driver = None
    _initialized = False

    def __new__(cls):
        """シングルトンパターンの実装"""
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            if not cls._initialized:
                cls._initialize()
        return cls._instance

    @classmethod
    def _initialize(cls):
        """データベース接続の初期化"""
        if cls._initialized:
            return

        # Firestore初期化
        if not firebase_admin._apps:
            try:
                firestore_config = get_db_config(DatabaseType.FIRESTORE)
                additional_params = firestore_config.additional_params or {}

                if os.getenv("FIRESTORE_EMULATOR_HOST"):
                    # エミュレータが設定されている場合
                    initialize_app(options={
                        'projectId': additional_params.get("project_id", "startup-wellness-analyze"),
                    })
                elif "credentials_path" in additional_params and additional_params["credentials_path"]:
                    # 認証情報ファイルを使用
                    cred = credentials.Certificate(additional_params["credentials_path"])
                    initialize_app(credential=cred)
                else:
                    # デフォルト認証情報を使用
                    initialize_app()

                cls._firestore_client = firestore.client()
                logger.info("Firestoreクライアントの初期化に成功しました。")
            except Exception as e:
                logger.error(f"Firestore初期化エラー: {str(e)}")
                cls._firestore_client = None
                if os.getenv("ENVIRONMENT") != "development":
                    raise
        else:
            cls._firestore_client = firestore.client()

        # PostgreSQL初期化
        try:
            postgres_config = get_db_config(DatabaseType.POSTGRESQL)
            database_url = f"postgresql://{postgres_config.username}:{postgres_config.password}@{postgres_config.host}:{postgres_config.port}/{postgres_config.database_name}"
            cls._sql_engine = create_engine(
                database_url,
                pool_size=postgres_config.pool_size,
                connect_args={
                    "connect_timeout": postgres_config.connection_timeout,
                    "sslmode": "require" if postgres_config.ssl_enabled else "disable"
                }
            )
            cls._sql_session_factory = sessionmaker(autocommit=False, autoflush=False, bind=cls._sql_engine)
            logger.info("PostgreSQLの初期化に成功しました。")
        except Exception as e:
            logger.error(f"PostgreSQL初期化エラー: {str(e)}")
            cls._sql_engine = None
            cls._sql_session_factory = None

        # Neo4j初期化
        try:
            neo4j_config = get_db_config(DatabaseType.NEO4J)
            additional_params = neo4j_config.additional_params or {}

            uri = f"neo4j://{neo4j_config.host}:{neo4j_config.port}"
            cls._neo4j_driver = GraphDatabase.driver(
                uri,
                auth=(neo4j_config.username, neo4j_config.password),
                encrypted=additional_params.get("encryption", True),
                trust=additional_params.get("trust", "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"),
                connection_timeout=neo4j_config.connection_timeout
            )
            logger.info("Neo4jの初期化に成功しました。")
        except Exception as e:
            logger.error(f"Neo4j初期化エラー: {str(e)}")
            cls._neo4j_driver = None

        cls._initialized = True

    @classmethod
    def get_firestore_client(cls) -> firestore.Client:
        """
        Firestoreクライアントを取得します

        Returns:
            firestore.Client: Firestoreクライアントインスタンス
        """
        if not cls._initialized:
            cls._initialize()
        return cls._firestore_client

    @classmethod
    @contextmanager
    def get_sql_session(cls) -> Generator[Session, None, None]:
        """
        PostgreSQLセッションを取得します

        Yields:
            Session: SQLAlchemyセッションインスタンス

        Example:
            with Database.get_sql_session() as session:
                users = session.query(User).all()
        """
        if not cls._initialized:
            cls._initialize()

        if cls._sql_session_factory is None:
            raise ValueError("PostgreSQLセッションファクトリが初期化されていません")

        session = cls._sql_session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()

    @classmethod
    def get_neo4j_driver(cls):
        """
        Neo4jドライバーを取得します

        Returns:
            Driver: Neo4jドライバーインスタンス
        """
        if not cls._initialized:
            cls._initialize()
        return cls._neo4j_driver

    @classmethod
    def get_repository(cls, entity_class: Type[T], data_category: Optional[DataCategory] = None):
        """
        エンティティに対応するリポジトリを取得します

        Args:
            entity_class: エンティティクラス
            data_category: データカテゴリ（オプション）

        Returns:
            Repository: リポジトリインスタンス
        """
        from .repositories.factory import repository_factory
        return repository_factory.get_repository(entity_class, data_category)

# 以下は利便性のための関数
def get_firestore_client() -> firestore.Client:
    """Firestoreクライアントを取得"""
    return Database.get_firestore_client()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """SQLセッションを取得"""
    with Database.get_sql_session() as session:
        yield session

def get_neo4j_driver():
    """Neo4jドライバーを取得"""
    return Database.get_neo4j_driver()

def get_repository(entity_class: Type[T], data_category: Optional[DataCategory] = None):
    """エンティティに対応するリポジトリを取得"""
    return Database.get_repository(entity_class, data_category)

# FastAPIのDependsで使用するための関数
def get_db():
    """
    FastAPIのDependsで使用するためのデータベース依存関数

    Example:
        @app.get("/items/")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    with get_db_session() as session:
        yield session

# データベース初期化
def init_db():
    """
    すべてのデータベースを初期化します。
    アプリケーション起動時に呼び出されます。
    """
    # データベース接続を初期化
    Database()

    # PostgreSQL: テーブルスキーマの作成
    try:
        from . import models_sql
        engine = Database._sql_engine
        if engine:
            # 1. マイグレーションの実行
            run_migrations()

            # 2. スキーマの適用（マイグレーションシステムが設定されるまでの暫定措置）
            apply_schemas(engine)

            # 3. シードデータの読み込み
            load_seed_data(engine)

            logger.info("SQLデータベースが正常に初期化されました")
    except ImportError:
        logger.warning("models_sqlモジュールが見つかりません。SQLテーブル初期化をスキップします。")
    except Exception as e:
        logger.error(f"SQLデータベース初期化エラー: {str(e)}")

    # Neo4j: インデックスの作成など（必要に応じて）
    try:
        driver = Database.get_neo4j_driver()
        if driver:
            with Neo4jService(driver).get_session() as session:
                # インデックス作成などの初期化処理
                pass
            logger.info("Neo4jデータベースの初期化が完了しました")
    except Exception as e:
        logger.error(f"Neo4jデータベース初期化エラー: {str(e)}")

    logger.info("データベース初期化完了")

def run_migrations():
    """
    マイグレーションを実行します。
    Alembicなどのマイグレーションツールを使用します。
    """
    try:
        import alembic.config
        from alembic import command

        # Alembicの設定ファイルのパスを取得
        alembic_ini_path = os.path.join(os.path.dirname(__file__), 'migrations', 'alembic.ini')

        if os.path.exists(alembic_ini_path):
            # Alembic設定の読み込み
            alembic_cfg = alembic.config.Config(alembic_ini_path)

            # マイグレーションの実行
            command.upgrade(alembic_cfg, "head")
            logger.info("マイグレーションが正常に実行されました")
        else:
            logger.warning("alembic.iniが見つかりません。マイグレーションをスキップします。")
    except ImportError:
        logger.warning("Alembicがインストールされていません。マイグレーションをスキップします。")
    except Exception as e:
        logger.error(f"マイグレーション実行エラー: {str(e)}")
        raise

def apply_schemas(engine):
    """
    スキーマファイルを適用します。
    マイグレーションシステムが完全に設定されるまでの暫定措置です。

    Args:
        engine: SQLAlchemyエンジン
    """
    try:
        # スキーマファイルのディレクトリ
        schema_dir = os.path.join(os.path.dirname(__file__), 'schemas')

        if not os.path.exists(schema_dir):
            logger.warning(f"スキーマディレクトリが見つかりません: {schema_dir}")
            return

        # スキーマファイルを取得
        schema_files = glob.glob(os.path.join(schema_dir, '*.sql'))

        if not schema_files:
            logger.info("適用するスキーマファイルが見つかりません")
            return

        # SQLファイルを実行
        with engine.connect() as connection:
            for schema_file in sorted(schema_files):
                logger.info(f"スキーマファイルを適用: {os.path.basename(schema_file)}")
                with open(schema_file, 'r', encoding='utf-8') as f:
                    sql_statements = f.read()
                    # 複数のステートメントを個別に実行
                    for statement in sql_statements.split(';'):
                        if statement.strip():
                            connection.execute(text(statement))
                    connection.commit()

        logger.info(f"{len(schema_files)}個のスキーマファイルが正常に適用されました")
    except Exception as e:
        logger.error(f"スキーマ適用エラー: {str(e)}")
        raise

def load_seed_data(engine):
    """
    シードデータをロードします。

    Args:
        engine: SQLAlchemyエンジン
    """
    try:
        # シードデータのディレクトリ
        seed_dir = os.path.join(os.path.dirname(__file__), 'seed')

        if not os.path.exists(seed_dir):
            logger.warning(f"シードデータディレクトリが見つかりません: {seed_dir}")
            return

        # マスターデータの読み込み
        load_seed_data_category(engine, os.path.join(seed_dir, 'master'), "マスター")

        # 重みデータの読み込み
        load_seed_data_category(engine, os.path.join(seed_dir, 'weights'), "重み")

        logger.info("シードデータの読み込みが完了しました")
    except Exception as e:
        logger.error(f"シードデータ読み込みエラー: {str(e)}")
        raise

def load_seed_data_category(engine, category_dir, category_name):
    """
    カテゴリごとのシードデータを読み込みます。

    Args:
        engine: SQLAlchemyエンジン
        category_dir: カテゴリディレクトリのパス
        category_name: カテゴリの名前（ログ用）
    """
    if not os.path.exists(category_dir):
        logger.info(f"{category_name}データディレクトリが見つかりません: {category_dir}")
        return

    # シードファイルを取得
    seed_files = glob.glob(os.path.join(category_dir, '*.sql'))

    if not seed_files:
        logger.info(f"読み込む{category_name}データファイルが見つかりません")
        return

    # SQLファイルを実行
    with engine.connect() as connection:
        for seed_file in sorted(seed_files):
            logger.info(f"{category_name}データを読み込み: {os.path.basename(seed_file)}")
            with open(seed_file, 'r', encoding='utf-8') as f:
                sql_statements = f.read()
                # 複数のステートメントを個別に実行
                for statement in sql_statements.split(';'):
                    if statement.strip():
                        connection.execute(text(statement))
                connection.commit()

    logger.info(f"{len(seed_files)}個の{category_name}データファイルが正常に読み込まれました")

# Neo4jサービスクラス
class Neo4jService:
    """
    Neo4jデータベースへのアクセスを提供するサービスクラス
    アプリケーション固有のグラフデータベース操作を実装
    """

    def __init__(self, driver: Optional[Driver] = None):
        """
        Neo4jサービスを初期化

        Args:
            driver: Neo4jドライバー（Noneの場合はデフォルトを使用）
        """
        self.driver = driver or get_neo4j_driver()

    @contextmanager
    def get_session(self) -> Generator[Neo4jSession, None, None]:
        """セッションを取得するコンテキストマネージャー"""
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None):
        """
        Cypherクエリを実行し結果を返す

        Args:
            query: 実行するCypherクエリ
            params: クエリパラメータ

        Returns:
            クエリ実行結果
        """
        with self.get_session() as session:
            return session.run(query, params or {})

    def create_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        新しいノードを作成

        Args:
            label: ノードのラベル
            properties: ノードのプロパティ

        Returns:
            作成されたノードの情報
        """
        query = f"CREATE (n:{label} $props) RETURN n"
        result = self.execute_query(query, {"props": properties})
        return result.single()[0].as_dict()

    def find_nodes(self, label: str, conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        条件に一致するノードを検索

        Args:
            label: 検索対象ノードのラベル
            conditions: 検索条件

        Returns:
            ノードのリスト
        """
        where_clause = ""
        if conditions:
            where_conditions = [f"n.{k} = ${k}" for k in conditions.keys()]
            where_clause = f"WHERE {' AND '.join(where_conditions)}"

        query = f"MATCH (n:{label}) {where_clause} RETURN n"
        result = self.execute_query(query, conditions or {})

        return [record["n"].as_dict() for record in result]

    def create_relationship(
        self,
        from_label: str,
        from_properties: Dict[str, Any],
        to_label: str,
        to_properties: Dict[str, Any],
        relationship_type: str,
        relationship_properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        2つのノード間にリレーションシップを作成

        Args:
            from_label: 開始ノードのラベル
            from_properties: 開始ノードの識別プロパティ
            to_label: 終了ノードのラベル
            to_properties: 終了ノードの識別プロパティ
            relationship_type: リレーションシップの種類
            relationship_properties: リレーションシップのプロパティ

        Returns:
            作成されたリレーションシップの情報
        """
        # 開始ノードと終了ノードのマッチング条件を構築
        from_match = " AND ".join([f"a.{k} = ${k}_from" for k in from_properties.keys()])
        to_match = " AND ".join([f"b.{k} = ${k}_to" for k in to_properties.keys()])

        # パラメータを構築
        params = {}
        for k, v in from_properties.items():
            params[f"{k}_from"] = v
        for k, v in to_properties.items():
            params[f"{k}_to"] = v

        if relationship_properties:
            params["rel_props"] = relationship_properties

        # リレーションシップ作成クエリ
        rel_props = " $rel_props" if relationship_properties else ""
        query = f"""
        MATCH (a:{from_label}), (b:{to_label})
        WHERE {from_match} AND {to_match}
        CREATE (a)-[r:{relationship_type}{rel_props}]->(b)
        RETURN r
        """

        result = self.execute_query(query, params)
        return result.single()[0].as_dict()