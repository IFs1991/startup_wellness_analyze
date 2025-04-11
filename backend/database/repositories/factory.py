# -*- coding: utf-8 -*-
"""
リポジトリファクトリ実装
データカテゴリや型に基づいて適切なリポジトリインスタンスを提供します。
"""
import logging
from typing import Any, Dict, Generic, List, Optional, TypeVar, Type, Union, cast

from firebase_admin import firestore
from sqlalchemy.orm import Session
from neo4j import Driver

from ..repository import Repository, RepositoryFactory, DataCategory as RepoDataCategory
from ..models.base import BaseEntity, ModelType
from ..models.adapters import get_adapter_for_model_type
from .firestore import FirestoreRepository
from .sql import SQLRepository
from .neo4j import Neo4jRepository
from ..config import (
    DatabaseType,
    get_db_config,
    get_db_type_for_category,
    DataCategory as ConfigDataCategory
)

T = TypeVar('T', bound=BaseEntity)

# ロガーの設定
logger = logging.getLogger(__name__)

# データカテゴリマッピング（リポジトリのDataCategoryとconfig.pyのDataCategoryの変換）
DATA_CATEGORY_MAPPING = {
    # PostgreSQL向きカテゴリ
    RepoDataCategory.STRUCTURED: ConfigDataCategory.HEALTH_METRICS,
    RepoDataCategory.TRANSACTIONAL: ConfigDataCategory.ANALYTICS,
    RepoDataCategory.USER_MASTER: ConfigDataCategory.USER_PROFILE,
    RepoDataCategory.COMPANY_MASTER: ConfigDataCategory.USER_PROFILE,
    RepoDataCategory.EMPLOYEE_MASTER: ConfigDataCategory.USER_PROFILE,
    RepoDataCategory.FINANCIAL: ConfigDataCategory.ANALYTICS,
    RepoDataCategory.BILLING: ConfigDataCategory.ANALYTICS,
    RepoDataCategory.AUDIT_LOG: ConfigDataCategory.ACTIVITY_LOG,

    # Firestore向きカテゴリ
    RepoDataCategory.REALTIME: ConfigDataCategory.ACTIVITY_LOG,
    RepoDataCategory.SCALABLE: ConfigDataCategory.ACTIVITY_LOG,
    RepoDataCategory.CHAT: ConfigDataCategory.ACTIVITY_LOG,
    RepoDataCategory.ANALYTICS_CACHE: ConfigDataCategory.ACTIVITY_LOG,
    RepoDataCategory.USER_SESSION: ConfigDataCategory.USER_PROFILE,
    RepoDataCategory.SURVEY: ConfigDataCategory.HEALTH_METRICS,
    RepoDataCategory.TREATMENT: ConfigDataCategory.HEALTH_METRICS,
    RepoDataCategory.REPORT: ConfigDataCategory.HEALTH_METRICS,

    # Neo4j向きカテゴリ
    RepoDataCategory.GRAPH: ConfigDataCategory.SOCIAL_GRAPH,
    RepoDataCategory.RELATIONSHIP: ConfigDataCategory.SOCIAL_GRAPH,
    RepoDataCategory.NETWORK: ConfigDataCategory.SOCIAL_GRAPH,
    RepoDataCategory.PATH: ConfigDataCategory.RECOMMENDATIONS,
}

class ConcreteRepositoryFactory(RepositoryFactory):
    """
    具象リポジトリファクトリ

    エンティティタイプとデータカテゴリに基づいて適切なリポジトリインスタンスを提供します。
    """

    _firestore_client: Optional[firestore.Client] = None
    _sql_session_factory: Optional[Any] = None  # SessionMaker
    _neo4j_driver: Optional[Driver] = None

    # データカテゴリとリポジトリタイプのマッピング
    _repository_mapping = {
        # PostgreSQL向きカテゴリ
        RepoDataCategory.STRUCTURED: ModelType.SQL,
        RepoDataCategory.TRANSACTIONAL: ModelType.SQL,
        RepoDataCategory.USER_MASTER: ModelType.SQL,
        RepoDataCategory.COMPANY_MASTER: ModelType.SQL,
        RepoDataCategory.EMPLOYEE_MASTER: ModelType.SQL,
        RepoDataCategory.FINANCIAL: ModelType.SQL,
        RepoDataCategory.BILLING: ModelType.SQL,
        RepoDataCategory.AUDIT_LOG: ModelType.SQL,

        # Firestore向きカテゴリ
        RepoDataCategory.REALTIME: ModelType.FIRESTORE,
        RepoDataCategory.SCALABLE: ModelType.FIRESTORE,
        RepoDataCategory.CHAT: ModelType.FIRESTORE,
        RepoDataCategory.ANALYTICS_CACHE: ModelType.FIRESTORE,
        RepoDataCategory.USER_SESSION: ModelType.FIRESTORE,
        RepoDataCategory.SURVEY: ModelType.FIRESTORE,
        RepoDataCategory.TREATMENT: ModelType.FIRESTORE,
        RepoDataCategory.REPORT: ModelType.FIRESTORE,

        # Neo4j向きカテゴリ
        RepoDataCategory.GRAPH: ModelType.NEO4J,
        RepoDataCategory.RELATIONSHIP: ModelType.NEO4J,
        RepoDataCategory.NETWORK: ModelType.NEO4J,
        RepoDataCategory.PATH: ModelType.NEO4J,
    }

    @classmethod
    def set_firestore_client(cls, client: firestore.Client) -> None:
        """Firestoreクライアントを設定"""
        cls._firestore_client = client

    @classmethod
    def set_sql_session_factory(cls, session_factory: Any) -> None:
        """SQLセッションファクトリを設定"""
        cls._sql_session_factory = session_factory

    @classmethod
    def set_neo4j_driver(cls, driver: Driver) -> None:
        """Neo4jドライバーを設定"""
        cls._neo4j_driver = driver

    @classmethod
    def get_firestore_client(cls) -> firestore.Client:
        """Firestoreクライアントを取得"""
        if cls._firestore_client is None:
            from ..connection import get_firestore_client
            cls._firestore_client = get_firestore_client()
        return cls._firestore_client

    @classmethod
    def get_sql_session(cls) -> Session:
        """SQLセッションを取得"""
        if cls._sql_session_factory is None:
            from ..connection import get_db_session
            session = next(get_db_session())
            return session
        else:
            return cls._sql_session_factory()

    @classmethod
    def get_neo4j_driver(cls) -> Driver:
        """Neo4jドライバーを取得"""
        if cls._neo4j_driver is None:
            from ..connection import get_neo4j_driver
            cls._neo4j_driver = get_neo4j_driver()
        return cls._neo4j_driver

    @classmethod
    def determine_repository_type(cls, entity_class: Type[T], data_category: Optional[RepoDataCategory] = None) -> ModelType:
        """
        リポジトリタイプを決定

        Args:
            entity_class: エンティティクラス
            data_category: データカテゴリ

        Returns:
            ModelType: 適切なモデルタイプ
        """
        # 1. データカテゴリが指定されている場合はそれを使用
        if data_category and data_category in cls._repository_mapping:
            return cls._repository_mapping[data_category]

        # 2. エンティティクラスに定義されているモデルタイプを使用
        if hasattr(entity_class, "model_type"):
            return entity_class.model_type

        # 3. デフォルトのリポジトリタイプを使用（新しいconfigモジュールから取得）
        try:
            # 新しいconfigモジュールを使用
            config_data_category = ConfigDataCategory.USER_PROFILE  # デフォルト
            db_type = get_db_type_for_category(config_data_category)

            if db_type == DatabaseType.POSTGRESQL:
                return ModelType.SQL
            elif db_type == DatabaseType.FIRESTORE:
                return ModelType.FIRESTORE
            elif db_type == DatabaseType.NEO4J:
                return ModelType.NEO4J
            else:
                return ModelType.FIRESTORE
        except Exception as e:
            logger.warning(f"新しいconfigモジュールの使用中にエラーが発生しました: {e}")
            return ModelType.FIRESTORE

    @classmethod
    def get_repository(cls, entity_class: Type[T], data_category: Optional[RepoDataCategory] = None) -> Repository[T, Any]:
        """
        エンティティタイプに対応するリポジトリを取得

        Args:
            entity_class: エンティティの型
            data_category: データカテゴリ（Noneの場合は型から推測）

        Returns:
            Repository: 対応するリポジトリインスタンス

        Raises:
            ValueError: 不明なエンティティタイプまたはデータカテゴリ
        """
        repo_type = cls.determine_repository_type(entity_class, data_category)

        # モデルタイプに応じたアダプターを準備（必要に応じて）
        adapter_class = get_adapter_for_model_type(repo_type)

        if repo_type == ModelType.FIRESTORE:
            client = cls.get_firestore_client()
            return FirestoreRepository(client, entity_class)

        elif repo_type == ModelType.SQL:
            session = cls.get_sql_session()

            # SQLモデルのマッピングを取得
            try:
                # 新システム: エンティティタイプからORMクラスへのマッピング
                from ..models_sql import get_orm_model_for_entity
                sql_model_class = get_orm_model_for_entity(entity_class)
            except (ImportError, ValueError):
                # フォールバック: エンティティのコレクション名とORMクラスのテーブル名で一致を試みる
                from sqlalchemy.ext.declarative import DeclarativeMeta
                from sqlalchemy import inspect as sqla_inspect
                from ..connection import Base

                collection_name = entity_class.get_collection_name()

                # SQLAlchemyモデルクラスを検索
                for model_cls in Base.__subclasses__():
                    if hasattr(model_cls, '__tablename__') and model_cls.__tablename__ == collection_name:
                        sql_model_class = model_cls
                        break
                else:
                    raise ValueError(f"エンティティクラス {entity_class.__name__} に対応するSQLモデルが見つかりません")

            return SQLRepository(session, sql_model_class, entity_class)

        elif repo_type == ModelType.NEO4J:
            driver = cls.get_neo4j_driver()
            return Neo4jRepository(driver, entity_class)

        else:
            raise ValueError(f"不明なリポジトリタイプ: {repo_type}")

# シングルトンとしてのファクトリインスタンスを提供
repository_factory = ConcreteRepositoryFactory