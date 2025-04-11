# -*- coding: utf-8 -*-
"""
データリポジトリ抽象インターフェース
異なるデータベースバックエンド間で共通のインターフェースを提供します。
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar, Type, Union
from enum import Enum

# モデル型変数の定義
T = TypeVar('T')
ID = TypeVar('ID')

class DataCategory(Enum):
    """データカテゴリを定義する列挙型"""
    # 構造化データ（PostgreSQL向き）
    STRUCTURED = "structured"
    TRANSACTIONAL = "transactional"
    USER_MASTER = "user_master"
    COMPANY_MASTER = "company_master"
    EMPLOYEE_MASTER = "employee_master"
    FINANCIAL = "financial"
    BILLING = "billing"
    AUDIT_LOG = "audit_log"

    # リアルタイム/スケーラブルデータ（Firestore向き）
    REALTIME = "realtime"
    SCALABLE = "scalable"
    CHAT = "chat"
    ANALYTICS_CACHE = "analytics_cache"
    USER_SESSION = "user_session"
    SURVEY = "survey"
    TREATMENT = "treatment"
    REPORT = "report"

    # グラフデータ（Neo4j向き）
    GRAPH = "graph"
    RELATIONSHIP = "relationship"
    NETWORK = "network"
    PATH = "path"

class RepositoryException(Exception):
    """リポジトリ操作に関する例外の基底クラス"""
    pass

class EntityNotFoundException(RepositoryException):
    """エンティティが見つからない場合の例外"""
    pass

class EntityDuplicateException(RepositoryException):
    """エンティティの重複がある場合の例外"""
    pass

class ValidationException(RepositoryException):
    """データバリデーションに失敗した場合の例外"""
    pass

class Repository(Generic[T, ID], ABC):
    """
    データリポジトリの抽象基底クラス

    データストアに対するCRUD操作の標準インターフェースを定義します。
    具体的なデータベース実装はこのインターフェースを実装します。
    """

    @abstractmethod
    def find_by_id(self, id: ID) -> Optional[T]:
        """
        IDによるエンティティの取得

        Args:
            id: エンティティID

        Returns:
            エンティティまたはNone

        Raises:
            RepositoryException: リポジトリ操作エラー
        """
        pass

    @abstractmethod
    def find_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """
        全てのエンティティを取得

        Args:
            skip: スキップする件数
            limit: 取得する最大件数

        Returns:
            エンティティのリスト

        Raises:
            RepositoryException: リポジトリ操作エラー
        """
        pass

    @abstractmethod
    def find_by_criteria(self, criteria: Dict[str, Any], skip: int = 0, limit: int = 100) -> List[T]:
        """
        条件に合致するエンティティを取得

        Args:
            criteria: 検索条件
            skip: スキップする件数
            limit: 取得する最大件数

        Returns:
            条件に合致するエンティティのリスト

        Raises:
            RepositoryException: リポジトリ操作エラー
        """
        pass

    @abstractmethod
    def save(self, entity: T) -> T:
        """
        エンティティを保存

        Args:
            entity: 保存するエンティティ

        Returns:
            保存されたエンティティ（IDが割り当てられる場合あり）

        Raises:
            ValidationException: エンティティが無効
            RepositoryException: リポジトリ操作エラー
        """
        pass

    @abstractmethod
    def update(self, id: ID, data: Dict[str, Any]) -> T:
        """
        エンティティを更新

        Args:
            id: 更新するエンティティのID
            data: 更新するフィールドと値

        Returns:
            更新されたエンティティ

        Raises:
            EntityNotFoundException: エンティティが存在しない
            ValidationException: 更新データが無効
            RepositoryException: リポジトリ操作エラー
        """
        pass

    @abstractmethod
    def delete(self, id: ID) -> bool:
        """
        エンティティを削除

        Args:
            id: 削除するエンティティのID

        Returns:
            削除に成功した場合はTrue

        Raises:
            EntityNotFoundException: エンティティが存在しない
            RepositoryException: リポジトリ操作エラー
        """
        pass

    @abstractmethod
    def count(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """
        条件に合致するエンティティの数を取得

        Args:
            criteria: 検索条件（Noneの場合は全件数）

        Returns:
            エンティティの数

        Raises:
            RepositoryException: リポジトリ操作エラー
        """
        pass

class RepositoryFactory:
    """
    リポジトリファクトリ

    エンティティタイプに応じた適切なリポジトリインスタンスを提供します。
    """

    @staticmethod
    def get_repository(entity_type: Type[T], data_category: Optional[DataCategory] = None) -> Repository[T, Any]:
        """
        エンティティタイプに対応するリポジトリを取得

        Args:
            entity_type: エンティティの型
            data_category: データカテゴリ（Noneの場合は型から推測）

        Returns:
            Repository: 対応するリポジトリインスタンス

        Raises:
            ValueError: 不明なエンティティタイプまたはデータカテゴリ
        """
        # 具象クラスでオーバーライド
        raise NotImplementedError("具象クラスで実装する必要があります")