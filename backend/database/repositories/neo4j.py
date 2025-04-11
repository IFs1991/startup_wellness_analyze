# -*- coding: utf-8 -*-
"""
Neo4jリポジトリ実装
Neo4jグラフデータベース向けリポジトリの具象実装を提供します。
"""
import logging
from typing import Any, Dict, Generic, List, Optional, TypeVar, Type, cast
from neo4j import Driver, Session, Transaction

from ..repository import Repository, RepositoryException, EntityNotFoundException, ValidationException
from ..models.base import BaseEntity

T = TypeVar('T', bound=BaseEntity)
ID = TypeVar('ID')

# ロガーの設定
logger = logging.getLogger(__name__)

class Neo4jRepository(Repository[T, ID], Generic[T, ID]):
    """
    Neo4jリポジトリの具象実装

    Neo4jグラフデータベースをデータストアとして使用するリポジトリです。
    """

    def __init__(self, driver: Driver, entity_class: Type[T], label: Optional[str] = None):
        """
        リポジトリの初期化

        Args:
            driver: Neo4jドライバー
            entity_class: エンティティクラス
            label: Neo4jノードラベル（指定がなければコレクション名を使用）
        """
        self._driver = driver
        self._entity_class = entity_class
        self._label = label or entity_class.get_collection_name()

    def _process_result(self, result: Dict[str, Any]) -> Optional[T]:
        """Neo4j結果をモデルに変換"""
        if not result or "n" not in result:
            return None

        node = result["n"]
        # Neo4jノードのプロパティをディクショナリに変換
        props = dict(node.items())
        return self._entity_class.from_dict(props)

    def find_by_id(self, id: ID) -> Optional[T]:
        """IDによるエンティティの取得"""
        try:
            cypher = f"MATCH (n:{self._label}) WHERE n.id = $id RETURN n"
            params = {"id": str(id)}

            with self._driver.session() as session:
                result = session.run(cypher, params).single()
                return self._process_result(result) if result else None
        except Exception as e:
            logger.error(f"Neo4jからのエンティティ取得エラー: {str(e)}")
            raise RepositoryException(f"エンティティ取得エラー: {str(e)}") from e

    def find_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """全てのエンティティを取得"""
        try:
            cypher = f"MATCH (n:{self._label}) RETURN n ORDER BY n.created_at DESC SKIP $skip LIMIT $limit"
            params = {"skip": skip, "limit": limit}

            with self._driver.session() as session:
                results = session.run(cypher, params).data()
                return [self._process_result(result) for result in results if result]
        except Exception as e:
            logger.error(f"Neo4jからの全エンティティ取得エラー: {str(e)}")
            raise RepositoryException(f"全エンティティ取得エラー: {str(e)}") from e

    def find_by_criteria(self, criteria: Dict[str, Any], skip: int = 0, limit: int = 100) -> List[T]:
        """条件に合致するエンティティを取得"""
        try:
            # 条件の構築
            where_clauses = []
            params = {"skip": skip, "limit": limit}

            for field, value in criteria.items():
                if isinstance(value, dict) and "operator" in value:
                    # 演算子指定あり: {"field": {"operator": ">=", "value": 100}}
                    operator = value["operator"]
                    field_value = value["value"]
                    param_name = f"{field}_value"
                    params[param_name] = field_value

                    where_clauses.append(f"n.{field} {operator} ${param_name}")
                else:
                    # デフォルトは等価比較
                    param_name = f"{field}_value"
                    params[param_name] = value
                    where_clauses.append(f"n.{field} = ${param_name}")

            # WHERE句の構築
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

            cypher = f"MATCH (n:{self._label}) WHERE {where_clause} RETURN n ORDER BY n.created_at DESC SKIP $skip LIMIT $limit"

            with self._driver.session() as session:
                results = session.run(cypher, params).data()
                return [self._process_result(result) for result in results if result]
        except Exception as e:
            logger.error(f"Neo4jからの条件付きエンティティ取得エラー: {str(e)}")
            raise RepositoryException(f"条件付きエンティティ取得エラー: {str(e)}") from e

    def save(self, entity: T) -> T:
        """エンティティを保存"""
        try:
            data = entity.to_dict()
            id_value = entity.entity_id

            if not id_value:
                # 新規作成の場合は自動生成IDを設定
                import uuid
                id_value = str(uuid.uuid4())
                data["id"] = id_value

            # プロパティのリスト作成
            props_list = ", ".join([f"n.{k} = ${k}" for k in data.keys()])

            # MERGE文でノードを作成または更新
            cypher = f"MERGE (n:{self._label} {{id: $id}}) SET {props_list} RETURN n"

            with self._driver.session() as session:
                result = session.run(cypher, data).single()
                return self._process_result(result) if result else None
        except Exception as e:
            logger.error(f"Neo4jへのエンティティ保存エラー: {str(e)}")
            raise RepositoryException(f"エンティティ保存エラー: {str(e)}") from e

    def update(self, id: ID, data: Dict[str, Any]) -> T:
        """エンティティを更新"""
        try:
            # 既存のエンティティを確認
            entity = self.find_by_id(id)
            if not entity:
                raise EntityNotFoundException(f"ID {id} のエンティティが存在しません")

            # 更新データにIDを追加
            update_data = dict(data)
            update_data["id"] = str(id)

            # プロパティのリスト作成
            props_list = ", ".join([f"n.{k} = ${k}" for k in update_data.keys()])

            # MATCH文でノードを検索し更新
            cypher = f"MATCH (n:{self._label} {{id: $id}}) SET {props_list} RETURN n"

            with self._driver.session() as session:
                result = session.run(cypher, update_data).single()
                return self._process_result(result) if result else None
        except EntityNotFoundException:
            raise
        except Exception as e:
            logger.error(f"Neo4jでのエンティティ更新エラー: {str(e)}")
            raise RepositoryException(f"エンティティ更新エラー: {str(e)}") from e

    def delete(self, id: ID) -> bool:
        """エンティティを削除"""
        try:
            # 既存のエンティティを確認
            entity = self.find_by_id(id)
            if not entity:
                raise EntityNotFoundException(f"ID {id} のエンティティが存在しません")

            # ノードを削除
            cypher = f"MATCH (n:{self._label} {{id: $id}}) DETACH DELETE n"
            params = {"id": str(id)}

            with self._driver.session() as session:
                session.run(cypher, params)

            return True
        except EntityNotFoundException:
            raise
        except Exception as e:
            logger.error(f"Neo4jでのエンティティ削除エラー: {str(e)}")
            raise RepositoryException(f"エンティティ削除エラー: {str(e)}") from e

    def count(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """条件に合致するエンティティの数を取得"""
        try:
            # 条件の構築
            where_clauses = []
            params = {}

            if criteria:
                for field, value in criteria.items():
                    if isinstance(value, dict) and "operator" in value:
                        operator = value["operator"]
                        field_value = value["value"]
                        param_name = f"{field}_value"
                        params[param_name] = field_value

                        where_clauses.append(f"n.{field} {operator} ${param_name}")
                    else:
                        param_name = f"{field}_value"
                        params[param_name] = value
                        where_clauses.append(f"n.{field} = ${param_name}")

            # WHERE句の構築
            where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            cypher = f"MATCH (n:{self._label}) {where_clause} RETURN count(n) as count"

            with self._driver.session() as session:
                result = session.run(cypher, params).single()
                return result["count"] if result else 0
        except Exception as e:
            logger.error(f"Neo4jでのエンティティ件数取得エラー: {str(e)}")
            raise RepositoryException(f"エンティティ件数取得エラー: {str(e)}") from e