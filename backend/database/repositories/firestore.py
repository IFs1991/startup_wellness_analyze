# -*- coding: utf-8 -*-
"""
Firestoreリポジトリ実装
Firestoreをバックエンドとするリポジトリの具象実装を提供します。
"""
import logging
from typing import Any, Dict, Generic, List, Optional, TypeVar, Type, cast
from firebase_admin import firestore
import asyncio

from ..repository import Repository, RepositoryException, EntityNotFoundException, ValidationException
from ..models.base import BaseEntity

T = TypeVar('T', bound=BaseEntity)
ID = TypeVar('ID')

# ロガーの設定
logger = logging.getLogger(__name__)

class FirestoreRepository(Repository[T, ID], Generic[T, ID]):
    """
    Firestoreリポジトリの具象実装

    Firestoreをデータストアとして使用するリポジトリです。
    """

    def __init__(self, client: firestore.Client, entity_class: Type[T]):
        """
        リポジトリの初期化

        Args:
            client: Firestoreクライアント
            entity_class: エンティティクラス
        """
        self._client = client
        self._entity_class = entity_class
        self._collection_name = entity_class.get_collection_name()
        self._collection_ref = client.collection(self._collection_name)

    def find_by_id(self, id: ID) -> Optional[T]:
        """IDによるエンティティの取得"""
        try:
            doc_ref = self._collection_ref.document(str(id))
            doc = doc_ref.get()

            if not doc.exists:
                return None

            data = doc.to_dict()
            if not data:
                return None

            return self._entity_class.from_dict(data)
        except Exception as e:
            logger.error(f"Firestoreからのエンティティ取得エラー: {str(e)}")
            raise RepositoryException(f"エンティティ取得エラー: {str(e)}") from e

    def find_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """全てのエンティティを取得"""
        try:
            query = self._collection_ref.offset(skip).limit(limit)
            docs = query.stream()

            result: List[T] = []
            for doc in docs:
                data = doc.to_dict()
                if data:
                    entity = self._entity_class.from_dict(data)
                    if entity:
                        result.append(entity)

            return result
        except Exception as e:
            logger.error(f"Firestoreからの全エンティティ取得エラー: {str(e)}")
            raise RepositoryException(f"全エンティティ取得エラー: {str(e)}") from e

    def find_by_criteria(self, criteria: Dict[str, Any], skip: int = 0, limit: int = 100) -> List[T]:
        """条件に合致するエンティティを取得"""
        try:
            query = self._collection_ref

            # 条件の適用
            for field, value in criteria.items():
                if isinstance(value, dict) and "operator" in value:
                    # 演算子指定あり: {"field": {"operator": ">=", "value": 100}}
                    operator = value["operator"]
                    field_value = value["value"]
                    query = query.where(field, operator, field_value)
                else:
                    # デフォルトは等価比較
                    query = query.where(field, "==", value)

            # ページネーション
            query = query.offset(skip).limit(limit)
            docs = query.stream()

            result: List[T] = []
            for doc in docs:
                data = doc.to_dict()
                if data:
                    entity = self._entity_class.from_dict(data)
                    if entity:
                        result.append(entity)

            return result
        except Exception as e:
            logger.error(f"Firestoreからの条件付きエンティティ取得エラー: {str(e)}")
            raise RepositoryException(f"条件付きエンティティ取得エラー: {str(e)}") from e

    def save(self, entity: T) -> T:
        """エンティティを保存"""
        try:
            data = entity.to_dict()
            id_value = entity.entity_id

            if not id_value:
                # 新規作成の場合は自動生成IDを使用
                doc_ref = self._collection_ref.document()
                doc_ref.set(data)
                # TODO: エンティティにIDを設定する方法が必要
                return entity
            else:
                # 既存エンティティの更新
                doc_ref = self._collection_ref.document(str(id_value))
                doc_ref.set(data, merge=True)
                return entity
        except Exception as e:
            logger.error(f"Firestoreへのエンティティ保存エラー: {str(e)}")
            raise RepositoryException(f"エンティティ保存エラー: {str(e)}") from e

    def update(self, id: ID, data: Dict[str, Any]) -> T:
        """エンティティを更新"""
        try:
            # 既存のエンティティを確認
            entity = self.find_by_id(id)
            if not entity:
                raise EntityNotFoundException(f"ID {id} のエンティティが存在しません")

            # ドキュメント参照を取得
            doc_ref = self._collection_ref.document(str(id))

            # 部分更新を実行
            doc_ref.update(data)

            # 更新後のエンティティを取得して返す
            updated_entity = self.find_by_id(id)
            if not updated_entity:
                raise RepositoryException("更新後のエンティティの取得に失敗しました")

            return updated_entity
        except EntityNotFoundException:
            raise
        except Exception as e:
            logger.error(f"Firestoreでのエンティティ更新エラー: {str(e)}")
            raise RepositoryException(f"エンティティ更新エラー: {str(e)}") from e

    def delete(self, id: ID) -> bool:
        """エンティティを削除"""
        try:
            # 既存のエンティティを確認
            entity = self.find_by_id(id)
            if not entity:
                raise EntityNotFoundException(f"ID {id} のエンティティが存在しません")

            # ドキュメント参照を取得して削除
            doc_ref = self._collection_ref.document(str(id))
            doc_ref.delete()

            return True
        except EntityNotFoundException:
            raise
        except Exception as e:
            logger.error(f"Firestoreでのエンティティ削除エラー: {str(e)}")
            raise RepositoryException(f"エンティティ削除エラー: {str(e)}") from e

    def count(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """条件に合致するエンティティの数を取得"""
        # Firestoreには直接的なカウント機能がないため、
        # 件数取得のためにドキュメントを実際に取得する必要がある
        try:
            if criteria:
                # 条件付きクエリ
                query = self._collection_ref
                for field, value in criteria.items():
                    if isinstance(value, dict) and "operator" in value:
                        operator = value["operator"]
                        field_value = value["value"]
                        query = query.where(field, operator, field_value)
                    else:
                        query = query.where(field, "==", value)

                # Note: これは効率的でない方法。大量のデータがある場合は注意
                docs = list(query.stream())
                return len(docs)
            else:
                # 全件数
                docs = list(self._collection_ref.stream())
                return len(docs)
        except Exception as e:
            logger.error(f"Firestoreでのエンティティ件数取得エラー: {str(e)}")
            raise RepositoryException(f"エンティティ件数取得エラー: {str(e)}") from e