# -*- coding: utf-8 -*-
"""
SQLリポジトリ実装
SQLAlchemyを使用したSQLデータベース向けリポジトリの具象実装を提供します。
"""
import logging
from typing import Any, Dict, Generic, List, Optional, TypeVar, Type, cast, get_type_hints
from sqlalchemy.orm import Session
from sqlalchemy.inspection import inspect

from ..repository import Repository, RepositoryException, EntityNotFoundException, ValidationException
from ..models.base import BaseEntity
from ..models.adapters import SQLAdapter

T = TypeVar('T', bound=BaseEntity)
ID = TypeVar('ID')

# ロガーの設定
logger = logging.getLogger(__name__)

class SQLRepository(Repository[T, ID], Generic[T, ID]):
    """
    SQLリポジトリの具象実装

    SQLAlchemyを使用してSQL関係データベースにアクセスするリポジトリです。
    """

    def __init__(self, session: Session, entity_class: Type[Any], model_class: Type[T]):
        """
        リポジトリの初期化

        Args:
            session: SQLAlchemyセッション
            entity_class: SQLAlchemyエンティティクラス（モデル）
            model_class: 抽象化層のエンティティクラス
        """
        self._session = session
        self._entity_class = entity_class
        self._model_class = model_class
        self._adapter = SQLAdapter()

    def _to_orm_entity(self, model: T) -> Any:
        """モデルをORMエンティティに変換"""
        # アダプターを使用してデータを変換
        data = self._adapter.to_db_model(model)

        # ID関連フィールドを処理（必要に応じて）
        primary_key_name = inspect(self._entity_class).primary_key[0].name
        if hasattr(model, "entity_id") and model.entity_id:
            data[primary_key_name] = model.entity_id

        # 既存エンティティの場合は更新、それ以外は新規作成
        if hasattr(model, "entity_id") and model.entity_id:
            existing = self._session.query(self._entity_class).get(model.entity_id)
            if existing:
                for key, value in data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                return existing

        # データフィールドをエンティティクラスで利用可能なものに制限
        valid_fields = {c.name for c in inspect(self._entity_class).mapper.column_attrs}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return self._entity_class(**filtered_data)

    def _to_model(self, orm_entity: Any) -> Optional[T]:
        """ORMエンティティをモデルに変換"""
        if not orm_entity:
            return None

        # ORMエンティティをディクショナリに変換
        data = {c.name: getattr(orm_entity, c.name) for c in inspect(orm_entity).mapper.column_attrs}

        # アダプターを使用してモデルを生成
        return self._adapter.from_db_model(self._model_class, data)

    def find_by_id(self, id: ID) -> Optional[T]:
        """IDによるエンティティの取得"""
        try:
            orm_entity = self._session.query(self._entity_class).get(id)
            return self._to_model(orm_entity)
        except Exception as e:
            logger.error(f"SQLからのエンティティ取得エラー: {str(e)}")
            raise RepositoryException(f"エンティティ取得エラー: {str(e)}") from e

    def find_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """全てのエンティティを取得"""
        try:
            orm_entities = self._session.query(self._entity_class)\
                .offset(skip).limit(limit).all()

            return [self._to_model(entity) for entity in orm_entities if entity]
        except Exception as e:
            logger.error(f"SQLからの全エンティティ取得エラー: {str(e)}")
            raise RepositoryException(f"全エンティティ取得エラー: {str(e)}") from e

    def find_by_criteria(self, criteria: Dict[str, Any], skip: int = 0, limit: int = 100) -> List[T]:
        """条件に合致するエンティティを取得"""
        try:
            query = self._session.query(self._entity_class)

            # 条件の適用
            for field, value in criteria.items():
                if hasattr(self._entity_class, field):
                    column = getattr(self._entity_class, field)
                    if isinstance(value, dict) and "operator" in value:
                        # 演算子指定あり: {"field": {"operator": ">=", "value": 100}}
                        operator = value["operator"]
                        field_value = value["value"]

                        if operator == "==":
                            query = query.filter(column == field_value)
                        elif operator == "!=":
                            query = query.filter(column != field_value)
                        elif operator == ">":
                            query = query.filter(column > field_value)
                        elif operator == ">=":
                            query = query.filter(column >= field_value)
                        elif operator == "<":
                            query = query.filter(column < field_value)
                        elif operator == "<=":
                            query = query.filter(column <= field_value)
                        elif operator == "like":
                            query = query.filter(column.like(field_value))
                        elif operator == "in":
                            query = query.filter(column.in_(field_value))
                    else:
                        # デフォルトは等価比較
                        query = query.filter(column == value)

            # ページネーション
            orm_entities = query.offset(skip).limit(limit).all()
            return [self._to_model(entity) for entity in orm_entities if entity]
        except Exception as e:
            logger.error(f"SQLからの条件付きエンティティ取得エラー: {str(e)}")
            raise RepositoryException(f"条件付きエンティティ取得エラー: {str(e)}") from e

    def save(self, entity: T) -> T:
        """エンティティを保存"""
        try:
            orm_entity = self._to_orm_entity(entity)

            # IDがない場合は新規作成
            if not hasattr(entity, "entity_id") or not entity.entity_id:
                self._session.add(orm_entity)
                self._session.flush()  # IDを生成するためにフラッシュ

                # 生成されたIDを取得
                pk_name = inspect(self._entity_class).primary_key[0].name
                generated_id = getattr(orm_entity, pk_name)

                # モデルを再取得（生成されたIDで）
                self._session.commit()
                return self.find_by_id(generated_id)
            else:
                # 既存エンティティの更新
                self._session.merge(orm_entity)
                self._session.commit()
                return self.find_by_id(entity.entity_id)
        except Exception as e:
            self._session.rollback()
            logger.error(f"SQLへのエンティティ保存エラー: {str(e)}")
            raise RepositoryException(f"エンティティ保存エラー: {str(e)}") from e

    def update(self, id: ID, data: Dict[str, Any]) -> T:
        """エンティティを更新"""
        try:
            # 既存のエンティティを取得
            orm_entity = self._session.query(self._entity_class).get(id)
            if not orm_entity:
                raise EntityNotFoundException(f"ID {id} のエンティティが存在しません")

            # 属性を更新（有効なフィールドのみ）
            valid_fields = {c.name for c in inspect(self._entity_class).mapper.column_attrs}
            for key, value in data.items():
                if key in valid_fields and hasattr(orm_entity, key):
                    setattr(orm_entity, key, value)

            self._session.commit()
            return self.find_by_id(id)
        except EntityNotFoundException:
            raise
        except Exception as e:
            self._session.rollback()
            logger.error(f"SQLでのエンティティ更新エラー: {str(e)}")
            raise RepositoryException(f"エンティティ更新エラー: {str(e)}") from e

    def delete(self, id: ID) -> bool:
        """エンティティを削除"""
        try:
            # 既存のエンティティを取得
            orm_entity = self._session.query(self._entity_class).get(id)
            if not orm_entity:
                raise EntityNotFoundException(f"ID {id} のエンティティが存在しません")

            # エンティティを削除
            self._session.delete(orm_entity)
            self._session.commit()

            return True
        except EntityNotFoundException:
            raise
        except Exception as e:
            self._session.rollback()
            logger.error(f"SQLでのエンティティ削除エラー: {str(e)}")
            raise RepositoryException(f"エンティティ削除エラー: {str(e)}") from e

    def count(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """条件に合致するエンティティの数を取得"""
        try:
            query = self._session.query(self._entity_class)

            # 検索条件の適用
            if criteria:
                for field, value in criteria.items():
                    if hasattr(self._entity_class, field):
                        column = getattr(self._entity_class, field)
                        if isinstance(value, dict) and "operator" in value:
                            # 演算子指定ありの処理
                            operator = value["operator"]
                            field_value = value["value"]

                            if operator == "==":
                                query = query.filter(column == field_value)
                            elif operator == "!=":
                                query = query.filter(column != field_value)
                            # 他の演算子も同様に処理
                        else:
                            # デフォルトは等価比較
                            query = query.filter(column == value)

            return query.count()
        except Exception as e:
            logger.error(f"SQLでのエンティティカウントエラー: {str(e)}")
            raise RepositoryException(f"エンティティカウントエラー: {str(e)}") from e