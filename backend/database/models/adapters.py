"""
モデルアダプター
エンティティモデルと各データベース固有の実装の間のアダプターを提供します。
"""
from typing import Dict, Any, Type, TypeVar, Optional, List, cast
from datetime import datetime
import uuid

from .base import BaseEntity, ModelType
from .entities import (
    UserEntity,
    StartupEntity,
    VASDataEntity,
    FinancialDataEntity,
    NoteEntity
)

T = TypeVar('T', bound=BaseEntity)

class ModelAdapter:
    """
    モデルアダプターの基底クラス
    エンティティモデルとデータベース固有の表現の間の変換を行います。
    """

    @staticmethod
    def to_db_model(entity: BaseEntity) -> Dict[str, Any]:
        """
        エンティティモデルをデータベース固有の表現に変換します。

        Args:
            entity: 変換するエンティティモデル

        Returns:
            Dict[str, Any]: データベース固有の表現
        """
        return entity.to_dict()

    @staticmethod
    def from_db_model(entity_cls: Type[T], data: Dict[str, Any]) -> Optional[T]:
        """
        データベース固有の表現からエンティティモデルを作成します。

        Args:
            entity_cls: エンティティモデルのクラス
            data: データベース固有の表現

        Returns:
            Optional[T]: エンティティモデルまたはNone
        """
        if not data:
            return None
        return entity_cls.from_dict(data)


class FirestoreAdapter(ModelAdapter):
    """
    Firestore用アダプター
    エンティティモデルとFirestoreドキュメントの間の変換を行います。
    """

    @staticmethod
    def to_db_model(entity: BaseEntity) -> Dict[str, Any]:
        """エンティティモデルをFirestoreドキュメントに変換"""
        data = entity.to_dict()
        # 日付型をFirestoreで扱いやすい形式に変換
        for key, value in data.items():
            if isinstance(value, datetime):
                # そのままでOK - Firestoreクライアントが変換を処理
                pass
        return data

    @staticmethod
    def get_document_id(entity: BaseEntity) -> str:
        """エンティティからFirestoreドキュメントIDを取得"""
        return str(entity.entity_id)


class SQLAdapter(ModelAdapter):
    """
    SQL用アダプター
    エンティティモデルとSQLAlchemyモデルの間の変換を行います。
    """

    @staticmethod
    def to_db_model(entity: BaseEntity) -> Dict[str, Any]:
        """エンティティモデルをSQLデータに変換"""
        data = entity.to_dict()
        # 必要なフィールドを追加/変換
        if 'id' not in data and hasattr(entity, 'entity_id'):
            data['id'] = str(entity.entity_id)
        return data

    @staticmethod
    def get_primary_key(entity: BaseEntity) -> str:
        """エンティティからSQLプライマリキー値を取得"""
        if hasattr(entity, 'id'):
            return getattr(entity, 'id')
        return str(entity.entity_id)


class Neo4jAdapter(ModelAdapter):
    """
    Neo4j用アダプター
    エンティティモデルとNeo4jノードの間の変換を行います。
    """

    @staticmethod
    def to_db_model(entity: BaseEntity) -> Dict[str, Any]:
        """エンティティモデルをNeo4jノードプロパティに変換"""
        data = entity.to_dict()
        # 日付型をNeo4jで扱いやすい形式に変換
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    @staticmethod
    def get_node_id(entity: BaseEntity) -> str:
        """エンティティからNeo4jノードIDを取得"""
        return str(entity.entity_id)

    @staticmethod
    def get_node_labels(entity: BaseEntity) -> List[str]:
        """エンティティからNeo4jノードラベルを取得"""
        # 基本的なラベル + エンティティタイプ名
        return ["Entity", entity.__class__.__name__]


def get_adapter_for_model_type(model_type: ModelType) -> Type[ModelAdapter]:
    """
    モデルタイプに応じたアダプタークラスを取得します。

    Args:
        model_type: モデルタイプ

    Returns:
        Type[ModelAdapter]: アダプタークラス

    Raises:
        ValueError: サポートされていないモデルタイプ
    """
    if model_type == ModelType.FIRESTORE:
        return FirestoreAdapter
    elif model_type == ModelType.SQL:
        return SQLAdapter
    elif model_type == ModelType.NEO4J:
        return Neo4jAdapter
    else:
        return ModelAdapter  # デフォルトアダプター