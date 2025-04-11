# -*- coding: utf-8 -*-
"""
データモデル基底クラス
異なるデータベース間で一貫したデータモデルインターフェースを提供します。
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, TypeVar, Type, ClassVar
from enum import Enum
from pydantic import BaseModel, Field

# 型変数の定義
T = TypeVar('T', bound='BaseEntity')

class ModelType(Enum):
    """モデルタイプの列挙型"""
    FIRESTORE = "firestore"
    SQL = "sql"
    NEO4J = "neo4j"
    MEMORY = "memory"

class BaseEntity(BaseModel):
    """
    全データモデルの基底クラス

    全てのモデルに共通するフィールドと挙動を定義します。
    """
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # クラス変数
    model_type: ClassVar[ModelType] = ModelType.MEMORY

    @classmethod
    @abstractmethod
    def get_collection_name(cls) -> str:
        """
        このモデルに対応するコレクション/テーブル名を取得

        Returns:
            str: コレクション/テーブル名
        """
        pass

    @property
    @abstractmethod
    def entity_id(self) -> Any:
        """
        エンティティを一意に識別するID

        Returns:
            Any: エンティティID
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        モデルをディクショナリに変換

        Returns:
            Dict[str, Any]: モデルのディクショナリ表現
        """
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> Optional[T]:
        """
        ディクショナリからモデルを生成

        Args:
            data: ディクショナリ形式のデータ

        Returns:
            Optional[T]: モデルインスタンスまたはNone
        """
        if not data:
            return None
        return cls(**data)

    def __str__(self) -> str:
        """文字列表現"""
        return f"{self.__class__.__name__}(id={self.entity_id}, updated_at={self.updated_at})"