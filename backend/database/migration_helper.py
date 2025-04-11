# -*- coding: utf-8 -*-
"""
移行ヘルパーモジュール
古いCRUDモジュールから新しいリポジトリパターンへの移行を支援します。
"""
import warnings
from typing import Any, Dict, List, Optional, Type, Union, TypeVar

from .repository import Repository, DataCategory
from .repositories import repository_factory
from .models.base import BaseEntity
from .models import (
    UserEntity,
    StartupEntity,
    VASDataEntity,
    FinancialDataEntity,
    NoteEntity
)

T = TypeVar('T', bound=BaseEntity)

class MigrationHelper:
    """
    移行ヘルパークラス

    古いCRUD関数呼び出しを新しいリポジトリパターンに変換する静的メソッドを提供します。
    """

    # データカテゴリマッピング
    _entity_category_map = {
        UserEntity: DataCategory.USER_MASTER,
        StartupEntity: DataCategory.REALTIME,
        VASDataEntity: DataCategory.REALTIME,
        FinancialDataEntity: DataCategory.STRUCTURED,
        NoteEntity: DataCategory.RELATIONSHIP
    }

    @staticmethod
    def get_repository_for_entity(entity_class: Type[T]) -> Repository[T, Any]:
        """
        エンティティクラスに対応するリポジトリを取得

        Args:
            entity_class: エンティティクラス

        Returns:
            対応するリポジトリインスタンス
        """
        data_category = MigrationHelper._entity_category_map.get(entity_class)
        return repository_factory.get_repository(entity_class, data_category)

    @staticmethod
    def entity_to_dict(entity: BaseEntity) -> Dict[str, Any]:
        """エンティティをディクショナリに変換"""
        return entity.dict()

    @staticmethod
    def dict_to_entity(data: Dict[str, Any], entity_class: Type[T]) -> T:
        """ディクショナリからエンティティを作成"""
        return entity_class(**data)

    # 古いAPIと新しいAPIの橋渡し関数（移行期間中のみ使用）

    @staticmethod
    def get_user(user_id: str) -> Optional[Dict[str, Any]]:
        """ユーザーIDによるユーザー取得（移行用）"""
        warnings.warn(
            "この関数は移行用です。直接リポジトリパターンを使用してください。",
            DeprecationWarning,
            stacklevel=2
        )
        repo = MigrationHelper.get_repository_for_entity(UserEntity)
        user = repo.find_by_id(user_id)
        return MigrationHelper.entity_to_dict(user) if user else None

    @staticmethod
    def get_users(skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """全ユーザー取得（移行用）"""
        warnings.warn(
            "この関数は移行用です。直接リポジトリパターンを使用してください。",
            DeprecationWarning,
            stacklevel=2
        )
        repo = MigrationHelper.get_repository_for_entity(UserEntity)
        users = repo.find_all(skip=skip, limit=limit)
        return [MigrationHelper.entity_to_dict(user) for user in users]

    @staticmethod
    def create_user(user_data: Dict[str, Any]) -> Dict[str, Any]:
        """ユーザー作成（移行用）"""
        warnings.warn(
            "この関数は移行用です。直接リポジトリパターンを使用してください。",
            DeprecationWarning,
            stacklevel=2
        )
        entity = MigrationHelper.dict_to_entity(user_data, UserEntity)
        repo = MigrationHelper.get_repository_for_entity(UserEntity)
        saved_entity = repo.save(entity)
        return MigrationHelper.entity_to_dict(saved_entity)

    # 以下、他のエンティティタイプに対する同様の移行メソッドを実装
    # スタートアップ、VASデータ、財務データ、メモなど

    @staticmethod
    def get_startup(startup_id: str) -> Optional[Dict[str, Any]]:
        """スタートアップID検索（移行用）"""
        warnings.warn(
            "この関数は移行用です。直接リポジトリパターンを使用してください。",
            DeprecationWarning,
            stacklevel=2
        )
        repo = MigrationHelper.get_repository_for_entity(StartupEntity)
        startup = repo.find_by_id(startup_id)
        return MigrationHelper.entity_to_dict(startup) if startup else None

    @staticmethod
    def demonstrate_repository_usage():
        """
        リポジトリパターンの使用例を示す

        この関数は実際には使用されず、移行時の参考コードとして提供されています。
        """
        # ユーザーリポジトリの取得
        user_repo = repository_factory.get_repository(UserEntity, DataCategory.USER_MASTER)

        # ユーザーの取得
        user = user_repo.find_by_id("user123")

        # ユーザーのリスト取得
        users = user_repo.find_all(skip=0, limit=10)

        # 条件によるユーザー検索
        active_users = user_repo.find_by_criteria({"is_active": True})

        # 新規ユーザー作成
        new_user = UserEntity(
            email="new@example.com",
            display_name="新規ユーザー",
            is_active=True
        )
        saved_user = user_repo.save(new_user)

        # ユーザー更新
        updated_user = user_repo.update("user123", {"display_name": "更新済みユーザー"})

        # ユーザー削除
        deleted = user_repo.delete("user123")