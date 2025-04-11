"""
ユーザーリポジトリインターフェース
ユーザーエンティティのデータアクセスを抽象化します。
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from domain.entities.user import User, UserRole


class UserRepositoryInterface(ABC):
    """
    ユーザーデータへのアクセスを抽象化するリポジトリインターフェース
    """

    @abstractmethod
    async def get_by_id(self, user_id: str) -> Optional[User]:
        """
        IDによりユーザーを取得

        Args:
            user_id: 取得するユーザーのID

        Returns:
            ユーザーエンティティ、存在しない場合はNone
        """
        pass

    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        """
        メールアドレスによりユーザーを取得

        Args:
            email: 取得するユーザーのメールアドレス

        Returns:
            ユーザーエンティティ、存在しない場合はNone
        """
        pass

    @abstractmethod
    async def create(self, user: User) -> User:
        """
        ユーザーを作成

        Args:
            user: 作成するユーザーエンティティ

        Returns:
            作成されたユーザーエンティティ

        Raises:
            UserRepositoryError: ユーザー作成に失敗した場合
        """
        pass

    @abstractmethod
    async def update(self, user: User) -> User:
        """
        ユーザーを更新

        Args:
            user: 更新するユーザーエンティティ

        Returns:
            更新されたユーザーエンティティ

        Raises:
            UserRepositoryError: ユーザー更新に失敗した場合
        """
        pass

    @abstractmethod
    async def delete(self, user_id: str) -> bool:
        """
        ユーザーを削除

        Args:
            user_id: 削除するユーザーのID

        Returns:
            削除が成功したかどうか

        Raises:
            UserRepositoryError: ユーザー削除に失敗した場合
        """
        pass

    @abstractmethod
    async def list_by_company(self, company_id: str) -> List[User]:
        """
        企業に所属するユーザーを一覧取得

        Args:
            company_id: 企業ID

        Returns:
            ユーザーエンティティのリスト
        """
        pass

    @abstractmethod
    async def list_by_role(self, role: UserRole) -> List[User]:
        """
        特定の役割を持つユーザーを一覧取得

        Args:
            role: 検索する役割

        Returns:
            ユーザーエンティティのリスト
        """
        pass

    @abstractmethod
    async def search(self, query: Dict[str, Any], limit: int = 100) -> List[User]:
        """
        条件でユーザーを検索

        Args:
            query: 検索条件
            limit: 取得する最大件数

        Returns:
            条件に合致するユーザーエンティティのリスト
        """
        pass

    @abstractmethod
    async def get_active_users_count(self) -> int:
        """
        アクティブなユーザー数を取得

        Returns:
            アクティブなユーザーの数
        """
        pass

    @abstractmethod
    async def update_last_login(self, user_id: str, timestamp: datetime) -> bool:
        """
        最終ログイン時刻を更新

        Args:
            user_id: ユーザーID
            timestamp: 最終ログイン時刻

        Returns:
            更新が成功したかどうか
        """
        pass