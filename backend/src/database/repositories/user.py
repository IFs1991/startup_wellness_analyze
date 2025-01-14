from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from .base import BaseRepository
from ..models import User, UserRole

class UserRepository(BaseRepository[User]):
    """ユーザーリポジトリクラス"""

    def __init__(self, session: AsyncSession):
        super().__init__(User, session)

    async def get_by_username(self, username: str) -> Optional[User]:
        """ユーザー名でユーザーを取得する"""
        return await self.get_by_field("username", username)

    async def get_by_email(self, email: str) -> Optional[User]:
        """メールアドレスでユーザーを取得する"""
        return await self.get_by_field("email", email)

    async def get_by_role(self, role: UserRole) -> List[User]:
        """ロールでユーザーを取得する"""
        return await self.get_many_by_field("role", role)

    async def get_active_users(self) -> List[User]:
        """アクティブなユーザーを取得する"""
        query = select(User).where(User.is_active == True)
        result = await self.session.execute(query)
        return result.scalars().all()

    async def deactivate_user(self, id: str) -> Optional[User]:
        """ユーザーを非アクティブにする"""
        return await self.update(id, is_active=False)

    async def activate_user(self, id: str) -> Optional[User]:
        """ユーザーをアクティブにする"""
        return await self.update(id, is_active=True)

    async def update_role(self, id: str, role: UserRole) -> Optional[User]:
        """ユーザーのロールを更新する"""
        return await self.update(id, role=role)

    async def update_password(self, id: str, hashed_password: str) -> Optional[User]:
        """ユーザーのパスワードを更新する"""
        return await self.update(id, hashed_password=hashed_password)