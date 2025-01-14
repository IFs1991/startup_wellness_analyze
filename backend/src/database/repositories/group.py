from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from .base import BaseRepository
from ..models import Group, Tag, GroupMember

class GroupRepository(BaseRepository[Group]):
    """グループリポジトリクラス"""

    def __init__(self, session: AsyncSession):
        super().__init__(Group, session)

    async def get_by_owner(self, owner_id: str) -> List[Group]:
        """オーナーIDでグループを取得する"""
        return await self.get_many_by_field("owner_id", owner_id)

    async def get_with_details(self, group_id: str) -> Optional[Group]:
        """詳細情報付きでグループを取得する"""
        query = (
            select(Group)
            .options(
                joinedload(Group.members),
                joinedload(Group.tags)
            )
            .where(Group.id == group_id)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_user_groups(self, user_id: str) -> List[Group]:
        """ユーザーが所属するグループを取得する"""
        query = (
            select(Group)
            .join(GroupMember)
            .where(GroupMember.user_id == user_id)
        )
        result = await self.session.execute(query)
        return result.scalars().all()

    async def add_member(self, group_id: str, user_id: str, role: str = "member") -> GroupMember:
        """グループにメンバーを追加する"""
        member = GroupMember(
            group_id=group_id,
            user_id=user_id,
            role=role
        )
        self.session.add(member)
        await self.session.commit()
        await self.session.refresh(member)
        return member

    async def remove_member(self, group_id: str, user_id: str) -> bool:
        """グループからメンバーを削除する"""
        query = (
            select(GroupMember)
            .where(
                GroupMember.group_id == group_id,
                GroupMember.user_id == user_id
            )
        )
        result = await self.session.execute(query)
        member = result.scalar_one_or_none()

        if member:
            await self.session.delete(member)
            await self.session.commit()
            return True
        return False

    async def update_member_role(self, group_id: str, user_id: str, role: str) -> Optional[GroupMember]:
        """メンバーのロールを更新する"""
        query = (
            select(GroupMember)
            .where(
                GroupMember.group_id == group_id,
                GroupMember.user_id == user_id
            )
        )
        result = await self.session.execute(query)
        member = result.scalar_one_or_none()

        if member:
            member.role = role
            await self.session.commit()
            await self.session.refresh(member)
            return member
        return None

    async def add_tag(self, group_id: str, name: str, description: Optional[str] = None) -> Tag:
        """グループにタグを追加する"""
        tag = Tag(
            group_id=group_id,
            name=name,
            description=description
        )
        self.session.add(tag)
        await self.session.commit()
        await self.session.refresh(tag)
        return tag

    async def remove_tag(self, tag_id: str) -> bool:
        """グループからタグを削除する"""
        query = select(Tag).where(Tag.id == tag_id)
        result = await self.session.execute(query)
        tag = result.scalar_one_or_none()

        if tag:
            await self.session.delete(tag)
            await self.session.commit()
            return True
        return False

    async def get_group_tags(self, group_id: str) -> List[Tag]:
        """グループのタグを取得する"""
        query = select(Tag).where(Tag.group_id == group_id)
        result = await self.session.execute(query)
        return result.scalars().all()

    async def search_groups(
        self,
        name: Optional[str] = None,
        owner_id: Optional[str] = None,
        member_id: Optional[str] = None,
        tag_name: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Group]:
        """グループを検索する"""
        query = select(Group)

        if name:
            query = query.where(Group.name.ilike(f"%{name}%"))
        if owner_id:
            query = query.where(Group.owner_id == owner_id)
        if member_id:
            query = query.join(GroupMember).where(GroupMember.user_id == member_id)
        if tag_name:
            query = query.join(Tag).where(Tag.name.ilike(f"%{tag_name}%"))

        query = query.order_by(Group.created_at.desc())
        query = query.limit(limit).offset(offset)

        result = await self.session.execute(query)
        return result.scalars().all()