from typing import Generic, TypeVar, Type, Optional, List, Any, Dict, cast
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.sql import Select
from sqlalchemy.orm import DeclarativeBase, Mapped
from ..models import Base

ModelType = TypeVar('ModelType', bound=DeclarativeBase)

class BaseRepository(Generic[ModelType]):
    """基本リポジトリクラス"""

    def __init__(self, model: Type[ModelType], session: AsyncSession):
        self.model = model
        self.session = session

    async def create(self, **kwargs) -> ModelType:
        """新しいエンティティを作成する"""
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.commit()
        await self.session.refresh(instance)
        return instance

    async def update(self, id: str, **kwargs) -> Optional[ModelType]:
        """エンティティを更新する"""
        query = (
            update(self.model)
            .where(self.model.id == id)
            .values(**kwargs)
            .returning(self.model)
        )
        result = await self.session.execute(query)
        await self.session.commit()
        return result.scalar_one_or_none()

    async def delete(self, id: str) -> bool:
        """エンティティを削除する"""
        query = delete(self.model).where(self.model.id == id)
        result = await self.session.execute(query)
        await self.session.commit()
        return result.rowcount > 0

    async def get_by_id(self, id: str) -> Optional[ModelType]:
        """IDでエンティティを取得する"""
        query = select(self.model).where(self.model.id == id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_all(self) -> List[ModelType]:
        """全てのエンティティを取得する"""
        query = select(self.model)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_by_field(self, field: str, value: Any) -> Optional[ModelType]:
        """フィールドの値でエンティティを取得する"""
        query = select(self.model).where(getattr(self.model, field) == value)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_many_by_field(self, field: str, value: Any) -> List[ModelType]:
        """フィールドの値で複数のエンティティを取得する"""
        query = select(self.model).where(getattr(self.model, field) == value)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def exists(self, **kwargs) -> bool:
        """条件に一致するエンティティが存在するか確認する"""
        conditions = [getattr(self.model, k) == v for k, v in kwargs.items()]
        query = select(self.model).where(*conditions)
        result = await self.session.execute(query)
        return result.first() is not None

    async def count(self, **kwargs) -> int:
        """条件に一致するエンティティの数を取得する"""
        conditions = [getattr(self.model, k) == v for k, v in kwargs.items()]
        query = select(self.model).where(*conditions)
        result = await self.session.execute(query)
        return len(result.all())

    def _prepare_query(self, **kwargs) -> Select:
        """クエリを準備する"""
        query = select(self.model)

        # フィルタリング
        if filters := kwargs.get("filters"):
            conditions = [
                getattr(self.model, field) == value
                for field, value in filters.items()
            ]
            query = query.where(*conditions)

        # ソート
        if sort_by := kwargs.get("sort_by"):
            direction = kwargs.get("sort_direction", "asc")
            column = getattr(self.model, sort_by)
            query = query.order_by(
                column.asc() if direction == "asc" else column.desc()
            )

        # ページネーション
        if limit := kwargs.get("limit"):
            query = query.limit(limit)
        if offset := kwargs.get("offset"):
            query = query.offset(offset)

        return query

    async def find(self, **kwargs) -> List[ModelType]:
        """条件に一致するエンティティを検索する"""
        query = self._prepare_query(**kwargs)
        result = await self.session.execute(query)
        return list(result.scalars().all())