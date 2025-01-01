from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from .base import BaseRepository
from ..models import Company, Status, Stage

class CompanyRepository(BaseRepository[Company]):
    """会社リポジトリクラス"""

    def __init__(self, session: AsyncSession):
        super().__init__(Company, session)

    async def get_by_owner(self, owner_id: str) -> List[Company]:
        """オーナーIDで会社を取得する"""
        return await self.get_many_by_field("owner_id", owner_id)

    async def get_by_industry(self, industry: str) -> List[Company]:
        """業種で会社を取得する"""
        return await self.get_many_by_field("industry", industry)

    async def get_with_details(self, company_id: str) -> Optional[Company]:
        """詳細情報付きで会社を取得する"""
        query = (
            select(Company)
            .options(
                joinedload(Company.statuses),
                joinedload(Company.stages),
                joinedload(Company.metrics)
            )
            .where(Company.id == company_id)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_latest_status(self, company_id: str) -> Optional[Status]:
        """最新のステータスを取得する"""
        query = (
            select(Status)
            .where(Status.company_id == company_id)
            .order_by(Status.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_latest_stage(self, company_id: str) -> Optional[Stage]:
        """最新のステージを取得する"""
        query = (
            select(Stage)
            .where(Stage.company_id == company_id)
            .order_by(Stage.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_companies_by_stage(self, stage_type: str) -> List[Company]:
        """ステージタイプで会社を取得する"""
        query = (
            select(Company)
            .join(Stage)
            .where(Stage.stage_type == stage_type)
            .order_by(Stage.created_at.desc())
        )
        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_companies_by_status(self, status_type: str) -> List[Company]:
        """ステータスタイプで会社を取得する"""
        query = (
            select(Company)
            .join(Status)
            .where(Status.status_type == status_type)
            .order_by(Status.created_at.desc())
        )
        result = await self.session.execute(query)
        return result.scalars().all()

    async def search_companies(
        self,
        name: Optional[str] = None,
        industry: Optional[str] = None,
        stage_type: Optional[str] = None,
        status_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Company]:
        """会社を検索する"""
        query = select(Company)

        if name:
            query = query.where(Company.name.ilike(f"%{name}%"))
        if industry:
            query = query.where(Company.industry == industry)
        if stage_type:
            query = query.join(Stage).where(Stage.stage_type == stage_type)
        if status_type:
            query = query.join(Status).where(Status.status_type == status_type)
        if owner_id:
            query = query.where(Company.owner_id == owner_id)

        query = query.order_by(Company.created_at.desc())
        query = query.limit(limit).offset(offset)

        result = await self.session.execute(query)
        return result.scalars().all()