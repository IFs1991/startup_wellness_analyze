from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from .base import BaseRepository
from ..models import Report, ReportTemplate

class ReportRepository(BaseRepository[Report]):
    """レポートリポジトリクラス"""

    def __init__(self, session: AsyncSession):
        super().__init__(Report, session)

    async def get_with_template(self, report_id: str) -> Optional[Report]:
        """テンプレート情報付きでレポートを取得する"""
        query = (
            select(Report)
            .options(joinedload(Report.template))
            .where(Report.id == report_id)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_by_template(self, template_id: str) -> List[Report]:
        """テンプレートIDでレポートを取得する"""
        return await self.get_many_by_field("template_id", template_id)

    async def get_reports_by_format(self, format: str) -> List[Report]:
        """フォーマットでレ��ートを取得する"""
        return await self.get_many_by_field("format", format)

    async def search_reports(
        self,
        template_id: Optional[str] = None,
        format: Optional[str] = None,
        metadata: Optional[dict] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Report]:
        """レポートを検索する"""
        query = select(Report)

        if template_id:
            query = query.where(Report.template_id == template_id)
        if format:
            query = query.where(Report.format == format)
        if metadata:
            for key, value in metadata.items():
                query = query.where(Report.metadata[key].astext == str(value))

        query = query.order_by(Report.created_at.desc())
        query = query.limit(limit).offset(offset)

        result = await self.session.execute(query)
        return result.scalars().all()

class ReportTemplateRepository(BaseRepository[ReportTemplate]):
    """レポートテンプレートリポジトリクラス"""

    def __init__(self, session: AsyncSession):
        super().__init__(ReportTemplate, session)

    async def get_by_name(self, name: str) -> Optional[ReportTemplate]:
        """名前でテンプレートを取得する"""
        return await self.get_by_field("name", name)

    async def get_by_format(self, format: str) -> List[ReportTemplate]:
        """フォーマットでテンプレートを取得する"""
        return await self.get_many_by_field("format", format)

    async def search_templates(
        self,
        name: Optional[str] = None,
        format: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[ReportTemplate]:
        """テンプレートを検索する"""
        query = select(ReportTemplate)

        if name:
            query = query.where(ReportTemplate.name.ilike(f"%{name}%"))
        if format:
            query = query.where(ReportTemplate.format == format)

        query = query.order_by(ReportTemplate.created_at.desc())
        query = query.limit(limit).offset(offset)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_with_reports(self, template_id: str) -> Optional[ReportTemplate]:
        """レポート情報付きでテンプレートを取得する"""
        query = (
            select(ReportTemplate)
            .options(joinedload(ReportTemplate.reports))
            .where(ReportTemplate.id == template_id)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def update_template_content(
        self,
        template_id: str,
        content: str,
        variables: Optional[dict] = None
    ) -> Optional[ReportTemplate]:
        """テンプレートの内容を更新する"""
        update_data = {"template_content": content}
        if variables is not None:
            update_data["variables"] = variables
        return await self.update(template_id, **update_data)