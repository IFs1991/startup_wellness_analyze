# -*- coding: utf-8 -*-
"""
企業情報サービス
企業情報管理に関するビジネスロジックを提供します。
"""
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from backend.database.repository import DataCategory
from backend.database.repositories import repository_factory
from backend.database.models.entities import CompanyEntity
from backend.database.connection import get_db
from backend.database.repositories.company_repository import CompanyRepository

class CompanyService:
    """
    企業情報サービスクラス
    企業情報の取得、登録、更新などの機能を提供します。
    """
    def __init__(self, session=None):
        """初期化"""
        self.session = session or next(get_db())
        self.company_repo = CompanyRepository(self.session)

    async def get_companies(
        self,
        search: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[CompanyEntity]:
        """
        企業情報の一覧を取得

        Args:
            search: 検索キーワード（会社名の部分一致）
            filters: フィルター条件
            limit: 取得する最大件数

        Returns:
            List[CompanyEntity]: 企業情報のリスト
        """
        # 検索条件の適用
        if search:
            return await self.company_repo.find_by_name_contains(search, limit)

        # フィルター条件の適用
        if filters:
            return await self.company_repo.find_by_criteria(filters, limit=limit)

        # 条件なしの場合は全件取得
        return await self.company_repo.find_all(limit=limit)

    async def get_company_by_id(self, company_id: str) -> Optional[CompanyEntity]:
        """
        IDによる企業情報の取得

        Args:
            company_id: 企業ID

        Returns:
            Optional[CompanyEntity]: 企業情報またはNone
        """
        return await self.company_repo.find_by_id(company_id)

    async def create_company(self, company_data: Dict[str, Any]) -> CompanyEntity:
        """
        企業情報の新規作成

        Args:
            company_data: 企業情報データ

        Returns:
            CompanyEntity: 作成された企業情報
        """
        company = CompanyEntity(**company_data)
        return await self.company_repo.save(company)

    async def update_company(
        self,
        company_id: str,
        company_data: Dict[str, Any]
    ) -> CompanyEntity:
        """
        企業情報の更新

        Args:
            company_id: 企業ID
            company_data: 更新データ

        Returns:
            CompanyEntity: 更新された企業情報
        """
        return await self.company_repo.update(company_id, company_data)

    async def delete_company(self, company_id: str) -> bool:
        """
        企業情報の削除

        Args:
            company_id: 企業ID

        Returns:
            bool: 削除成功時True
        """
        return await self.company_repo.delete(company_id)