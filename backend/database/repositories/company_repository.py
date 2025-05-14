# -*- coding: utf-8 -*-
"""
企業情報リポジトリ
企業情報の検索、保存、更新、削除機能を提供します。
"""
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc

from ..models_sql import Company
from ..models.entities import CompanyEntity
from ..repositories.sql import SQLRepository
from ..repository import EntityNotFoundException, ValidationException

class CompanyRepository(SQLRepository):
    """
    企業情報リポジトリクラス
    SQLRepositoryを拡張し、企業情報特有の操作を提供します。
    """
    def __init__(self, session: Session):
        """初期化"""
        super().__init__(session, Company, CompanyEntity)

    async def find_by_name_contains(self, name_fragment: str, limit: int = 10) -> List[CompanyEntity]:
        """
        名前の一部が一致する企業を検索します（部分一致検索）
        """
        query = self._session.query(self._entity_class).filter(
            self._entity_class.name.ilike(f"%{name_fragment}%")
        ).limit(limit)

        orm_entities = query.all()
        return [self._to_model(entity) for entity in orm_entities if entity]

    async def find_by_industry(self, industry: str, limit: int = 50) -> List[CompanyEntity]:
        """
        指定された業界の企業を検索します
        """
        query = self._session.query(self._entity_class).filter(
            self._entity_class.industry == industry
        ).limit(limit)

        orm_entities = query.all()
        return [self._to_model(entity) for entity in orm_entities if entity]

    async def find_by_criteria(self, criteria: Dict[str, Any], limit: int = 50) -> List[CompanyEntity]:
        """
        複数の条件で企業を検索します

        Args:
            criteria: 検索条件の辞書 (フィールド名: 値)
            limit: 取得する最大件数

        Returns:
            条件に一致する企業エンティティのリスト
        """
        query = self._session.query(self._entity_class)

        # 各条件を適用
        for field, value in criteria.items():
            if hasattr(self._entity_class, field):
                query = query.filter(getattr(self._entity_class, field) == value)

        query = query.limit(limit)
        orm_entities = query.all()
        return [self._to_model(entity) for entity in orm_entities if entity]