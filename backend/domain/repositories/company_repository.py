"""
企業リポジトリインターフェース

企業データへのアクセスを抽象化するインターフェース定義。
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from backend.domain.entities.company import Company, CompanyStatus


class CompanyRepositoryInterface(ABC):
    """
    企業リポジトリのインターフェース

    企業データの取得・保存・更新・削除などの操作を定義します。
    """

    @abstractmethod
    async def get_by_id(self, company_id: str) -> Company:
        """
        IDで企業を取得

        Args:
            company_id: 企業ID

        Returns:
            Company: 取得した企業エンティティ

        Raises:
            CompanyNotFoundError: 企業が見つからない場合
        """
        pass

    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[Company]:
        """
        企業名で企業を検索

        Args:
            name: 企業名

        Returns:
            Optional[Company]: 検索した企業エンティティ。見つからない場合はNone
        """
        pass

    @abstractmethod
    async def create(self, company: Company) -> Company:
        """
        企業を新規作成

        Args:
            company: 作成する企業エンティティ

        Returns:
            Company: 作成された企業エンティティ（IDが設定される）

        Raises:
            CompanyAlreadyExistsError: 既に同じIDの企業が存在する場合
        """
        pass

    @abstractmethod
    async def update(self, company: Company) -> Company:
        """
        企業情報を更新

        Args:
            company: 更新する企業エンティティ

        Returns:
            Company: 更新された企業エンティティ

        Raises:
            CompanyNotFoundError: 企業が見つからない場合
        """
        pass

    @abstractmethod
    async def delete(self, company_id: str) -> None:
        """
        企業を削除

        Args:
            company_id: 削除する企業ID

        Raises:
            CompanyNotFoundError: 企業が見つからない場合
        """
        pass

    @abstractmethod
    async def list_all(self, limit: int = 100, offset: int = 0) -> List[Company]:
        """
        全企業をリスト取得

        Args:
            limit: 取得する最大数
            offset: オフセット（スキップする件数）

        Returns:
            List[Company]: 企業エンティティのリスト
        """
        pass

    @abstractmethod
    async def find_by_status(self, status: CompanyStatus, limit: int = 100) -> List[Company]:
        """
        ステータスで企業を検索

        Args:
            status: 検索するステータス
            limit: 取得する最大数

        Returns:
            List[Company]: 条件に一致する企業エンティティのリスト
        """
        pass

    @abstractmethod
    async def find_by_industry(self, industry: str, limit: int = 100) -> List[Company]:
        """
        業種で企業を検索

        Args:
            industry: 検索する業種
            limit: 取得する最大数

        Returns:
            List[Company]: 条件に一致する企業エンティティのリスト
        """
        pass

    @abstractmethod
    async def find_updated_since(self, since: datetime, limit: int = 100) -> List[Company]:
        """
        指定日時以降に更新された企業を検索

        Args:
            since: 検索の基準日時
            limit: 取得する最大数

        Returns:
            List[Company]: 条件に一致する企業エンティティのリスト
        """
        pass