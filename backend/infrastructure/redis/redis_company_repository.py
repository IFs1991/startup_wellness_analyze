"""
Redis企業リポジトリ

企業情報をRedisにキャッシュするリポジトリの実装。
"""

import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from backend.domain.entities.company import Company, CompanyAddress, CompanySize, CompanyStatus
from backend.domain.repositories.company_repository import CompanyRepositoryInterface
from backend.core.exceptions import CompanyNotFoundError
from backend.infrastructure.redis.redis_service import RedisService

logger = logging.getLogger(__name__)


class RedisCompanyRepository(CompanyRepositoryInterface):
    """
    Redisを使用した企業データのキャッシュリポジトリ実装。
    デコレータパターンを使用して、メインリポジトリの前にキャッシュ層として機能します。
    """

    # キャッシュキーのプレフィックス
    COMPANY_KEY_PREFIX = "company:"
    NAME_INDEX_PREFIX = "company:name:"
    INDUSTRY_INDEX_PREFIX = "company:industry:"
    STATUS_INDEX_PREFIX = "company:status:"
    UPDATED_INDEX_PREFIX = "company:updated:"

    def __init__(
        self,
        redis_service: RedisService,
        main_repository: CompanyRepositoryInterface,
        ttl_seconds: int = 3600  # デフォルト: 1時間
    ):
        """
        初期化メソッド

        Args:
            redis_service: Redisサービスインスタンス
            main_repository: メインの企業リポジトリ実装
            ttl_seconds: キャッシュの有効期限（秒）
        """
        self.redis = redis_service
        self.main_repository = main_repository
        self.ttl = ttl_seconds

    def _get_company_key(self, company_id: str) -> str:
        """
        企業IDからRedisキーを生成します

        Args:
            company_id: 企業ID

        Returns:
            Redisキー文字列
        """
        return f"{self.COMPANY_KEY_PREFIX}{company_id}"

    def _get_name_key(self, name: str) -> str:
        """
        企業名からインデックスキーを生成します

        Args:
            name: 企業名

        Returns:
            Redisキー文字列
        """
        return f"{self.NAME_INDEX_PREFIX}{name}"

    def _get_industry_key(self, industry: str) -> str:
        """
        業種からインデックスキーを生成します

        Args:
            industry: 業種

        Returns:
            Redisキー文字列
        """
        return f"{self.INDUSTRY_INDEX_PREFIX}{industry}"

    def _get_status_key(self, status: CompanyStatus) -> str:
        """
        ステータスからインデックスキーを生成します

        Args:
            status: ステータス

        Returns:
            Redisキー文字列
        """
        return f"{self.STATUS_INDEX_PREFIX}{status.value}"

    def _get_updated_key(self, timestamp: str) -> str:
        """
        更新日時からインデックスキーを生成します

        Args:
            timestamp: ISO形式の日時文字列

        Returns:
            Redisキー文字列
        """
        return f"{self.UPDATED_INDEX_PREFIX}{timestamp}"

    def _serialize_company(self, company: Company) -> Dict[str, Any]:
        """
        企業オブジェクトを辞書に変換します

        Args:
            company: 企業オブジェクト

        Returns:
            企業データの辞書
        """
        address_dict = None
        if company.address:
            address_dict = {
                "postal_code": company.address.postal_code,
                "prefecture": company.address.prefecture,
                "city": company.address.city,
                "street_address": company.address.street_address,
                "building": company.address.building,
                "country": company.address.country
            }

        return {
            "id": company.id,
            "name": company.name,
            "size": company.size.value if isinstance(company.size, CompanySize) else company.size,
            "status": company.status.value if isinstance(company.status, CompanyStatus) else company.status,
            "address": address_dict,
            "industry": company.industry,
            "established_date": company.established_date.isoformat() if company.established_date else None,
            "employee_count": company.employee_count,
            "created_at": company.created_at.isoformat() if company.created_at else None,
            "updated_at": company.updated_at.isoformat() if company.updated_at else None,
            "contact_email": company.contact_email,
            "contact_phone": company.contact_phone,
            "website": company.website,
            "departments": list(company.departments) if company.departments else [],
            "metadata": company.metadata,
            "admin_user_ids": list(company.admin_user_ids) if company.admin_user_ids else []
        }

    def _deserialize_company(self, data: Dict[str, Any]) -> Company:
        """
        辞書から企業オブジェクトを復元します

        Args:
            data: 企業データの辞書

        Returns:
            企業オブジェクト
        """
        # 日時文字列をdatetimeオブジェクトに変換
        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
        updated_at = datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
        established_date = datetime.fromisoformat(data["established_date"]) if data.get("established_date") else None

        # 住所オブジェクトの構築
        address = None
        if data.get("address"):
            address = CompanyAddress(
                postal_code=data["address"].get("postal_code", ""),
                prefecture=data["address"].get("prefecture", ""),
                city=data["address"].get("city", ""),
                street_address=data["address"].get("street_address", ""),
                building=data["address"].get("building"),
                country=data["address"].get("country", "日本")
            )

        # 企業オブジェクトを構築
        return Company(
            id=data.get("id", ""),
            name=data.get("name", ""),
            size=CompanySize(data.get("size", "small")),
            status=CompanyStatus(data.get("status", "active")),
            address=address,
            industry=data.get("industry"),
            established_date=established_date,
            employee_count=data.get("employee_count", 0),
            created_at=created_at,
            updated_at=updated_at,
            contact_email=data.get("contact_email"),
            contact_phone=data.get("contact_phone"),
            website=data.get("website"),
            departments=data.get("departments", []),
            metadata=data.get("metadata", {}),
            admin_user_ids=set(data.get("admin_user_ids", []))
        )

    async def get_by_id(self, company_id: str) -> Company:
        """
        IDにより企業を取得します。キャッシュにあればそこから、なければメインリポジトリから取得します。

        Args:
            company_id: 取得する企業のID

        Returns:
            企業オブジェクト

        Raises:
            CompanyNotFoundError: 企業が見つからない場合
        """
        # キャッシュからの取得を試みる
        cache_key = self._get_company_key(company_id)
        cached_data = await self.redis.get_json(cache_key)

        if cached_data:
            # キャッシュヒット
            logger.debug(f"企業(ID: {company_id})のキャッシュヒット")
            return self._deserialize_company(cached_data)

        # キャッシュミス - メインリポジトリから取得
        try:
            logger.debug(f"企業(ID: {company_id})のキャッシュミス、メインリポジトリから取得")
            company = await self.main_repository.get_by_id(company_id)

            # キャッシュに保存
            company_data = self._serialize_company(company)
            await self.redis.set_json(cache_key, company_data, self.ttl)

            # インデックスの更新
            await self._update_indices(company)

            return company
        except CompanyNotFoundError:
            # メインリポジトリでも見つからない場合は例外を再発生
            logger.warning(f"企業(ID: {company_id})が見つかりません")
            raise

    async def get_by_name(self, name: str) -> Optional[Company]:
        """
        企業名により企業を取得します。

        Args:
            name: 企業名

        Returns:
            企業オブジェクト。見つからない場合はNone
        """
        # 名前インデックスからIDを取得
        name_key = self._get_name_key(name)
        company_id = await self.redis.get(name_key)

        if company_id:
            # キャッシュヒット - IDを使用して企業データを取得
            try:
                return await self.get_by_id(company_id)
            except CompanyNotFoundError:
                # IDは見つかったが、企業データがキャッシュにない場合
                pass

        # キャッシュミス - メインリポジトリから取得
        company = await self.main_repository.get_by_name(name)
        if company:
            # キャッシュに保存
            await self._cache_company_with_indices(company)

        return company

    async def create(self, company: Company) -> Company:
        """
        企業を新規作成します。

        Args:
            company: 作成する企業エンティティ

        Returns:
            Company: 作成された企業エンティティ（IDが設定される）
        """
        # メインリポジトリで作成
        created_company = await self.main_repository.create(company)

        # キャッシュに保存
        await self._cache_company_with_indices(created_company)

        return created_company

    async def update(self, company: Company) -> Company:
        """
        企業情報を更新します。

        Args:
            company: 更新する企業エンティティ

        Returns:
            Company: 更新された企業エンティティ
        """
        # メインリポジトリで更新
        updated_company = await self.main_repository.update(company)

        # 古いインデックスを削除（必要に応じて）
        cache_key = self._get_company_key(company.id)
        old_data = await self.redis.get_json(cache_key)
        if old_data:
            old_company = self._deserialize_company(old_data)
            await self._remove_indices(old_company)

        # 更新されたデータをキャッシュに保存
        await self._cache_company_with_indices(updated_company)

        return updated_company

    async def delete(self, company_id: str) -> None:
        """
        企業を削除します。

        Args:
            company_id: 削除する企業ID
        """
        # キャッシュからインデックスを削除するため、まず企業データを取得
        try:
            cache_key = self._get_company_key(company_id)
            cached_data = await self.redis.get_json(cache_key)
            if cached_data:
                company = self._deserialize_company(cached_data)
                await self._remove_indices(company)

            # キャッシュからも削除
            await self.redis.delete(cache_key)
        except Exception as e:
            logger.warning(f"企業(ID: {company_id})のキャッシュ削除中にエラー発生: {e}")

        # メインリポジトリで削除
        await self.main_repository.delete(company_id)

    async def list_all(self, limit: int = 100, offset: int = 0) -> List[Company]:
        """
        全企業をリスト取得します。

        Args:
            limit: 取得する最大数
            offset: オフセット（スキップする件数）

        Returns:
            List[Company]: 企業エンティティのリスト
        """
        # この操作はキャッシュせず、メインリポジトリに委譲
        companies = await self.main_repository.list_all(limit, offset)

        # 個々の企業をキャッシュに保存
        for company in companies:
            await self._cache_company_with_indices(company)

        return companies

    async def find_by_status(self, status: CompanyStatus, limit: int = 100) -> List[Company]:
        """
        ステータスで企業を検索します。

        Args:
            status: 検索するステータス
            limit: 取得する最大数

        Returns:
            List[Company]: 条件に一致する企業エンティティのリスト
        """
        # メインリポジトリから検索
        companies = await self.main_repository.find_by_status(status, limit)

        # 個々の企業をキャッシュに保存
        for company in companies:
            await self._cache_company_with_indices(company)

        return companies

    async def find_by_industry(self, industry: str, limit: int = 100) -> List[Company]:
        """
        業種で企業を検索します。

        Args:
            industry: 検索する業種
            limit: 取得する最大数

        Returns:
            List[Company]: 条件に一致する企業エンティティのリスト
        """
        # メインリポジトリから検索
        companies = await self.main_repository.find_by_industry(industry, limit)

        # 個々の企業をキャッシュに保存
        for company in companies:
            await self._cache_company_with_indices(company)

        return companies

    async def find_updated_since(self, since: datetime, limit: int = 100) -> List[Company]:
        """
        指定日時以降に更新された企業を検索します。

        Args:
            since: 検索の基準日時
            limit: 取得する最大数

        Returns:
            List[Company]: 条件に一致する企業エンティティのリスト
        """
        # メインリポジトリから検索
        companies = await self.main_repository.find_updated_since(since, limit)

        # 個々の企業をキャッシュに保存
        for company in companies:
            await self._cache_company_with_indices(company)

        return companies

    async def _cache_company_with_indices(self, company: Company) -> None:
        """
        企業データとそのインデックスをキャッシュに保存します。

        Args:
            company: キャッシュする企業エンティティ
        """
        company_data = self._serialize_company(company)
        company_key = self._get_company_key(company.id)

        # 企業データをキャッシュに保存
        await self.redis.set_json(company_key, company_data, self.ttl)

        # インデックスを更新
        await self._update_indices(company)

    async def _update_indices(self, company: Company) -> None:
        """
        企業に関連するインデックスを更新します。

        Args:
            company: インデックスを更新する企業エンティティ
        """
        # 名前インデックス
        if company.name:
            name_key = self._get_name_key(company.name)
            await self.redis.set(name_key, company.id, self.ttl)

        # 業種インデックス
        if company.industry:
            industry_key = self._get_industry_key(company.industry)
            await self.redis.set(industry_key, company.id, self.ttl)

        # ステータスインデックス
        status_key = self._get_status_key(company.status)
        companies_by_status = await self.redis.get_list(status_key) or []
        if company.id not in companies_by_status:
            companies_by_status.append(company.id)
            await self.redis.set_list(status_key, companies_by_status, self.ttl)

        # 更新日時インデックス
        if company.updated_at:
            updated_key = self._get_updated_key(company.updated_at.isoformat()[:10])  # YYYY-MM-DD形式
            companies_by_date = await self.redis.get_list(updated_key) or []
            if company.id not in companies_by_date:
                companies_by_date.append(company.id)
                await self.redis.set_list(updated_key, companies_by_date, self.ttl)

    async def _remove_indices(self, company: Company) -> None:
        """
        企業に関連するインデックスを削除します。

        Args:
            company: インデックスを削除する企業エンティティ
        """
        # 名前インデックス
        if company.name:
            name_key = self._get_name_key(company.name)
            await self.redis.delete(name_key)

        # 業種インデックス
        if company.industry:
            industry_key = self._get_industry_key(company.industry)
            await self.redis.delete(industry_key)

        # ステータスインデックスから削除
        status_key = self._get_status_key(company.status)
        companies_by_status = await self.redis.get_list(status_key) or []
        if company.id in companies_by_status:
            companies_by_status.remove(company.id)
            if companies_by_status:
                await self.redis.set_list(status_key, companies_by_status, self.ttl)
            else:
                await self.redis.delete(status_key)

        # 更新日時インデックスから削除
        if company.updated_at:
            updated_key = self._get_updated_key(company.updated_at.isoformat()[:10])
            companies_by_date = await self.redis.get_list(updated_key) or []
            if company.id in companies_by_date:
                companies_by_date.remove(company.id)
                if companies_by_date:
                    await self.redis.set_list(updated_key, companies_by_date, self.ttl)
                else:
                    await self.redis.delete(updated_key)


def create_redis_company_repository(
    redis_service: RedisService,
    main_repository: CompanyRepositoryInterface,
    ttl: int = 3600
) -> RedisCompanyRepository:
    """
    Redis企業リポジトリを作成するファクトリ関数

    Args:
        redis_service: Redisサービスインスタンス
        main_repository: メインの企業リポジトリ実装
        ttl: キャッシュの有効期限（秒、デフォルト: 1時間）

    Returns:
        RedisCompanyRepository: 設定されたRedis企業リポジトリ
    """
    return RedisCompanyRepository(redis_service, main_repository, ttl)