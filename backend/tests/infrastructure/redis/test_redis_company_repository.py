"""
RedisCompanyRepositoryのテスト

Redis企業リポジトリの機能（モックと実際の接続）をテストします。
"""

import json
import pytest
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from backend.domain.entities.company import Company, CompanyAddress, CompanySize, CompanyStatus
from backend.domain.repositories.company_repository import CompanyRepositoryInterface
from backend.infrastructure.redis.redis_company_repository import RedisCompanyRepository
from backend.infrastructure.redis.redis_service import RedisService
from backend.infrastructure.redis.redis_bulk_operations import RedisBulkOperations
from backend.core.exceptions import CompanyNotFoundError
from backend.core.async_utils import gather_with_concurrency

# モックを使用したユニットテスト

@pytest.fixture
def mock_redis_service():
    """Redisサービスのモックを作成します"""
    service = AsyncMock()
    return service

@pytest.fixture
def mock_main_repository():
    """メインリポジトリのモックを作成します"""
    repo = AsyncMock(spec=CompanyRepositoryInterface)
    return repo

@pytest.fixture
def redis_company_repository(mock_redis_service, mock_main_repository):
    """テスト用のRedisCompanyRepositoryインスタンスを作成します"""
    return RedisCompanyRepository(
        redis_service=mock_redis_service,
        main_repository=mock_main_repository,
        ttl_seconds=3600
    )

@pytest.fixture
def sample_company():
    """テスト用の企業データを作成します"""
    now = datetime.now()
    return Company(
        id="test-company-id",
        name="テスト株式会社",
        size=CompanySize.MEDIUM,
        status=CompanyStatus.ACTIVE,
        address=CompanyAddress(
            postal_code="123-4567",
            prefecture="東京都",
            city="千代田区",
            street_address="1-1-1"
        ),
        industry="テクノロジー",
        established_date=now - timedelta(days=365*5),  # 5年前
        employee_count=150,
        created_at=now - timedelta(days=90),
        updated_at=now,
        contact_email="contact@test-company.example.com",
        contact_phone="03-1234-5678",
        website="https://test-company.example.com",
        departments=["開発", "営業", "管理"],
        admin_user_ids={"admin-user-1", "admin-user-2"}
    )

@pytest.fixture
def sample_companies():
    """複数のテスト用企業データを作成します"""
    now = datetime.now()
    companies = []

    for i in range(5):
        company = Company(
            id=f"test-company-{i}",
            name=f"テスト企業{i}",
            size=CompanySize.MEDIUM if i % 2 == 0 else CompanySize.SMALL,
            status=CompanyStatus.ACTIVE if i % 3 != 0 else CompanyStatus.INACTIVE,
            address=CompanyAddress(
                postal_code=f"123-{i}567",
                prefecture="東京都",
                city="千代田区",
                street_address=f"1-1-{i}"
            ),
            industry="テクノロジー" if i % 2 == 0 else "金融",
            established_date=now - timedelta(days=365*(3+i)),
            employee_count=50 + (i * 30),
            created_at=now - timedelta(days=90 - i*5),
            updated_at=now - timedelta(days=i*3),
            contact_email=f"contact{i}@example.com",
            contact_phone=f"03-1234-{i}678",
            website=f"https://company{i}.example.com",
            departments=["開発", "営業"] if i % 2 == 0 else ["管理", "人事"],
            admin_user_ids={f"admin-{i}-1", f"admin-{i}-2"}
        )
        companies.append(company)

    return companies

@pytest.mark.asyncio
async def test_get_by_id_cached(redis_company_repository, mock_redis_service, sample_company):
    """キャッシュから企業をIDで取得するテスト"""
    # 準備
    company_id = sample_company.id
    company_data = {
        "id": company_id,
        "name": sample_company.name,
        "size": sample_company.size.value,
        "status": sample_company.status.value,
        "address": {
            "postal_code": sample_company.address.postal_code,
            "prefecture": sample_company.address.prefecture,
            "city": sample_company.address.city,
            "street_address": sample_company.address.street_address,
            "building": None,
            "country": "日本"
        },
        "industry": sample_company.industry,
        "established_date": sample_company.established_date.isoformat(),
        "employee_count": sample_company.employee_count,
        "created_at": sample_company.created_at.isoformat(),
        "updated_at": sample_company.updated_at.isoformat(),
        "contact_email": sample_company.contact_email,
        "contact_phone": sample_company.contact_phone,
        "website": sample_company.website,
        "departments": list(sample_company.departments),
        "metadata": {},
        "admin_user_ids": list(sample_company.admin_user_ids)
    }

    # Redisからの応答をモック
    mock_redis_service.get_json.return_value = company_data

    # 実行
    result = await redis_company_repository.get_by_id(company_id)

    # 検証
    mock_redis_service.get_json.assert_called_once_with(f"company:{company_id}")
    assert isinstance(result, Company)
    assert result.id == company_id
    assert result.name == sample_company.name
    assert result.size == sample_company.size
    assert result.industry == sample_company.industry

@pytest.mark.asyncio
async def test_get_by_id_not_cached(redis_company_repository, mock_redis_service, mock_main_repository, sample_company):
    """キャッシュにない企業をメインリポジトリから取得するテスト"""
    # 準備
    company_id = sample_company.id

    # Redisからの応答をモック
    mock_redis_service.get_json.return_value = None

    # メインリポジトリからの応答をモック
    mock_main_repository.get_by_id.return_value = sample_company

    # 実行
    result = await redis_company_repository.get_by_id(company_id)

    # 検証
    mock_redis_service.get_json.assert_called_once_with(f"company:{company_id}")
    mock_main_repository.get_by_id.assert_called_once_with(company_id)
    assert result == sample_company

    # キャッシュが更新されたことを確認
    mock_redis_service.set_json.assert_called_once()

@pytest.mark.asyncio
async def test_create_company(redis_company_repository, mock_redis_service, mock_main_repository, sample_company):
    """企業作成のテスト"""
    # メインリポジトリからの応答をモック
    mock_main_repository.create.return_value = sample_company

    # 実行
    result = await redis_company_repository.create(sample_company)

    # 検証
    mock_main_repository.create.assert_called_once_with(sample_company)
    assert result == sample_company

    # キャッシュが更新されたことを確認
    mock_redis_service.set_json.assert_called_once()

@pytest.mark.asyncio
async def test_update_company(redis_company_repository, mock_redis_service, mock_main_repository, sample_company):
    """企業更新のテスト"""
    # 古いデータを持つ企業をキャッシュに想定
    old_company_data = {
        "id": sample_company.id,
        "name": "古い企業名",
        "size": "small",
        "status": "active",
        "address": {
            "postal_code": "100-0001",
            "prefecture": "東京都",
            "city": "千代田区",
            "street_address": "1-1-1",
            "building": None,
            "country": "日本"
        },
        "industry": "旧業種",
        "employee_count": 50,
        "created_at": (datetime.now() - timedelta(days=90)).isoformat(),
        "updated_at": (datetime.now() - timedelta(days=30)).isoformat(),
        "departments": ["旧部署"],
        "admin_user_ids": []
    }

    # Redisからの古いデータ応答をモック
    mock_redis_service.get_json.return_value = old_company_data

    # メインリポジトリからの応答をモック
    mock_main_repository.update.return_value = sample_company

    # 実行
    result = await redis_company_repository.update(sample_company)

    # 検証
    mock_main_repository.update.assert_called_once_with(sample_company)
    assert result == sample_company

    # 古いインデックスが削除され、新しいキャッシュが更新されたことを確認
    assert mock_redis_service.delete.call_count >= 1
    assert mock_redis_service.set_json.call_count >= 1

@pytest.mark.asyncio
async def test_delete_company(redis_company_repository, mock_redis_service, mock_main_repository, sample_company):
    """企業削除のテスト"""
    # 準備
    company_id = sample_company.id

    # キャッシュに既存データがある想定
    mock_redis_service.get_json.return_value = {
        "id": company_id,
        "name": sample_company.name,
        "size": sample_company.size.value,
        "status": sample_company.status.value,
        # その他のフィールド
    }

    # 実行
    await redis_company_repository.delete(company_id)

    # 検証
    mock_main_repository.delete.assert_called_once_with(company_id)

    # キャッシュとインデックスが削除されたことを確認
    assert mock_redis_service.delete.call_count >= 1

# 実際のRedis接続を使用した統合テスト（オプション、CI環境では実行しない）

# テストでRedis接続を使用するかどうかの環境変数
USE_REAL_REDIS = os.environ.get("TEST_USE_REAL_REDIS", "False").lower() == "true"

@pytest.fixture
def real_redis_service():
    """実際のRedis接続を行うRedisServiceを提供します（環境変数で制御）"""
    if not USE_REAL_REDIS:
        pytest.skip("実際のRedis接続テストがスキップされました (TEST_USE_REAL_REDIS=True に設定してください)")

    # テスト用Redisホスト設定
    redis_host = os.environ.get("TEST_REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("TEST_REDIS_PORT", "6379"))
    redis_db = int(os.environ.get("TEST_REDIS_DB", "1"))  # テスト用に別DBを使用

    service = RedisService(
        host=redis_host,
        port=redis_port,
        db=redis_db
    )
    return service

@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_redis_company_repository(real_redis_service, mock_main_repository, sample_company):
    """Redis接続を使った企業リポジトリの統合テスト"""
    # 準備
    await real_redis_service.connect()

    # テスト前にテスト用キーをクリア
    await real_redis_service.delete_pattern("company:*")

    # モックメインリポジトリのセットアップ
    mock_main_repository.get_by_id.return_value = sample_company
    mock_main_repository.create.return_value = sample_company

    # テスト対象のリポジトリ作成
    repo = RedisCompanyRepository(
        redis_service=real_redis_service,
        main_repository=mock_main_repository,
        ttl_seconds=60  # テスト用に短いTTL
    )

    # テスト実行: 企業作成
    created_company = await repo.create(sample_company)
    assert created_company.id == sample_company.id

    # キャッシュからの取得を確認
    cached_company = await repo.get_by_id(sample_company.id)
    assert cached_company.id == sample_company.id
    assert cached_company.name == sample_company.name

    # メインリポジトリから再取得されないことを確認
    mock_main_repository.get_by_id.assert_called_once()  # 最初の呼び出しだけ

    # テスト終了後のクリーンアップ
    await real_redis_service.delete_pattern("company:*")

@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_indices(real_redis_service, mock_main_repository, sample_company):
    """Redisインデックスを使った検索機能の統合テスト"""
    # 準備
    await real_redis_service.connect()

    # テスト前にテスト用キーをクリア
    await real_redis_service.delete_pattern("company:*")

    # テスト対象のリポジトリ作成
    mock_main_repository.get_by_name.return_value = sample_company

    repo = RedisCompanyRepository(
        redis_service=real_redis_service,
        main_repository=mock_main_repository,
        ttl_seconds=60  # テスト用に短いTTL
    )

    # テスト実行: 企業の保存
    await repo._cache_company_with_indices(sample_company)

    # 名前インデックスからIDを取得できることを確認
    name_key = repo._get_name_key(sample_company.name)
    company_id = await real_redis_service.get(name_key)
    assert company_id == sample_company.id

    # 業種インデックスからIDを取得できることを確認
    if sample_company.industry:
        industry_key = repo._get_industry_key(sample_company.industry)
        company_id = await real_redis_service.get(industry_key)
        assert company_id == sample_company.id

    # ステータスインデックスに企業IDが含まれることを確認
    status_key = repo._get_status_key(sample_company.status)
    companies_by_status = await real_redis_service.get_list(status_key)
    assert sample_company.id in companies_by_status

    # テスト終了後のクリーンアップ
    await real_redis_service.delete_pattern("company:*")

@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_find_by_status(real_redis_service, mock_main_repository, sample_companies):
    """Redisを使ったステータスによる企業検索の統合テスト"""
    # 準備
    await real_redis_service.connect()

    # テスト前にテスト用キーをクリア
    await real_redis_service.delete_pattern("company:*")

    # テスト用のリポジトリ作成
    repo = RedisCompanyRepository(
        redis_service=real_redis_service,
        main_repository=mock_main_repository,
        ttl_seconds=60
    )

    # 企業をキャッシュに保存
    for company in sample_companies:
        await repo._cache_company_with_indices(company)

    # ステータスインデックスの作成を確認
    active_companies = []
    inactive_companies = []

    for company in sample_companies:
        if company.status == CompanyStatus.ACTIVE:
            active_companies.append(company)
        else:
            inactive_companies.append(company)

    # モックの設定
    mock_main_repository.find_by_status.side_effect = lambda status, limit=100: asyncio.Future().set_result(
        active_companies if status == CompanyStatus.ACTIVE else inactive_companies
    )

    # アクティブな企業を検索
    result_active = await repo.find_by_status(CompanyStatus.ACTIVE)

    # 検証
    assert len(result_active) == len(active_companies)
    assert all(c.status == CompanyStatus.ACTIVE for c in result_active)

    # 非アクティブな企業を検索
    result_inactive = await repo.find_by_status(CompanyStatus.INACTIVE)

    # 検証
    assert len(result_inactive) == len(inactive_companies)
    assert all(c.status == CompanyStatus.INACTIVE for c in result_inactive)

    # テスト終了後のクリーンアップ
    await real_redis_service.delete_pattern("company:*")

@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_find_by_industry(real_redis_service, mock_main_repository, sample_companies):
    """Redisを使った業種による企業検索の統合テスト"""
    # 準備
    await real_redis_service.connect()

    # テスト前にテスト用キーをクリア
    await real_redis_service.delete_pattern("company:*")

    # テスト用のリポジトリ作成
    repo = RedisCompanyRepository(
        redis_service=real_redis_service,
        main_repository=mock_main_repository,
        ttl_seconds=60
    )

    # 企業をキャッシュに保存
    for company in sample_companies:
        await repo._cache_company_with_indices(company)

    # 業種ごとに企業を分類
    tech_companies = [c for c in sample_companies if c.industry == "テクノロジー"]
    finance_companies = [c for c in sample_companies if c.industry == "金融"]

    # モックの設定
    mock_main_repository.find_by_industry.side_effect = lambda industry, limit=100: asyncio.Future().set_result(
        tech_companies if industry == "テクノロジー" else finance_companies
    )

    # テクノロジー企業を検索
    result_tech = await repo.find_by_industry("テクノロジー")

    # 検証
    assert len(result_tech) == len(tech_companies)
    assert all(c.industry == "テクノロジー" for c in result_tech)

    # 金融企業を検索
    result_finance = await repo.find_by_industry("金融")

    # 検証
    assert len(result_finance) == len(finance_companies)
    assert all(c.industry == "金融" for c in result_finance)

    # テスト終了後のクリーンアップ
    await real_redis_service.delete_pattern("company:*")

@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_async_optimization(real_redis_service, mock_main_repository, sample_companies):
    """非同期処理の最適化をテストする統合テスト"""
    # 準備
    await real_redis_service.connect()

    # テスト前にテスト用キーをクリア
    await real_redis_service.delete_pattern("company:*")

    # テスト用のリポジトリ作成
    repo = RedisCompanyRepository(
        redis_service=real_redis_service,
        main_repository=mock_main_repository,
        ttl_seconds=60
    )

    # RedisBulkOperationsを作成
    bulk_ops = RedisBulkOperations(
        redis_service=real_redis_service,
        max_concurrency=5,
        batch_size=2
    )

    # 複数の企業を一括でキャッシュ
    company_dict = {}
    for company in sample_companies:
        key = f"company:{company.id}"
        company_dict[key] = repo._serialize_company(company)

    # バルク操作で保存
    await bulk_ops.set_many_json(company_dict, expire=60)

    # 一括でキーを取得
    keys = [f"company:{company.id}" for company in sample_companies]
    cached_data = await bulk_ops.get_many_json(keys)

    # 検証
    assert len(cached_data) == len(sample_companies)
    assert all(data is not None for data in cached_data)

    # 並列処理で企業を取得
    async def get_company(company_id):
        return await repo.get_by_id(company_id)

    company_ids = [company.id for company in sample_companies]
    mock_main_repository.get_by_id.side_effect = lambda id: next((c for c in sample_companies if c.id == id), None)

    results = await gather_with_concurrency(3, *(get_company(id) for id in company_ids))

    # 検証
    assert len(results) == len(sample_companies)
    assert all(isinstance(company, Company) for company in results)
    assert set(company.id for company in results) == set(company_ids)

    # テスト終了後のクリーンアップ
    await real_redis_service.delete_pattern("company:*")

@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_bulk_delete(real_redis_service, mock_main_repository, sample_companies):
    """バルク削除操作の統合テスト"""
    # 準備
    await real_redis_service.connect()

    # テスト前にテスト用キーをクリア
    await real_redis_service.delete_pattern("company:*")

    # テスト用のリポジトリ作成
    repo = RedisCompanyRepository(
        redis_service=real_redis_service,
        main_repository=mock_main_repository,
        ttl_seconds=60
    )

    # RedisBulkOperationsを作成
    bulk_ops = RedisBulkOperations(
        redis_service=real_redis_service,
        max_concurrency=5,
        batch_size=2
    )

    # すべての企業をキャッシュ
    for company in sample_companies:
        await repo._cache_company_with_indices(company)

    # すべての企業がキャッシュされたことを確認
    for company in sample_companies:
        company_key = f"company:{company.id}"
        data = await real_redis_service.get_json(company_key)
        assert data is not None

    # 半分の企業を削除する
    companies_to_delete = sample_companies[:len(sample_companies)//2]
    delete_keys = [f"company:{company.id}" for company in companies_to_delete]

    # バルク削除
    deleted_count = await bulk_ops.delete_many(delete_keys)

    # 検証
    assert deleted_count == len(companies_to_delete)

    # 削除されたキーが存在しないことを確認
    for company in companies_to_delete:
        company_key = f"company:{company.id}"
        data = await real_redis_service.get_json(company_key)
        assert data is None

    # 残りの企業はまだキャッシュに存在することを確認
    remaining_companies = sample_companies[len(sample_companies)//2:]
    for company in remaining_companies:
        company_key = f"company:{company.id}"
        data = await real_redis_service.get_json(company_key)
        assert data is not None

    # テスト終了後のクリーンアップ
    await real_redis_service.delete_pattern("company:*")