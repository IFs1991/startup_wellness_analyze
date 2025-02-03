import pytest
import asyncio
from datetime import datetime
from google.cloud import firestore
from google.cloud.firestore import AsyncClient
from backend.src.database.repositories.company import CompanyRepository

@pytest.fixture
async def firestore_client():
    """Firestoreエミュレータに接続するクライアントを作成"""
    import os
    os.environ["FIRESTORE_EMULATOR_HOST"] = "localhost:8080"
    client = firestore.AsyncClient(project="test-project")
    yield client
    # テスト後のクリーンアップ
    collections = await client.collections()
    for collection in collections:
        docs = await collection.get()
        for doc in docs:
            await doc.reference.delete()

@pytest.fixture
async def company_repository(firestore_client):
    """会社リポジトリのインスタンスを作成"""
    return CompanyRepository(firestore_client)

@pytest.mark.asyncio
async def test_company_repository(company_repository):
    """会社リポジトリの基本的なCRUD操作をテスト"""
    # 会社を作成
    company_data = {
        "name": "Test Company",
        "description": "Test Description",
        "industry": "Technology",
        "owner_id": "test_owner",
        "founded_date": datetime.now(),
        "employee_count": 100,
        "website": "https://example.com",
        "location": "Tokyo, Japan"
    }

    company = await company_repository.create(**company_data)
    assert company.name == "Test Company"
    assert company.industry == "Technology"
    assert company.owner_id == "test_owner"

    # IDで取得
    retrieved_company = await company_repository.get_by_id(company.id)
    assert retrieved_company is not None
    assert retrieved_company.name == company.name

    # オーナーIDで取得
    companies_by_owner = await company_repository.get_by_owner("test_owner")
    assert len(companies_by_owner) == 1
    assert companies_by_owner[0].id == company.id

    # 業界で取得
    companies_by_industry = await company_repository.get_by_industry("Technology")
    assert len(companies_by_industry) == 1
    assert companies_by_industry[0].id == company.id

    # ステータスを追加
    status_data = {
        "type": "ACTIVE",
        "description": "Company is active",
        "created_at": datetime.now()
    }
    await company_repository.add_status(company.id, **status_data)

    # ステージを追加
    stage_data = {
        "type": "SEED",
        "description": "Seed stage",
        "created_at": datetime.now()
    }
    await company_repository.add_stage(company.id, **stage_data)

    # 詳細付きで取得
    company_with_details = await company_repository.get_with_details(company.id)
    assert company_with_details is not None
    assert company_with_details.statuses[0].type == "ACTIVE"
    assert company_with_details.stages[0].type == "SEED"

    # 最新のステータスを取得
    latest_status = await company_repository.get_latest_status(company.id)
    assert latest_status is not None
    assert latest_status.type == "ACTIVE"

    # 最新のステージを取得
    latest_stage = await company_repository.get_latest_stage(company.id)
    assert latest_stage is not None
    assert latest_stage.type == "SEED"

    # 会社を更新
    updated_data = {
        "name": "Updated Company",
        "employee_count": 200
    }
    updated_company = await company_repository.update(company.id, **updated_data)
    assert updated_company is not None
    assert updated_company.name == "Updated Company"
    assert updated_company.employee_count == 200

    # 会社を削除
    deleted = await company_repository.delete(company.id)
    assert deleted is True

    # 削除されたことを確認
    deleted_company = await company_repository.get_by_id(company.id)
    assert deleted_company is None

@pytest.mark.asyncio
async def test_company_repository_search(company_repository):
    """会社リポジトリの検索機能をテスト"""
    # テスト用の会社を複数作成
    companies_data = [
        {
            "name": f"Company {i}",
            "description": f"Description {i}",
            "industry": "Technology" if i % 2 == 0 else "Finance",
            "owner_id": f"owner_{i}",
            "founded_date": datetime.now(),
            "employee_count": 100 * (i + 1),
            "website": f"https://example{i}.com",
            "location": f"Location {i}"
        } for i in range(5)
    ]

    for company_data in companies_data:
        company = await company_repository.create(**company_data)

        # ステータスとステージを追加
        await company_repository.add_status(
            company.id,
            type="ACTIVE" if company_data["industry"] == "Technology" else "INACTIVE",
            description=f"Status for {company.name}",
            created_at=datetime.now()
        )

        await company_repository.add_stage(
            company.id,
            type="SEED" if company_data["industry"] == "Technology" else "SERIES_A",
            description=f"Stage for {company.name}",
            created_at=datetime.now()
        )

    # 業界でフィルター
    tech_companies = await company_repository.search_companies(
        industry="Technology",
        page=1,
        per_page=10
    )
    assert len(tech_companies) == 3

    # ステージでフィルター
    seed_companies = await company_repository.search_companies(
        stage_type="SEED",
        page=1,
        per_page=10
    )
    assert len(seed_companies) == 3

    # ステータスでフィルター
    active_companies = await company_repository.search_companies(
        status_type="ACTIVE",
        page=1,
        per_page=10
    )
    assert len(active_companies) == 3

    # 複数の条件でフィルター
    filtered_companies = await company_repository.search_companies(
        industry="Technology",
        stage_type="SEED",
        status_type="ACTIVE",
        page=1,
        per_page=10
    )
    assert len(filtered_companies) == 3

    # ページネーション
    paginated_companies = await company_repository.search_companies(
        page=1,
        per_page=2
    )
    assert len(paginated_companies) == 2

@pytest.mark.asyncio
async def test_company_repository_error_handling(company_repository):
    """会社リポジトリのエラーハンドリングをテスト"""
    # 存在しない会社IDでの操作
    non_existent_id = "non_existent_id"

    # 存在しない会社の取得
    company = await company_repository.get_by_id(non_existent_id)
    assert company is None

    # 存在しない会社の詳細取得
    company_with_details = await company_repository.get_with_details(non_existent_id)
    assert company_with_details is None

    # 存在しない会社の最新ステータス取得
    latest_status = await company_repository.get_latest_status(non_existent_id)
    assert latest_status is None

    # 存在しない会社の最新ステージ取得
    latest_stage = await company_repository.get_latest_stage(non_existent_id)
    assert latest_stage is None