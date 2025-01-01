import pytest
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from backend.src.database.models import Base, Company, Status, Stage
from backend.src.database.repositories.company import CompanyRepository

# テスト用のデータベースURL
TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/test_startup_wellness"

@pytest.fixture
async def engine():
    """テスト用のデータベースエンジンを作成する"""
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()

@pytest.fixture
async def session(engine):
    """テスト用のセッションを作成する"""
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session

@pytest.mark.asyncio
async def test_company_repository(session):
    """会社リポジトリのテスト"""
    repo = CompanyRepository(session)

    # 会社を作成
    company = await repo.create(
        name="Test Company",
        description="Test Description",
        industry="Technology",
        owner_id="test_owner",
        founded_date=datetime.now(),
        employee_count=100,
        website="https://example.com",
        location="Tokyo, Japan"
    )

    assert company.name == "Test Company"
    assert company.industry == "Technology"
    assert company.owner_id == "test_owner"

    # オーナーIDで取得
    companies_by_owner = await repo.get_by_owner("test_owner")
    assert len(companies_by_owner) == 1
    assert companies_by_owner[0].id == company.id

    # 業界で取得
    companies_by_industry = await repo.get_by_industry("Technology")
    assert len(companies_by_industry) == 1
    assert companies_by_industry[0].id == company.id

    # ステータスを追加
    status = Status(
        company_id=company.id,
        type="ACTIVE",
        description="Company is active",
        created_at=datetime.now()
    )
    session.add(status)
    await session.commit()

    # ステージを追加
    stage = Stage(
        company_id=company.id,
        type="SEED",
        description="Seed stage",
        created_at=datetime.now()
    )
    session.add(stage)
    await session.commit()

    # 詳細付きで取得
    company_with_details = await repo.get_with_details(company.id)
    assert company_with_details is not None
    assert company_with_details.statuses[0].type == "ACTIVE"
    assert company_with_details.stages[0].type == "SEED"

    # 最新のステータスを取得
    latest_status = await repo.get_latest_status(company.id)
    assert latest_status is not None
    assert latest_status.type == "ACTIVE"

    # 最新のステージを取得
    latest_stage = await repo.get_latest_stage(company.id)
    assert latest_stage is not None
    assert latest_stage.type == "SEED"

    # ステージで会社を取得
    companies_by_stage = await repo.get_companies_by_stage("SEED")
    assert len(companies_by_stage) == 1
    assert companies_by_stage[0].id == company.id

    # ステータスで会社を取得
    companies_by_status = await repo.get_companies_by_status("ACTIVE")
    assert len(companies_by_status) == 1
    assert companies_by_status[0].id == company.id

    # 会社を検索
    search_results = await repo.search_companies(
        name="Test",
        industry="Technology",
        stage_type="SEED",
        status_type="ACTIVE",
        owner_id="test_owner",
        page=1,
        per_page=10
    )
    assert len(search_results) == 1
    assert search_results[0].id == company.id

    # 会社を更新
    updated_company = await repo.update(
        company.id,
        name="Updated Company",
        employee_count=200
    )
    assert updated_company is not None
    assert updated_company.name == "Updated Company"
    assert updated_company.employee_count == 200

    # 会社を削除
    deleted = await repo.delete(company.id)
    assert deleted is True

    # 削除されたことを確認
    deleted_company = await repo.get_by_id(company.id)
    assert deleted_company is None

@pytest.mark.asyncio
async def test_company_repository_error_handling(session):
    """会社リポジトリのエラーハンドリングをテスト"""
    repo = CompanyRepository(session)

    # 存在しない会社IDでの操作
    non_existent_id = "non_existent_id"

    # 存在しない会社の取得
    company = await repo.get_by_id(non_existent_id)
    assert company is None

    # 存在しない会社の詳細取得
    company_with_details = await repo.get_with_details(non_existent_id)
    assert company_with_details is None

    # 存在しない会社の最新ステータス取得
    latest_status = await repo.get_latest_status(non_existent_id)
    assert latest_status is None

    # 存在���ない会社の最新ステージ取得
    latest_stage = await repo.get_latest_stage(non_existent_id)
    assert latest_stage is None

@pytest.mark.asyncio
async def test_company_repository_search_filters(session):
    """会社リポジトリの検索フィルターをテスト"""
    repo = CompanyRepository(session)

    # テスト用の会社を複数作成
    companies = []
    for i in range(5):
        company = await repo.create(
            name=f"Company {i}",
            description=f"Description {i}",
            industry="Technology" if i % 2 == 0 else "Finance",
            owner_id=f"owner_{i}",
            founded_date=datetime.now(),
            employee_count=100 * (i + 1),
            website=f"https://example{i}.com",
            location=f"Location {i}"
        )
        companies.append(company)

        # ステータスとステージを追加
        status = Status(
            company_id=company.id,
            type="ACTIVE" if i % 2 == 0 else "INACTIVE",
            description=f"Status {i}",
            created_at=datetime.now()
        )
        stage = Stage(
            company_id=company.id,
            type="SEED" if i % 2 == 0 else "SERIES_A",
            description=f"Stage {i}",
            created_at=datetime.now()
        )
        session.add(status)
        session.add(stage)
        await session.commit()

    # 業界でフィルター
    tech_companies = await repo.search_companies(
        industry="Technology",
        page=1,
        per_page=10
    )
    assert len(tech_companies) == 3

    # ステージでフィルター
    seed_companies = await repo.search_companies(
        stage_type="SEED",
        page=1,
        per_page=10
    )
    assert len(seed_companies) == 3

    # ステータスでフィルター
    active_companies = await repo.search_companies(
        status_type="ACTIVE",
        page=1,
        per_page=10
    )
    assert len(active_companies) == 3

    # 複数の条件でフィルター
    filtered_companies = await repo.search_companies(
        industry="Technology",
        stage_type="SEED",
        status_type="ACTIVE",
        page=1,
        per_page=10
    )
    assert len(filtered_companies) == 3

    # ページネーション
    paginated_companies = await repo.search_companies(
        page=1,
        per_page=2
    )
    assert len(paginated_companies) == 2