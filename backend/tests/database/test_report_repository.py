import pytest
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from backend.src.database.models import Base, Report, ReportTemplate
from backend.src.database.repositories.report import ReportRepository, ReportTemplateRepository

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
async def test_report_template_repository(session):
    """レポートテンプレートリポジトリのテスト"""
    repo = ReportTemplateRepository(session)

    # テンプレートを作成
    template = await repo.create(
        name="Test Template",
        description="Test Description",
        format="pdf",
        template_content="Test Content",
        variables={"key": "value"}
    )

    assert template.name == "Test Template"
    assert template.format == "pdf"

    # テンプレートを取得
    retrieved_template = await repo.get_by_id(template.id)
    assert retrieved_template is not None
    assert retrieved_template.name == template.name

    # 名前で取得
    template_by_name = await repo.get_by_name("Test Template")
    assert template_by_name is not None
    assert template_by_name.id == template.id

    # フォーマットで取得
    templates = await repo.get_by_format("pdf")
    assert len(templates) == 1
    assert templates[0].id == template.id

    # テンプレートを検索
    search_results = await repo.search_templates(
        name="Test",
        format="pdf"
    )
    assert len(search_results) == 1
    assert search_results[0].id == template.id

    # テンプレートの内容を更新
    updated_template = await repo.update_template_content(
        template.id,
        content="Updated Content",
        variables={"new_key": "new_value"}
    )
    assert updated_template is not None
    assert updated_template.template_content == "Updated Content"
    assert updated_template.variables == {"new_key": "new_value"}

    # テンプレートを削除
    deleted = await repo.delete(template.id)
    assert deleted is True

    # 削除されたことを確認
    deleted_template = await repo.get_by_id(template.id)
    assert deleted_template is None

@pytest.mark.asyncio
async def test_report_repository(session):
    """レポートリポジトリのテスト"""
    template_repo = ReportTemplateRepository(session)
    report_repo = ReportRepository(session)

    # テンプレートを作成
    template = await template_repo.create(
        name="Test Template",
        description="Test Description",
        format="pdf",
        template_content="Test Content",
        variables={"key": "value"}
    )

    # レポートを作成
    report = await report_repo.create(
        template_id=template.id,
        content="Test Report Content",
        format="pdf",
        report_metadata={"status": "completed"}
    )

    assert report.template_id == template.id
    assert report.format == "pdf"

    # レポートを取得
    retrieved_report = await report_repo.get_by_id(report.id)
    assert retrieved_report is not None
    assert retrieved_report.content == "Test Report Content"

    # テンプレート情報付きでレポートを取得
    report_with_template = await report_repo.get_with_template(report.id)
    assert report_with_template is not None
    assert report_with_template.template.name == "Test Template"

    # テンプレートIDでレポートを取得
    reports = await report_repo.get_by_template(template.id)
    assert len(reports) == 1
    assert reports[0].id == report.id

    # フォーマットでレポートを取得
    reports = await report_repo.get_reports_by_format("pdf")
    assert len(reports) == 1
    assert reports[0].id == report.id

    # レポートを検索
    search_results = await report_repo.search_reports(
        template_id=template.id,
        format="pdf",
        report_metadata={"status": "completed"}
    )
    assert len(search_results) == 1
    assert search_results[0].id == report.id

    # レポートを更新
    updated_report = await report_repo.update(
        report.id,
        content="Updated Report Content"
    )
    assert updated_report is not None
    assert updated_report.content == "Updated Report Content"

    # レポートを削除
    deleted = await report_repo.delete(report.id)
    assert deleted is True

    # 削除されたことを確認
    deleted_report = await report_repo.get_by_id(report.id)
    assert deleted_report is None