import pytest
import asyncio
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from backend.src.database.models import Base, User, UserRole, Company, Group, Tag
from backend.src.database.repositories.user import UserRepository
from backend.src.database.repositories.company import CompanyRepository
from backend.src.database.repositories.group import GroupRepository

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
async def test_user_repository(session):
    """ユーザーリポジトリのテスト"""
    repo = UserRepository(session)

    # ユーザーを作成
    user = await repo.create(
        username="test_user",
        email="test@example.com",
        hashed_password="hashed_password",
        role=UserRole.USER,
        is_active=True
    )

    assert user.username == "test_user"
    assert user.email == "test@example.com"

    # ユーザーを取得
    retrieved_user = await repo.get_by_id(user.id)
    assert retrieved_user is not None
    assert retrieved_user.username == user.username

    # ユーザー名で取得
    user_by_username = await repo.get_by_username("test_user")
    assert user_by_username is not None
    assert user_by_username.id == user.id

    # メールアドレスで取得
    user_by_email = await repo.get_by_email("test@example.com")
    assert user_by_email is not None
    assert user_by_email.id == user.id

    # ユーザーを更新
    updated_user = await repo.update(user.id, username="updated_user")
    assert updated_user is not None
    assert updated_user.username == "updated_user"

    # ユーザーを削除
    deleted = await repo.delete(user.id)
    assert deleted is True

    # 削除されたことを確認
    deleted_user = await repo.get_by_id(user.id)
    assert deleted_user is None

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
        website="https://example.com",
        contact_info={"email": "contact@example.com"}
    )

    assert company.name == "Test Company"
    assert company.industry == "Technology"

    # 会社を取得
    retrieved_company = await repo.get_by_id(company.id)
    assert retrieved_company is not None
    assert retrieved_company.name == company.name

    # オーナーIDで取得
    companies = await repo.get_by_owner("test_owner")
    assert len(companies) == 1
    assert companies[0].id == company.id

    # 業種で取得
    companies = await repo.get_by_industry("Technology")
    assert len(companies) == 1
    assert companies[0].id == company.id

    # 会社を更新
    updated_company = await repo.update(company.id, name="Updated Company")
    assert updated_company is not None
    assert updated_company.name == "Updated Company"

    # 会社を削除
    deleted = await repo.delete(company.id)
    assert deleted is True

    # 削除されたことを確認
    deleted_company = await repo.get_by_id(company.id)
    assert deleted_company is None

@pytest.mark.asyncio
async def test_group_repository(session):
    """グループリポジトリのテスト"""
    repo = GroupRepository(session)

    # グループを作成
    group = await repo.create(
        name="Test Group",
        description="Test Description",
        owner_id="test_owner"
    )

    assert group.name == "Test Group"
    assert group.owner_id == "test_owner"

    # グループを取得
    retrieved_group = await repo.get_by_id(group.id)
    assert retrieved_group is not None
    assert retrieved_group.name == group.name

    # オーナーIDで取得
    groups = await repo.get_by_owner("test_owner")
    assert len(groups) == 1
    assert groups[0].id == group.id

    # タグを追加
    tag = await repo.add_tag(
        group_id=group.id,
        name="Test Tag",
        description="Test Tag Description"
    )
    assert tag.name == "Test Tag"
    assert tag.group_id == group.id

    # グループのタグを取得
    tags = await repo.get_group_tags(group.id)
    assert len(tags) == 1
    assert tags[0].name == "Test Tag"

    # タグを削除
    deleted_tag = await repo.remove_tag(tag.id)
    assert deleted_tag is True

    # グループを更新
    updated_group = await repo.update(group.id, name="Updated Group")
    assert updated_group is not None
    assert updated_group.name == "Updated Group"

    # グループを削除
    deleted = await repo.delete(group.id)
    assert deleted is True

    # 削除されたことを確認
    deleted_group = await repo.get_by_id(group.id)
    assert deleted_group is None