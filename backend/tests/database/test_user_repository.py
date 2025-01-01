import pytest
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from backend.src.database.models import Base, User, UserRole
from backend.src.database.repositories.user import UserRepository

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
        password_hash="hashed_password",
        role="user",
        is_active=True
    )

    assert user.username == "test_user"
    assert user.email == "test@example.com"
    assert user.role == "user"
    assert user.is_active is True

    # ユーザー名で取得
    user_by_username = await repo.get_by_username("test_user")
    assert user_by_username is not None
    assert user_by_username.id == user.id

    # メールアドレスで取得
    user_by_email = await repo.get_by_email("test@example.com")
    assert user_by_email is not None
    assert user_by_email.id == user.id

    # ロールで取得
    users_by_role = await repo.get_by_role("user")
    assert len(users_by_role) == 1
    assert users_by_role[0].id == user.id

    # アクティブユーザーを取得
    active_users = await repo.get_active_users()
    assert len(active_users) == 1
    assert active_users[0].id == user.id

    # ユーザーを非アクティブ化
    deactivated_user = await repo.deactivate_user(user.id)
    assert deactivated_user is not None
    assert deactivated_user.is_active is False

    # ユーザーをアクティブ化
    activated_user = await repo.activate_user(user.id)
    assert activated_user is not None
    assert activated_user.is_active is True

    # ロールを更新
    updated_user = await repo.update_role(user.id, "admin")
    assert updated_user is not None
    assert updated_user.role == "admin"

    # パスワードを更新
    updated_user = await repo.update_password(user.id, "new_hashed_password")
    assert updated_user is not None
    assert updated_user.password_hash == "new_hashed_password"

    # ユーザーを削除
    deleted = await repo.delete(user.id)
    assert deleted is True

    # 削除されたことを確認
    deleted_user = await repo.get_by_id(user.id)
    assert deleted_user is None

@pytest.mark.asyncio
async def test_user_repository_error_handling(session):
    """ユーザーリポジトリのエラーハンドリングをテスト"""
    repo = UserRepository(session)

    # 存在しないユーザーIDでの操作
    non_existent_id = "non_existent_id"

    # 存在しないユーザーの取得
    user = await repo.get_by_id(non_existent_id)
    assert user is None

    # 存在しないユーザーの非アクティブ化
    with pytest.raises(Exception):
        await repo.deactivate_user(non_existent_id)

    # 存在しないユーザーのアクティブ化
    with pytest.raises(Exception):
        await repo.activate_user(non_existent_id)

    # 存在しないユーザーのロール更新
    with pytest.raises(Exception):
        await repo.update_role(non_existent_id, "admin")

    # 存在しないユーザーのパスワード更新
    with pytest.raises(Exception):
        await repo.update_password(non_existent_id, "new_password")

@pytest.mark.asyncio
async def test_user_repository_duplicate_handling(session):
    """重複ユーザーの処理をテスト"""
    repo = UserRepository(session)

    # 最初のユーザーを作成
    user1 = await repo.create(
        username="test_user",
        email="test@example.com",
        password_hash="hashed_password",
        role="user",
        is_active=True
    )

    # 同じユーザー名で別のユーザーを作成しようとする
    with pytest.raises(Exception):
        await repo.create(
            username="test_user",
            email="different@example.com",
            password_hash="hashed_password",
            role="user",
            is_active=True
        )

    # 同じメールアドレスで別のユーザーを作成しようとする
    with pytest.raises(Exception):
        await repo.create(
            username="different_user",
            email="test@example.com",
            password_hash="hashed_password",
            role="user",
            is_active=True
        )