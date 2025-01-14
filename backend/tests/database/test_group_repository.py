import pytest
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from backend.src.database.models import Base, Group, User, Tag
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

@pytest.fixture
async def test_user(session):
    """テスト用のユーザーを作成��る"""
    user = User(
        username="test_user",
        email="test@example.com",
        password_hash="hashed_password",
        role="user",
        is_active=True
    )
    session.add(user)
    await session.commit()
    return user

@pytest.mark.asyncio
async def test_group_repository(session, test_user):
    """グループリポジトリのテスト"""
    repo = GroupRepository(session)

    # グループを作成
    group = await repo.create(
        name="Test Group",
        description="Test Description",
        owner_id=test_user.id,
        is_private=False
    )

    assert group.name == "Test Group"
    assert group.owner_id == test_user.id
    assert group.is_private is False

    # オーナーIDで取得
    groups_by_owner = await repo.get_by_owner(test_user.id)
    assert len(groups_by_owner) == 1
    assert groups_by_owner[0].id == group.id

    # メンバーを追加
    member_user = User(
        username="member_user",
        email="member@example.com",
        password_hash="hashed_password",
        role="user",
        is_active=True
    )
    session.add(member_user)
    await session.commit()

    await repo.add_member(group.id, member_user.id, "member")

    # タグを追加
    tag = Tag(name="test_tag")
    session.add(tag)
    await session.commit()

    await repo.add_tag(group.id, tag.id)

    # 詳細付きで取得
    group_with_details = await repo.get_with_details(group.id)
    assert group_with_details is not None
    assert len(group_with_details.members) == 2  # オーナーとメンバー
    assert len(group_with_details.tags) == 1

    # ユーザーのグループを取得
    user_groups = await repo.get_user_groups(member_user.id)
    assert len(user_groups) == 1
    assert user_groups[0].id == group.id

    # メンバーのロールを更新
    await repo.update_member_role(group.id, member_user.id, "admin")
    group_with_details = await repo.get_with_details(group.id)
    member = next(m for m in group_with_details.members if m.user_id == member_user.id)
    assert member.role == "admin"

    # グループのタグを取得
    group_tags = await repo.get_group_tags(group.id)
    assert len(group_tags) == 1
    assert group_tags[0].name == "test_tag"

    # グループを検索
    search_results = await repo.search_groups(
        name="Test",
        owner_id=test_user.id,
        member_id=member_user.id,
        tag_name="test_tag",
        page=1,
        per_page=10
    )
    assert len(search_results) == 1
    assert search_results[0].id == group.id

    # タグを削除
    await repo.remove_tag(group.id, tag.id)
    group_with_details = await repo.get_with_details(group.id)
    assert len(group_with_details.tags) == 0

    # メンバーを削除
    await repo.remove_member(group.id, member_user.id)
    group_with_details = await repo.get_with_details(group.id)
    assert len(group_with_details.members) == 1  # オーナーのみ

    # グループを更新
    updated_group = await repo.update(
        group.id,
        name="Updated Group",
        is_private=True
    )
    assert updated_group is not None
    assert updated_group.name == "Updated Group"
    assert updated_group.is_private is True

    # グループを削除
    deleted = await repo.delete(group.id)
    assert deleted is True

    # 削除されたことを確認
    deleted_group = await repo.get_by_id(group.id)
    assert deleted_group is None

@pytest.mark.asyncio
async def test_group_repository_error_handling(session):
    """グループリポジトリのエラーハンドリングをテスト"""
    repo = GroupRepository(session)

    # 存在しないグループIDでの操作
    non_existent_id = "non_existent_id"

    # 存在しないグループの取得
    group = await repo.get_by_id(non_existent_id)
    assert group is None

    # 存在しないグループの詳細取得
    group_with_details = await repo.get_with_details(non_existent_id)
    assert group_with_details is None

    # 存在しないグループへのメンバー追加
    with pytest.raises(Exception):
        await repo.add_member(non_existent_id, "user_id", "member")

    # 存在しないグループ��のタグ追加
    with pytest.raises(Exception):
        await repo.add_tag(non_existent_id, "tag_id")

@pytest.mark.asyncio
async def test_group_repository_member_management(session, test_user):
    """グループメンバー管理のテスト"""
    repo = GroupRepository(session)

    # グループを作成
    group = await repo.create(
        name="Test Group",
        description="Test Description",
        owner_id=test_user.id,
        is_private=False
    )

    # 複数のメンバーを追加
    members = []
    for i in range(5):
        user = User(
            username=f"member_{i}",
            email=f"member{i}@example.com",
            password_hash="hashed_password",
            role="user",
            is_active=True
        )
        session.add(user)
        await session.commit()
        members.append(user)

        await repo.add_member(group.id, user.id, "member")

    # メンバー数を確認
    group_with_details = await repo.get_with_details(group.id)
    assert len(group_with_details.members) == 6  # オーナー + 5メンバー

    # メンバーのロールを一括更新
    for member in members[:2]:
        await repo.update_member_role(group.id, member.id, "admin")

    group_with_details = await repo.get_with_details(group.id)
    admin_count = sum(1 for m in group_with_details.members if m.role == "admin")
    assert admin_count == 2

    # メンバーを一括削除
    for member in members[3:]:
        await repo.remove_member(group.id, member.id)

    group_with_details = await repo.get_with_details(group.id)
    assert len(group_with_details.members) == 4  # オーナー + 3メンバー