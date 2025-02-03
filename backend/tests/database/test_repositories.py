import pytest
import asyncio
from datetime import datetime
from google.cloud import firestore
from google.cloud.firestore import AsyncClient
from backend.src.database.repositories.user import UserRepository
from backend.src.database.repositories.company import CompanyRepository
from backend.src.database.repositories.group import GroupRepository

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
async def user_repository(firestore_client):
    """ユーザーリポジトリのインスタンスを作成"""
    return UserRepository(firestore_client)

@pytest.fixture
async def company_repository(firestore_client):
    """会社リポジトリのインスタンスを作成"""
    return CompanyRepository(firestore_client)

@pytest.fixture
async def group_repository(firestore_client):
    """グループリポジトリのインスタンスを作成"""
    return GroupRepository(firestore_client)

@pytest.mark.asyncio
async def test_user_repository(user_repository):
    """ユーザーリポジトリのテスト"""
    # ユーザーを作成
    user_data = {
        "username": "test_user",
        "email": "test@example.com",
        "password_hash": "hashed_password",
        "role": "user",
        "is_active": True
    }

    user = await user_repository.create(**user_data)
    assert user.username == "test_user"
    assert user.email == "test@example.com"
    assert user.role == "user"

    # ユーザーを取得
    retrieved_user = await user_repository.get_by_id(user.id)
    assert retrieved_user is not None
    assert retrieved_user.username == user.username

    # ユーザー名で取得
    user_by_username = await user_repository.get_by_username("test_user")
    assert user_by_username is not None
    assert user_by_username.id == user.id

    # メールアドレスで取得
    user_by_email = await user_repository.get_by_email("test@example.com")
    assert user_by_email is not None
    assert user_by_email.id == user.id

    # ユーザーを更新
    updated_user = await user_repository.update(
        user.id,
        username="updated_user"
    )
    assert updated_user is not None
    assert updated_user.username == "updated_user"

    # ユーザーを削除
    deleted = await user_repository.delete(user.id)
    assert deleted is True

    # 削除されたことを確認
    deleted_user = await user_repository.get_by_id(user.id)
    assert deleted_user is None

@pytest.mark.asyncio
async def test_company_repository(company_repository):
    """会社リポジトリのテスト"""
    # 会社を作成
    company_data = {
        "name": "Test Company",
        "description": "Test Description",
        "industry": "Technology",
        "owner_id": "test_owner",
        "website": "https://example.com",
        "contact_info": {"email": "contact@example.com"}
    }

    company = await company_repository.create(**company_data)
    assert company.name == "Test Company"
    assert company.industry == "Technology"

    # 会社を取得
    retrieved_company = await company_repository.get_by_id(company.id)
    assert retrieved_company is not None
    assert retrieved_company.name == company.name

    # オーナーIDで取得
    companies = await company_repository.get_by_owner("test_owner")
    assert len(companies) == 1
    assert companies[0].id == company.id

    # 業種で取得
    companies = await company_repository.get_by_industry("Technology")
    assert len(companies) == 1
    assert companies[0].id == company.id

    # 会社を更新
    updated_company = await company_repository.update(
        company.id,
        name="Updated Company"
    )
    assert updated_company is not None
    assert updated_company.name == "Updated Company"

    # 会社を削除
    deleted = await company_repository.delete(company.id)
    assert deleted is True

    # 削除されたことを確認
    deleted_company = await company_repository.get_by_id(company.id)
    assert deleted_company is None

@pytest.mark.asyncio
async def test_group_repository(group_repository):
    """グループリポジトリのテスト"""
    # グループを作成
    group_data = {
        "name": "Test Group",
        "description": "Test Description",
        "owner_id": "test_owner",
        "is_private": False
    }

    group = await group_repository.create(**group_data)
    assert group.name == "Test Group"
    assert group.owner_id == "test_owner"

    # グループを取得
    retrieved_group = await group_repository.get_by_id(group.id)
    assert retrieved_group is not None
    assert retrieved_group.name == group.name

    # オーナーIDで取得
    groups = await group_repository.get_by_owner("test_owner")
    assert len(groups) == 1
    assert groups[0].id == group.id

    # タグを追加
    tag_data = {
        "name": "Test Tag",
        "description": "Test Tag Description"
    }
    tag = await group_repository.add_tag(group.id, **tag_data)
    assert tag.name == "Test Tag"
    assert tag.group_id == group.id

    # グループのタグを取得
    tags = await group_repository.get_group_tags(group.id)
    assert len(tags) == 1
    assert tags[0].name == "Test Tag"

    # タグを削除
    deleted_tag = await group_repository.remove_tag(group.id, tag.name)
    assert deleted_tag is True

    # グループを更新
    updated_group = await group_repository.update(
        group.id,
        name="Updated Group"
    )
    assert updated_group is not None
    assert updated_group.name == "Updated Group"

    # グループを削除
    deleted = await group_repository.delete(group.id)
    assert deleted is True

    # 削除されたことを確認
    deleted_group = await group_repository.get_by_id(group.id)
    assert deleted_group is None