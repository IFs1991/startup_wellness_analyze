import pytest
import asyncio
from datetime import datetime
from google.cloud import firestore
from google.cloud.firestore import AsyncClient
from backend.src.database.repositories.user import UserRepository

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

@pytest.mark.asyncio
async def test_user_repository(user_repository):
    """ユーザーリポジトリの基本的なCRUD操作をテスト"""
    # ユーザーを作成
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password_hash": "hashed_password",
        "role": "user",
        "is_active": True
    }

    user = await user_repository.create(**user_data)
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.role == "user"

    # IDで取得
    retrieved_user = await user_repository.get_by_id(user.id)
    assert retrieved_user is not None
    assert retrieved_user.username == user.username

    # メールアドレスで取得
    user_by_email = await user_repository.get_by_email("test@example.com")
    assert user_by_email is not None
    assert user_by_email.id == user.id

    # ユーザー名で取得
    user_by_username = await user_repository.get_by_username("testuser")
    assert user_by_username is not None
    assert user_by_username.id == user.id

    # ユーザーを更新
    updated_data = {
        "username": "updated_user",
        "is_active": False
    }
    updated_user = await user_repository.update(user.id, **updated_data)
    assert updated_user is not None
    assert updated_user.username == "updated_user"
    assert updated_user.is_active is False

    # ユーザーを削除
    deleted = await user_repository.delete(user.id)
    assert deleted is True

    # 削除されたことを確認
    deleted_user = await user_repository.get_by_id(user.id)
    assert deleted_user is None

@pytest.mark.asyncio
async def test_user_repository_search(user_repository):
    """ユーザーリポジトリの検索機能をテスト"""
    # テスト用のユーザーを複数作成
    users_data = [
        {
            "username": f"user_{i}",
            "email": f"user{i}@example.com",
            "password_hash": f"hash_{i}",
            "role": "admin" if i % 2 == 0 else "user",
            "is_active": True
        } for i in range(5)
    ]

    for user_data in users_data:
        await user_repository.create(**user_data)

    # ロールでフィルター
    admin_users = await user_repository.search_users(
        role="admin",
        page=1,
        per_page=10
    )
    assert len(admin_users) == 3

    # アクティブステータスでフィルター
    active_users = await user_repository.search_users(
        is_active=True,
        page=1,
        per_page=10
    )
    assert len(active_users) == 5

    # ページネーション
    paginated_users = await user_repository.search_users(
        page=1,
        per_page=2
    )
    assert len(paginated_users) == 2

@pytest.mark.asyncio
async def test_user_repository_error_handling(user_repository):
    """ユーザーリポジトリのエラーハンドリングをテスト"""
    # 存在しないユーザーIDでの操作
    non_existent_id = "non_existent_id"

    # 存在しないユーザーの取得
    user = await user_repository.get_by_id(non_existent_id)
    assert user is None

    # 存在しないメールアドレスでの取得
    user_by_email = await user_repository.get_by_email("nonexistent@example.com")
    assert user_by_email is None

    # 存在しないユーザー名での取得
    user_by_username = await user_repository.get_by_username("nonexistent")
    assert user_by_username is None

    # 存在しないユーザーの更新
    updated_user = await user_repository.update(
        non_existent_id,
        username="updated"
    )
    assert updated_user is None

    # 存在しないユーザーの削除
    deleted = await user_repository.delete(non_existent_id)
    assert deleted is False