import pytest
import asyncio
from datetime import datetime
from google.cloud import firestore
from google.cloud.firestore import AsyncClient
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
async def group_repository(firestore_client):
    """グループリポジトリのインスタンスを作成"""
    return GroupRepository(firestore_client)

@pytest.mark.asyncio
async def test_group_repository(group_repository):
    """グループリポジトリの基本的なCRUD操作をテスト"""
    # グループを作成
    group_data = {
        "name": "Test Group",
        "description": "Test Description",
        "owner_id": "test_owner",
        "is_private": False
    }

    group = await group_repository.create(**group_data)
    assert group.name == "Test Group"
    assert group.description == "Test Description"
    assert group.owner_id == "test_owner"

    # IDで取得
    retrieved_group = await group_repository.get_by_id(group.id)
    assert retrieved_group is not None
    assert retrieved_group.name == group.name

    # オーナーIDで取得
    groups_by_owner = await group_repository.get_by_owner("test_owner")
    assert len(groups_by_owner) == 1
    assert groups_by_owner[0].id == group.id

    # メンバーを追加
    member_data = {
        "user_id": "test_member",
        "role": "member"
    }
    await group_repository.add_member(group.id, **member_data)

    # グループメンバーを取得
    members = await group_repository.get_members(group.id)
    assert len(members) == 1
    assert members[0].user_id == "test_member"
    assert members[0].role == "member"

    # タグを追加
    tag_data = {
        "name": "test_tag"
    }
    await group_repository.add_tag(group.id, **tag_data)

    # グループのタグを取得
    tags = await group_repository.get_tags(group.id)
    assert len(tags) == 1
    assert tags[0].name == "test_tag"

    # グループを更新
    updated_data = {
        "name": "Updated Group",
        "is_private": True
    }
    updated_group = await group_repository.update(group.id, **updated_data)
    assert updated_group is not None
    assert updated_group.name == "Updated Group"
    assert updated_group.is_private is True

    # メンバーを削除
    removed = await group_repository.remove_member(group.id, "test_member")
    assert removed is True

    # タグを削除
    removed = await group_repository.remove_tag(group.id, "test_tag")
    assert removed is True

    # グループを削除
    deleted = await group_repository.delete(group.id)
    assert deleted is True

    # 削除されたことを確認
    deleted_group = await group_repository.get_by_id(group.id)
    assert deleted_group is None

@pytest.mark.asyncio
async def test_group_repository_search(group_repository):
    """グループリポジトリの検索機能をテスト"""
    # テスト用のグループを複数作成
    groups_data = [
        {
            "name": f"Group {i}",
            "description": f"Description {i}",
            "owner_id": f"owner_{i}",
            "is_private": i % 2 == 0
        } for i in range(5)
    ]

    created_groups = []
    for group_data in groups_data:
        group = await group_repository.create(**group_data)
        created_groups.append(group)

        # メンバーとタグを追加
        await group_repository.add_member(
            group.id,
            user_id=f"member_{i}",
            role="member"
        )

        await group_repository.add_tag(
            group.id,
            name=f"tag_{i}"
        )

    # プライベート設定でフィルター
    private_groups = await group_repository.search_groups(
        is_private=True,
        page=1,
        per_page=10
    )
    assert len(private_groups) == 3

    # タグでフィルター
    groups_by_tag = await group_repository.get_groups_by_tag("tag_0")
    assert len(groups_by_tag) == 1

    # メンバーでフィルター
    groups_by_member = await group_repository.get_groups_by_member("member_0")
    assert len(groups_by_member) == 1

    # ページネーション
    paginated_groups = await group_repository.search_groups(
        page=1,
        per_page=2
    )
    assert len(paginated_groups) == 2

@pytest.mark.asyncio
async def test_group_repository_error_handling(group_repository):
    """グループリポジトリのエラーハンドリングをテスト"""
    # 存在しないグループIDでの操作
    non_existent_id = "non_existent_id"

    # 存在しないグループの取得
    group = await group_repository.get_by_id(non_existent_id)
    assert group is None

    # 存在しないグループのメンバー取得
    members = await group_repository.get_members(non_existent_id)
    assert len(members) == 0

    # 存在しないグループのタグ取得
    tags = await group_repository.get_tags(non_existent_id)
    assert len(tags) == 0

    # 存在しないグループの更新
    updated_group = await group_repository.update(
        non_existent_id,
        name="updated"
    )
    assert updated_group is None

    # 存在しないグループの削除
    deleted = await group_repository.delete(non_existent_id)
    assert deleted is False