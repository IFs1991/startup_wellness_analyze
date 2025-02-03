import pytest
from httpx import AsyncClient
from backend.main import app
from backend.src.database.models import User, Group, Tag

pytestmark = pytest.mark.asyncio

@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
async def test_user():
    """テスト用のユーザーデータを作成する"""
    return {
        "username": "test_user",
        "email": "test@example.com",
        "password": "test_password",
        "role": "user"
    }

@pytest.fixture
async def test_group():
    """テスト用のグループデータを作成する"""
    return {
        "name": "Test Group",
        "description": "Test Description",
        "is_private": False
    }

async def test_create_group(async_client, test_user_data, test_company_data, test_group_data, auth_headers):
    """グループ作成のテスト"""
    # 会社を作成
    company_response = await async_client.post("/api/companies/", json=test_company_data, headers=auth_headers)
    company_id = company_response.json()["id"]

    # グループデータに会社IDを追加
    group_data = {**test_group_data, "company_id": company_id}

    # グループを作成
    response = await async_client.post("/api/groups/", json=group_data, headers=auth_headers)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == test_group_data["name"]
    assert data["description"] == test_group_data["description"]
    assert data["company_id"] == company_id

async def test_get_group(async_client, test_user_data, test_company_data, test_group_data, auth_headers):
    """グループ取得のテスト"""
    # 会社を作成
    company_response = await async_client.post("/api/companies/", json=test_company_data, headers=auth_headers)
    company_id = company_response.json()["id"]

    # グループを作成
    group_data = {**test_group_data, "company_id": company_id}
    create_response = await async_client.post("/api/groups/", json=group_data, headers=auth_headers)
    group_id = create_response.json()["id"]

    # グループを取得
    response = await async_client.get(f"/api/groups/{group_id}", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_group_data["name"]
    assert data["description"] == test_group_data["description"]

async def test_update_group(async_client, test_user_data, test_company_data, test_group_data, auth_headers):
    """グループ更新のテスト"""
    # 会社を作成
    company_response = await async_client.post("/api/companies/", json=test_company_data, headers=auth_headers)
    company_id = company_response.json()["id"]

    # グループを作成
    group_data = {**test_group_data, "company_id": company_id}
    create_response = await async_client.post("/api/groups/", json=group_data, headers=auth_headers)
    group_id = create_response.json()["id"]

    # グループを更新
    update_data = {
        "name": "Updated Group",
        "description": "Updated Description",
        "is_private": True
    }
    response = await async_client.put(f"/api/groups/{group_id}", json=update_data, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == update_data["name"]
    assert data["description"] == update_data["description"]
    assert data["is_private"] == update_data["is_private"]

async def test_delete_group(async_client, test_user_data, test_company_data, test_group_data, auth_headers):
    """グループ削除のテスト"""
    # 会社を作成
    company_response = await async_client.post("/api/companies/", json=test_company_data, headers=auth_headers)
    company_id = company_response.json()["id"]

    # グループを作成
    group_data = {**test_group_data, "company_id": company_id}
    create_response = await async_client.post("/api/groups/", json=group_data, headers=auth_headers)
    group_id = create_response.json()["id"]

    # グループを削除
    response = await async_client.delete(f"/api/groups/{group_id}", headers=auth_headers)
    assert response.status_code == 204

    # 削除されたグループを取得しようとする
    get_response = await async_client.get(f"/api/groups/{group_id}", headers=auth_headers)
    assert get_response.status_code == 404

async def test_add_member(async_client, test_user_data, test_company_data, test_group_data, auth_headers):
    """メンバー追加のテスト"""
    # 会社を作成
    company_response = await async_client.post("/api/companies/", json=test_company_data, headers=auth_headers)
    company_id = company_response.json()["id"]

    # グループを作成
    group_data = {**test_group_data, "company_id": company_id}
    create_response = await async_client.post("/api/groups/", json=group_data, headers=auth_headers)
    group_id = create_response.json()["id"]

    # 新しいメンバーを作成
    new_member = {
        "username": "new_member",
        "email": "new_member@example.com",
        "password": "member_password",
        "role": "user"
    }
    member_response = await async_client.post("/api/users/register", json=new_member)
    member_id = member_response.json()["id"]

    # メンバーを追加
    response = await async_client.post(f"/api/groups/{group_id}/members/{member_id}", headers=auth_headers)
    assert response.status_code == 200

async def test_remove_member(async_client, test_user_data, test_company_data, test_group_data, auth_headers):
    """メンバー削除のテスト"""
    # 会社を作成
    company_response = await async_client.post("/api/companies/", json=test_company_data, headers=auth_headers)
    company_id = company_response.json()["id"]

    # グループを作成
    group_data = {**test_group_data, "company_id": company_id}
    create_response = await async_client.post("/api/groups/", json=group_data, headers=auth_headers)
    group_id = create_response.json()["id"]

    # メンバーを削除
    member_id = test_user_data["id"]
    response = await async_client.delete(f"/api/groups/{group_id}/members/{member_id}", headers=auth_headers)
    assert response.status_code == 200

async def test_get_group_members(async_client, test_user_data, test_company_data, test_group_data, auth_headers):
    """グループメンバー取得のテスト"""
    # 会社を作成
    company_response = await async_client.post("/api/companies/", json=test_company_data, headers=auth_headers)
    company_id = company_response.json()["id"]

    # グループを作成
    group_data = {**test_group_data, "company_id": company_id}
    create_response = await async_client.post("/api/groups/", json=group_data, headers=auth_headers)
    group_id = create_response.json()["id"]

    # メンバーリストを取得
    response = await async_client.get(f"/api/groups/{group_id}/members", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # 作成者が自動的にメンバーとして追加されているはず
    assert len(data) >= 1

async def test_unauthorized_access(async_client):
    """未認証アクセスのテスト"""
    response = await async_client.get("/api/groups/")
    assert response.status_code == 401