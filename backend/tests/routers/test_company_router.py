import pytest
from datetime import datetime
from httpx import AsyncClient
from backend.main import app
from backend.src.database.models import User, Company, Status, Stage

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
async def test_company():
    """テスト用の会社データを作成する"""
    return {
        "name": "Test Company",
        "description": "Test Description",
        "industry": "Technology",
        "founded_date": datetime.now().isoformat(),
        "employee_count": 100,
        "website": "https://example.com",
        "location": "Tokyo, Japan"
    }

async def test_create_company(async_client, test_user_data, test_company_data, auth_headers):
    """会社作成のテスト"""
    response = await async_client.post("/api/companies/", json=test_company_data, headers=auth_headers)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == test_company_data["name"]
    assert data["industry"] == test_company_data["industry"]
    assert data["employee_count"] == test_company_data["employee_count"]

async def test_get_company(async_client, test_user_data, test_company_data, auth_headers):
    """会社取得のテスト"""
    # 会社を作成
    create_response = await async_client.post("/api/companies/", json=test_company_data, headers=auth_headers)
    company_id = create_response.json()["id"]

    # 会社を取得
    response = await async_client.get(f"/api/companies/{company_id}", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_company_data["name"]
    assert data["industry"] == test_company_data["industry"]

async def test_update_company(async_client, test_user_data, test_company_data, auth_headers):
    """会社更新のテスト"""
    # 会社を作成
    create_response = await async_client.post("/api/companies/", json=test_company_data, headers=auth_headers)
    company_id = create_response.json()["id"]

    # 会社を更新
    update_data = {
        "name": "Updated Company",
        "industry": "Updated Industry",
        "employee_count": 200
    }
    response = await async_client.put(f"/api/companies/{company_id}", json=update_data, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == update_data["name"]
    assert data["industry"] == update_data["industry"]
    assert data["employee_count"] == update_data["employee_count"]

async def test_add_company_status(async_client, test_user_data, test_company_data, test_status_data, auth_headers):
    """会社のステータス追加テスト"""
    # 会社を作成
    create_response = await async_client.post("/api/companies/", json=test_company_data, headers=auth_headers)
    company_id = create_response.json()["id"]

    # ステータスを追加
    response = await async_client.post(
        f"/api/companies/{company_id}/status",
        json=test_status_data,
        headers=auth_headers
    )
    assert response.status_code == 201
    data = response.json()
    assert data["type"] == test_status_data["type"]
    assert data["description"] == test_status_data["description"]

async def test_add_company_stage(async_client, test_user_data, test_company_data, test_stage_data, auth_headers):
    """会社のステージ追加テスト"""
    # 会社を作成
    create_response = await async_client.post("/api/companies/", json=test_company_data, headers=auth_headers)
    company_id = create_response.json()["id"]

    # ステージを追加
    response = await async_client.post(
        f"/api/companies/{company_id}/stage",
        json=test_stage_data,
        headers=auth_headers
    )
    assert response.status_code == 201
    data = response.json()
    assert data["type"] == test_stage_data["type"]
    assert data["description"] == test_stage_data["description"]

async def test_search_companies(async_client, test_user_data, test_company_data, auth_headers):
    """会社検索のテスト"""
    # 会社を作成
    await async_client.post("/api/companies/", json=test_company_data, headers=auth_headers)

    # 会社を検索
    search_params = {
        "industry": test_company_data["industry"],
        "min_employees": 50,
        "max_employees": 150
    }
    response = await async_client.get("/api/companies/search", params=search_params, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert data[0]["industry"] == test_company_data["industry"]
    assert data[0]["employee_count"] >= search_params["min_employees"]
    assert data[0]["employee_count"] <= search_params["max_employees"]

async def test_unauthorized_access(async_client):
    """未認証アクセスのテスト"""
    # 認証なしで保護されたエンドポイントにアクセス
    response = await async_client.get("/api/companies/")
    assert response.status_code == 401

async def test_delete_company(async_client, test_user_data, test_company_data, auth_headers):
    """会社削除のテスト"""
    # 会社を作成
    create_response = await async_client.post("/api/companies/", json=test_company_data, headers=auth_headers)
    company_id = create_response.json()["id"]

    # 会社を削除
    response = await async_client.delete(f"/api/companies/{company_id}", headers=auth_headers)
    assert response.status_code == 204

    # 削除された会社を取得しようとする
    get_response = await async_client.get(f"/api/companies/{company_id}", headers=auth_headers)
    assert get_response.status_code == 404