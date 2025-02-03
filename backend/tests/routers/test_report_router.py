import pytest
from datetime import datetime
from httpx import AsyncClient
from backend.main import app
from backend.src.database.models import User, Report, ReportTemplate as Template

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
async def test_template():
    """テスト用のレポートテンプレートデータを作成する"""
    return {
        "name": "Test Template",
        "description": "Test Description",
        "format": "pdf",
        "template_content": "Test Content",
        "variables": {"key": "value"}
    }

@pytest.fixture
async def test_report():
    """テスト用のレポートデータを作成する"""
    return {
        "content": "Test Report Content",
        "format": "pdf",
        "report_metadata": {"status": "completed"}
    }

async def test_create_template(async_client, test_user, test_template, auth_headers):
    """テンプレート作成のテスト"""
    # テンプレートを作成
    response = await async_client.post("/api/v1/reports/templates/", json=test_template, headers=auth_headers)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == test_template["name"]
    assert data["format"] == test_template["format"]
    assert data["variables"] == test_template["variables"]

async def test_get_template(async_client, test_user, test_template, auth_headers):
    """テンプレート取得のテスト"""
    # テンプレートを作成
    create_response = await async_client.post("/api/v1/reports/templates/", json=test_template, headers=auth_headers)
    template_id = create_response.json()["id"]

    # テンプレートを取得
    response = await async_client.get(f"/api/v1/reports/templates/{template_id}", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_template["name"]
    assert data["format"] == test_template["format"]

async def test_update_template(async_client, test_user, test_template, auth_headers):
    """テンプレート更新のテスト"""
    # テンプレートを作成
    create_response = await async_client.post("/api/v1/reports/templates/", json=test_template, headers=auth_headers)
    template_id = create_response.json()["id"]

    # テンプレートを更新
    update_data = {
        "name": "Updated Template",
        "description": "Updated Description",
        "variables": {"updated_key": "updated_value"}
    }
    response = await async_client.put(f"/api/v1/reports/templates/{template_id}", json=update_data, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == update_data["name"]
    assert data["description"] == update_data["description"]
    assert data["variables"] == update_data["variables"]

async def test_generate_report(async_client, test_user, test_template, test_report, auth_headers):
    """レポート生成のテスト"""
    # テンプレートを作成
    template_response = await async_client.post("/api/v1/reports/templates/", json=test_template, headers=auth_headers)
    template_id = template_response.json()["id"]

    # レポートを生成
    response = await async_client.post(
        f"/api/v1/reports/generate/{template_id}",
        json={"variables": {"test": "value"}},
        headers=auth_headers
    )
    assert response.status_code == 201
    data = response.json()
    assert "content" in data
    assert data["format"] == test_template["format"]

async def test_get_report(async_client, test_user, test_template, test_report, auth_headers):
    """レポート取得のテスト"""
    # レポートを作成
    create_response = await async_client.post("/api/v1/reports/", json=test_report, headers=auth_headers)
    report_id = create_response.json()["id"]

    # レポートを取得
    response = await async_client.get(f"/api/v1/reports/{report_id}", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == test_report["content"]
    assert data["format"] == test_report["format"]

async def test_search_reports(async_client, test_user, test_template, auth_headers):
    """レポート検索のテスト"""
    # 複数のレポートを作成
    reports = [
        {
            "content": f"Test Report {i}",
            "format": "pdf",
            "report_metadata": {"status": "completed"}
        }
        for i in range(3)
    ]

    for report in reports:
        await async_client.post("/api/v1/reports/", json=report, headers=auth_headers)

    # レポートを検索
    search_params = {
        "format": "pdf",
        "status": "completed"
    }
    response = await async_client.get("/api/v1/reports/search", params=search_params, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 3
    for report in data:
        assert report["format"] == "pdf"
        assert report["report_metadata"]["status"] == "completed"

async def test_delete_report(async_client, test_user, test_template, test_report, auth_headers):
    """レポート削除のテスト"""
    # レポートを作成
    create_response = await async_client.post("/api/v1/reports/", json=test_report, headers=auth_headers)
    report_id = create_response.json()["id"]

    # レポートを削除
    response = await async_client.delete(f"/api/v1/reports/{report_id}", headers=auth_headers)
    assert response.status_code == 204

    # 削除されたレポートを取得しようとする
    get_response = await async_client.get(f"/api/v1/reports/{report_id}", headers=auth_headers)
    assert get_response.status_code == 404

async def test_unauthorized_access(async_client):
    """未認証アクセスのテスト"""
    response = await async_client.get("/api/v1/reports/")
    assert response.status_code == 401