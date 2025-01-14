import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from backend.src.main import app
from backend.src.database.models import User, Report, ReportTemplate

client = TestClient(app)

@pytest.fixture
def test_user():
    """テスト用のユーザーデータを作成する"""
    return {
        "username": "test_user",
        "email": "test@example.com",
        "password": "test_password",
        "role": "user"
    }

@pytest.fixture
def test_template():
    """テスト用のレポートテンプレートデータを作成する"""
    return {
        "name": "Test Template",
        "description": "Test Description",
        "format": "pdf",
        "template_content": "Test Content",
        "variables": {"key": "value"}
    }

@pytest.fixture
def test_report():
    """テスト用のレポートデータを作成する"""
    return {
        "content": "Test Report Content",
        "format": "pdf",
        "report_metadata": {"status": "completed"}
    }

def test_create_template(test_user, test_template):
    """テンプレート作成のテスト"""
    # ユーザーを登録
    client.post("/api/users/register", json=test_user)

    # ログイン
    login_data = {
        "username": test_user["username"],
        "password": test_user["password"]
    }
    login_response = client.post("/api/users/login", json=login_data)
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # テンプレートを作成
    response = client.post("/api/reports/templates/", json=test_template, headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == test_template["name"]
    assert data["format"] == test_template["format"]
    assert data["variables"] == test_template["variables"]

def test_get_template(test_user, test_template):
    """テンプレート取得のテスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # テンプレートを作成
    create_response = client.post("/api/reports/templates/", json=test_template, headers=headers)
    template_id = create_response.json()["id"]

    # テンプレートを取得
    response = client.get(f"/api/reports/templates/{template_id}", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_template["name"]
    assert data["format"] == test_template["format"]

def test_update_template(test_user, test_template):
    """テンプレート更新のテスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # テンプレートを作成
    create_response = client.post("/api/reports/templates/", json=test_template, headers=headers)
    template_id = create_response.json()["id"]

    # テンプレートを更新
    update_data = {
        "name": "Updated Template",
        "template_content": "Updated Content",
        "variables": {"new_key": "new_value"}
    }
    response = client.put(f"/api/reports/templates/{template_id}", json=update_data, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == update_data["name"]
    assert data["template_content"] == update_data["template_content"]
    assert data["variables"] == update_data["variables"]

def test_generate_report(test_user, test_template, test_report):
    """レポート生成のテスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # テンプレートを作成
    create_template_response = client.post("/api/reports/templates/", json=test_template, headers=headers)
    template_id = create_template_response.json()["id"]

    # レポートを生成
    test_report["template_id"] = template_id
    response = client.post("/api/reports/", json=test_report, headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert data["template_id"] == template_id
    assert data["format"] == test_report["format"]
    assert data["report_metadata"] == test_report["report_metadata"]

def test_get_report(test_user, test_template, test_report):
    """レポート取得のテスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # テンプレートを作成
    create_template_response = client.post("/api/reports/templates/", json=test_template, headers=headers)
    template_id = create_template_response.json()["id"]

    # レポートを生成
    test_report["template_id"] = template_id
    create_report_response = client.post("/api/reports/", json=test_report, headers=headers)
    report_id = create_report_response.json()["id"]

    # レポートを取得
    response = client.get(f"/api/reports/{report_id}", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["template_id"] == template_id
    assert data["format"] == test_report["format"]

def test_search_reports(test_user, test_template):
    """レポート検索のテスト"""
    # ユーザーを��録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # テンプレートを作成
    create_template_response = client.post("/api/reports/templates/", json=test_template, headers=headers)
    template_id = create_template_response.json()["id"]

    # 複数のレポートを生成
    reports = [
        {
            "template_id": template_id,
            "content": f"Report Content {i}",
            "format": "pdf" if i % 2 == 0 else "docx",
            "report_metadata": {"status": "completed" if i % 2 == 0 else "pending"}
        }
        for i in range(5)
    ]

    for report in reports:
        client.post("/api/reports/", json=report, headers=headers)

    # フォーマットで検索
    response = client.get("/api/reports/search?format=pdf", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3

    # メタデータで検索
    response = client.get("/api/reports/search?status=completed", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3

def test_delete_report(test_user, test_template, test_report):
    """レポート削除のテスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # テンプレートを作成
    create_template_response = client.post("/api/reports/templates/", json=test_template, headers=headers)
    template_id = create_template_response.json()["id"]

    # レポートを生成
    test_report["template_id"] = template_id
    create_report_response = client.post("/api/reports/", json=test_report, headers=headers)
    report_id = create_report_response.json()["id"]

    # レポートを削除
    response = client.delete(f"/api/reports/{report_id}", headers=headers)
    assert response.status_code == 200

    # 削除されたレポートの取得を試みる
    response = client.get(f"/api/reports/{report_id}", headers=headers)
    assert response.status_code == 404

def test_unauthorized_access():
    """未認���アクセスのテスト"""
    # 認証なしで保護されたエンドポイントにアクセス
    response = client.get("/api/reports/templates/")
    assert response.status_code == 401