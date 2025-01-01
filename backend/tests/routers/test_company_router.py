import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from backend.src.main import app
from backend.src.database.models import User, Company, Status, Stage

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
def test_company():
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

def test_create_company(test_user, test_company):
    """会社作成のテスト"""
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

    # 会社を作成
    response = client.post("/api/companies/", json=test_company, headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == test_company["name"]
    assert data["industry"] == test_company["industry"]
    assert data["owner_id"] is not None

def test_get_company(test_user, test_company):
    """会社取得のテスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 会社を作成
    create_response = client.post("/api/companies/", json=test_company, headers=headers)
    company_id = create_response.json()["id"]

    # 会社を取得
    response = client.get(f"/api/companies/{company_id}", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_company["name"]
    assert data["industry"] == test_company["industry"]

def test_update_company(test_user, test_company):
    """会社更新のテスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 会社を作成
    create_response = client.post("/api/companies/", json=test_company, headers=headers)
    company_id = create_response.json()["id"]

    # 会社を更新
    update_data = {
        "name": "Updated Company",
        "employee_count": 200
    }
    response = client.put(f"/api/companies/{company_id}", json=update_data, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == update_data["name"]
    assert data["employee_count"] == update_data["employee_count"]

def test_add_company_status(test_user, test_company):
    """会社のステータス追加テスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 会社を作成
    create_response = client.post("/api/companies/", json=test_company, headers=headers)
    company_id = create_response.json()["id"]

    # ステータスを追加
    status_data = {
        "type": "ACTIVE",
        "description": "Company is active"
    }
    response = client.post(f"/api/companies/{company_id}/status", json=status_data, headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert data["type"] == status_data["type"]
    assert data["description"] == status_data["description"]

def test_add_company_stage(test_user, test_company):
    """会社のステージ追加テスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 会社を作成
    create_response = client.post("/api/companies/", json=test_company, headers=headers)
    company_id = create_response.json()["id"]

    # ステージを追加
    stage_data = {
        "type": "SEED",
        "description": "Seed stage"
    }
    response = client.post(f"/api/companies/{company_id}/stage", json=stage_data, headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert data["type"] == stage_data["type"]
    assert data["description"] == stage_data["description"]

def test_search_companies(test_user):
    """会社検索のテスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 複数の会社を作成
    companies = [
        {
            "name": f"Company {i}",
            "description": f"Description {i}",
            "industry": "Technology" if i % 2 == 0 else "Finance",
            "founded_date": datetime.now().isoformat(),
            "employee_count": 100 * (i + 1),
            "website": f"https://example{i}.com",
            "location": f"Location {i}"
        }
        for i in range(5)
    ]

    for company in companies:
        client.post("/api/companies/", json=company, headers=headers)

    # 業界で検索
    response = client.get("/api/companies/search?industry=Technology", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3

    # 従業員数で検索
    response = client.get("/api/companies/search?min_employees=200", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 4

def test_unauthorized_access():
    """未認証アクセスのテスト"""
    # 認証なしで保護されたエンドポイントにアクセス
    response = client.get("/api/companies/")
    assert response.status_code == 401

def test_delete_company(test_user, test_company):
    """会社削除のテスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 会社を作成
    create_response = client.post("/api/companies/", json=test_company, headers=headers)
    company_id = create_response.json()["id"]

    # 会社を削除
    response = client.delete(f"/api/companies/{company_id}", headers=headers)
    assert response.status_code == 200

    # 削除された会社の取得を試みる
    response = client.get(f"/api/companies/{company_id}", headers=headers)
    assert response.status_code == 404