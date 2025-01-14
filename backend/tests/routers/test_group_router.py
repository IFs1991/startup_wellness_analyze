import pytest
from fastapi.testclient import TestClient
from backend.src.main import app
from backend.src.database.models import User, Group, Tag

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
def test_group():
    """テスト用のグループデータを作成する"""
    return {
        "name": "Test Group",
        "description": "Test Description",
        "is_private": False
    }

def test_create_group(test_user, test_group):
    """グループ作成のテスト"""
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

    # グループを作成
    response = client.post("/api/groups/", json=test_group, headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == test_group["name"]
    assert data["owner_id"] is not None
    assert data["is_private"] == test_group["is_private"]

def test_get_group(test_user, test_group):
    """グループ取得のテスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # グループを作成
    create_response = client.post("/api/groups/", json=test_group, headers=headers)
    group_id = create_response.json()["id"]

    # グループを取得
    response = client.get(f"/api/groups/{group_id}", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_group["name"]
    assert data["description"] == test_group["description"]

def test_update_group(test_user, test_group):
    """グループ更新のテスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # グループを作成
    create_response = client.post("/api/groups/", json=test_group, headers=headers)
    group_id = create_response.json()["id"]

    # グループを更新
    update_data = {
        "name": "Updated Group",
        "is_private": True
    }
    response = client.put(f"/api/groups/{group_id}", json=update_data, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == update_data["name"]
    assert data["is_private"] == update_data["is_private"]

def test_add_group_member(test_user, test_group):
    """グループメンバー追加のテスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 新しいメンバーを作成
    member_data = {
        "username": "member_user",
        "email": "member@example.com",
        "password": "password",
        "role": "user"
    }
    client.post("/api/users/register", json=member_data)

    # グループを作成
    create_response = client.post("/api/groups/", json=test_group, headers=headers)
    group_id = create_response.json()["id"]

    # メンバーを追加
    member_add_data = {
        "username": member_data["username"],
        "role": "member"
    }
    response = client.post(f"/api/groups/{group_id}/members", json=member_add_data, headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert data["role"] == member_add_data["role"]

def test_add_group_tag(test_user, test_group):
    """グループタグ追加のテスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # グループを作成
    create_response = client.post("/api/groups/", json=test_group, headers=headers)
    group_id = create_response.json()["id"]

    # タグを追加
    tag_data = {
        "name": "test_tag"
    }
    response = client.post(f"/api/groups/{group_id}/tags", json=tag_data, headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == tag_data["name"]

def test_search_groups(test_user):
    """グループ検索のテスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 複数のグループを作成
    groups = [
        {
            "name": f"Group {i}",
            "description": f"Description {i}",
            "is_private": i % 2 == 0
        }
        for i in range(5)
    ]

    for group in groups:
        client.post("/api/groups/", json=group, headers=headers)

    # 名前で検索
    response = client.get("/api/groups/search?name=Group", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5

    # プライベート設定で検索
    response = client.get("/api/groups/search?is_private=false", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2

def test_update_member_role(test_user, test_group):
    """メンバーロール更新のテスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 新しいメンバーを作成
    member_data = {
        "username": "member_user",
        "email": "member@example.com",
        "password": "password",
        "role": "user"
    }
    client.post("/api/users/register", json=member_data)

    # グループを作成
    create_response = client.post("/api/groups/", json=test_group, headers=headers)
    group_id = create_response.json()["id"]

    # メンバーを追加
    member_add_data = {
        "username": member_data["username"],
        "role": "member"
    }
    client.post(f"/api/groups/{group_id}/members", json=member_add_data, headers=headers)

    # メンバーのロールを更新
    role_update_data = {
        "role": "admin"
    }
    response = client.put(
        f"/api/groups/{group_id}/members/{member_data['username']}/role",
        json=role_update_data,
        headers=headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == role_update_data["role"]

def test_remove_member(test_user, test_group):
    """メンバー削除のテスト"""
    # ユーザーを登録してログイン
    client.post("/api/users/register", json=test_user)
    login_response = client.post("/api/users/login", json={
        "username": test_user["username"],
        "password": test_user["password"]
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 新しいメンバーを作成
    member_data = {
        "username": "member_user",
        "email": "member@example.com",
        "password": "password",
        "role": "user"
    }
    client.post("/api/users/register", json=member_data)

    # グループを作成
    create_response = client.post("/api/groups/", json=test_group, headers=headers)
    group_id = create_response.json()["id"]

    # メンバーを追加
    member_add_data = {
        "username": member_data["username"],
        "role": "member"
    }
    client.post(f"/api/groups/{group_id}/members", json=member_add_data, headers=headers)

    # メンバーを削除
    response = client.delete(
        f"/api/groups/{group_id}/members/{member_data['username']}",
        headers=headers
    )
    assert response.status_code == 200

    # グループの詳細を取得して確認
    response = client.get(f"/api/groups/{group_id}", headers=headers)
    data = response.json()
    members = [m for m in data["members"] if m["username"] == member_data["username"]]
    assert len(members) == 0

def test_unauthorized_access():
    """未認証アクセスのテスト"""
    # 認証なしで保護されたエンドポイントにアクセス
    response = client.get("/api/groups/")
    assert response.status_code == 401