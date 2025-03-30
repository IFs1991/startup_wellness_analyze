import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
from datetime import datetime

from core.auth_manager import User, UserRole

# 前提条件を確認するテスト
def test_router_exists():
    """データ入力ルーターが正しくインポートされていることを確認"""
    from api.routers import data_input
    assert data_input.router is not None
    assert "data_input" in data_input.router.tags

# データアップロードテスト
@patch("api.routers.data_input.DataStorage")
@patch("api.routers.data_input.get_current_active_user")
def test_upload_data(mock_get_user, mock_data_storage, client, mock_user, token_header):
    """データアップロードエンドポイントが正しく動作することを確認"""
    # モックの設定
    mock_get_user.return_value = mock_user

    storage_instance = MagicMock()
    mock_data_storage.return_value = storage_instance
    storage_instance.store_data.return_value = {"stored_items": 10, "collection": "financial_data"}

    # テストデータ
    upload_data = {
        "collection_name": "financial_data",
        "data": [
            {"company_id": "test_company_id", "revenue": 1000000, "expenses": 800000, "year": 2023},
            {"company_id": "test_company_id", "revenue": 1200000, "expenses": 900000, "year": 2022}
        ]
    }

    # リクエスト実行
    response = client.post("/data-input/upload", json=upload_data, headers=token_header)

    # 結果確認
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["stored_items"] == 10
    assert data["data"]["collection"] == "financial_data"
    storage_instance.store_data.assert_called_once()

# CSVアップロードテスト
@patch("api.routers.data_input.DataStorage")
@patch("api.routers.data_input.CSVProcessor")
@patch("api.routers.data_input.get_current_active_user")
def test_upload_csv(mock_get_user, mock_csv_processor, mock_data_storage, client, mock_user, token_header):
    """CSVファイルアップロードエンドポイントが正しく動作することを確認"""
    # モックの設定
    mock_get_user.return_value = mock_user

    csv_processor_instance = MagicMock()
    mock_csv_processor.return_value = csv_processor_instance
    csv_processor_instance.process.return_value = [
        {"company_id": "test_company_id", "revenue": 1000000, "expenses": 800000, "year": 2023},
        {"company_id": "test_company_id", "revenue": 1200000, "expenses": 900000, "year": 2022}
    ]

    storage_instance = MagicMock()
    mock_data_storage.return_value = storage_instance
    storage_instance.store_data.return_value = {"stored_items": 2, "collection": "financial_data"}

    # CSVファイルをモック
    csv_content = "company_id,revenue,expenses,year\ntest_company_id,1000000,800000,2023\ntest_company_id,1200000,900000,2022"
    files = {
        "file": ("test.csv", csv_content, "text/csv")
    }

    # フォームデータ
    form_data = {"collection_name": "financial_data"}

    # リクエスト実行
    response = client.post("/data-input/upload-csv", files=files, data=form_data, headers=token_header)

    # 結果確認
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["stored_items"] == 2
    assert data["data"]["collection"] == "financial_data"
    csv_processor_instance.process.assert_called_once()
    storage_instance.store_data.assert_called_once()

# データ削除テスト
@patch("api.routers.data_input.DataStorage")
@patch("api.routers.data_input.get_current_active_user")
def test_delete_data(mock_get_user, mock_data_storage, client, mock_user, token_header):
    """データ削除エンドポイントが正しく動作することを確認"""
    # モックの設定
    mock_get_user.return_value = mock_user

    storage_instance = MagicMock()
    mock_data_storage.return_value = storage_instance
    storage_instance.delete_data.return_value = {"deleted_items": 5, "collection": "financial_data"}

    # テストデータ
    delete_data = {
        "collection_name": "financial_data",
        "conditions": [
            {"field": "company_id", "operator": "==", "value": "test_company_id"},
            {"field": "year", "operator": "<", "value": 2022}
        ]
    }

    # リクエスト実行
    response = client.post("/data-input/delete", json=delete_data, headers=token_header)

    # 結果確認
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["deleted_items"] == 5
    assert data["data"]["collection"] == "financial_data"
    storage_instance.delete_data.assert_called_once()

# データ取得テスト
@patch("api.routers.data_input.DataStorage")
@patch("api.routers.data_input.get_current_active_user")
def test_get_data(mock_get_user, mock_data_storage, client, mock_user, token_header):
    """データ取得エンドポイントが正しく動作することを確認"""
    # モックの設定
    mock_get_user.return_value = mock_user

    storage_instance = MagicMock()
    mock_data_storage.return_value = storage_instance
    storage_instance.get_data.return_value = [
        {"company_id": "test_company_id", "revenue": 1000000, "expenses": 800000, "year": 2023},
        {"company_id": "test_company_id", "revenue": 1200000, "expenses": 900000, "year": 2022}
    ]

    # テストデータ
    query_params = {
        "collection_name": "financial_data",
        "company_id": "test_company_id",
        "limit": 10
    }

    # リクエスト実行
    response = client.get("/data-input/data", params=query_params, headers=token_header)

    # 結果確認
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["data"]) == 2
    assert data["data"][0]["revenue"] == 1000000
    storage_instance.get_data.assert_called_once()