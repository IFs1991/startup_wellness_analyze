# -*- coding: utf-8 -*-
"""
企業情報 API の統合テスト
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import json
from datetime import datetime, timedelta

from backend.app.main import app
from backend.services.company_service import CompanyService
from backend.database.models.entities import CompanyEntity

@pytest.fixture
def client():
    """テストクライアント"""
    return TestClient(app)

class TestCompaniesAPI:
    """企業情報 API のテストクラス"""

    @patch('backend.services.company_service.CompanyService.get_companies')
    def test_get_companies_no_params(self, mock_get_companies, client):
        """パラメータなしでの企業一覧取得テスト"""
        # モックの設定
        mock_companies = [
            CompanyEntity(
                id="1",
                name="テスト企業A",
                industry="テクノロジー",
                employee_count=50,
                created_at=datetime.now() - timedelta(days=10),
                updated_at=datetime.now() - timedelta(days=1)
            ),
            CompanyEntity(
                id="2",
                name="テスト企業B",
                industry="ヘルスケア",
                employee_count=100,
                created_at=datetime.now() - timedelta(days=5),
                updated_at=datetime.now()
            )
        ]
        mock_get_companies.return_value = mock_companies

        # APIリクエスト
        response = client.get("/companies/")

        # 検証
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "テスト企業A"
        assert data[1]["name"] == "テスト企業B"

        # モックの呼び出し確認
        mock_get_companies.assert_called_once()

    @patch('backend.services.company_service.CompanyService.get_companies')
    def test_get_companies_with_search(self, mock_get_companies, client):
        """検索条件付きでの企業一覧取得テスト"""
        # モックの設定
        mock_companies = [
            CompanyEntity(
                id="1",
                name="検索テスト企業",
                industry="テクノロジー"
            )
        ]
        mock_get_companies.return_value = mock_companies

        # APIリクエスト
        response = client.get("/companies/?search=検索")

        # 検証
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "検索テスト企業"

        # モックの呼び出し確認
        mock_get_companies.assert_called_once()

    @patch('backend.services.company_service.CompanyService.get_companies')
    def test_get_companies_with_filters(self, mock_get_companies, client):
        """フィルタ条件付きでの企業一覧取得テスト"""
        # モックの設定
        mock_companies = [
            CompanyEntity(
                id="1",
                name="テスト企業A",
                industry="テクノロジー",
                location="東京都"
            ),
            CompanyEntity(
                id="2",
                name="テスト企業B",
                industry="テクノロジー",
                location="東京都"
            )
        ]
        mock_get_companies.return_value = mock_companies

        # APIリクエスト
        response = client.get("/companies/?filters=industry=テクノロジー,location=東京都")

        # 検証
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["industry"] == "テクノロジー"
        assert data[0]["location"] == "東京都"

        # モックの呼び出し確認
        mock_get_companies.assert_called_once()

    @patch('backend.services.company_service.CompanyService.create_company')
    def test_create_company(self, mock_create_company, client):
        """企業作成テスト"""
        # モックの設定
        company_data = {
            "name": "新規テスト企業",
            "industry": "テクノロジー",
            "employee_count": 50,
            "location": "東京都"
        }

        mock_created_company = CompanyEntity(
            id="generated_id",
            name="新規テスト企業",
            industry="テクノロジー",
            employee_count=50,
            location="東京都",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        mock_create_company.return_value = mock_created_company

        # APIリクエスト
        response = client.post("/companies/", json=company_data)

        # 検証
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "新規テスト企業"
        assert data["industry"] == "テクノロジー"
        assert data["id"] == "generated_id"

        # モックの呼び出し確認
        mock_create_company.assert_called_once()

    @patch('backend.services.company_service.CompanyService.get_company_by_id')
    def test_get_company_by_id(self, mock_get_company, client):
        """企業ID指定での取得テスト"""
        # モックの設定
        company_id = "test123"
        mock_company = CompanyEntity(
            id=company_id,
            name="取得テスト企業",
            industry="テクノロジー",
            created_at=datetime.now() - timedelta(days=10),
            updated_at=datetime.now()
        )
        mock_get_company.return_value = mock_company

        # APIリクエスト
        response = client.get(f"/companies/{company_id}")

        # 検証
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == company_id
        assert data["name"] == "取得テスト企業"

        # モックの呼び出し確認
        mock_get_company.assert_called_once_with(company_id)

    @patch('backend.services.company_service.CompanyService.get_company_by_id')
    def test_get_company_not_found(self, mock_get_company, client):
        """存在しない企業ID指定での取得テスト"""
        # モックの設定
        company_id = "nonexistent"
        mock_get_company.return_value = None

        # APIリクエスト
        response = client.get(f"/companies/{company_id}")

        # 検証
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"]

        # モックの呼び出し確認
        mock_get_company.assert_called_once_with(company_id)

    @patch('backend.services.company_service.CompanyService.update_company')
    def test_update_company(self, mock_update_company, client):
        """企業更新テスト"""
        # モックの設定
        company_id = "test123"
        update_data = {
            "name": "更新テスト企業",
            "employee_count": 100
        }

        mock_updated_company = CompanyEntity(
            id=company_id,
            name="更新テスト企業",
            industry="テクノロジー",
            employee_count=100,
            updated_at=datetime.now()
        )
        mock_update_company.return_value = mock_updated_company

        # APIリクエスト
        response = client.put(f"/companies/{company_id}", json=update_data)

        # 検証
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == company_id
        assert data["name"] == "更新テスト企業"
        assert data["employee_count"] == 100

        # モックの呼び出し確認
        mock_update_company.assert_called_once_with(company_id, update_data)

    @patch('backend.services.company_service.CompanyService.delete_company')
    def test_delete_company(self, mock_delete_company, client):
        """企業削除テスト"""
        # モックの設定
        company_id = "test123"
        mock_delete_company.return_value = True

        # APIリクエスト
        response = client.delete(f"/companies/{company_id}")

        # 検証
        assert response.status_code == 204

        # モックの呼び出し確認
        mock_delete_company.assert_called_once_with(company_id)