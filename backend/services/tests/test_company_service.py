# -*- coding: utf-8 -*-
"""
企業サービスの単体テスト
"""
import pytest
from unittest.mock import MagicMock, patch
import datetime

from backend.services.company_service import CompanyService
from backend.database.models.entities import CompanyEntity
from backend.database.repository import EntityNotFoundException

class TestCompanyService:
    """企業サービスのテストクラス"""

    def setup_method(self):
        """各テストメソッド実行前のセットアップ"""
        # リポジトリのモック
        self.mock_repo = MagicMock()

        # セッションのモック
        self.mock_session = MagicMock()

        # パッチ適用
        self.repo_patcher = patch('backend.services.company_service.CompanyRepository')
        self.repo_class_mock = self.repo_patcher.start()
        self.repo_class_mock.return_value = self.mock_repo

        # サービスのインスタンス作成
        self.service = CompanyService(session=self.mock_session)

    def teardown_method(self):
        """各テストメソッド実行後のクリーンアップ"""
        self.repo_patcher.stop()

    async def test_get_companies_no_filters(self):
        """フィルタなしでの企業一覧取得テスト"""
        # モックデータ
        mock_companies = [
            CompanyEntity(id="1", name="企業A"),
            CompanyEntity(id="2", name="企業B")
        ]
        self.mock_repo.find_all.return_value = mock_companies

        # テスト対象のメソッド呼び出し
        result = await self.service.get_companies()

        # 検証
        self.mock_repo.find_all.assert_called_once_with(limit=50)
        assert result == mock_companies
        assert len(result) == 2

    async def test_get_companies_with_search(self):
        """検索条件での企業一覧取得テスト"""
        # モックデータ
        search_term = "テスト"
        mock_companies = [CompanyEntity(id="1", name="テスト企業")]
        self.mock_repo.find_by_name_contains.return_value = mock_companies

        # テスト対象のメソッド呼び出し
        result = await self.service.get_companies(search=search_term)

        # 検証
        self.mock_repo.find_by_name_contains.assert_called_once_with(search_term, 50)
        assert result == mock_companies
        assert len(result) == 1

    async def test_get_companies_with_filters(self):
        """フィルタ条件での企業一覧取得テスト"""
        # モックデータ
        filters = {"industry": "テクノロジー"}
        mock_companies = [
            CompanyEntity(id="1", name="企業A", industry="テクノロジー"),
            CompanyEntity(id="2", name="企業B", industry="テクノロジー")
        ]
        self.mock_repo.find_by_criteria.return_value = mock_companies

        # テスト対象のメソッド呼び出し
        result = await self.service.get_companies(filters=filters)

        # 検証
        self.mock_repo.find_by_criteria.assert_called_once_with(filters, limit=50)
        assert result == mock_companies
        assert len(result) == 2

    async def test_get_company_by_id_existing(self):
        """既存企業IDでの取得テスト"""
        # モックデータ
        company_id = "test123"
        mock_company = CompanyEntity(
            id=company_id,
            name="テスト企業",
            industry="テクノロジー"
        )
        self.mock_repo.find_by_id.return_value = mock_company

        # テスト対象のメソッド呼び出し
        result = await self.service.get_company_by_id(company_id)

        # 検証
        self.mock_repo.find_by_id.assert_called_once_with(company_id)
        assert result == mock_company
        assert result.id == company_id
        assert result.name == "テスト企業"

    async def test_get_company_by_id_nonexistent(self):
        """存在しない企業IDでの取得テスト"""
        # モックデータ
        company_id = "nonexistent"
        self.mock_repo.find_by_id.return_value = None

        # テスト対象のメソッド呼び出し
        result = await self.service.get_company_by_id(company_id)

        # 検証
        self.mock_repo.find_by_id.assert_called_once_with(company_id)
        assert result is None

    async def test_create_company(self):
        """企業作成テスト"""
        # モックデータ
        company_data = {
            "name": "新規企業",
            "industry": "テクノロジー",
            "employee_count": 50
        }

        # 作成されたエンティティのモック
        mock_company = CompanyEntity(
            id="generated_id",
            name="新規企業",
            industry="テクノロジー",
            employee_count=50,
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )
        self.mock_repo.save.return_value = mock_company

        # テスト対象のメソッド呼び出し
        result = await self.service.create_company(company_data)

        # 検証
        self.mock_repo.save.assert_called_once()
        assert result == mock_company
        assert result.name == "新規企業"
        assert result.industry == "テクノロジー"
        assert result.employee_count == 50

    async def test_update_company(self):
        """企業更新テスト"""
        # モックデータ
        company_id = "test123"
        update_data = {
            "name": "更新企業",
            "employee_count": 100
        }

        # 更新されたエンティティのモック
        mock_updated_company = CompanyEntity(
            id=company_id,
            name="更新企業",
            employee_count=100,
            updated_at=datetime.datetime.now()
        )
        self.mock_repo.update.return_value = mock_updated_company

        # テスト対象のメソッド呼び出し
        result = await self.service.update_company(company_id, update_data)

        # 検証
        self.mock_repo.update.assert_called_once_with(company_id, update_data)
        assert result == mock_updated_company
        assert result.name == "更新企業"
        assert result.employee_count == 100

    async def test_delete_company(self):
        """企業削除テスト"""
        # モックデータ
        company_id = "test123"
        self.mock_repo.delete.return_value = True

        # テスト対象のメソッド呼び出し
        result = await self.service.delete_company(company_id)

        # 検証
        self.mock_repo.delete.assert_called_once_with(company_id)
        assert result is True