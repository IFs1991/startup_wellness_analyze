import pytest
import asyncio
from datetime import datetime
from google.cloud import firestore
from google.cloud.firestore import AsyncClient
from backend.src.database.repositories.report import ReportRepository, ReportTemplateRepository

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
async def report_template_repository(firestore_client):
    """レポートテンプレートリポジトリのインスタンスを作成"""
    return ReportTemplateRepository(firestore_client)

@pytest.fixture
async def report_repository(firestore_client):
    """レポートリポジトリのインスタンスを作成"""
    return ReportRepository(firestore_client)

@pytest.mark.asyncio
async def test_report_template_repository(report_template_repository):
    """レポートテンプレートリポジトリのテスト"""
    # テンプレートを作成
    template_data = {
        "name": "Test Template",
        "description": "Test Description",
        "format": "pdf",
        "template_content": "Test Content",
        "variables": {"key": "value"}
    }

    template = await report_template_repository.create(**template_data)
    assert template.name == "Test Template"
    assert template.format == "pdf"

    # テンプレートを取得
    retrieved_template = await report_template_repository.get_by_id(template.id)
    assert retrieved_template is not None
    assert retrieved_template.name == template.name

    # 名前で取得
    template_by_name = await report_template_repository.get_by_name("Test Template")
    assert template_by_name is not None
    assert template_by_name.id == template.id

    # フォーマットで取得
    templates = await report_template_repository.get_by_format("pdf")
    assert len(templates) == 1
    assert templates[0].id == template.id

    # テンプレートを検索
    search_results = await report_template_repository.search_templates(
        name="Test",
        format="pdf"
    )
    assert len(search_results) == 1
    assert search_results[0].id == template.id

    # テンプレートの内容を更新
    updated_template = await report_template_repository.update_template_content(
        template.id,
        content="Updated Content",
        variables={"new_key": "new_value"}
    )
    assert updated_template is not None
    assert updated_template.template_content == "Updated Content"
    assert updated_template.variables == {"new_key": "new_value"}

    # テンプレートを削除
    deleted = await report_template_repository.delete(template.id)
    assert deleted is True

    # 削除されたことを確認
    deleted_template = await report_template_repository.get_by_id(template.id)
    assert deleted_template is None

@pytest.mark.asyncio
async def test_report_repository(report_repository, report_template_repository):
    """レポートリポジトリのテスト"""
    # テンプレートを作成
    template_data = {
        "name": "Test Template",
        "description": "Test Description",
        "format": "pdf",
        "template_content": "Test Content",
        "variables": {"key": "value"}
    }
    template = await report_template_repository.create(**template_data)

    # レポートを作成
    report_data = {
        "template_id": template.id,
        "content": "Test Report Content",
        "format": "pdf",
        "report_metadata": {"status": "completed"}
    }
    report = await report_repository.create(**report_data)

    assert report.template_id == template.id
    assert report.format == "pdf"

    # レポートを取得
    retrieved_report = await report_repository.get_by_id(report.id)
    assert retrieved_report is not None
    assert retrieved_report.content == "Test Report Content"

    # テンプレート情報付きでレポートを取得
    report_with_template = await report_repository.get_with_template(report.id)
    assert report_with_template is not None
    assert report_with_template.template.name == "Test Template"

    # テンプレートIDでレポートを取得
    reports = await report_repository.get_by_template(template.id)
    assert len(reports) == 1
    assert reports[0].id == report.id

    # フォーマットでレポートを取得
    reports = await report_repository.get_reports_by_format("pdf")
    assert len(reports) == 1
    assert reports[0].id == report.id

    # レポートを検索
    search_results = await report_repository.search_reports(
        template_id=template.id,
        format="pdf",
        report_metadata={"status": "completed"}
    )
    assert len(search_results) == 1
    assert search_results[0].id == report.id

    # レポートを更新
    updated_report = await report_repository.update(
        report.id,
        content="Updated Report Content"
    )
    assert updated_report is not None
    assert updated_report.content == "Updated Report Content"

    # レポートを削除
    deleted = await report_repository.delete(report.id)
    assert deleted is True

    # 削除されたことを確認
    deleted_report = await report_repository.get_by_id(report.id)
    assert deleted_report is None

@pytest.mark.asyncio
async def test_report_repository_error_handling(report_repository):
    """レポートリポジトリのエラーハンドリングをテスト"""
    # 存在しないレポートIDでの操作
    non_existent_id = "non_existent_id"

    # 存在しないレポートの取得
    report = await report_repository.get_by_id(non_existent_id)
    assert report is None

    # 存在しないレポートのテンプレート情報付き取得
    report_with_template = await report_repository.get_with_template(non_existent_id)
    assert report_with_template is None

    # 存在しないテンプレートIDでのレポート取得
    reports = await report_repository.get_by_template(non_existent_id)
    assert len(reports) == 0

    # 存在しないレポートの更新
    updated_report = await report_repository.update(
        non_existent_id,
        content="Updated Content"
    )
    assert updated_report is None

    # 存在しないレポートの削除
    deleted = await report_repository.delete(non_existent_id)
    assert deleted is False