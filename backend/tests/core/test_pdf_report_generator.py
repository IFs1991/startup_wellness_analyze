import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime
import json
import io
from pathlib import Path

from core.pdf_report_generator import (
    PDFReportService,
    PDFGenerationError,
    ReportMetadata,
    get_pdf_report_service
)

@pytest.fixture
def mock_firestore_db():
    """Firestoreデータベースのモックを提供します"""
    db = MagicMock()
    return db

@pytest.fixture
def mock_storage_client():
    """Cloud Storageクライアントのモックを提供します"""
    client = MagicMock()
    # バケットのモック
    bucket = MagicMock()
    client.bucket.return_value = bucket
    # blobのモック
    blob = MagicMock()
    bucket.blob.return_value = blob
    return client

@pytest.fixture
def sample_report_data():
    """レポート生成用のサンプルデータを提供します"""
    return {
        'company_id': 'comp1',
        'company_name': 'テスト株式会社',
        'overall_score': 78.5,
        'category_scores': {
            'employee_satisfaction': 85,
            'work_life_balance': 78,
            'team_collaboration': 82,
            'leadership_quality': 75,
            'career_growth': 76
        },
        'timestamp': datetime(2022, 5, 15),
        'industry': 'Technology',
        'stage': 'Growth',
        'recommendations': [
            '従業員満足度は高いが、リーダーシップの質とキャリア成長に改善の余地があります。',
            'ワークライフバランスの向上プログラムを検討してください。',
            'チームコラボレーションの強みを活かして新規プロジェクトを推進することを推奨します。'
        ],
        'trend_data': {
            'months': ['1月', '2月', '3月', '4月', '5月'],
            'scores': [70.2, 72.5, 74.8, 76.3, 78.5]
        }
    }

@pytest.fixture
def mock_jinja_env():
    """Jinja2環境のモックを提供します"""
    env = MagicMock()
    template = MagicMock()
    env.get_template.return_value = template
    template.render.return_value = '<html><body><h1>テストレポート</h1></body></html>'
    return env

@pytest.mark.asyncio
async def test_generate_and_save_report(mock_firestore_db, mock_storage_client, sample_report_data):
    """レポート生成と保存機能をテスト"""
    # ReportLabのキャンバスをモック
    mock_canvas = MagicMock()

    with patch('core.pdf_report_generator.canvas') as mock_canvas_module, \
         patch('core.pdf_report_generator.io.BytesIO') as mock_bytesio:

        # モックの設定
        mock_canvas_module.Canvas.return_value = mock_canvas
        mock_pdf_content = MagicMock()
        mock_bytesio.return_value = mock_pdf_content
        mock_pdf_content.getvalue.return_value = b'mock_pdf_bytes'

        # PDFReportServiceインスタンスの作成
        service = PDFReportService(
            db=mock_firestore_db,
            storage_client=mock_storage_client,
            bucket_name='test-bucket',
            templates_dir='templates'
        )

        # _generate_reportメソッドをモック
        with patch.object(service, '_generate_report', new=AsyncMock()) as mock_generate, \
             patch.object(service, '_save_to_storage', new=AsyncMock()) as mock_save_storage, \
             patch.object(service, '_save_metadata', new=AsyncMock()) as mock_save_metadata:

            # モックの戻り値を設定
            mock_generate.return_value = mock_pdf_content

            # レポート生成と保存を実行
            result = await service.generate_and_save_report(
                data=sample_report_data,
                user_id='user123',
                report_type='wellness',
                metadata={'company_id': 'comp1'}
            )

            # 結果を検証
            assert isinstance(result, ReportMetadata)
            assert result.report_type == 'wellness'
            assert result.user_id == 'user123'
            assert result.metadata == {'company_id': 'comp1'}

            # 各メソッドが正しく呼び出されたか検証
            mock_generate.assert_called_once_with(sample_report_data)
            mock_save_storage.assert_called_once()
            mock_save_metadata.assert_called_once()

@pytest.mark.asyncio
async def test_generate_html_report(mock_firestore_db, mock_storage_client, sample_report_data, mock_jinja_env):
    """HTMLレポート生成機能をテスト"""
    with patch('core.pdf_report_generator.jinja2.Environment', return_value=mock_jinja_env), \
         patch('core.pdf_report_generator.HTML') as mock_html, \
         patch('core.pdf_report_generator.CSS') as mock_css:

        # WeasyPrintのHTMLとCSSをモック
        html_instance = MagicMock()
        mock_html.return_value = html_instance
        pdf_bytes = b'mock_pdf_from_html'
        html_instance.write_pdf.return_value = pdf_bytes

        # PDFReportServiceインスタンスの作成
        service = PDFReportService(
            db=mock_firestore_db,
            storage_client=mock_storage_client,
            bucket_name='test-bucket',
            templates_dir='templates'
        )

        # _save_to_storageと_save_metadataをモック
        with patch.object(service, '_save_to_storage', new=AsyncMock()) as mock_save_storage, \
             patch.object(service, '_save_metadata', new=AsyncMock()) as mock_save_metadata, \
             patch.object(service, '_html_to_pdf', new=AsyncMock(return_value=pdf_bytes)) as mock_html_to_pdf:

            # HTMLレポート生成を実行
            result = await service.generate_html_report(
                data=sample_report_data,
                template_name='wellness_template.html',
                user_id='user123',
                report_type='web_report',
                metadata={'company_id': 'comp1'},
                css_files=['style.css']
            )

            # 結果を検証
            assert isinstance(result, ReportMetadata)
            assert result.report_type == 'web_report'
            assert result.user_id == 'user123'

            # テンプレートレンダリングが正しく呼び出されたか検証
            mock_jinja_env.get_template.assert_called_once_with('wellness_template.html')
            mock_jinja_env.get_template().render.assert_called_once_with(sample_report_data)

            # HTMLからPDFへの変換が正しく呼び出されたか検証
            mock_html_to_pdf.assert_called_once()

            # ストレージへの保存が正しく呼び出されたか検証
            mock_save_storage.assert_called_once()
            mock_save_metadata.assert_called_once()

@pytest.mark.asyncio
async def test_html_to_pdf():
    """HTML to PDF変換機能をテスト"""
    with patch('core.pdf_report_generator.HTML') as mock_html, \
         patch('core.pdf_report_generator.CSS') as mock_css:

        # WeasyPrintのモックを設定
        html_instance = MagicMock()
        mock_html.return_value = html_instance
        html_instance.write_pdf.return_value = b'pdf_content'

        # PDFReportServiceインスタンスの作成
        service = PDFReportService(
            db=MagicMock(),
            storage_client=MagicMock(),
            bucket_name='test-bucket'
        )

        # HTML to PDF変換を実行
        html_content = '<html><body><h1>テスト</h1></body></html>'
        css_files = ['style.css']

        result = await service._html_to_pdf(html_content, css_files)

        # 結果を検証
        assert result == b'pdf_content'
        mock_html.assert_called_once_with(string=html_content)
        mock_css.assert_called_once_with(filename='style.css')
        html_instance.write_pdf.assert_called_once()

@pytest.mark.asyncio
async def test_save_to_storage(mock_storage_client):
    """ストレージ保存機能をテスト"""
    # PDFReportServiceインスタンスの作成
    service = PDFReportService(
        db=MagicMock(),
        storage_client=mock_storage_client,
        bucket_name='test-bucket'
    )

    # BytesIOオブジェクトをモック
    content = io.BytesIO(b'test_content')
    storage_path = 'reports/test.pdf'

    # ストレージへの保存を実行
    await service._save_to_storage(content, storage_path)

    # 結果を検証
    mock_storage_client.bucket.assert_called_once_with('test-bucket')
    mock_storage_client.bucket().blob.assert_called_once_with(storage_path)

    # blobのuploadとcontentの扱いをチェック
    blob = mock_storage_client.bucket().blob()
    if isinstance(content, io.BytesIO):
        blob.upload_from_file.assert_called_once_with(content)
    else:
        blob.upload_from_string.assert_called_once_with(content)

@pytest.mark.asyncio
async def test_save_metadata(mock_firestore_db):
    """メタデータ保存機能をテスト"""
    # Firestoreコレクション参照をモック
    collection_ref = MagicMock()
    mock_firestore_db.collection.return_value = collection_ref

    # PDFReportServiceインスタンスの作成
    service = PDFReportService(
        db=mock_firestore_db,
        storage_client=MagicMock(),
        bucket_name='test-bucket'
    )

    # テスト用メタデータを作成
    metadata = ReportMetadata(
        report_type='wellness',
        user_id='user123',
        created_at=datetime.now(),
        storage_path='reports/test.pdf',
        filename='test.pdf',
        metadata={'company_id': 'comp1'}
    )

    # メタデータ保存を実行
    await service._save_metadata(metadata)

    # 結果を検証
    mock_firestore_db.collection.assert_called_once_with('report_metadata')
    collection_ref.add.assert_called_once()

@pytest.mark.asyncio
async def test_generate_report_error_handling(mock_firestore_db, mock_storage_client, sample_report_data):
    """レポート生成のエラーハンドリングをテスト"""
    # PDFReportServiceインスタンスの作成
    service = PDFReportService(
        db=mock_firestore_db,
        storage_client=mock_storage_client,
        bucket_name='test-bucket'
    )

    # _generate_reportメソッドでエラーを発生させる
    with patch.object(service, '_generate_report', side_effect=Exception("PDF generation error")):

        # 例外が発生することを確認
        with pytest.raises(PDFGenerationError) as excinfo:
            await service.generate_and_save_report(
                data=sample_report_data,
                user_id='user123'
            )

        # エラーメッセージを検証
        assert "Failed to generate PDF report" in str(excinfo.value)

def test_pdf_report_service_factory():
    """PDFレポートサービスのファクトリ関数をテスト"""
    with patch('core.pdf_report_generator.PDFReportService') as mock_service_class:
        # モックインスタンスを設定
        mock_instance = MagicMock()
        mock_service_class.return_value = mock_instance

        # モックオブジェクト
        mock_db = MagicMock()
        mock_storage_client = MagicMock()

        # ファクトリ関数を呼び出し
        service = get_pdf_report_service(
            db=mock_db,
            storage_client=mock_storage_client,
            bucket_name='test-bucket',
            templates_dir='templates'
        )

        # 結果を検証
        assert service == mock_instance
        mock_service_class.assert_called_once_with(
            db=mock_db,
            storage_client=mock_storage_client,
            bucket_name='test-bucket',
            templates_dir='templates'
        )