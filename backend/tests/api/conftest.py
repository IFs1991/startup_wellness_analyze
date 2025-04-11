"""
API テスト用の共通フィクスチャー定義
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

# プロジェクトルートへのパスを追加してインポートを可能にする
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.main import app
from backend.core.config import get_settings, Settings

# テスト用の設定オーバーライド
@pytest.fixture
def test_settings():
    """テスト用の設定を返す"""
    return Settings(
        # テスト用の設定をオーバーライド
        database_url="sqlite:///./test.db",
        testing=True,
        # テスト用のGemini APIキー（モック用）
        gemini_api_key="test_gemini_api_key"
    )

# テスト用のクライアント
@pytest.fixture
def client(test_settings):
    """FastAPIのテストクライアント"""
    # 設定の依存関係をオーバーライド
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as client:
        yield client
    # テスト後にクリーンアップ
    app.dependency_overrides = {}

# レポートジェネレーターのモック
@pytest.fixture
def mock_report_generator(monkeypatch):
    """レポートジェネレーターのモック"""
    class MockReportGenerator:
        async def generate_report(self, *args, **kwargs):
            return {
                "success": True,
                "format": "pdf",
                "report_id": "test_report_123",
                "file_path": "/tmp/test_report.pdf"
            }

        async def generate_and_save_report(self, *args, **kwargs):
            return "test_report_123"

    # モックをインジェクト
    from unittest.mock import MagicMock
    mock = MagicMock(return_value=MockReportGenerator())

    # インポートパスによってモック対象を変える
    try:
        from backend.api.routers.reports import get_report_generator
        monkeypatch.setattr("backend.api.routers.reports.get_report_generator", mock)
    except ImportError:
        pass

    try:
        from backend.api.routes.reports import get_report_generator
        monkeypatch.setattr("backend.api.routes.reports.get_report_generator", mock)
    except ImportError:
        pass

    return MockReportGenerator()

# 可視化ジェネレーターのモック
@pytest.fixture
def mock_chart_generator(monkeypatch):
    """チャートジェネレーターのモック"""
    class MockChartGenerator:
        async def generate_chart(self, *args, **kwargs):
            return {
                "success": True,
                "image_data": "base64_encoded_image_data",
                "format": "png",
                "width": 800,
                "height": 600,
                "cached": False
            }

        async def generate_multiple_charts(self, *args, **kwargs):
            return [
                {
                    "success": True,
                    "image_data": "base64_encoded_image_data",
                    "format": "png",
                    "width": 800,
                    "height": 600,
                    "cached": False
                }
            ]

        async def generate_dashboard(self, *args, **kwargs):
            return {
                "success": True,
                "html": "<html><body>Dashboard</body></html>",
                "width": 1200,
                "height": 800
            }

        def _generate_cache_key(self, *args, **kwargs):
            return "test_cache_key_123"

    # モックをインジェクト
    from unittest.mock import MagicMock
    mock = MagicMock(return_value=MockChartGenerator())

    # インポートパスによってモック対象を変える
    try:
        from backend.api.routers.gemini_visualization import get_chart_generator
        monkeypatch.setattr("backend.api.routers.gemini_visualization.get_chart_generator", mock)
    except ImportError:
        pass

    try:
        from backend.api.routes.visualization import get_chart_generator
        monkeypatch.setattr("backend.api.routes.visualization.get_chart_generator", mock)
    except ImportError:
        pass

    return MockChartGenerator()