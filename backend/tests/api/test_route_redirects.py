"""
ルーターリダイレクトのテスト

routes/ディレクトリからrouters/ディレクトリへのリダイレクトが
正しく機能することを確認するテスト
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# 可視化関連のリダイレクトテスト
def test_visualization_chart_redirect():
    """可視化チャートエンドポイントのリダイレクトをテスト"""
    response = client.post(
        "/api/v1/visualizations/chart",
        json={
            "data": {"sample": "data"},
            "chart_type": "bar",
            "title": "テストチャート",
            "width": 800,
            "height": 500,
            "theme": "light",
            "use_cache": True
        },
        allow_redirects=False
    )
    assert response.status_code == 307
    assert response.headers["location"] == "/api/visualization/chart"

def test_visualization_multiple_charts_redirect():
    """複数チャートエンドポイントのリダイレクトをテスト"""
    response = client.post(
        "/api/v1/visualizations/multiple-charts",
        json={
            "chart_configs": [
                {
                    "data": {"sample": "data"},
                    "chart_type": "bar",
                    "title": "テストチャート1"
                }
            ],
            "use_cache": True
        },
        allow_redirects=False
    )
    assert response.status_code == 307
    assert response.headers["location"] == "/api/visualization/multiple-charts"

def test_visualization_dashboard_redirect():
    """ダッシュボードエンドポイントのリダイレクトをテスト"""
    response = client.post(
        "/api/v1/visualizations/dashboard",
        json={
            "dashboard_data": {"sample": "data"},
            "title": "テストダッシュボード",
            "layout": [],
            "width": 1200,
            "height": 800,
            "theme": "light"
        },
        allow_redirects=False
    )
    assert response.status_code == 307
    assert response.headers["location"] == "/api/visualization/dashboard"

# レポート関連のリダイレクトテスト
def test_report_generate_redirect():
    """レポート生成エンドポイントのリダイレクトをテスト"""
    response = client.post(
        "/api/v1/reports/generate",
        json={
            "template_id": "test_template",
            "company_data": {"company_name": "テスト株式会社"},
            "period": "2025-Q1",
            "include_sections": ["summary", "metrics"],
            "format": "pdf"
        },
        allow_redirects=False
    )
    assert response.status_code == 307
    assert response.headers["location"] == "/api/reports/generate"

def test_report_templates_redirect():
    """レポートテンプレート一覧エンドポイントのリダイレクトをテスト"""
    response = client.get(
        "/api/v1/reports/templates",
        allow_redirects=False
    )
    assert response.status_code == 307
    assert response.headers["location"] == "/api/reports/templates"

def test_report_download_redirect():
    """レポートダウンロードエンドポイントのリダイレクトをテスト"""
    response = client.get(
        "/api/v1/reports/download/test_report.pdf",
        allow_redirects=False
    )
    assert response.status_code == 307
    assert response.headers["location"] == "/api/reports/download/test_report.pdf"

# クエリパラメータを含むリダイレクトテスト
def test_redirect_with_query_params():
    """クエリパラメータを含むURLのリダイレクトをテスト"""
    response = client.get(
        "/api/v1/visualizations/chart/status/abc123?format=png&width=800",
        allow_redirects=False
    )
    assert response.status_code == 307
    assert response.headers["location"] == "/api/visualization/chart/status/abc123?format=png&width=800"

# 存在しないパスのリダイレクトテスト（キャッチオールルート）
def test_non_existent_path_redirect():
    """存在しないパスのリダイレクトをテスト（キャッチオールルート）"""
    response = client.get(
        "/api/v1/reports/non-existent-path",
        allow_redirects=False
    )
    assert response.status_code == 307
    assert response.headers["location"] == "/api/reports/non-existent-path"