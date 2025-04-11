"""
Gemini可視化APIエンドポイントのテスト
"""
import pytest
import json
from fastapi.testclient import TestClient

# テスト対象のルート
GEMINI_VISUALIZATION_ENDPOINT = "/visualization/gemini"

# テスト用のデータ
CHART_DATA = {
    "data": {
        "labels": ["A", "B", "C", "D"],
        "datasets": [
            {
                "label": "Dataset 1",
                "data": [10, 20, 30, 40]
            }
        ]
    },
    "chart_type": "bar",
    "title": "Test Chart",
    "description": "Test Description",
    "width": 800,
    "height": 500,
    "theme": "light",
    "use_cache": False
}

DASHBOARD_DATA = {
    "dashboard_data": {
        "charts": [
            {
                "title": "Chart 1",
                "data": {
                    "labels": ["A", "B", "C"],
                    "datasets": [{"label": "Dataset", "data": [10, 20, 30]}]
                },
                "chart_type": "bar"
            }
        ]
    },
    "title": "Test Dashboard",
    "width": 1200,
    "height": 800,
    "theme": "professional"
}

@pytest.mark.asyncio
async def test_gemini_generate_chart(client, mock_chart_generator):
    """Geminiチャート生成エンドポイントのテスト"""
    response = client.post(f"{GEMINI_VISUALIZATION_ENDPOINT}/chart", json=CHART_DATA)

    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "image_data" in response.json()
    assert response.json()["format"] == "png"

@pytest.mark.asyncio
async def test_gemini_generate_multiple_charts(client, mock_chart_generator):
    """Gemini複数チャート生成エンドポイントのテスト"""
    multiple_charts_data = {
        "chart_configs": [CHART_DATA, CHART_DATA],
        "use_cache": False
    }

    response = client.post(f"{GEMINI_VISUALIZATION_ENDPOINT}/multiple-charts", json=multiple_charts_data)

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0
    assert response.json()[0]["success"] == True

@pytest.mark.asyncio
async def test_gemini_generate_dashboard(client, mock_chart_generator):
    """Geminiダッシュボード生成エンドポイントのテスト"""
    response = client.post(f"{GEMINI_VISUALIZATION_ENDPOINT}/dashboard", json=DASHBOARD_DATA)

    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "html" in response.json()

@pytest.mark.asyncio
async def test_gemini_generate_chart_background(client, mock_chart_generator):
    """Geminiバックグラウンドチャート生成エンドポイントのテスト"""
    response = client.post(f"{GEMINI_VISUALIZATION_ENDPOINT}/chart/background", json=CHART_DATA)

    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "cache_key" in response.json()

@pytest.mark.asyncio
async def test_gemini_get_chart_status(client, mock_chart_generator):
    """Geminiチャートステータス取得エンドポイントのテスト"""
    # まずバックグラウンド処理を開始
    response = client.post(f"{GEMINI_VISUALIZATION_ENDPOINT}/chart/background", json=CHART_DATA)
    cache_key = response.json()["cache_key"]

    # ステータスをチェック
    response = client.get(f"{GEMINI_VISUALIZATION_ENDPOINT}/chart/status/{cache_key}")

    # モックの場合は常に成功するはず
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_gemini_error_handling(client, monkeypatch, mock_chart_generator):
    """Geminiエラーハンドリングのテスト"""
    # 不正なデータでリクエスト
    invalid_data = {
        "data": {},  # 必要なデータがない
        "chart_type": "invalid_type",
        "title": "Error Test"
    }

    response = client.post(f"{GEMINI_VISUALIZATION_ENDPOINT}/chart", json=invalid_data)

    # バリデーションエラーか、生成エラーのどちらかになるはず
    assert response.status_code in [400, 422, 500]

@pytest.mark.asyncio
async def test_gemini_api_compatibility(client, mock_chart_generator):
    """Gemini API互換性のテスト - routes版とrouters版の等価性確認"""
    # Gemini可視化エンドポイントにリクエスト
    gemini_response = client.post(f"{GEMINI_VISUALIZATION_ENDPOINT}/chart", json=CHART_DATA)

    # 標準可視化エンドポイントにも同じリクエスト
    std_response = client.post("/api/v1/visualizations/chart", json=CHART_DATA)

    # レスポンス形式の互換性をチェック
    assert "success" in gemini_response.json() and "success" in std_response.json()
    assert "image_data" in gemini_response.json() and "image_data" in std_response.json()
    assert "format" in gemini_response.json() and "format" in std_response.json()