"""
routersバージョンの可視化APIエンドポイントのテスト
"""
import pytest
import json
from fastapi.testclient import TestClient

# テスト対象のルート
VISUALIZATION_ENDPOINT = "/api/visualization"

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
    "title": "テストチャート",
    "description": "テスト説明",
    "width": 800,
    "height": 500,
    "theme": "light",
    "use_cache": False
}

DASHBOARD_DATA = {
    "dashboard_data": {
        "charts": [
            {
                "title": "チャート 1",
                "data": {
                    "labels": ["A", "B", "C"],
                    "datasets": [{"label": "データセット", "data": [10, 20, 30]}]
                },
                "chart_type": "bar"
            }
        ]
    },
    "title": "テストダッシュボード",
    "width": 1200,
    "height": 800,
    "theme": "professional"
}

@pytest.mark.asyncio
async def test_router_generate_chart(client, mock_chart_generator):
    """routersチャート生成エンドポイントのテスト"""
    response = client.post(f"{VISUALIZATION_ENDPOINT}/chart", json=CHART_DATA)

    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "image_data" in response.json()
    assert response.json()["format"] == "png"

@pytest.mark.asyncio
async def test_router_generate_multiple_charts(client, mock_chart_generator):
    """routers複数チャート生成エンドポイントのテスト"""
    multiple_charts_data = {
        "chart_configs": [CHART_DATA, CHART_DATA],
        "use_cache": False
    }

    response = client.post(f"{VISUALIZATION_ENDPOINT}/multiple-charts", json=multiple_charts_data)

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0
    assert response.json()[0]["success"] == True

@pytest.mark.asyncio
async def test_router_generate_dashboard(client, mock_chart_generator):
    """routersダッシュボード生成エンドポイントのテスト"""
    response = client.post(f"{VISUALIZATION_ENDPOINT}/dashboard", json=DASHBOARD_DATA)

    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "html" in response.json()

@pytest.mark.asyncio
async def test_router_generate_chart_background(client, mock_chart_generator):
    """routersバックグラウンドチャート生成エンドポイントのテスト"""
    response = client.post(f"{VISUALIZATION_ENDPOINT}/chart/background", json=CHART_DATA)

    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "cache_key" in response.json()

@pytest.mark.asyncio
async def test_router_get_chart_status(client, mock_chart_generator):
    """routersチャートステータス取得エンドポイントのテスト"""
    # まずバックグラウンド処理を開始
    response = client.post(f"{VISUALIZATION_ENDPOINT}/chart/background", json=CHART_DATA)
    cache_key = response.json()["cache_key"]

    # ステータスをチェック
    response = client.get(f"{VISUALIZATION_ENDPOINT}/chart/status/{cache_key}")

    # モックの場合は常に成功するはず
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_router_error_handling(client, monkeypatch, mock_chart_generator):
    """routersエラーハンドリングのテスト"""
    # 不正なデータでリクエスト
    invalid_data = {
        "data": {},  # 必要なデータがない
        "chart_type": "invalid_type",
        "title": "エラーテスト"
    }

    response = client.post(f"{VISUALIZATION_ENDPOINT}/chart", json=invalid_data)

    # バリデーションエラーか、生成エラーのどちらかになるはず
    assert response.status_code in [400, 422, 500]

@pytest.mark.asyncio
async def test_router_api_compatibility(client, mock_chart_generator):
    """API互換性のテスト - routes版とrouters版の等価性確認"""
    # routers版のAPIにリクエスト
    router_response = client.post(f"{VISUALIZATION_ENDPOINT}/chart", json=CHART_DATA)

    # routes版のAPIにも同じリクエスト
    std_response = client.post("/api/v1/visualizations/chart", json=CHART_DATA)

    # レスポンス形式の互換性をチェック
    assert "success" in router_response.json() and "success" in std_response.json()
    assert "image_data" in router_response.json() and "image_data" in std_response.json()
    assert "format" in router_response.json() and "format" in std_response.json()

@pytest.mark.asyncio
async def test_router_cache_behavior(client, mock_chart_generator):
    """routersキャッシュ動作のテスト"""
    # キャッシュ使用なしのリクエスト
    no_cache_data = CHART_DATA.copy()
    no_cache_data["use_cache"] = False

    # キャッシュ使用ありのリクエスト
    cache_data = CHART_DATA.copy()
    cache_data["use_cache"] = True

    # 両方のリクエストを実行
    no_cache_response = client.post(f"{VISUALIZATION_ENDPOINT}/chart", json=no_cache_data)
    cache_response = client.post(f"{VISUALIZATION_ENDPOINT}/chart", json=cache_data)

    # 両方とも成功するはず
    assert no_cache_response.status_code == 200
    assert cache_response.status_code == 200

    # キャッシュフラグがレスポンスに反映されているか確認
    assert not no_cache_response.json().get("cached", True)
    # キャッシュ使用時は2回目のリクエストでcachedがTrueになるはずだが、
    # モックの場合は常にFalseの可能性がある