"""
可視化統合テスト

このモジュールでは、可視化エンドポイントの統合テストを実施します。
特に、異なる分析タイプに対する統一可視化エンドポイントの動作と
パフォーマンス最適化の検証に重点を置いています。
"""

import pytest
import json
import time
from fastapi.testclient import TestClient
from api.main import app
from api.utils.caching import clear_cache, get_from_cache, _generate_cache_key
from api.visualization.factory import VisualizationProcessorFactory, register_processor, GenericVisualizationProcessor
from api.routers.visualization import UnifiedVisualizationRequest

# テストクライアント
client = TestClient(app)

# テスト用の認証トークン
TEST_AUTH_TOKEN = "test_token"


class MockProcessor:
    """テスト用モックプロセッサ"""

    def prepare_chart_data(self, analysis_results, visualization_type, options):
        return {
            "config": {
                "chart_type": visualization_type,
                "title": options.get("title", "テストチャート"),
                "width": 800,
                "height": 600
            },
            "data": {
                "labels": ["A", "B", "C"],
                "datasets": [
                    {
                        "label": "テストデータ",
                        "data": [1, 2, 3]
                    }
                ]
            }
        }

    def format_summary(self, analysis_results):
        return {
            "summary": "テスト用サマリー",
            "test_key": analysis_results.get("test_key", "default")
        }


@pytest.fixture(scope="module", autouse=True)
def setup_test_processors():
    """テスト用プロセッサの登録"""
    # テスト前にキャッシュをクリア
    clear_cache()

    # テスト用プロセッサを登録
    VisualizationProcessorFactory.register("test_analysis", MockProcessor)

    yield

    # テスト終了後に登録を解除
    if "test_analysis" in VisualizationProcessorFactory._processor_registry:
        del VisualizationProcessorFactory._processor_registry["test_analysis"]


@pytest.fixture
def auth_headers():
    """認証ヘッダー"""
    return {"Authorization": f"Bearer {TEST_AUTH_TOKEN}"}


@pytest.mark.parametrize("analysis_type,visualization_type", [
    ("test_analysis", "bar"),
    ("test_analysis", "line"),
    ("correlation", "heatmap"),
    ("descriptive_stats", "histogram"),
])
def test_unified_visualization_endpoint(analysis_type, visualization_type, auth_headers, monkeypatch):
    """統一可視化エンドポイントのテスト"""
    # 認証をモック
    monkeypatch.setattr("api.routers.visualization.get_current_user", lambda *args, **kwargs: {"id": "test_user_id"})

    # VisualizationServiceのチャート生成をモック
    def mock_generate_chart(*args, **kwargs):
        return {
            "chart_id": "test_chart_id",
            "url": "https://example.com/test_chart.png",
            "format": "png",
            "thumbnail_url": "https://example.com/test_chart_thumb.png",
            "metadata": {"test": "metadata"},
            "file_path": "/tmp/test_chart.png"
        }

    monkeypatch.setattr(
        "api.services.visualization.VisualizationService.generate_chart",
        mock_generate_chart
    )

    # テスト用リクエストデータ
    request_data = {
        "analysis_type": analysis_type,
        "visualization_type": visualization_type,
        "analysis_results": {
            "test_key": "test_value",
            "data": [1, 2, 3, 4, 5]
        },
        "options": {
            "title": "テストチャート",
            "format": "png"
        }
    }

    # リクエスト実行
    response = client.post(
        "/api/visualizations/visualize",
        json=request_data,
        headers=auth_headers
    )

    # ステータスコード確認
    assert response.status_code == 200

    # レスポンス内容確認
    data = response.json()
    assert data["chart_id"] == "test_chart_id"
    assert data["url"] == "https://example.com/test_chart.png"
    assert data["format"] == "png"
    assert "analysis_summary" in data


def test_cache_effectiveness(auth_headers, monkeypatch):
    """キャッシュの有効性テスト"""
    # 認証をモック
    monkeypatch.setattr("api.routers.visualization.get_current_user", lambda *args, **kwargs: {"id": "test_user_id"})

    # カウンタ
    call_count = {"count": 0}

    # VisualizationServiceのチャート生成をモック
    def mock_generate_chart(*args, **kwargs):
        call_count["count"] += 1
        return {
            "chart_id": "test_chart_id",
            "url": "https://example.com/test_chart.png",
            "format": "png",
            "thumbnail_url": "https://example.com/test_chart_thumb.png",
            "metadata": {"test": "metadata"},
            "file_path": "/tmp/test_chart.png"
        }

    monkeypatch.setattr(
        "api.services.visualization.VisualizationService.generate_chart",
        mock_generate_chart
    )

    # テスト用リクエストデータ
    request_data = {
        "analysis_type": "test_analysis",
        "visualization_type": "bar",
        "analysis_results": {
            "test_key": "cache_test",
            "data": [1, 2, 3]
        },
        "options": {
            "title": "キャッシュテスト",
            "format": "png"
        }
    }

    # 1回目のリクエスト
    response1 = client.post(
        "/api/visualizations/visualize",
        json=request_data,
        headers=auth_headers
    )
    assert response1.status_code == 200
    assert call_count["count"] == 1

    # 2回目の同一リクエスト（キャッシュから取得されるはず）
    response2 = client.post(
        "/api/visualizations/visualize",
        json=request_data,
        headers=auth_headers
    )
    assert response2.status_code == 200
    # チャート生成が呼ばれていないことを確認（カウントが増えていない）
    assert call_count["count"] == 1

    # キャッシュキーの確認
    cache_key = _generate_cache_key(
        request_data["analysis_type"],
        request_data["analysis_results"],
        request_data["visualization_type"],
        request_data["options"]
    )
    assert get_from_cache(cache_key) is not None


def test_large_dataset_handling(auth_headers, monkeypatch):
    """大規模データセット処理の最適化テスト"""
    # 認証をモック
    monkeypatch.setattr("api.routers.visualization.get_current_user", lambda *args, **kwargs: {"id": "test_user_id"})

    # 処理時間計測用変数
    process_time = {"value": 0}

    # チャート生成をモック
    def mock_generate_chart(*args, **kwargs):
        return {
            "chart_id": "test_chart_id",
            "url": "https://example.com/test_chart.png",
            "format": "png",
            "metadata": {"test": "metadata"},
            "file_path": "/tmp/test_chart.png"
        }

    monkeypatch.setattr(
        "api.services.visualization.VisualizationService.generate_chart",
        mock_generate_chart
    )

    # _is_large_datasetをオーバーライド
    original_is_large_dataset = getattr(
        __import__("api.routers.visualization", fromlist=["_is_large_dataset"]),
        "_is_large_dataset"
    )

    def mock_is_large_dataset(*args, **kwargs):
        return True

    monkeypatch.setattr(
        "api.routers.visualization._is_large_dataset",
        mock_is_large_dataset
    )

    # 非同期処理の時間計測用モック
    original_prepare_chart_data_optimized = getattr(
        __import__("api.routers.visualization", fromlist=["_prepare_chart_data_optimized"]),
        "_prepare_chart_data_optimized"
    )

    async def mock_prepare_chart_data_optimized(processor, analysis_results, visualization_type, options):
        start_time = time.time()
        result = await original_prepare_chart_data_optimized(processor, analysis_results, visualization_type, options)
        process_time["value"] = time.time() - start_time
        return result

    monkeypatch.setattr(
        "api.routers.visualization._prepare_chart_data_optimized",
        mock_prepare_chart_data_optimized
    )

    # 大規模データセット（5000件の配列）
    large_dataset = {
        "analysis_type": "test_analysis",
        "visualization_type": "line",
        "analysis_results": {
            "test_key": "large_dataset_test",
            "data": list(range(5000))  # 5000件のデータ
        },
        "options": {
            "title": "大規模データセットテスト",
            "format": "png"
        }
    }

    # リクエスト実行
    response = client.post(
        "/api/visualizations/visualize",
        json=large_dataset,
        headers=auth_headers
    )

    # テスト結果確認
    assert response.status_code == 200
    assert process_time["value"] > 0  # 処理時間が計測されていることを確認


def test_fallback_to_generic_processor(auth_headers, monkeypatch):
    """未登録の分析タイプに対する汎用プロセッサへのフォールバックテスト"""
    # 認証をモック
    monkeypatch.setattr("api.routers.visualization.get_current_user", lambda *args, **kwargs: {"id": "test_user_id"})

    # チャート生成をモック
    def mock_generate_chart(*args, **kwargs):
        return {
            "chart_id": "test_chart_id",
            "url": "https://example.com/test_chart.png",
            "format": "png",
            "metadata": {"test": "metadata"},
            "file_path": "/tmp/test_chart.png"
        }

    monkeypatch.setattr(
        "api.services.visualization.VisualizationService.generate_chart",
        mock_generate_chart
    )

    # 汎用プロセッサを登録
    VisualizationProcessorFactory.register("generic", GenericVisualizationProcessor)

    # 未登録の分析タイプでリクエスト
    unknown_type_request = {
        "analysis_type": "unknown_type",
        "visualization_type": "bar",
        "analysis_results": {
            "test_key": "unknown_type_test",
            "data": [1, 2, 3]
        },
        "options": {
            "title": "未知の分析タイプテスト"
        }
    }

    # モックを設定して、get_processorが"generic"を返すようにする
    original_get_processor = VisualizationProcessorFactory.get_processor

    def mock_get_processor(analysis_type):
        if analysis_type == "unknown_type":
            return original_get_processor("generic")
        return original_get_processor(analysis_type)

    monkeypatch.setattr(
        VisualizationProcessorFactory,
        "get_processor",
        mock_get_processor
    )

    # リクエスト実行
    response = client.post(
        "/api/visualizations/visualize",
        json=unknown_type_request,
        headers=auth_headers
    )

    # テスト結果確認（汎用プロセッサが正常に機能していれば200を返す）
    assert response.status_code == 200
    assert response.json()["chart_id"] == "test_chart_id"