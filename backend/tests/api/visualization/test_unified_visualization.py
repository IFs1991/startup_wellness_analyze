"""
統一可視化エンドポイントの統合テスト

このモジュールでは、統一された可視化エンドポイントを通じて
各種可視化プロセッサが正しく連携・動作することを確認します。
"""

import pytest
import json
import numpy as np
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

# エンドポイント定義
VISUALIZATION_ENDPOINT = "/api/visualizations/visualize"

# 各分析タイプのテストデータ生成関数
def get_association_test_data():
    """アソシエーション分析のテストデータを生成"""
    return {
        "analysis_type": "association",
        "visualization_type": "network",
        "analysis_results": {
            "rules": [
                {"antecedent": ["A"], "consequent": ["B"], "confidence": 0.8, "support": 0.5, "lift": 2.1},
                {"antecedent": ["B"], "consequent": ["C"], "confidence": 0.7, "support": 0.4, "lift": 1.9},
                {"antecedent": ["A", "B"], "consequent": ["D"], "confidence": 0.6, "support": 0.3, "lift": 1.7}
            ],
            "metrics": {
                "total_rules": 3,
                "min_support": 0.3,
                "min_confidence": 0.6
            }
        },
        "options": {
            "format": "png",
            "title": "アソシエーションテスト",
            "width": 800,
            "height": 600
        }
    }

def get_correlation_test_data():
    """相関分析のテストデータを生成"""
    return {
        "analysis_type": "correlation",
        "visualization_type": "heatmap",
        "analysis_results": {
            "correlation_matrix": [
                [1.0, 0.7, -0.5],
                [0.7, 1.0, 0.2],
                [-0.5, 0.2, 1.0]
            ],
            "variables": ["A", "B", "C"],
            "metadata": {
                "dataset_name": "テストデータセット",
                "sample_size": 100
            }
        },
        "options": {
            "format": "png",
            "title": "相関分析テスト",
            "colormap": "viridis"
        }
    }

def get_descriptive_stats_test_data():
    """記述統計のテストデータを生成"""
    return {
        "analysis_type": "descriptive_stats",
        "visualization_type": "histogram",
        "analysis_results": {
            "stats": {
                "A": {
                    "count": 100,
                    "mean": 50.0,
                    "std": 10.0,
                    "min": 25.0,
                    "q1": 45.0,
                    "median": 50.0,
                    "q3": 55.0,
                    "max": 75.0,
                    "data": [
                        *np.random.normal(50, 10, 100).tolist()
                    ]
                }
            }
        },
        "options": {
            "variable": "A",
            "bins": 10,
            "format": "png",
            "title": "記述統計テスト"
        }
    }

def get_predictive_model_test_data():
    """予測モデルのテストデータを生成"""
    return {
        "analysis_type": "predictive_model",
        "visualization_type": "feature_importance",
        "analysis_results": {
            "model_info": {
                "model_type": "RandomForest"
            },
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.8,
                "recall": 0.75,
                "f1_score": 0.77,
                "auc": 0.9
            },
            "feature_importance": {
                "features": ["feature1", "feature2", "feature3", "feature4"],
                "importance": [0.4, 0.3, 0.2, 0.1]
            }
        },
        "options": {
            "format": "png",
            "title": "特徴量重要度",
            "sort_order": "descending"
        }
    }

def get_survival_analysis_test_data():
    """生存分析のテストデータを生成"""
    return {
        "analysis_type": "survival_analysis",
        "visualization_type": "survival_curve",
        "analysis_results": {
            "survival_curve": {
                "time": [0, 10, 20, 30, 40, 50],
                "survival": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                "lower_ci": [1.0, 0.85, 0.75, 0.65, 0.55, 0.45],
                "upper_ci": [1.0, 0.95, 0.85, 0.75, 0.65, 0.55]
            },
            "model_info": {
                "model_type": "kaplan_meier"
            },
            "samples": {
                "total": 100,
                "events": 40,
                "censored": 60
            },
            "survival_metrics": {
                "median_survival": 45.2
            }
        },
        "options": {
            "format": "png",
            "title": "生存曲線",
            "show_confidence_interval": True
        }
    }

def get_timeseries_test_data():
    """時系列分析のテストデータを生成"""
    # テスト用日付データ
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30, 0, -1)]
    future_dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 11)]

    return {
        "analysis_type": "timeseries",
        "visualization_type": "line",
        "analysis_results": {
            "original_data": {
                "dates": dates,
                "values": [100 + i + np.random.normal(0, 5) for i in range(30)]
            },
            "fitted_values": {
                "dates": dates,
                "values": [100 + i for i in range(30)]
            },
            "forecast_values": {
                "dates": future_dates,
                "values": [130 + i for i in range(10)],
                "lower_ci": [125 + i for i in range(10)],
                "upper_ci": [135 + i for i in range(10)]
            },
            "metadata": {
                "target_variable": "Sales"
            },
            "model_info": {
                "model_type": "ARIMA",
                "aic": 234.5,
                "bic": 245.6
            }
        },
        "options": {
            "format": "png",
            "title": "時系列予測",
            "show_confidence_interval": True
        }
    }

# 統合テスト
@pytest.mark.asyncio
async def test_unified_visualization_association(client, mocked_visualization_service):
    """アソシエーション分析の統一可視化エンドポイントテスト"""
    # テストデータ
    test_data = get_association_test_data()

    # リクエスト実行
    response = client.post(VISUALIZATION_ENDPOINT, json=test_data)

    # 検証
    assert response.status_code == 200
    assert response.json()["chart_id"] is not None
    assert response.json()["url"] is not None
    assert response.json()["format"] == "png"
    assert "analysis_summary" in response.json()
    assert "rule_count" in response.json()["analysis_summary"]

@pytest.mark.asyncio
async def test_unified_visualization_correlation(client, mocked_visualization_service):
    """相関分析の統一可視化エンドポイントテスト"""
    # テストデータ
    test_data = get_correlation_test_data()

    # リクエスト実行
    response = client.post(VISUALIZATION_ENDPOINT, json=test_data)

    # 検証
    assert response.status_code == 200
    assert response.json()["chart_id"] is not None
    assert response.json()["url"] is not None
    assert "analysis_summary" in response.json()
    assert "variable_count" in response.json()["analysis_summary"]

@pytest.mark.asyncio
async def test_unified_visualization_descriptive_stats(client, mocked_visualization_service):
    """記述統計の統一可視化エンドポイントテスト"""
    # テストデータ
    test_data = get_descriptive_stats_test_data()

    # リクエスト実行
    response = client.post(VISUALIZATION_ENDPOINT, json=test_data)

    # 検証
    assert response.status_code == 200
    assert response.json()["chart_id"] is not None
    assert response.json()["url"] is not None
    assert "analysis_summary" in response.json()
    assert "numeric_variables" in response.json()["analysis_summary"]

@pytest.mark.asyncio
async def test_unified_visualization_predictive_model(client, mocked_visualization_service):
    """予測モデルの統一可視化エンドポイントテスト"""
    # テストデータ
    test_data = get_predictive_model_test_data()

    # リクエスト実行
    response = client.post(VISUALIZATION_ENDPOINT, json=test_data)

    # 検証
    assert response.status_code == 200
    assert response.json()["chart_id"] is not None
    assert response.json()["url"] is not None
    assert "analysis_summary" in response.json()
    assert "model_type" in response.json()["analysis_summary"]
    assert "top_features" in response.json()["analysis_summary"]

@pytest.mark.asyncio
async def test_unified_visualization_survival_analysis(client, mocked_visualization_service):
    """生存分析の統一可視化エンドポイントテスト"""
    # テストデータ
    test_data = get_survival_analysis_test_data()

    # リクエスト実行
    response = client.post(VISUALIZATION_ENDPOINT, json=test_data)

    # 検証
    assert response.status_code == 200
    assert response.json()["chart_id"] is not None
    assert response.json()["url"] is not None
    assert "analysis_summary" in response.json()
    assert "model_type" in response.json()["analysis_summary"]
    assert "sample_info" in response.json()["analysis_summary"]

@pytest.mark.asyncio
async def test_unified_visualization_timeseries(client, mocked_visualization_service):
    """時系列分析の統一可視化エンドポイントテスト"""
    # テストデータ
    test_data = get_timeseries_test_data()

    # リクエスト実行
    response = client.post(VISUALIZATION_ENDPOINT, json=test_data)

    # 検証
    assert response.status_code == 200
    assert response.json()["chart_id"] is not None
    assert response.json()["url"] is not None
    assert "analysis_summary" in response.json()
    assert "model_type" in response.json()["analysis_summary"]
    assert "forecast_info" in response.json()["analysis_summary"]

@pytest.mark.asyncio
async def test_unified_visualization_invalid_type(client, mocked_visualization_service):
    """無効な分析タイプの統一可視化エンドポイントテスト"""
    # テストデータ
    test_data = {
        "analysis_type": "invalid_type",
        "visualization_type": "chart",
        "analysis_results": {},
        "options": {}
    }

    # リクエスト実行
    response = client.post(VISUALIZATION_ENDPOINT, json=test_data)

    # 検証：エラーになることを確認
    assert response.status_code == 400

@pytest.mark.asyncio
async def test_unified_visualization_missing_data(client, mocked_visualization_service):
    """必要なデータが欠けている場合の統一可視化エンドポイントテスト"""
    # テストデータ（必要なフィールドが欠けている）
    test_data = {
        "analysis_type": "correlation",
        "visualization_type": "heatmap",
        "analysis_results": {}  # 必要なデータがない
    }

    # リクエスト実行
    response = client.post(VISUALIZATION_ENDPOINT, json=test_data)

    # 検証：エラーになることを確認
    assert response.status_code in [400, 500]

# テスト用フィクスチャ
@pytest.fixture
def mocked_visualization_service(monkeypatch):
    """可視化サービスのモック"""
    # モック実装
    async def mock_generate_chart(*args, **kwargs):
        return {
            "chart_id": "test-chart-id",
            "url": "https://example.com/charts/test-chart-id.png",
            "format": kwargs.get("format", "png"),
            "thumbnail_url": "https://example.com/charts/thumbnails/test-chart-id.png",
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "chart_type": kwargs.get("config", {}).get("chart_type", "unknown")
            }
        }

    # モックの設定
    if hasattr(monkeypatch, "setattr"):
        # visualization_serviceのgenerate_chartメソッドをモック
        from api.services.visualization import VisualizationService
        monkeypatch.setattr(VisualizationService, "generate_chart", mock_generate_chart)

    return mock_generate_chart