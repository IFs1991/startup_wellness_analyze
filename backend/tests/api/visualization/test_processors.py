"""
可視化プロセッサのテスト

このモジュールでは、以下のプロセッサのテストを行います：
- Association Visualization Processor
- Correlation Visualization Processor
- Descriptive Stats Visualization Processor
- Predictive Model Visualization Processor
- Survival Analysis Visualization Processor
- Time Series Visualization Processor
"""

import pytest
import json
import numpy as np
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from api.visualization.factory import VisualizationProcessorFactory
from api.visualization.association_processor import AssociationVisualizationProcessor
from api.visualization.correlation_processor import CorrelationVisualizationProcessor
from api.visualization.descriptive_stats_processor import DescriptiveStatsVisualizationProcessor
from api.visualization.predictive_model_processor import PredictiveModelVisualizationProcessor
from api.visualization.survival_analysis_processor import SurvivalAnalysisVisualizationProcessor
from api.visualization.timeseries_processor import TimeSeriesVisualizationProcessor

# --- ファクトリーテスト ---

def test_factory_get_processor():
    """ファクトリーが正しいプロセッサを返すことをテスト"""
    # 各種プロセッサの取得テスト
    association_processor = VisualizationProcessorFactory.get_processor("association")
    assert isinstance(association_processor, AssociationVisualizationProcessor)

    correlation_processor = VisualizationProcessorFactory.get_processor("correlation")
    assert isinstance(correlation_processor, CorrelationVisualizationProcessor)

    descriptive_stats_processor = VisualizationProcessorFactory.get_processor("descriptive_stats")
    assert isinstance(descriptive_stats_processor, DescriptiveStatsVisualizationProcessor)

    predictive_model_processor = VisualizationProcessorFactory.get_processor("predictive_model")
    assert isinstance(predictive_model_processor, PredictiveModelVisualizationProcessor)

    survival_analysis_processor = VisualizationProcessorFactory.get_processor("survival_analysis")
    assert isinstance(survival_analysis_processor, SurvivalAnalysisVisualizationProcessor)

    timeseries_processor = VisualizationProcessorFactory.get_processor("timeseries")
    assert isinstance(timeseries_processor, TimeSeriesVisualizationProcessor)

def test_factory_unknown_processor():
    """未知の分析タイプに対してNoneを返すことをテスト"""
    unknown_processor = VisualizationProcessorFactory.get_processor("unknown_type")
    assert unknown_processor is None

# --- AssociationVisualizationProcessorテスト ---

def test_association_processor_network():
    """アソシエーション分析の可視化：ネットワーク図"""
    processor = VisualizationProcessorFactory.get_processor("association")

    # テスト用データ
    test_data = {
        "rules": [
            {"antecedent": ["A"], "consequent": ["B"], "confidence": 0.8, "support": 0.5, "lift": 2.1},
            {"antecedent": ["B"], "consequent": ["C"], "confidence": 0.7, "support": 0.4, "lift": 1.9},
            {"antecedent": ["A", "B"], "consequent": ["D"], "confidence": 0.6, "support": 0.3, "lift": 1.7}
        ]
    }

    # ネットワーク図用のデータを準備
    result = processor.prepare_chart_data(test_data, "network", {})

    # 基本的な検証
    assert "config" in result
    assert "data" in result
    assert result["config"]["chart_type"] == "network"
    assert "nodes" in result["data"]
    assert "edges" in result["data"]
    assert len(result["data"]["nodes"]) > 0
    assert len(result["data"]["edges"]) > 0

def test_association_processor_heatmap():
    """アソシエーション分析の可視化：ヒートマップ"""
    processor = VisualizationProcessorFactory.get_processor("association")

    # テスト用データ
    test_data = {
        "rules": [
            {"antecedent": ["A"], "consequent": ["B"], "confidence": 0.8, "support": 0.5, "lift": 2.1},
            {"antecedent": ["B"], "consequent": ["C"], "confidence": 0.7, "support": 0.4, "lift": 1.9},
            {"antecedent": ["A", "B"], "consequent": ["D"], "confidence": 0.6, "support": 0.3, "lift": 1.7}
        ]
    }

    # ヒートマップ用のデータを準備
    result = processor.prepare_chart_data(test_data, "heatmap", {})

    # 基本的な検証
    assert "config" in result
    assert "data" in result
    assert result["config"]["chart_type"] == "heatmap"
    assert "matrix" in result["data"]
    assert "x_labels" in result["data"]
    assert "y_labels" in result["data"]

def test_association_processor_format_summary():
    """アソシエーション分析の要約フォーマットをテスト"""
    processor = VisualizationProcessorFactory.get_processor("association")

    # テスト用データ
    test_data = {
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
    }

    # サマリーを生成
    summary = processor.format_summary(test_data)

    # 基本的な検証
    assert "rule_count" in summary
    assert summary["rule_count"] == 3
    assert "top_rules" in summary
    assert len(summary["top_rules"]) > 0

# --- CorrelationVisualizationProcessorテスト ---

def test_correlation_processor_heatmap():
    """相関分析の可視化：ヒートマップ"""
    processor = VisualizationProcessorFactory.get_processor("correlation")

    # テスト用データ
    test_data = {
        "correlation_matrix": [
            [1.0, 0.7, -0.5],
            [0.7, 1.0, 0.2],
            [-0.5, 0.2, 1.0]
        ],
        "variables": ["A", "B", "C"]
    }

    # ヒートマップ用のデータを準備
    result = processor.prepare_chart_data(test_data, "heatmap", {})

    # 基本的な検証
    assert "config" in result
    assert "data" in result
    assert result["config"]["chart_type"] == "heatmap"
    assert "matrix" in result["data"]
    assert len(result["data"]["matrix"]) == 3
    assert len(result["data"]["matrix"][0]) == 3
    assert "x_labels" in result["data"]
    assert "y_labels" in result["data"]

def test_correlation_processor_network():
    """相関分析の可視化：ネットワーク図"""
    processor = VisualizationProcessorFactory.get_processor("correlation")

    # テスト用データ
    test_data = {
        "correlation_matrix": [
            [1.0, 0.7, -0.5],
            [0.7, 1.0, 0.2],
            [-0.5, 0.2, 1.0]
        ],
        "variables": ["A", "B", "C"]
    }

    # ネットワーク図用のデータを準備
    result = processor.prepare_chart_data(test_data, "network", {})

    # 基本的な検証
    assert "config" in result
    assert "data" in result
    assert result["config"]["chart_type"] == "network"
    assert "nodes" in result["data"]
    assert "edges" in result["data"]
    assert len(result["data"]["nodes"]) == 3

# --- DescriptiveStatsVisualizationProcessorテスト ---

def test_descriptive_stats_processor_histogram():
    """記述統計の可視化：ヒストグラム"""
    processor = VisualizationProcessorFactory.get_processor("descriptive_stats")

    # テスト用データ
    test_data = {
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
                    # 100個のランダムなデータ
                    *np.random.normal(50, 10, 100).tolist()
                ]
            }
        }
    }

    # ヒストグラム用のデータを準備
    result = processor.prepare_chart_data(test_data, "histogram", {"variable": "A"})

    # 基本的な検証
    assert "config" in result
    assert "data" in result
    assert result["config"]["chart_type"] == "bar"
    assert "labels" in result["data"]
    assert "datasets" in result["data"]
    assert len(result["data"]["datasets"]) > 0
    assert len(result["data"]["labels"]) > 0

def test_descriptive_stats_processor_boxplot():
    """記述統計の可視化：箱ひげ図"""
    processor = VisualizationProcessorFactory.get_processor("descriptive_stats")

    # テスト用データ
    test_data = {
        "stats": {
            "A": {
                "count": 100,
                "mean": 50.0,
                "std": 10.0,
                "min": 25.0,
                "q1": 45.0,
                "median": 50.0,
                "q3": 55.0,
                "max": 75.0
            },
            "B": {
                "count": 100,
                "mean": 40.0,
                "std": 5.0,
                "min": 30.0,
                "q1": 35.0,
                "median": 40.0,
                "q3": 45.0,
                "max": 50.0
            }
        }
    }

    # 箱ひげ図用のデータを準備
    result = processor.prepare_chart_data(test_data, "boxplot", {"variables": ["A", "B"]})

    # 基本的な検証
    assert "config" in result
    assert "data" in result
    assert result["config"]["chart_type"] == "boxplot"
    assert "datasets" in result["data"]
    assert len(result["data"]["datasets"]) == 2

def test_descriptive_stats_processor_format_summary():
    """記述統計の要約フォーマットをテスト"""
    processor = VisualizationProcessorFactory.get_processor("descriptive_stats")

    # テスト用データ
    test_data = {
        "stats": {
            "A": {
                "count": 100,
                "mean": 50.0,
                "std": 10.0,
                "min": 25.0,
                "q1": 45.0,
                "median": 50.0,
                "q3": 55.0,
                "max": 75.0
            },
            "B": {
                "count": 100,
                "mean": 40.0,
                "std": 5.0,
                "min": 30.0,
                "q1": 35.0,
                "median": 40.0,
                "q3": 45.0,
                "max": 50.0
            }
        },
        "category_counts": {
            "C": {
                "Value1": 30,
                "Value2": 40,
                "Value3": 30
            }
        }
    }

    # サマリーを生成
    summary = processor.format_summary(test_data)

    # 基本的な検証
    assert "numeric_variables" in summary
    assert "categorical_variables" in summary
    assert "total_variables" in summary
    assert summary["total_variables"] == 3

# --- PredictiveModelVisualizationProcessorテスト ---

def test_predictive_model_processor_feature_importance():
    """予測モデルの可視化：特徴量重要度"""
    processor = VisualizationProcessorFactory.get_processor("predictive_model")

    # テスト用データ
    test_data = {
        "feature_importance": {
            "features": ["feature1", "feature2", "feature3", "feature4"],
            "importance": [0.4, 0.3, 0.2, 0.1]
        },
        "model_info": {
            "model_type": "RandomForest",
            "parameters": {"n_estimators": 100, "max_depth": 5}
        }
    }

    # 特徴量重要度用のデータを準備
    result = processor.prepare_chart_data(test_data, "feature_importance", {})

    # 基本的な検証
    assert "config" in result
    assert "data" in result
    assert result["config"]["chart_type"] == "bar"
    assert "labels" in result["data"]
    assert "datasets" in result["data"]
    assert len(result["data"]["labels"]) == 4
    assert len(result["data"]["datasets"]) == 1

def test_predictive_model_processor_roc_curve():
    """予測モデルの可視化：ROC曲線"""
    processor = VisualizationProcessorFactory.get_processor("predictive_model")

    # テスト用データ
    test_data = {
        "roc_curve": {
            "fpr": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            "tpr": [0, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0],
            "auc": 0.75
        },
        "model_info": {
            "model_type": "LogisticRegression"
        }
    }

    # ROC曲線用のデータを準備
    result = processor.prepare_chart_data(test_data, "roc_curve", {})

    # 基本的な検証
    assert "config" in result
    assert "data" in result
    assert result["config"]["chart_type"] == "scatter"
    assert "datasets" in result["data"]
    assert len(result["data"]["datasets"]) == 2  # ROC曲線と対角線

def test_predictive_model_processor_format_summary():
    """予測モデルの要約フォーマットをテスト"""
    processor = VisualizationProcessorFactory.get_processor("predictive_model")

    # テスト用データ
    test_data = {
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
    }

    # サマリーを生成
    summary = processor.format_summary(test_data)

    # 基本的な検証
    assert "model_type" in summary
    assert "classification_metrics" in summary
    assert "top_features" in summary
    assert len(summary["top_features"]) > 0
    assert summary["classification_metrics"]["accuracy"] == 0.85

# --- SurvivalAnalysisVisualizationProcessorテスト ---

def test_survival_analysis_processor_survival_curve():
    """生存分析の可視化：生存曲線"""
    processor = VisualizationProcessorFactory.get_processor("survival_analysis")

    # テスト用データ
    test_data = {
        "survival_curve": {
            "time": [0, 10, 20, 30, 40, 50],
            "survival": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            "lower_ci": [1.0, 0.85, 0.75, 0.65, 0.55, 0.45],
            "upper_ci": [1.0, 0.95, 0.85, 0.75, 0.65, 0.55]
        },
        "model_info": {
            "model_type": "kaplan_meier"
        }
    }

    # 生存曲線用のデータを準備
    result = processor.prepare_chart_data(test_data, "survival_curve", {})

    # 基本的な検証
    assert "config" in result
    assert "data" in result
    assert result["config"]["chart_type"] == "line"
    assert "datasets" in result["data"]
    assert len(result["data"]["datasets"]) > 0

def test_survival_analysis_processor_format_summary():
    """生存分析の要約フォーマットをテスト"""
    processor = VisualizationProcessorFactory.get_processor("survival_analysis")

    # テスト用データ
    test_data = {
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
    }

    # サマリーを生成
    summary = processor.format_summary(test_data)

    # 基本的な検証
    assert "model_type" in summary
    assert "sample_info" in summary
    assert "survival_metrics" in summary
    assert summary["sample_info"]["total_samples"] == 100
    assert summary["sample_info"]["events"] == 40
    assert summary["sample_info"]["censored"] == 60

# --- TimeSeriesVisualizationProcessorテスト ---

def test_timeseries_processor_line():
    """時系列分析の可視化：ライングラフ"""
    processor = VisualizationProcessorFactory.get_processor("timeseries")

    # テスト用日付データ
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30, 0, -1)]
    future_dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 11)]

    # テスト用データ
    test_data = {
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
        }
    }

    # ライングラフ用のデータを準備
    result = processor.prepare_chart_data(test_data, "line", {})

    # 基本的な検証
    assert "config" in result
    assert "data" in result
    assert result["config"]["chart_type"] == "line"
    assert "datasets" in result["data"]
    assert len(result["data"]["datasets"]) >= 3  # 実測値、フィッティング、予測

def test_timeseries_processor_format_summary():
    """時系列分析の要約フォーマットをテスト"""
    processor = VisualizationProcessorFactory.get_processor("timeseries")

    # テスト用日付データ
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30, 0, -1)]
    future_dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 11)]

    # テスト用データ
    test_data = {
        "original_data": {
            "dates": dates,
            "values": [100 + i + np.random.normal(0, 5) for i in range(30)]
        },
        "forecast_values": {
            "dates": future_dates,
            "values": [130 + i for i in range(10)]
        },
        "metadata": {
            "target_variable": "Sales"
        },
        "model_info": {
            "model_type": "ARIMA",
            "aic": 234.5,
            "bic": 245.6
        }
    }

    # サマリーを生成
    summary = processor.format_summary(test_data)

    # 基本的な検証
    assert "model_type" in summary
    assert "target_variable" in summary
    assert "model_summary" in summary
    assert "forecast_info" in summary
    assert "data_info" in summary
    assert summary["target_variable"] == "Sales"
    assert summary["model_type"] == "ARIMA"