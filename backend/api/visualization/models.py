"""
共通可視化モデル定義

このモジュールでは、可視化機能で使用される共通のリクエスト/レスポンスモデルを定義します。
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class BaseVisualizationRequest(BaseModel):
    """基本可視化リクエストモデル"""
    analysis_results: Dict[str, Any] = Field(..., description="分析結果データ")
    visualization_type: str = Field(..., description="可視化タイプ")
    options: Dict[str, Any] = Field(default_factory=dict, description="可視化設定オプション")


class BaseVisualizationResponse(BaseModel):
    """基本可視化レスポンスモデル"""
    chart_id: str = Field(..., description="チャートID")
    url: str = Field(..., description="チャートURL")
    format: str = Field(..., description="フォーマット")
    thumbnail_url: Optional[str] = Field(None, description="サムネイルURL")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="メタデータ")
    analysis_summary: Dict[str, Any] = Field(default_factory=dict, description="分析サマリー")


# 分析タイプ別の具体的リクエストモデル
class AssociationVisualizationRequest(BaseVisualizationRequest):
    """アソシエーション分析可視化リクエスト"""
    class Config:
        schema_extra = {
            "example": {
                "analysis_results": {
                    "rules": [
                        {"antecedents": ["A"], "consequents": ["B"], "support": 0.5, "confidence": 0.8, "lift": 1.2}
                    ]
                },
                "visualization_type": "network",
                "options": {"min_confidence": 0.5}
            }
        }


class ClusterVisualizationRequest(BaseVisualizationRequest):
    """クラスター分析可視化リクエスト"""
    class Config:
        schema_extra = {
            "example": {
                "analysis_results": {
                    "cluster_centers": [[1.2, 3.4], [5.6, 7.8]],
                    "labels": [0, 1, 0, 1],
                    "data": [[1.0, 3.0], [5.0, 7.0], [1.5, 3.5], [6.0, 8.0]]
                },
                "visualization_type": "scatter",
                "options": {"show_centers": True}
            }
        }


class CorrelationVisualizationRequest(BaseVisualizationRequest):
    """相関分析可視化リクエスト"""
    class Config:
        schema_extra = {
            "example": {
                "analysis_results": {
                    "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
                    "variables": ["X", "Y"]
                },
                "visualization_type": "heatmap",
                "options": {"colormap": "viridis"}
            }
        }


class TimeseriesVisualizationRequest(BaseVisualizationRequest):
    """時系列分析可視化リクエスト"""
    class Config:
        schema_extra = {
            "example": {
                "analysis_results": {
                    "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03"],
                    "values": [10, 15, 13],
                    "forecast": [14, 16]
                },
                "visualization_type": "line",
                "options": {"show_forecast": True}
            }
        }


class StatisticalAnalysisVisualizationRequest(BaseVisualizationRequest):
    """統計分析可視化リクエスト"""
    class Config:
        schema_extra = {
            "example": {
                "analysis_results": {
                    "group_data": {"Group A": [1, 2, 3], "Group B": [4, 5, 6]},
                    "p_value": 0.03
                },
                "visualization_type": "boxplot",
                "options": {"show_significance": True}
            }
        }


class PCAVisualizationRequest(BaseVisualizationRequest):
    """主成分分析可視化リクエスト"""
    class Config:
        schema_extra = {
            "example": {
                "analysis_results": {
                    "explained_variance_ratio": [0.7, 0.2, 0.1],
                    "components": [[0.5, 0.5], [0.7, -0.7]],
                    "transformed_data": [[1.0, 0.5], [0.5, 1.0]]
                },
                "visualization_type": "biplot",
                "options": {"n_components": 2}
            }
        }


class DescriptiveStatsVisualizationRequest(BaseVisualizationRequest):
    """記述統計可視化リクエスト"""
    class Config:
        schema_extra = {
            "example": {
                "analysis_results": {
                    "mean": 5.5,
                    "median": 5.0,
                    "std": 1.2,
                    "data": [4, 5, 5, 6, 7]
                },
                "visualization_type": "histogram",
                "options": {"bins": 10}
            }
        }


class PredictiveModelVisualizationRequest(BaseVisualizationRequest):
    """予測モデル可視化リクエスト"""
    class Config:
        schema_extra = {
            "example": {
                "analysis_results": {
                    "y_true": [0, 1, 0, 1],
                    "y_pred": [0.1, 0.9, 0.2, 0.8],
                    "feature_importance": {"X1": 0.7, "X2": 0.3}
                },
                "visualization_type": "roc_curve",
                "options": {"show_thresholds": True}
            }
        }