"""
相関分析可視化プロセッサ

このモジュールでは、相関分析の可視化を処理するプロセッサクラスを実装します。
"""

from typing import Dict, List, Any, Optional
import logging
import numpy as np
from .factory import VisualizationProcessor, register_processor

logger = logging.getLogger(__name__)


@register_processor("correlation")
class CorrelationVisualizationProcessor(VisualizationProcessor):
    """相関分析の可視化プロセッサ"""

    def prepare_chart_data(self, analysis_results: Dict[str, Any],
                          visualization_type: str,
                          options: Dict[str, Any]) -> Dict[str, Any]:
        """
        相関分析のチャートデータを準備する

        Args:
            analysis_results: 相関分析結果
            visualization_type: 可視化タイプ
            options: オプション設定

        Returns:
            チャートデータ
        """
        if visualization_type == "heatmap":
            return self._prepare_heatmap_data(analysis_results, options)
        elif visualization_type == "network":
            return self._prepare_network_data(analysis_results, options)
        elif visualization_type == "matrix":
            return self._prepare_matrix_data(analysis_results, options)
        else:
            # デフォルトはヒートマップ
            return self._prepare_heatmap_data(analysis_results, options)

    def _prepare_heatmap_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """ヒートマップ用データを準備する"""
        correlation_matrix = analysis_results.get("correlation_matrix", [])
        variables = analysis_results.get("variables", [])

        # 変数名が提供されていない場合、デフォルト名を使用
        if not variables and correlation_matrix:
            variables = [f"変数{i+1}" for i in range(len(correlation_matrix))]

        # クリッピング値を適用（オプション）
        clip_min = options.get("clip_min", -1)
        clip_max = options.get("clip_max", 1)

        if correlation_matrix:
            clipped_matrix = np.clip(correlation_matrix, clip_min, clip_max).tolist()
        else:
            clipped_matrix = correlation_matrix

        chart_data = {
            "matrix": clipped_matrix,
            "x_labels": variables,
            "y_labels": variables
        }

        chart_config = {
            "chart_type": "heatmap",
            "title": options.get("title", "相関マトリックス"),
            "width": options.get("width", 800),
            "height": options.get("height", 800),
            "color_scheme": options.get("color_scheme", "correlation"),
            "min_value": clip_min,
            "max_value": clip_max,
            "show_values": options.get("show_values", True)
        }

        return {"config": chart_config, "data": chart_data}

    def _prepare_network_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """ネットワーク図用データを準備する"""
        correlation_matrix = analysis_results.get("correlation_matrix", [])
        variables = analysis_results.get("variables", [])

        # 変数名が提供されていない場合、デフォルト名を使用
        if not variables and correlation_matrix:
            variables = [f"変数{i+1}" for i in range(len(correlation_matrix))]

        threshold = options.get("correlation_threshold", 0.5)
        nodes = [{"id": var, "label": var} for var in variables]
        links = []

        for i in range(len(correlation_matrix)):
            for j in range(i+1, len(correlation_matrix[i])):
                corr_value = correlation_matrix[i][j]
                if abs(corr_value) >= threshold:
                    links.append({
                        "source": variables[i],
                        "target": variables[j],
                        "value": abs(corr_value),
                        "is_positive": corr_value > 0
                    })

        chart_data = {
            "nodes": nodes,
            "links": links
        }

        chart_config = {
            "chart_type": "network",
            "title": options.get("title", "相関ネットワーク"),
            "width": options.get("width", 800),
            "height": options.get("height", 600),
            "positive_color": options.get("positive_color", "rgba(66, 133, 244, 0.7)"),
            "negative_color": options.get("negative_color", "rgba(219, 68, 55, 0.7)"),
            "node_size": options.get("node_size", "degree"),
            "link_width": options.get("link_width", "value")
        }

        return {"config": chart_config, "data": chart_data}

    def _prepare_matrix_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """相関係数マトリックス表用データを準備する"""
        correlation_matrix = analysis_results.get("correlation_matrix", [])
        variables = analysis_results.get("variables", [])

        # 変数名が提供されていない場合、デフォルト名を使用
        if not variables and correlation_matrix:
            variables = [f"変数{i+1}" for i in range(len(correlation_matrix))]

        chart_data = {
            "variables": variables,
            "matrix": correlation_matrix
        }

        chart_config = {
            "chart_type": "table",
            "title": options.get("title", "相関係数マトリックス"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "cell_style": "correlation",
            "show_header": True,
            "row_header": True
        }

        return {"config": chart_config, "data": chart_data}

    def format_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        相関分析結果のサマリーをフォーマットする

        Args:
            analysis_results: 相関分析結果

        Returns:
            フォーマット済みサマリー
        """
        correlation_matrix = analysis_results.get("correlation_matrix", [])
        variables = analysis_results.get("variables", [])

        # 変数名が提供されていない場合、デフォルト名を使用
        if not variables and correlation_matrix:
            variables = [f"変数{i+1}" for i in range(len(correlation_matrix))]

        # 相関マトリックスがない場合
        if not correlation_matrix or not variables:
            return {
                "message": "有効な相関データが見つかりませんでした"
            }

        # 強い相関関係を持つ変数ペアを検出
        strong_correlations = []
        threshold = 0.7  # 強い相関の閾値

        for i in range(len(variables)):
            for j in range(i+1, len(variables)):
                if i < len(correlation_matrix) and j < len(correlation_matrix[i]):
                    corr_value = correlation_matrix[i][j]
                    if abs(corr_value) >= threshold:
                        strong_correlations.append({
                            "pair": f"{variables[i]} - {variables[j]}",
                            "correlation": corr_value,
                            "type": "正の相関" if corr_value > 0 else "負の相関"
                        })

        # 平均絶対相関値
        all_corrs = []
        for i in range(len(correlation_matrix)):
            for j in range(i+1, len(correlation_matrix[i])):
                all_corrs.append(abs(correlation_matrix[i][j]))

        avg_abs_corr = sum(all_corrs) / len(all_corrs) if all_corrs else 0

        # 最大正相関と最大負相関を計算
        max_positive = max_negative = 0
        if correlation_matrix:
            max_positive = max([correlation_matrix[i][j] for i in range(len(correlation_matrix)) for j in range(i+1, len(correlation_matrix[i]))], default=0)
            max_negative = min([correlation_matrix[i][j] for i in range(len(correlation_matrix)) for j in range(i+1, len(correlation_matrix[i]))], default=0)

        return {
            "variable_count": len(variables),
            "strong_correlations": strong_correlations[:5],  # 上位5つのみ表示
            "average_absolute_correlation": avg_abs_corr,
            "max_positive_correlation": max_positive,
            "max_negative_correlation": max_negative
        }