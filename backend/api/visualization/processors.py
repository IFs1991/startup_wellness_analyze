"""
可視化プロセッサの実装

このモジュールでは、各分析タイプに対応する可視化プロセッサクラスを実装します。
現在は、アソシエーション分析と相関分析の2つをサポートしています。
"""

from typing import Dict, List, Any, Optional
import logging
import numpy as np
from .factory import VisualizationProcessor, register_processor

logger = logging.getLogger(__name__)


@register_processor("association")
class AssociationVisualizationProcessor(VisualizationProcessor):
    """アソシエーション分析の可視化プロセッサ"""

    def prepare_chart_data(self, analysis_results: Dict[str, Any],
                          visualization_type: str,
                          options: Dict[str, Any]) -> Dict[str, Any]:
        """
        アソシエーション分析のチャートデータを準備する

        Args:
            analysis_results: アソシエーション分析結果
            visualization_type: 可視化タイプ
            options: オプション設定

        Returns:
            チャートデータ
        """
        rules = analysis_results.get("rules", [])
        min_confidence = options.get("min_confidence", 0.0)
        min_support = options.get("min_support", 0.0)

        # 閾値でフィルタリング
        filtered_rules = [
            rule for rule in rules
            if rule.get("confidence", 0) >= min_confidence
            and rule.get("support", 0) >= min_support
        ]

        if visualization_type == "network":
            return self._prepare_network_chart_data(filtered_rules, options)
        elif visualization_type == "heatmap":
            return self._prepare_heatmap_chart_data(filtered_rules, options)
        elif visualization_type == "bar":
            return self._prepare_bar_chart_data(filtered_rules, options)
        else:
            # デフォルトはテーブル表示
            return {
                "config": {
                    "chart_type": "table",
                    "title": options.get("title", "アソシエーションルール一覧"),
                    "width": options.get("width", 800),
                    "height": options.get("height", 500),
                    "columns": [
                        {"field": "antecedents", "headerName": "条件部"},
                        {"field": "consequents", "headerName": "結論部"},
                        {"field": "support", "headerName": "Support", "type": "number"},
                        {"field": "confidence", "headerName": "Confidence", "type": "number"},
                        {"field": "lift", "headerName": "Lift", "type": "number"}
                    ]
                },
                "data": {
                    "rules": filtered_rules
                }
            }

    def _prepare_network_chart_data(self, rules: List[Dict[str, Any]], options: Dict[str, Any]) -> Dict[str, Any]:
        """ネットワーク図用データを準備する"""
        nodes = set()
        links = []

        for rule in rules:
            antecedents = rule.get("antecedents", [])
            consequents = rule.get("consequents", [])

            for item in antecedents + consequents:
                nodes.add(item)

            for a_item in antecedents:
                for c_item in consequents:
                    links.append({
                        "source": a_item,
                        "target": c_item,
                        "confidence": rule.get("confidence", 0),
                        "support": rule.get("support", 0),
                        "lift": rule.get("lift", 1)
                    })

        chart_data = {
            "nodes": [{"id": node} for node in nodes],
            "links": links
        }

        chart_config = {
            "chart_type": "network",
            "title": options.get("title", "アソシエーションルールネットワーク"),
            "width": options.get("width", 800),
            "height": options.get("height", 600),
            "link_strength": options.get("link_strength", "confidence"),
            "node_size": options.get("node_size", "degree"),
            "color_scheme": options.get("color_scheme", "category10")
        }

        return {"config": chart_config, "data": chart_data}

    def _prepare_heatmap_chart_data(self, rules: List[Dict[str, Any]], options: Dict[str, Any]) -> Dict[str, Any]:
        """ヒートマップ用データを準備する"""
        items = set()
        for rule in rules:
            for item in rule.get("antecedents", []) + rule.get("consequents", []):
                items.add(item)

        items = list(items)
        matrix = np.zeros((len(items), len(items)))

        for rule in rules:
            antecedents = rule.get("antecedents", [])
            consequents = rule.get("consequents", [])

            for a_item in antecedents:
                a_idx = items.index(a_item)
                for c_item in consequents:
                    c_idx = items.index(c_item)
                    matrix[a_idx, c_idx] = rule.get(options.get("heatmap_value", "lift"), 0)

        chart_data = {
            "matrix": matrix.tolist(),
            "x_labels": items,
            "y_labels": items
        }

        chart_config = {
            "chart_type": "heatmap",
            "title": options.get("title", "アソシエーションルールヒートマップ"),
            "x_axis_label": options.get("x_axis_label", "アイテム"),
            "y_axis_label": options.get("y_axis_label", "アイテム"),
            "width": options.get("width", 800),
            "height": options.get("height", 800),
            "color_scheme": options.get("color_scheme", "blues")
        }

        return {"config": chart_config, "data": chart_data}

    def _prepare_bar_chart_data(self, rules: List[Dict[str, Any]], options: Dict[str, Any]) -> Dict[str, Any]:
        """バーチャート用データを準備する"""
        sorted_rules = sorted(
            rules,
            key=lambda x: x.get(options.get("sort_by", "lift"), 0),
            reverse=True
        )[:options.get("max_rules", 10)]

        labels = [
            f"{' & '.join(rule.get('antecedents', []))} => {' & '.join(rule.get('consequents', []))}"
            for rule in sorted_rules
        ]

        lifts = [rule.get("lift", 0) for rule in sorted_rules]
        confidences = [rule.get("confidence", 0) for rule in sorted_rules]
        supports = [rule.get("support", 0) for rule in sorted_rules]

        chart_data = {
            "labels": labels,
            "datasets": [
                {
                    "label": "Lift",
                    "data": lifts,
                    "backgroundColor": "rgba(66, 133, 244, 0.7)",
                },
                {
                    "label": "Confidence",
                    "data": confidences,
                    "backgroundColor": "rgba(219, 68, 55, 0.7)",
                },
                {
                    "label": "Support",
                    "data": supports,
                    "backgroundColor": "rgba(15, 157, 88, 0.7)",
                }
            ]
        }

        chart_config = {
            "chart_type": "bar",
            "title": options.get("title", "トップアソシエーションルール"),
            "x_axis_label": options.get("x_axis_label", "ルール"),
            "y_axis_label": options.get("y_axis_label", "メトリクス値"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "show_legend": True,
            "color_scheme": options.get("color_scheme", "google")
        }

        return {"config": chart_config, "data": chart_data}

    def format_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        アソシエーション分析結果のサマリーをフォーマットする

        Args:
            analysis_results: アソシエーション分析結果

        Returns:
            フォーマット済みサマリー
        """
        rules = analysis_results.get("rules", [])

        # ルールが存在しない場合
        if not rules:
            return {
                "rule_count": 0,
                "message": "有効なアソシエーションルールが見つかりませんでした"
            }

        # 上位ルールを抽出
        top_rules = self._get_top_rules(rules)

        # 基本統計
        lifts = [rule.get("lift", 0) for rule in rules]
        confidences = [rule.get("confidence", 0) for rule in rules]
        supports = [rule.get("support", 0) for rule in rules]

        avg_lift = sum(lifts) / len(lifts) if lifts else 0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        avg_support = sum(supports) / len(supports) if supports else 0

        return {
            "rule_count": len(rules),
            "top_rules": top_rules,
            "average_metrics": {
                "lift": avg_lift,
                "confidence": avg_confidence,
                "support": avg_support
            },
            "max_metrics": {
                "lift": max(lifts) if lifts else 0,
                "confidence": max(confidences) if confidences else 0,
                "support": max(supports) if supports else 0
            }
        }

    def _get_top_rules(self, rules: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
        """上位ルールを抽出する"""
        # リフト値でソート
        lift_sorted = sorted(rules, key=lambda x: x.get("lift", 0), reverse=True)[:limit]

        # 読みやすい形式に変換
        readable_rules = []
        for rule in lift_sorted:
            antecedents = " & ".join(rule.get("antecedents", []))
            consequents = " & ".join(rule.get("consequents", []))
            readable_rules.append({
                "rule": f"{antecedents} => {consequents}",
                "lift": rule.get("lift", 0),
                "confidence": rule.get("confidence", 0),
                "support": rule.get("support", 0)
            })

        return readable_rules


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

        return {
            "variable_count": len(variables),
            "strong_correlations": strong_correlations[:5],  # 上位5つのみ表示
            "average_absolute_correlation": avg_abs_corr,
            "max_positive_correlation": max([corr_matrix[i][j] for i in range(len(corr_matrix)) for j in range(i+1, len(corr_matrix[i]))], default=0),
            "max_negative_correlation": min([corr_matrix[i][j] for i in range(len(corr_matrix)) for j in range(i+1, len(corr_matrix[i]))], default=0)
        }