"""
アソシエーション分析可視化プロセッサ

このモジュールでは、アソシエーション分析の可視化を処理するプロセッサクラスを実装します。
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