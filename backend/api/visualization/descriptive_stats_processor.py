"""
記述統計可視化プロセッサ

このモジュールでは、記述統計の可視化を処理するプロセッサクラスを実装します。
"""

from typing import Dict, List, Any, Optional
import logging
import numpy as np
from .factory import VisualizationProcessor, register_processor

logger = logging.getLogger(__name__)


@register_processor("descriptive_stats")
class DescriptiveStatsVisualizationProcessor(VisualizationProcessor):
    """記述統計の可視化プロセッサ"""

    def prepare_chart_data(self, analysis_results: Dict[str, Any],
                          visualization_type: str,
                          options: Dict[str, Any]) -> Dict[str, Any]:
        """
        記述統計のチャートデータを準備する

        Args:
            analysis_results: 記述統計結果
            visualization_type: 可視化タイプ
            options: オプション設定

        Returns:
            チャートデータ
        """
        if visualization_type == "histogram":
            return self._prepare_histogram_data(analysis_results, options)
        elif visualization_type == "boxplot":
            return self._prepare_boxplot_data(analysis_results, options)
        elif visualization_type == "bar":
            return self._prepare_bar_data(analysis_results, options)
        elif visualization_type == "pie":
            return self._prepare_pie_data(analysis_results, options)
        elif visualization_type == "table":
            return self._prepare_table_data(analysis_results, options)
        else:
            # デフォルトはテーブル表示
            return self._prepare_table_data(analysis_results, options)

    def _prepare_histogram_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """ヒストグラム用データを準備する"""
        # データの取得
        stats = analysis_results.get("stats", {})
        variable = options.get("variable")

        if not variable and stats:
            # 変数が指定されていない場合は最初の変数を使用
            variable = next(iter(stats.keys()))

        variable_data = stats.get(variable, {})
        data = variable_data.get("data", [])

        # ヒストグラムのビン（階級）設定
        bin_count = options.get("bin_count", 10)

        if data:
            # データの最小値と最大値
            min_val = min(data)
            max_val = max(data)

            # ビン幅の計算
            bin_width = (max_val - min_val) / bin_count if max_val > min_val else 1

            # ビンの境界値と中央値を計算
            bins = [min_val + i * bin_width for i in range(bin_count + 1)]
            bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(bin_count)]
            bin_labels = [f"{bins[i]:.2f} - {bins[i+1]:.2f}" for i in range(bin_count)]

            # 各ビンの頻度をカウント
            hist, _ = np.histogram(data, bins=bins)

            chart_data = {
                "labels": bin_labels,
                "datasets": [{
                    "label": variable,
                    "data": hist.tolist(),
                    "backgroundColor": options.get("color", "rgba(66, 133, 244, 0.7)")
                }]
            }
        else:
            # データがない場合
            chart_data = {
                "labels": [f"ビン {i+1}" for i in range(bin_count)],
                "datasets": [{
                    "label": variable or "データなし",
                    "data": [0] * bin_count,
                    "backgroundColor": options.get("color", "rgba(66, 133, 244, 0.7)")
                }]
            }

        # チャート設定
        chart_config = {
            "chart_type": "bar",
            "title": options.get("title", f"ヒストグラム: {variable}"),
            "subtitle": options.get("subtitle", "頻度分布"),
            "x_axis_label": options.get("x_axis_label", "値"),
            "y_axis_label": options.get("y_axis_label", "頻度"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "show_grid": options.get("show_grid", True),
            "show_legend": options.get("show_legend", False)
        }

        return {"config": chart_config, "data": chart_data}

    def _prepare_boxplot_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """箱ひげ図用データを準備する"""
        # データの取得
        stats = analysis_results.get("stats", {})

        # 表示する変数の選択
        selected_vars = options.get("variables", [])
        if not selected_vars:
            # 指定がなければすべての変数を表示
            selected_vars = list(stats.keys())

        # 各変数のデータセット作成
        datasets = []
        for i, var in enumerate(selected_vars):
            var_stats = stats.get(var, {})

            # 基本統計量の取得またはデフォルト値の設定
            min_val = var_stats.get("min", 0)
            q1 = var_stats.get("q1", min_val)
            median = var_stats.get("median", q1)
            q3 = var_stats.get("q3", median)
            max_val = var_stats.get("max", q3)

            # 外れ値の計算（オプション）
            if "data" in var_stats and options.get("show_outliers", True):
                data = var_stats["data"]
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = [x for x in data if x < lower_bound or x > upper_bound]
            else:
                outliers = []

            # カラー設定
            color = options.get(f"color_{i}", self._get_color(i))

            datasets.append({
                "label": var,
                "data": [{
                    "min": min_val,
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "max": max_val,
                    "outliers": outliers
                }],
                "backgroundColor": color
            })

        # チャートデータ
        chart_data = {
            "labels": [""],  # 単一の箱ひげ図の場合は空ラベル
            "datasets": datasets
        }

        # チャート設定
        chart_config = {
            "chart_type": "boxplot",
            "title": options.get("title", "箱ひげ図"),
            "subtitle": options.get("subtitle", "分布の比較"),
            "x_axis_label": options.get("x_axis_label", "変数"),
            "y_axis_label": options.get("y_axis_label", "値"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "show_grid": options.get("show_grid", True),
            "show_legend": options.get("show_legend", True)
        }

        return {"config": chart_config, "data": chart_data}

    def _prepare_bar_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """棒グラフ用データを準備する"""
        # データの取得
        stats = analysis_results.get("stats", {})
        category_counts = analysis_results.get("category_counts", {})

        # カテゴリカルデータを表示
        if category_counts:
            # 表示するカテゴリの選択
            variable = options.get("variable")
            if not variable and category_counts:
                # 指定がなければ最初の変数を使用
                variable = next(iter(category_counts.keys()))

            # カテゴリデータの取得
            category_data = category_counts.get(variable, {})

            # ソート順の設定
            sort_by = options.get("sort_by", "value")  # value or alphabetical
            if sort_by == "value":
                sort_desc = options.get("sort_desc", True)
                sorted_items = sorted(category_data.items(), key=lambda x: x[1], reverse=sort_desc)
            else:
                sorted_items = sorted(category_data.items())

            # 上位N項目のみ表示（オプション）
            top_n = options.get("top_n")
            if top_n and isinstance(top_n, int) and top_n > 0:
                sorted_items = sorted_items[:top_n]

            # チャートデータの作成
            labels = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]

            chart_data = {
                "labels": labels,
                "datasets": [{
                    "label": variable,
                    "data": values,
                    "backgroundColor": options.get("colors", [self._get_color(i) for i in range(len(labels))])
                }]
            }

            chart_title = options.get("title", f"カテゴリ分布: {variable}")
        else:
            # 数値変数の統計量を表示
            # 表示する変数と統計量の選択
            selected_vars = options.get("variables", [])
            if not selected_vars:
                selected_vars = list(stats.keys())

            stat_type = options.get("stat_type", "mean")  # mean, median, etc.

            # 統計量の取得
            labels = selected_vars
            values = [stats.get(var, {}).get(stat_type, 0) for var in selected_vars]

            chart_data = {
                "labels": labels,
                "datasets": [{
                    "label": f"{stat_type}",
                    "data": values,
                    "backgroundColor": options.get("colors", [self._get_color(i) for i in range(len(labels))])
                }]
            }

            chart_title = options.get("title", f"変数の{stat_type}値")

        # チャート設定
        chart_config = {
            "chart_type": "bar",
            "title": chart_title,
            "subtitle": options.get("subtitle", ""),
            "x_axis_label": options.get("x_axis_label", "カテゴリ"),
            "y_axis_label": options.get("y_axis_label", "値"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "show_grid": options.get("show_grid", True),
            "show_legend": options.get("show_legend", False),
            "horizontal": options.get("horizontal", False)
        }

        return {"config": chart_config, "data": chart_data}

    def _prepare_pie_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """円グラフ用データを準備する"""
        # カテゴリデータの取得
        category_counts = analysis_results.get("category_counts", {})

        # 表示するカテゴリの選択
        variable = options.get("variable")
        if not variable and category_counts:
            # 指定がなければ最初の変数を使用
            variable = next(iter(category_counts.keys()))

        category_data = category_counts.get(variable, {})

        # ソート順の設定とデータ準備
        sort_by = options.get("sort_by", "value")  # value or alphabetical
        if sort_by == "value":
            sort_desc = options.get("sort_desc", True)
            sorted_items = sorted(category_data.items(), key=lambda x: x[1], reverse=sort_desc)
        else:
            sorted_items = sorted(category_data.items())

        # 上位N項目のみ表示、残りはその他にまとめる（オプション）
        top_n = options.get("top_n")
        if top_n and isinstance(top_n, int) and top_n > 0 and len(sorted_items) > top_n:
            top_items = sorted_items[:top_n]
            other_sum = sum([item[1] for item in sorted_items[top_n:]])

            if other_sum > 0 and options.get("show_others", True):
                labels = [item[0] for item in top_items] + ["その他"]
                values = [item[1] for item in top_items] + [other_sum]
            else:
                labels = [item[0] for item in top_items]
                values = [item[1] for item in top_items]
        else:
            labels = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]

        # カラー設定
        colors = options.get("colors", [self._get_color(i) for i in range(len(labels))])

        # チャートデータ
        chart_data = {
            "labels": labels,
            "datasets": [{
                "data": values,
                "backgroundColor": colors
            }]
        }

        # チャート設定
        chart_config = {
            "chart_type": "pie",
            "title": options.get("title", f"円グラフ: {variable}"),
            "subtitle": options.get("subtitle", "カテゴリ分布"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "show_legend": options.get("show_legend", True),
            "display_values": options.get("display_values", True),
            "value_format": options.get("value_format", "percentage")  # percentage or value
        }

        return {"config": chart_config, "data": chart_data}

    def _prepare_table_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """テーブル表示用データを準備する"""
        # データの取得
        stats = analysis_results.get("stats", {})

        # 表示する統計量の選択
        stat_metrics = options.get("metrics", ["count", "mean", "std", "min", "q1", "median", "q3", "max"])

        # テーブルデータの作成
        rows = []
        for var_name, var_stats in stats.items():
            row = {"variable": var_name}
            for metric in stat_metrics:
                value = var_stats.get(metric, "")
                # 数値フォーマット
                if isinstance(value, (int, float)):
                    if metric in ["count"]:
                        row[metric] = int(value)
                    else:
                        row[metric] = round(value, options.get("decimal_places", 2))
                else:
                    row[metric] = value
            rows.append(row)

        # テーブル設定
        columns = [{"field": "variable", "headerName": "変数"}]

        # 統計量の列定義
        metric_labels = {
            "count": "件数",
            "mean": "平均",
            "std": "標準偏差",
            "min": "最小値",
            "q1": "第1四分位",
            "median": "中央値",
            "q3": "第3四分位",
            "max": "最大値",
            "missing": "欠損値",
            "mode": "最頻値",
            "skewness": "歪度",
            "kurtosis": "尖度"
        }

        for metric in stat_metrics:
            columns.append({
                "field": metric,
                "headerName": metric_labels.get(metric, metric),
                "type": "number"
            })

        # チャートデータ
        chart_data = {
            "rows": rows,
            "columns": columns
        }

        # チャート設定
        chart_config = {
            "chart_type": "table",
            "title": options.get("title", "記述統計量"),
            "subtitle": options.get("subtitle", "基本統計指標"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "striped": options.get("striped", True),
            "sortable": options.get("sortable", True)
        }

        return {"config": chart_config, "data": chart_data}

    def format_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        記述統計結果のサマリーをフォーマットする

        Args:
            analysis_results: 記述統計結果

        Returns:
            フォーマット済みサマリー
        """
        stats = analysis_results.get("stats", {})
        category_counts = analysis_results.get("category_counts", {})

        # 結果がない場合
        if not stats and not category_counts:
            return {
                "message": "有効な記述統計データが見つかりませんでした"
            }

        # 数値変数のサマリー
        numeric_summary = {}
        if stats:
            # 変数数
            numeric_summary["variable_count"] = len(stats)

            # 各変数の基本統計量
            for var_name, var_stats in stats.items():
                mean = var_stats.get("mean", 0)
                median = var_stats.get("median", 0)
                std = var_stats.get("std", 0)

                numeric_summary[var_name] = {
                    "mean": mean,
                    "median": median,
                    "std": std,
                    "range": [var_stats.get("min", 0), var_stats.get("max", 0)]
                }

        # カテゴリカル変数のサマリー
        categorical_summary = {}
        if category_counts:
            # 変数数
            categorical_summary["variable_count"] = len(category_counts)

            # 各変数の上位カテゴリ
            for var_name, counts in category_counts.items():
                sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                top_categories = []

                # 上位3カテゴリまで取得
                for i, (category, count) in enumerate(sorted_items[:3]):
                    total = sum(counts.values())
                    percentage = count / total * 100 if total > 0 else 0
                    top_categories.append({
                        "category": category,
                        "count": count,
                        "percentage": round(percentage, 1)
                    })

                categorical_summary[var_name] = {
                    "category_count": len(counts),
                    "top_categories": top_categories
                }

        # 総合サマリー
        return {
            "numeric_variables": numeric_summary,
            "categorical_variables": categorical_summary,
            "total_variables": len(stats) + len(category_counts)
        }

    def _get_color(self, index: int) -> str:
        """インデックスに基づいて色を返す"""
        colors = [
            "rgba(66, 133, 244, 0.7)",   # Google Blue
            "rgba(219, 68, 55, 0.7)",    # Google Red
            "rgba(244, 180, 0, 0.7)",    # Google Yellow
            "rgba(15, 157, 88, 0.7)",    # Google Green
            "rgba(171, 71, 188, 0.7)",   # Purple
            "rgba(255, 112, 67, 0.7)",   # Deep Orange
            "rgba(0, 172, 193, 0.7)",    # Cyan
            "rgba(124, 179, 66, 0.7)",   # Light Green
        ]
        return colors[index % len(colors)]