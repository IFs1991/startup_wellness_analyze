"""
可視化ヘルパー関数

このモジュールは、さまざまな分析結果の可視化に使用されるヘルパー関数を提供します。
各グラフタイプ（箱ひげ図、ヒストグラム、散布図、棒グラフなど）のデータ準備関数を含みます。
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional

def prepare_boxplot_data(analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """
    箱ひげ図用のデータを準備する

    Args:
        analysis_results: 統計分析結果
        options: 可視化オプション

    Returns:
        チャート設定とデータ
    """
    # 分析結果からデータを抽出
    data = _extract_data_from_results(analysis_results)
    variables = analysis_results.get("metadata", {}).get("variables", [])
    groups = analysis_results.get("metadata", {}).get("groups", [])

    # ボックスプロットデータセットの構築
    datasets = []
    boxplot_data = {}

    if groups and len(groups) > 1:
        # グループごとのデータセット
        for variable in variables:
            var_data = []
            for group in groups:
                group_data = data.get(group, {}).get(variable, [])
                if not group_data and isinstance(data.get(group), list):
                    # データ形式が異なる場合の対応
                    for item in data.get(group, []):
                        if variable in item:
                            group_data.append(item[variable])

                # 基本統計量の計算
                if group_data:
                    q1 = np.percentile(group_data, 25)
                    median = np.percentile(group_data, 50)
                    q3 = np.percentile(group_data, 75)
                    iqr = q3 - q1
                    lower_bound = max(min(group_data), q1 - 1.5 * iqr)
                    upper_bound = min(max(group_data), q3 + 1.5 * iqr)

                    # 外れ値の検出
                    outliers = [x for x in group_data if x < lower_bound or x > upper_bound]

                    var_data.append({
                        "min": lower_bound,
                        "q1": q1,
                        "median": median,
                        "q3": q3,
                        "max": upper_bound,
                        "outliers": outliers
                    })
                else:
                    var_data.append({
                        "min": 0,
                        "q1": 0,
                        "median": 0,
                        "q3": 0,
                        "max": 0,
                        "outliers": []
                    })

            # データセット作成
            datasets.append({
                "label": options.get(f"{variable}_label", variable),
                "data": var_data,
                "backgroundColor": options.get(f"{variable}_color", _get_color(len(datasets))),
            })

        boxplot_data = {
            "labels": groups,
            "datasets": datasets
        }
    else:
        # 変数ごとのデータセット（グループなし）
        for variable in variables:
            var_data = data.get(variable, [])
            if not var_data and data:
                # 異なるデータ形式の場合の対応
                var_data = []
                for item in data:
                    if isinstance(item, dict) and variable in item:
                        var_data.append(item[variable])

            # 基本統計量の計算
            if var_data:
                q1 = np.percentile(var_data, 25)
                median = np.percentile(var_data, 50)
                q3 = np.percentile(var_data, 75)
                iqr = q3 - q1
                lower_bound = max(min(var_data), q1 - 1.5 * iqr)
                upper_bound = min(max(var_data), q3 + 1.5 * iqr)

                # 外れ値の検出
                outliers = [x for x in var_data if x < lower_bound or x > upper_bound]

                # データセット作成
                datasets.append({
                    "label": options.get(f"{variable}_label", variable),
                    "backgroundColor": options.get(f"{variable}_color", _get_color(len(datasets))),
                    "data": [{
                        "min": lower_bound,
                        "q1": q1,
                        "median": median,
                        "q3": q3,
                        "max": upper_bound,
                        "outliers": outliers
                    }]
                })
            else:
                datasets.append({
                    "label": options.get(f"{variable}_label", variable),
                    "backgroundColor": options.get(f"{variable}_color", _get_color(len(datasets))),
                    "data": [{
                        "min": 0,
                        "q1": 0,
                        "median": 0,
                        "q3": 0,
                        "max": 0,
                        "outliers": []
                    }]
                })

        boxplot_data = {
            "labels": [""],
            "datasets": datasets
        }

    # チャートの設定
    chart_config = {
        "chart_type": "boxplot",
        "title": options.get("title", "統計分析: 箱ひげ図"),
        "x_axis_label": options.get("x_axis_label", "グループ"),
        "y_axis_label": options.get("y_axis_label", "値"),
        "width": options.get("width", 800),
        "height": options.get("height", 500),
        "show_legend": True,
        "color_scheme": options.get("color_scheme", "google"),
    }

    return {"config": chart_config, "data": boxplot_data}

def prepare_histogram_data(analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """
    ヒストグラム用のデータを準備する

    Args:
        analysis_results: 統計分析結果
        options: 可視化オプション

    Returns:
        チャート設定とデータ
    """
    # 分析結果からデータを抽出
    data = _extract_data_from_results(analysis_results)
    variables = analysis_results.get("metadata", {}).get("variables", [])

    # 変数が指定されていない場合は最初の変数を使用
    target_variable = options.get("target_variable", variables[0] if variables else None)
    if not target_variable:
        # エラー処理またはデフォルト値設定
        target_variable = "value"
        var_data = list(range(10))  # ダミーデータ
    else:
        # 対象変数のデータを取得
        var_data = data.get(target_variable, [])
        if not var_data and data:
            # 異なるデータ形式の場合の対応
            var_data = []
            for item in data:
                if isinstance(item, dict) and target_variable in item:
                    var_data.append(item[target_variable])

    # ヒストグラムのビン（階級）を計算
    bin_count = options.get("bin_count", 10)

    if var_data:
        min_val = min(var_data)
        max_val = max(var_data)
        bin_width = (max_val - min_val) / bin_count if max_val > min_val else 1

        bins = [min_val + i * bin_width for i in range(bin_count + 1)]
        bin_labels = [f"{bins[i]:.2f} - {bins[i+1]:.2f}" for i in range(bin_count)]

        # ヒストグラムの頻度をカウント
        hist, _ = np.histogram(var_data, bins=bins)
    else:
        # データがない場合のダミー値
        bin_labels = [f"ビン {i+1}" for i in range(bin_count)]
        hist = np.zeros(bin_count)

    # チャートデータ
    chart_data = {
        "labels": bin_labels,
        "datasets": [
            {
                "label": options.get("label", target_variable),
                "data": hist.tolist(),
                "backgroundColor": options.get("bar_color", "rgba(66, 133, 244, 0.7)"),
                "borderColor": options.get("border_color", "rgba(66, 133, 244, 1)"),
                "borderWidth": 1
            }
        ]
    }

    # チャートの設定
    chart_config = {
        "chart_type": "bar",
        "title": options.get("title", f"ヒストグラム: {target_variable}"),
        "x_axis_label": options.get("x_axis_label", "値の範囲"),
        "y_axis_label": options.get("y_axis_label", "頻度"),
        "width": options.get("width", 800),
        "height": options.get("height", 500),
        "show_legend": False,
        "color_scheme": options.get("color_scheme", "blues")
    }

    return {"config": chart_config, "data": chart_data}

def prepare_scatter_data(analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """
    散布図用のデータを準備する

    Args:
        analysis_results: 統計分析結果またはデータ
        options: 可視化オプション

    Returns:
        チャート設定とデータ
    """
    # 分析結果からデータを抽出
    data = _extract_data_from_results(analysis_results)
    variables = analysis_results.get("metadata", {}).get("variables", [])

    # X軸とY軸の変数
    if len(variables) >= 2:
        x_variable = options.get("x_variable", variables[0])
        y_variable = options.get("y_variable", variables[1])
    else:
        x_variable = options.get("x_variable", "x")
        y_variable = options.get("y_variable", "y")

    # データポイントの作成
    points = []

    # 異なるデータ形式に対応
    if isinstance(data, list):
        # リスト形式の場合
        for item in data:
            if isinstance(item, dict) and x_variable in item and y_variable in item:
                points.append({
                    "x": item[x_variable],
                    "y": item[y_variable]
                })
    elif isinstance(data, dict):
        # 辞書形式の場合
        x_data = data.get(x_variable, [])
        y_data = data.get(y_variable, [])

        if len(x_data) == len(y_data):
            points = [{"x": x, "y": y} for x, y in zip(x_data, y_data)]

    # データポイントがない場合はダミーデータを作成
    if not points:
        points = [{"x": i, "y": np.random.normal(i, 1)} for i in range(10)]

    # チャートデータ
    chart_data = {
        "datasets": [
            {
                "label": options.get("label", f"{x_variable} vs {y_variable}"),
                "data": points,
                "backgroundColor": options.get("point_color", "rgba(66, 133, 244, 0.7)"),
                "borderColor": options.get("border_color", "rgba(66, 133, 244, 1)"),
                "borderWidth": 1,
                "pointRadius": options.get("point_radius", 4),
                "pointHoverRadius": options.get("point_hover_radius", 6),
            }
        ]
    }

    # 回帰線を追加
    if options.get("show_trendline", True) and len(points) > 1:
        # 回帰直線の計算
        x_values = [p["x"] for p in points]
        y_values = [p["y"] for p in points]

        try:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            x_min = min(x_values)
            x_max = max(x_values)

            trendline_points = [
                {"x": x_min, "y": slope * x_min + intercept},
                {"x": x_max, "y": slope * x_max + intercept}
            ]

            chart_data["datasets"].append({
                "label": "回帰直線",
                "data": trendline_points,
                "type": "line",
                "fill": False,
                "borderColor": options.get("trendline_color", "rgba(234, 67, 53, 0.7)"),
                "borderWidth": 2,
                "pointRadius": 0
            })
        except:
            # 回帰計算に失敗した場合は追加しない
            pass

    # チャートの設定
    chart_config = {
        "chart_type": "scatter",
        "title": options.get("title", f"散布図: {x_variable} vs {y_variable}"),
        "x_axis_label": options.get("x_axis_label", x_variable),
        "y_axis_label": options.get("y_axis_label", y_variable),
        "width": options.get("width", 800),
        "height": options.get("height", 500),
        "show_legend": options.get("show_trendline", True),
        "color_scheme": options.get("color_scheme", "google")
    }

    return {"config": chart_config, "data": chart_data}

def prepare_bar_data(analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """
    棒グラフ用のデータを準備する

    Args:
        analysis_results: 統計分析結果またはデータ
        options: 可視化オプション

    Returns:
        チャート設定とデータ
    """
    # 分析結果からデータを抽出
    data = _extract_data_from_results(analysis_results)
    test_type = analysis_results.get("metadata", {}).get("test_type", "")
    variables = analysis_results.get("metadata", {}).get("variables", [])
    groups = analysis_results.get("metadata", {}).get("groups", [])

    # ラベルとデータセットの初期化
    labels = []
    datasets = []

    # カイ二乗検定の場合
    if test_type == "chi2":
        # カテゴリカルデータの抽出
        contingency_table = analysis_results.get("contingency_table", {})
        if contingency_table and isinstance(contingency_table, dict):
            # カテゴリとその値を取得
            for category, values in contingency_table.items():
                if category not in labels:
                    labels.append(category)

            # グループごとのデータセット作成
            for i, group in enumerate(groups):
                group_data = []
                for category in labels:
                    value = contingency_table.get(category, {}).get(group, 0)
                    group_data.append(value)

                datasets.append({
                    "label": group,
                    "data": group_data,
                    "backgroundColor": options.get(f"color_{i}", _get_color(i)),
                    "borderColor": options.get(f"border_{i}", _get_border_color(i)),
                    "borderWidth": 1
                })
        else:
            # ダミーデータの作成
            labels = ["カテゴリA", "カテゴリB", "カテゴリC"]
            datasets = [{
                "label": "グループ1",
                "data": [10, 20, 15],
                "backgroundColor": "rgba(66, 133, 244, 0.7)",
                "borderColor": "rgba(66, 133, 244, 1)",
                "borderWidth": 1
            }]
    else:
        # 一般的な棒グラフの場合
        if groups and len(groups) > 0:
            # グループ化されたデータの場合
            labels = groups

            for i, variable in enumerate(variables):
                var_data = []
                for group in groups:
                    group_data = data.get(group, {}).get(variable, 0)
                    if isinstance(group_data, list):
                        # リスト形式の場合は平均値を使用
                        group_data = sum(group_data) / len(group_data) if group_data else 0
                    var_data.append(group_data)

                datasets.append({
                    "label": options.get(f"{variable}_label", variable),
                    "data": var_data,
                    "backgroundColor": options.get(f"color_{i}", _get_color(i)),
                    "borderColor": options.get(f"border_{i}", _get_border_color(i)),
                    "borderWidth": 1
                })
        else:
            # 単一変数の場合
            for variable in variables:
                labels.append(variable)

            var_data = []
            for variable in variables:
                var_value = data.get(variable, 0)
                if isinstance(var_value, list):
                    # リスト形式の場合は平均値を使用
                    var_value = sum(var_value) / len(var_value) if var_value else 0
                var_data.append(var_value)

            datasets.append({
                "label": options.get("label", "値"),
                "data": var_data,
                "backgroundColor": options.get("bar_color", "rgba(66, 133, 244, 0.7)"),
                "borderColor": options.get("border_color", "rgba(66, 133, 244, 1)"),
                "borderWidth": 1
            })

    # チャートデータ
    chart_data = {
        "labels": labels,
        "datasets": datasets
    }

    # チャートの設定
    title = options.get("title")
    if not title:
        if test_type == "chi2":
            title = "カイ二乗検定: カテゴリー分布"
        else:
            title = "統計分析: 棒グラフ"

    chart_config = {
        "chart_type": "bar",
        "title": title,
        "x_axis_label": options.get("x_axis_label", "グループ"),
        "y_axis_label": options.get("y_axis_label", "値"),
        "width": options.get("width", 800),
        "height": options.get("height", 500),
        "show_legend": len(datasets) > 1,
        "color_scheme": options.get("color_scheme", "google")
    }

    return {"config": chart_config, "data": chart_data}

# 内部ヘルパー関数
def _extract_data_from_results(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """分析結果からデータを抽出する"""
    # データの直接抽出
    data = analysis_results.get("data", {})

    # データが文字列形式の場合、JSONとして解析
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except:
            # 解析に失敗した場合は空の辞書を返す
            data = {}

    # データが見つからない場合は他の場所を探す
    if not data:
        # 記述統計からデータを抽出
        descriptive_stats = analysis_results.get("descriptive_stats", {})
        if isinstance(descriptive_stats, str):
            try:
                descriptive_stats = json.loads(descriptive_stats)
            except:
                descriptive_stats = {}

        # データフレームの構造を再構築
        reconstructed_data = {}
        for var, stats in descriptive_stats.items():
            if isinstance(stats, dict) and "count" in stats:
                # 基本的な擬似データの作成
                mean = stats.get("mean", 0)
                std = stats.get("std", 1)
                count = stats.get("count", 10)
                reconstructed_data[var] = list(np.random.normal(mean, std, int(count)))

        if reconstructed_data:
            data = reconstructed_data

    return data

def _get_color(index: int) -> str:
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

def _get_border_color(index: int) -> str:
    """インデックスに基づいて枠線色を返す"""
    border_colors = [
        "rgba(66, 133, 244, 1)",   # Google Blue
        "rgba(219, 68, 55, 1)",    # Google Red
        "rgba(244, 180, 0, 1)",    # Google Yellow
        "rgba(15, 157, 88, 1)",    # Google Green
        "rgba(171, 71, 188, 1)",   # Purple
        "rgba(255, 112, 67, 1)",   # Deep Orange
        "rgba(0, 172, 193, 1)",    # Cyan
        "rgba(124, 179, 66, 1)",   # Light Green
    ]
    return border_colors[index % len(border_colors)]

def prepare_chart_data_by_analysis_type(
    analysis_type: str,
    analysis_results: Dict[str, Any],
    visualization_type: str,
    options: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    分析タイプに基づいてチャートデータを準備する汎用関数

    Args:
        analysis_type: 分析の種類 (association, cluster, correlation, etc.)
        analysis_results: 分析結果
        visualization_type: 可視化タイプ (bar, line, scatter, etc.)
        options: 追加オプション

    Returns:
        チャート設定とデータ
    """
    options = options or {}

    if analysis_type == "association":
        return _prepare_association_chart_data(analysis_results, visualization_type, options)
    elif analysis_type == "cluster":
        return _prepare_cluster_chart_data(analysis_results, visualization_type, options)
    elif analysis_type == "correlation":
        return _prepare_correlation_chart_data(analysis_results, visualization_type, options)
    elif analysis_type == "timeseries":
        return _prepare_timeseries_chart_data(analysis_results, visualization_type, options)
    elif analysis_type == "statistical":
        return _prepare_statistical_chart_data(analysis_results, visualization_type, options)
    elif analysis_type == "pca":
        return _prepare_pca_chart_data(analysis_results, visualization_type, options)
    elif analysis_type == "descriptive_stats":
        return _prepare_descriptive_stats_chart_data(analysis_results, visualization_type, options)
    elif analysis_type == "predictive_model":
        return _prepare_predictive_model_chart_data(analysis_results, visualization_type, options)
    elif analysis_type == "survival_analysis":
        return _prepare_survival_analysis_chart_data(analysis_results, visualization_type, options)
    else:
        return _prepare_generic_chart_data(analysis_results, visualization_type, options)


def create_visualization_response(chart_result: Dict[str, Any], analysis_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    統一された可視化レスポンスを作成する

    Args:
        chart_result: チャート生成結果
        analysis_summary: 分析サマリー

    Returns:
        統一されたレスポンス
    """
    return {
        "chart_id": chart_result.get("chart_id", ""),
        "url": chart_result.get("url", ""),
        "format": chart_result.get("format", "png"),
        "thumbnail_url": chart_result.get("thumbnail_url"),
        "metadata": chart_result.get("metadata", {}),
        "analysis_summary": analysis_summary
    }


def _prepare_association_chart_data(
    analysis_results: Dict[str, Any],
    visualization_type: str,
    options: Dict[str, Any]
) -> Dict[str, Any]:
    """
    アソシエーション分析結果のチャートデータを準備する

    Args:
        analysis_results: アソシエーション分析結果
        visualization_type: 可視化タイプ
        options: オプション

    Returns:
        チャート設定とデータ
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
        # ネットワーク図用データ
        nodes = set()
        links = []

        for rule in filtered_rules:
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

    elif visualization_type == "heatmap":
        # ヒートマップ用データ
        items = set()
        for rule in filtered_rules:
            for item in rule.get("antecedents", []) + rule.get("consequents", []):
                items.add(item)

        items = list(items)
        matrix = np.zeros((len(items), len(items)))

        for rule in filtered_rules:
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

    elif visualization_type == "bar":
        # バーチャート用データ（ルールの強さを示す）
        sorted_rules = sorted(
            filtered_rules,
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

    else:
        # デフォルトはテーブル表示
        chart_data = {
            "rules": filtered_rules
        }

        chart_config = {
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
        }

    return {"config": chart_config, "data": chart_data}


def _prepare_cluster_chart_data(
    analysis_results: Dict[str, Any],
    visualization_type: str,
    options: Dict[str, Any]
) -> Dict[str, Any]:
    """
    クラスター分析結果のチャートデータを準備する

    Args:
        analysis_results: クラスター分析結果
        visualization_type: 可視化タイプ
        options: オプション

    Returns:
        チャート設定とデータ
    """
    cluster_centers = analysis_results.get("cluster_centers", [])
    labels = analysis_results.get("labels", [])
    data = analysis_results.get("data", [])

    if visualization_type == "scatter":
        if not data or len(data[0]) < 2:
            # 2次元以上のデータがない場合、ダミーデータを生成
            return _prepare_generic_chart_data(analysis_results, visualization_type, options)

        # 散布図データの準備
        datasets = []
        unique_labels = sorted(set(labels))

        for i, cluster in enumerate(unique_labels):
            cluster_data = [data[j] for j in range(len(data)) if labels[j] == cluster]

            # クラスターデータポイントのデータセット
            if cluster_data:
                datasets.append({
                    "label": f"クラスター {cluster}",
                    "data": [{"x": point[0], "y": point[1]} for point in cluster_data],
                    "backgroundColor": _get_color(i),
                    "pointRadius": options.get("point_radius", 5),
                    "pointStyle": "circle"
                })

        # クラスター中心のデータセット
        if cluster_centers and len(cluster_centers[0]) >= 2:
            datasets.append({
                "label": "クラスター中心",
                "data": [{"x": center[0], "y": center[1]} for center in cluster_centers],
                "backgroundColor": "rgba(0, 0, 0, 0.8)",
                "pointRadius": options.get("center_radius", 8),
                "pointStyle": "triangle"
            })

        chart_data = {"datasets": datasets}

        chart_config = {
            "chart_type": "scatter",
            "title": options.get("title", "クラスター分析"),
            "x_axis_label": options.get("x_axis_label", "特徴量1"),
            "y_axis_label": options.get("y_axis_label", "特徴量2"),
            "width": options.get("width", 800),
            "height": options.get("height", 600),
            "show_legend": True,
            "color_scheme": options.get("color_scheme", "category10")
        }

    elif visualization_type == "bar":
        # クラスターサイズの棒グラフ
        cluster_sizes = {}
        for label in labels:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

        sorted_clusters = sorted(cluster_sizes.keys())

        chart_data = {
            "labels": [f"クラスター {cluster}" for cluster in sorted_clusters],
            "datasets": [{
                "label": "クラスターサイズ",
                "data": [cluster_sizes[cluster] for cluster in sorted_clusters],
                "backgroundColor": [_get_color(i) for i in range(len(sorted_clusters))],
            }]
        }

        chart_config = {
            "chart_type": "bar",
            "title": options.get("title", "クラスターサイズ分布"),
            "x_axis_label": options.get("x_axis_label", "クラスター"),
            "y_axis_label": options.get("y_axis_label", "サイズ"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "show_legend": False,
            "color_scheme": options.get("color_scheme", "category10")
        }

    else:
        # デフォルトは特徴量ごとのクラスター平均値のレーダーチャート
        feature_names = analysis_results.get("feature_names", [f"特徴量{i+1}" for i in range(len(data[0]) if data and data[0] else 0)])

        datasets = []
        unique_labels = sorted(set(labels))

        for i, cluster in enumerate(unique_labels):
            cluster_data = [data[j] for j in range(len(data)) if labels[j] == cluster]

            if cluster_data:
                # 特徴量ごとの平均値を計算
                feature_means = []
                for feature_idx in range(len(feature_names)):
                    if feature_idx < len(cluster_data[0]):
                        feature_values = [point[feature_idx] for point in cluster_data]
                        feature_means.append(sum(feature_values) / len(feature_values))
                    else:
                        feature_means.append(0)

                datasets.append({
                    "label": f"クラスター {cluster}",
                    "data": feature_means,
                    "backgroundColor": _get_color(i, 0.2),
                    "borderColor": _get_color(i),
                    "borderWidth": 2
                })

        chart_data = {
            "labels": feature_names,
            "datasets": datasets
        }

        chart_config = {
            "chart_type": "radar",
            "title": options.get("title", "クラスター特徴量プロファイル"),
            "width": options.get("width", 800),
            "height": options.get("height", 600),
            "show_legend": True,
            "color_scheme": options.get("color_scheme", "category10")
        }

    return {"config": chart_config, "data": chart_data}


def _prepare_generic_chart_data(
    analysis_results: Dict[str, Any],
    visualization_type: str,
    options: Dict[str, Any]
) -> Dict[str, Any]:
    """
    汎用チャートデータを準備する

    Args:
        analysis_results: 分析結果
        visualization_type: 可視化タイプ
        options: オプション

    Returns:
        チャート設定とデータ
    """
    # 最低限のチャートデータ生成
    if visualization_type == "bar":
        # 基本的な棒グラフ用データ
        if isinstance(analysis_results, dict) and "data" in analysis_results:
            data = analysis_results["data"]
        else:
            data = analysis_results

        chart_data = {
            "labels": [f"項目 {i+1}" for i in range(10)],
            "datasets": [{
                "label": "サンプルデータ",
                "data": [0] * 10,
                "backgroundColor": "rgba(66, 133, 244, 0.7)"
            }]
        }

        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict) and "label" in data[0] and "value" in data[0]:
                # {label, value} 形式のデータ
                chart_data["labels"] = [item["label"] for item in data[:10]]
                chart_data["datasets"][0]["data"] = [item["value"] for item in data[:10]]
            elif len(data) > 0 and not isinstance(data[0], (list, dict)):
                # 単純な数値リスト
                chart_data["datasets"][0]["data"] = data[:10]

        chart_config = {
            "chart_type": "bar",
            "title": options.get("title", "データ可視化"),
            "x_axis_label": options.get("x_axis_label", "カテゴリ"),
            "y_axis_label": options.get("y_axis_label", "値"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "show_legend": True,
            "color_scheme": options.get("color_scheme", "blue")
        }

    elif visualization_type == "line":
        # 基本的な折れ線グラフ用データ
        if isinstance(analysis_results, dict) and "data" in analysis_results:
            data = analysis_results["data"]
        else:
            data = analysis_results

        labels = [f"ポイント {i+1}" for i in range(10)]
        values = [0] * 10

        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict) and "label" in data[0] and "value" in data[0]:
                # {label, value} 形式のデータ
                labels = [item["label"] for item in data[:10]]
                values = [item["value"] for item in data[:10]]
            elif len(data) > 0 and isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
                # [x, y] 形式のデータ
                labels = [item[0] for item in data[:10]]
                values = [item[1] for item in data[:10]]
            elif len(data) > 0 and not isinstance(data[0], (list, dict)):
                # 単純な数値リスト
                values = data[:10]

        chart_data = {
            "labels": labels,
            "datasets": [{
                "label": "データ系列",
                "data": values,
                "borderColor": "rgba(66, 133, 244, 1)",
                "backgroundColor": "rgba(66, 133, 244, 0.1)",
                "fill": true
            }]
        }

        chart_config = {
            "chart_type": "line",
            "title": options.get("title", "データ推移"),
            "x_axis_label": options.get("x_axis_label", "時間/カテゴリ"),
            "y_axis_label": options.get("y_axis_label", "値"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "show_legend": True,
            "color_scheme": options.get("color_scheme", "blue")
        }

    elif visualization_type == "pie":
        # 基本的な円グラフ用データ
        if isinstance(analysis_results, dict) and "data" in analysis_results:
            data = analysis_results["data"]
        else:
            data = analysis_results

        labels = [f"カテゴリ {i+1}" for i in range(5)]
        values = [20, 20, 20, 20, 20]  # デフォルトは均等分布

        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict) and "label" in data[0] and "value" in data[0]:
                # {label, value} 形式のデータ
                labels = [item["label"] for item in data[:5]]
                values = [item["value"] for item in data[:5]]
            elif len(data) > 0 and isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
                # [label, value] 形式のデータ
                labels = [item[0] for item in data[:5]]
                values = [item[1] for item in data[:5]]

        chart_data = {
            "labels": labels,
            "datasets": [{
                "data": values,
                "backgroundColor": [_get_color(i) for i in range(len(labels))]
            }]
        }

        chart_config = {
            "chart_type": "pie",
            "title": options.get("title", "カテゴリ分布"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "show_legend": True,
            "color_scheme": options.get("color_scheme", "category10")
        }

    elif visualization_type == "scatter":
        # 基本的な散布図用データ
        if isinstance(analysis_results, dict) and "data" in analysis_results:
            data = analysis_results["data"]
        else:
            data = analysis_results

        points = []

        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
                # [[x, y], ...] 形式のデータ
                points = [{"x": point[0], "y": point[1]} for point in data]
            elif len(data) > 0 and isinstance(data[0], dict) and "x" in data[0] and "y" in data[0]:
                # [{x, y}, ...] 形式のデータ
                points = data

        if not points:
            # ダミーデータ
            points = [{"x": i, "y": i * i} for i in range(10)]

        chart_data = {
            "datasets": [{
                "label": "データポイント",
                "data": points,
                "backgroundColor": "rgba(66, 133, 244, 0.7)",
                "pointRadius": options.get("point_radius", 5)
            }]
        }

        chart_config = {
            "chart_type": "scatter",
            "title": options.get("title", "データ分布"),
            "x_axis_label": options.get("x_axis_label", "X"),
            "y_axis_label": options.get("y_axis_label", "Y"),
            "width": options.get("width", 800),
            "height": options.get("height", 600),
            "show_legend": True,
            "color_scheme": options.get("color_scheme", "blue")
        }

    else:
        # そのほかのチャートタイプの場合はテーブル表示
        chart_data = {"data": analysis_results}

        chart_config = {
            "chart_type": "table",
            "title": options.get("title", "データ表示"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
        }

    return {"config": chart_config, "data": chart_data}

def _prepare_correlation_chart_data(
    analysis_results: Dict[str, Any],
    visualization_type: str,
    options: Dict[str, Any]
) -> Dict[str, Any]:
    """
    相関分析結果のチャートデータを準備する

    Args:
        analysis_results: 相関分析結果
        visualization_type: 可視化タイプ
        options: オプション

    Returns:
        チャート設定とデータ
    """
    correlation_matrix = analysis_results.get("correlation_matrix", [])
    variables = analysis_results.get("variables", [])

    # 変数名が提供されていない場合、デフォルト名を使用
    if not variables and correlation_matrix:
        variables = [f"変数{i+1}" for i in range(len(correlation_matrix))]

    if visualization_type == "heatmap":
        # ヒートマップ用データ
        chart_data = {
            "matrix": correlation_matrix,
            "x_labels": variables,
            "y_labels": variables
        }

        chart_config = {
            "chart_type": "heatmap",
            "title": options.get("title", "相関マトリックス"),
            "width": options.get("width", 800),
            "height": options.get("height", 800),
            "color_scheme": options.get("color_scheme", "correlation"),
            "min_value": -1,
            "max_value": 1,
            "show_values": options.get("show_values", True)
        }

    elif visualization_type == "network":
        # ネットワーク図用データ
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

    elif visualization_type == "matrix":
        # 相関係数マトリックス表
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

    else:
        # デフォルトはヒートマップ
        return _prepare_correlation_chart_data(analysis_results, "heatmap", options)

    return {"config": chart_config, "data": chart_data}


def _prepare_timeseries_chart_data(
    analysis_results: Dict[str, Any],
    visualization_type: str,
    options: Dict[str, Any]
) -> Dict[str, Any]:
    """
    時系列分析結果のチャートデータを準備する

    Args:
        analysis_results: 時系列分析結果
        visualization_type: 可視化タイプ
        options: オプション

    Returns:
        チャート設定とデータ
    """
    timestamps = analysis_results.get("timestamps", [])
    values = analysis_results.get("values", [])
    series_names = analysis_results.get("series_names", [])
    forecast = analysis_results.get("forecast", [])
    forecast_timestamps = analysis_results.get("forecast_timestamps", [])
    confidence_intervals = analysis_results.get("confidence_intervals", {})

    # 時系列データがない場合
    if not timestamps or not values:
        return _prepare_generic_chart_data(analysis_results, visualization_type, options)

    # データが複数系列かどうか判定
    is_multi_series = isinstance(values[0], (list, tuple)) if values else False

    if not series_names and is_multi_series:
        series_names = [f"系列{i+1}" for i in range(len(values[0]))]

    if visualization_type == "line":
        # 折れ線グラフ用データ
        datasets = []

        if is_multi_series:
            # 複数系列の場合
            for i, series_name in enumerate(series_names):
                series_data = [values[j][i] for j in range(len(values))]

                datasets.append({
                    "label": series_name,
                    "data": series_data,
                    "borderColor": _get_color(i),
                    "backgroundColor": _get_color(i, 0.1),
                    "fill": options.get("fill", False)
                })
        else:
            # 単一系列の場合
            datasets.append({
                "label": series_names[0] if series_names else "データ",
                "data": values,
                "borderColor": options.get("line_color", "rgba(66, 133, 244, 1)"),
                "backgroundColor": options.get("fill_color", "rgba(66, 133, 244, 0.1)"),
                "fill": options.get("fill", False)
            })

            # 予測値がある場合は追加
            if forecast:
                forecast_ts = forecast_timestamps if forecast_timestamps else timestamps[-1:] + [f"予測{i+1}" for i in range(len(forecast))]

                # 予測データセット
                datasets.append({
                    "label": "予測",
                    "data": [None] * (len(timestamps) - 1) + [values[-1]] + forecast,
                    "borderColor": options.get("forecast_color", "rgba(219, 68, 55, 1)"),
                    "backgroundColor": options.get("forecast_fill_color", "rgba(219, 68, 55, 0.1)"),
                    "borderDash": [5, 5],
                    "fill": options.get("fill_forecast", False)
                })

                # 信頼区間がある場合
                if confidence_intervals:
                    lower_bound = confidence_intervals.get("lower", [])
                    upper_bound = confidence_intervals.get("upper", [])

                    if lower_bound and upper_bound:
                        # 信頼区間の下限
                        datasets.append({
                            "label": "信頼区間 (下限)",
                            "data": [None] * (len(timestamps) - 1) + [values[-1]] + lower_bound,
                            "borderColor": options.get("ci_color", "rgba(219, 68, 55, 0.4)"),
                            "backgroundColor": "transparent",
                            "borderDash": [3, 3],
                            "fill": False
                        })

                        # 信頼区間の上限
                        datasets.append({
                            "label": "信頼区間 (上限)",
                            "data": [None] * (len(timestamps) - 1) + [values[-1]] + upper_bound,
                            "borderColor": options.get("ci_color", "rgba(219, 68, 55, 0.4)"),
                            "backgroundColor": "transparent",
                            "borderDash": [3, 3],
                            "fill": "-1" if options.get("fill_ci", True) else False
                        })

        chart_data = {
            "labels": timestamps + (forecast_timestamps if forecast and forecast_timestamps else []),
            "datasets": datasets
        }

        chart_config = {
            "chart_type": "line",
            "title": options.get("title", "時系列データ"),
            "x_axis_label": options.get("x_axis_label", "時間"),
            "y_axis_label": options.get("y_axis_label", "値"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "show_legend": True,
            "time_axis": True,
            "color_scheme": options.get("color_scheme", "google")
        }

    elif visualization_type == "bar":
        # 棒グラフ用データ
        datasets = []

        if is_multi_series:
            # 複数系列の場合
            for i, series_name in enumerate(series_names):
                series_data = [values[j][i] for j in range(len(values))]

                datasets.append({
                    "label": series_name,
                    "data": series_data,
                    "backgroundColor": _get_color(i)
                })
        else:
            # 単一系列の場合
            datasets.append({
                "label": series_names[0] if series_names else "データ",
                "data": values,
                "backgroundColor": options.get("bar_color", "rgba(66, 133, 244, 0.7)")
            })

        chart_data = {
            "labels": timestamps,
            "datasets": datasets
        }

        chart_config = {
            "chart_type": "bar",
            "title": options.get("title", "時系列データ"),
            "x_axis_label": options.get("x_axis_label", "時間"),
            "y_axis_label": options.get("y_axis_label", "値"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "show_legend": True,
            "time_axis": True,
            "color_scheme": options.get("color_scheme", "google")
        }

    elif visualization_type == "area":
        # エリアチャート用データ
        datasets = []

        if is_multi_series:
            # 複数系列の場合
            for i, series_name in enumerate(series_names):
                series_data = [values[j][i] for j in range(len(values))]

                datasets.append({
                    "label": series_name,
                    "data": series_data,
                    "borderColor": _get_color(i),
                    "backgroundColor": _get_color(i, 0.5),
                    "fill": options.get("fill", True)
                })
        else:
            # 単一系列の場合
            datasets.append({
                "label": series_names[0] if series_names else "データ",
                "data": values,
                "borderColor": options.get("line_color", "rgba(66, 133, 244, 1)"),
                "backgroundColor": options.get("fill_color", "rgba(66, 133, 244, 0.5)"),
                "fill": options.get("fill", True)
            })

        chart_data = {
            "labels": timestamps,
            "datasets": datasets
        }

        chart_config = {
            "chart_type": "line",
            "title": options.get("title", "時系列データ"),
            "x_axis_label": options.get("x_axis_label", "時間"),
            "y_axis_label": options.get("y_axis_label", "値"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "show_legend": True,
            "time_axis": True,
            "stacked": options.get("stacked", False),
            "color_scheme": options.get("color_scheme", "google")
        }

    else:
        # デフォルトは折れ線グラフ
        return _prepare_timeseries_chart_data(analysis_results, "line", options)

    return {"config": chart_config, "data": chart_data}