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