"""
生存分析可視化プロセッサ

このモジュールでは、生存分析の可視化を処理するプロセッサクラスを実装します。
"""

from typing import Dict, List, Any, Optional
import logging
import numpy as np
from .factory import VisualizationProcessor, register_processor

logger = logging.getLogger(__name__)


@register_processor("survival_analysis")
class SurvivalAnalysisVisualizationProcessor(VisualizationProcessor):
    """生存分析の可視化プロセッサ"""

    def prepare_chart_data(self, analysis_results: Dict[str, Any],
                          visualization_type: str,
                          options: Dict[str, Any]) -> Dict[str, Any]:
        """
        生存分析のチャートデータを準備する

        Args:
            analysis_results: 生存分析結果
            visualization_type: 可視化タイプ
            options: オプション設定

        Returns:
            チャートデータ
        """
        if visualization_type == "survival_curve":
            return self._prepare_survival_curve_data(analysis_results, options)
        elif visualization_type == "cumulative_hazard":
            return self._prepare_cumulative_hazard_data(analysis_results, options)
        elif visualization_type == "log_log":
            return self._prepare_log_log_data(analysis_results, options)
        elif visualization_type == "hazard_ratio":
            return self._prepare_hazard_ratio_data(analysis_results, options)
        else:
            # デフォルトは生存曲線
            return self._prepare_survival_curve_data(analysis_results, options)

    def _prepare_survival_curve_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """生存曲線データを準備する"""
        if "survival_curve" not in analysis_results:
            logger.warning("生存曲線データが分析結果に含まれていません")
            return {
                "config": {
                    "chart_type": "line",
                    "title": options.get("title", "生存曲線"),
                    "width": options.get("width", 800),
                    "height": options.get("height", 500),
                    "x_axis_label": "時間",
                    "y_axis_label": "生存率"
                },
                "data": {
                    "labels": [],
                    "datasets": []
                }
            }

        survival_data = analysis_results["survival_curve"]
        datasets = []

        # グループが存在する場合
        if "groups" in survival_data:
            for group_name, group_data in survival_data["groups"].items():
                if "time" not in group_data or "survival" not in group_data:
                    continue

                # 主要な生存曲線データセット
                group_color = self._get_color_for_group(group_name, options)

                # 信頼区間の処理
                show_ci = options.get("show_confidence_interval", True)
                if show_ci and "lower_ci" in group_data and "upper_ci" in group_data:
                    datasets.append({
                        "label": f"{group_name}",
                        "data": list(zip(group_data["time"], group_data["survival"])),
                        "borderColor": group_color,
                        "backgroundColor": "transparent",
                        "pointRadius": options.get("point_radius", 0),
                        "fill": False
                    })

                    # 信頼区間の領域データセット
                    datasets.append({
                        "label": f"{group_name} 信頼区間",
                        "data": self._create_confidence_band(
                            group_data["time"],
                            group_data["lower_ci"],
                            group_data["upper_ci"]
                        ),
                        "backgroundColor": self._get_transparent_color(group_color, 0.2),
                        "borderColor": "transparent",
                        "pointRadius": 0,
                        "fill": True,
                        "showInLegend": False
                    })
                else:
                    # 信頼区間なしの場合
                    datasets.append({
                        "label": f"{group_name}",
                        "data": list(zip(group_data["time"], group_data["survival"])),
                        "borderColor": group_color,
                        "backgroundColor": "transparent",
                        "pointRadius": options.get("point_radius", 0),
                        "fill": False
                    })
        else:
            # グループなしの場合（単一の生存曲線）
            if "time" not in survival_data or "survival" not in survival_data:
                logger.warning("生存曲線データの必要なフィールドが不足しています")
                return {
                    "config": {
                        "chart_type": "line",
                        "title": options.get("title", "生存曲線"),
                        "width": options.get("width", 800),
                        "height": options.get("height", 500),
                        "x_axis_label": "時間",
                        "y_axis_label": "生存率"
                    },
                    "data": {
                        "labels": [],
                        "datasets": []
                    }
                }

            # 主要な生存曲線データセット
            curve_color = options.get("color", "rgba(66, 133, 244, 0.7)")

            # 信頼区間の処理
            show_ci = options.get("show_confidence_interval", True)
            if show_ci and "lower_ci" in survival_data and "upper_ci" in survival_data:
                datasets.append({
                    "label": "生存率",
                    "data": list(zip(survival_data["time"], survival_data["survival"])),
                    "borderColor": curve_color,
                    "backgroundColor": "transparent",
                    "pointRadius": options.get("point_radius", 0),
                    "fill": False
                })

                # 信頼区間の領域データセット
                datasets.append({
                    "label": "信頼区間",
                    "data": self._create_confidence_band(
                        survival_data["time"],
                        survival_data["lower_ci"],
                        survival_data["upper_ci"]
                    ),
                    "backgroundColor": self._get_transparent_color(curve_color, 0.2),
                    "borderColor": "transparent",
                    "pointRadius": 0,
                    "fill": True,
                    "showInLegend": False
                })
            else:
                # 信頼区間なしの場合
                datasets.append({
                    "label": "生存率",
                    "data": list(zip(survival_data["time"], survival_data["survival"])),
                    "borderColor": curve_color,
                    "backgroundColor": "transparent",
                    "pointRadius": options.get("point_radius", 0),
                    "fill": False
                })

        # チャート設定
        chart_config = {
            "chart_type": "line",
            "title": options.get("title", "生存曲線"),
            "subtitle": options.get("subtitle", "カプランマイヤー推定による生存率"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "x_axis_label": options.get("x_axis_label", "時間"),
            "y_axis_label": options.get("y_axis_label", "生存率"),
            "x_min": 0,
            "y_min": 0,
            "y_max": 1,
            "show_grid": options.get("show_grid", True),
            "show_legend": options.get("show_legend", True),
            "tooltip_format": "時間: {x}, 生存率: {y:.3f}"
        }

        return {"config": chart_config, "data": {"datasets": datasets}}

    def _prepare_cumulative_hazard_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """累積ハザード曲線データを準備する"""
        if "cumulative_hazard" not in analysis_results:
            logger.warning("累積ハザードデータが分析結果に含まれていません")
            return {
                "config": {
                    "chart_type": "line",
                    "title": options.get("title", "累積ハザード曲線"),
                    "width": options.get("width", 800),
                    "height": options.get("height", 500),
                    "x_axis_label": "時間",
                    "y_axis_label": "累積ハザード"
                },
                "data": {
                    "labels": [],
                    "datasets": []
                }
            }

        hazard_data = analysis_results["cumulative_hazard"]
        datasets = []

        # グループが存在する場合
        if "groups" in hazard_data:
            for group_name, group_data in hazard_data["groups"].items():
                if "time" not in group_data or "hazard" not in group_data:
                    continue

                # 主要な累積ハザード曲線データセット
                group_color = self._get_color_for_group(group_name, options)

                # 信頼区間の処理
                show_ci = options.get("show_confidence_interval", True)
                if show_ci and "lower_ci" in group_data and "upper_ci" in group_data:
                    datasets.append({
                        "label": f"{group_name}",
                        "data": list(zip(group_data["time"], group_data["hazard"])),
                        "borderColor": group_color,
                        "backgroundColor": "transparent",
                        "pointRadius": options.get("point_radius", 0),
                        "fill": False
                    })

                    # 信頼区間の領域データセット
                    datasets.append({
                        "label": f"{group_name} 信頼区間",
                        "data": self._create_confidence_band(
                            group_data["time"],
                            group_data["lower_ci"],
                            group_data["upper_ci"]
                        ),
                        "backgroundColor": self._get_transparent_color(group_color, 0.2),
                        "borderColor": "transparent",
                        "pointRadius": 0,
                        "fill": True,
                        "showInLegend": False
                    })
                else:
                    # 信頼区間なしの場合
                    datasets.append({
                        "label": f"{group_name}",
                        "data": list(zip(group_data["time"], group_data["hazard"])),
                        "borderColor": group_color,
                        "backgroundColor": "transparent",
                        "pointRadius": options.get("point_radius", 0),
                        "fill": False
                    })
        else:
            # グループなしの場合（単一の累積ハザード曲線）
            if "time" not in hazard_data or "hazard" not in hazard_data:
                logger.warning("累積ハザードデータの必要なフィールドが不足しています")
                return {
                    "config": {
                        "chart_type": "line",
                        "title": options.get("title", "累積ハザード曲線"),
                        "width": options.get("width", 800),
                        "height": options.get("height", 500),
                        "x_axis_label": "時間",
                        "y_axis_label": "累積ハザード"
                    },
                    "data": {
                        "labels": [],
                        "datasets": []
                    }
                }

            # 主要な累積ハザード曲線データセット
            curve_color = options.get("color", "rgba(219, 68, 55, 0.7)")

            # 信頼区間の処理
            show_ci = options.get("show_confidence_interval", True)
            if show_ci and "lower_ci" in hazard_data and "upper_ci" in hazard_data:
                datasets.append({
                    "label": "累積ハザード",
                    "data": list(zip(hazard_data["time"], hazard_data["hazard"])),
                    "borderColor": curve_color,
                    "backgroundColor": "transparent",
                    "pointRadius": options.get("point_radius", 0),
                    "fill": False
                })

                # 信頼区間の領域データセット
                datasets.append({
                    "label": "信頼区間",
                    "data": self._create_confidence_band(
                        hazard_data["time"],
                        hazard_data["lower_ci"],
                        hazard_data["upper_ci"]
                    ),
                    "backgroundColor": self._get_transparent_color(curve_color, 0.2),
                    "borderColor": "transparent",
                    "pointRadius": 0,
                    "fill": True,
                    "showInLegend": False
                })
            else:
                # 信頼区間なしの場合
                datasets.append({
                    "label": "累積ハザード",
                    "data": list(zip(hazard_data["time"], hazard_data["hazard"])),
                    "borderColor": curve_color,
                    "backgroundColor": "transparent",
                    "pointRadius": options.get("point_radius", 0),
                    "fill": False
                })

        # チャート設定
        chart_config = {
            "chart_type": "line",
            "title": options.get("title", "累積ハザード曲線"),
            "subtitle": options.get("subtitle", "時間経過に伴うリスク累積"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "x_axis_label": options.get("x_axis_label", "時間"),
            "y_axis_label": options.get("y_axis_label", "累積ハザード"),
            "x_min": 0,
            "y_min": 0,
            "show_grid": options.get("show_grid", True),
            "show_legend": options.get("show_legend", True),
            "tooltip_format": "時間: {x}, 累積ハザード: {y:.3f}"
        }

        return {"config": chart_config, "data": {"datasets": datasets}}

    def _prepare_log_log_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """ログログ曲線データを準備する"""
        if "log_log" not in analysis_results:
            logger.warning("ログログ曲線データが分析結果に含まれていません")
            return {
                "config": {
                    "chart_type": "line",
                    "title": options.get("title", "ログログ曲線"),
                    "width": options.get("width", 800),
                    "height": options.get("height", 500),
                    "x_axis_label": "log(時間)",
                    "y_axis_label": "log(-log(生存率))"
                },
                "data": {
                    "labels": [],
                    "datasets": []
                }
            }

        log_log_data = analysis_results["log_log"]
        datasets = []

        # グループが存在する場合
        if "groups" in log_log_data:
            for group_name, group_data in log_log_data["groups"].items():
                if "log_time" not in group_data or "log_log_survival" not in group_data:
                    continue

                # グループ色の取得
                group_color = self._get_color_for_group(group_name, options)

                # データセットの追加
                datasets.append({
                    "label": f"{group_name}",
                    "data": list(zip(group_data["log_time"], group_data["log_log_survival"])),
                    "borderColor": group_color,
                    "backgroundColor": "transparent",
                    "pointRadius": options.get("point_radius", 1),
                    "fill": False
                })
        else:
            # グループなしの場合（単一のログログ曲線）
            if "log_time" not in log_log_data or "log_log_survival" not in log_log_data:
                logger.warning("ログログ曲線データの必要なフィールドが不足しています")
                return {
                    "config": {
                        "chart_type": "line",
                        "title": options.get("title", "ログログ曲線"),
                        "width": options.get("width", 800),
                        "height": options.get("height", 500),
                        "x_axis_label": "log(時間)",
                        "y_axis_label": "log(-log(生存率))"
                    },
                    "data": {
                        "labels": [],
                        "datasets": []
                    }
                }

            # 主要なログログ曲線データセット
            curve_color = options.get("color", "rgba(15, 157, 88, 0.7)")

            datasets.append({
                "label": "ログログ曲線",
                "data": list(zip(log_log_data["log_time"], log_log_data["log_log_survival"])),
                "borderColor": curve_color,
                "backgroundColor": "transparent",
                "pointRadius": options.get("point_radius", 1),
                "fill": False
            })

        # チャート設定
        chart_config = {
            "chart_type": "line",
            "title": options.get("title", "ログログ曲線"),
            "subtitle": options.get("subtitle", "比例ハザード仮定の評価"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "x_axis_label": options.get("x_axis_label", "log(時間)"),
            "y_axis_label": options.get("y_axis_label", "log(-log(生存率))"),
            "show_grid": options.get("show_grid", True),
            "show_legend": options.get("show_legend", True),
            "tooltip_format": "log(時間): {x:.3f}, log(-log(生存率)): {y:.3f}"
        }

        return {"config": chart_config, "data": {"datasets": datasets}}

    def _prepare_hazard_ratio_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """ハザード比データを準備する"""
        if "hazard_ratio" not in analysis_results:
            logger.warning("ハザード比データが分析結果に含まれていません")
            return {
                "config": {
                    "chart_type": "bar",
                    "title": options.get("title", "ハザード比"),
                    "width": options.get("width", 800),
                    "height": options.get("height", 500),
                    "x_axis_label": "変数",
                    "y_axis_label": "ハザード比"
                },
                "data": {
                    "labels": [],
                    "datasets": [{
                        "label": "ハザード比",
                        "data": []
                    }]
                }
            }

        hazard_ratio_data = analysis_results["hazard_ratio"]

        # 変数と対応するハザード比を抽出
        variables = hazard_ratio_data.get("variables", [])
        ratios = hazard_ratio_data.get("ratios", [])
        lower_ci = hazard_ratio_data.get("lower_ci", [])
        upper_ci = hazard_ratio_data.get("upper_ci", [])
        p_values = hazard_ratio_data.get("p_values", [])

        if not variables or not ratios or len(variables) != len(ratios):
            logger.warning("ハザード比データの形式が正しくありません")
            return {
                "config": {
                    "chart_type": "bar",
                    "title": options.get("title", "ハザード比"),
                    "width": options.get("width", 800),
                    "height": options.get("height", 500),
                    "x_axis_label": "変数",
                    "y_axis_label": "ハザード比"
                },
                "data": {
                    "labels": [],
                    "datasets": [{
                        "label": "ハザード比",
                        "data": []
                    }]
                }
            }

        # 誤差範囲データの準備（信頼区間）
        error_bars = []
        if lower_ci and upper_ci and len(lower_ci) == len(ratios) and len(upper_ci) == len(ratios):
            for i in range(len(ratios)):
                error_bars.append({
                    "plus": upper_ci[i] - ratios[i],
                    "minus": ratios[i] - lower_ci[i]
                })

        # 有意性に基づく色分け
        colors = []
        if p_values and len(p_values) == len(ratios):
            for p in p_values:
                if p < 0.01:
                    colors.append("rgba(219, 68, 55, 0.7)")  # 非常に有意 (p < 0.01)
                elif p < 0.05:
                    colors.append("rgba(244, 180, 0, 0.7)")  # 有意 (p < 0.05)
                else:
                    colors.append("rgba(66, 133, 244, 0.7)")  # 非有意
        else:
            colors = ["rgba(66, 133, 244, 0.7)" for _ in ratios]

        # チャートデータ
        chart_data = {
            "labels": variables,
            "datasets": [{
                "label": "ハザード比",
                "data": ratios,
                "backgroundColor": colors,
                "error_bars": error_bars if error_bars else None
            }]
        }

        # ログスケールと基準線（HR=1）の設定
        use_log_scale = options.get("log_scale", True)
        show_baseline = options.get("show_baseline", True)

        # チャート設定
        chart_config = {
            "chart_type": "bar",
            "title": options.get("title", "ハザード比"),
            "subtitle": options.get("subtitle", "Coxハザード分析結果"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "x_axis_label": options.get("x_axis_label", "変数"),
            "y_axis_label": options.get("y_axis_label", "ハザード比"),
            "y_log_scale": use_log_scale,
            "show_grid": options.get("show_grid", True),
            "show_legend": options.get("show_legend", False),
            "horizontal": options.get("horizontal", True),
            "baseline": 1 if show_baseline else None,
            "tooltip_format": "変数: {x}, ハザード比: {y:.3f}"
        }

        return {"config": chart_config, "data": chart_data}

    def format_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生存分析結果のサマリーをフォーマットする

        Args:
            analysis_results: 生存分析結果

        Returns:
            フォーマット済みサマリー
        """
        # 分析タイプの取得
        model_info = analysis_results.get("model_info", {})
        model_type = model_info.get("model_type", "不明")

        # 結果がない場合
        if not model_info:
            return {
                "message": "有効な生存分析データが見つかりませんでした"
            }

        # サンプル数情報
        sample_info = {}
        if "samples" in analysis_results:
            samples = analysis_results["samples"]
            total_samples = samples.get("total", 0)
            events = samples.get("events", 0)
            censored = samples.get("censored", 0)

            sample_info = {
                "total_samples": total_samples,
                "events": events,
                "censored": censored,
                "censoring_rate": round(censored / total_samples * 100, 1) if total_samples > 0 else 0
            }

        # グループ情報（存在する場合）
        group_info = {}
        if "groups" in analysis_results:
            groups = analysis_results["groups"]
            for group_name, group_data in groups.items():
                group_samples = group_data.get("samples", 0)
                group_events = group_data.get("events", 0)
                group_censored = group_data.get("censored", 0)

                group_info[group_name] = {
                    "samples": group_samples,
                    "events": group_events,
                    "censored": group_censored,
                    "censoring_rate": round(group_censored / group_samples * 100, 1) if group_samples > 0 else 0
                }

        # 統計的検定結果（グループ間比較）
        test_results = {}
        if "test_results" in analysis_results:
            tests = analysis_results["test_results"]

            # ログランク検定
            if "logrank" in tests:
                logrank = tests["logrank"]
                test_results["logrank"] = {
                    "statistic": logrank.get("statistic", 0),
                    "p_value": logrank.get("p_value", 1),
                    "significant": logrank.get("p_value", 1) < 0.05
                }

            # その他の検定（必要に応じて追加）

        # Coxモデル結果（存在する場合）
        cox_model = {}
        if "cox_model" in analysis_results:
            cox_data = analysis_results["cox_model"]

            # モデル適合度
            if "model_fit" in cox_data:
                fit = cox_data["model_fit"]
                cox_model["fit"] = {
                    "log_likelihood": fit.get("log_likelihood", 0),
                    "aic": fit.get("aic", 0),
                    "concordance": fit.get("concordance", 0)
                }

            # 共変量情報
            if "hazard_ratio" in analysis_results:
                hazard_ratio = analysis_results["hazard_ratio"]
                variables = hazard_ratio.get("variables", [])
                ratios = hazard_ratio.get("ratios", [])
                lower_ci = hazard_ratio.get("lower_ci", [])
                upper_ci = hazard_ratio.get("upper_ci", [])
                p_values = hazard_ratio.get("p_values", [])

                covariates = []
                for i in range(len(variables)):
                    if i < len(ratios):
                        covariate = {
                            "variable": variables[i],
                            "hazard_ratio": ratios[i]
                        }

                        if i < len(lower_ci) and i < len(upper_ci):
                            covariate["confidence_interval"] = [lower_ci[i], upper_ci[i]]

                        if i < len(p_values):
                            covariate["p_value"] = p_values[i]
                            covariate["significant"] = p_values[i] < 0.05

                        covariates.append(covariate)

                cox_model["covariates"] = covariates

        # 中央生存時間（存在する場合）
        survival_metrics = {}
        if "survival_metrics" in analysis_results:
            metrics = analysis_results["survival_metrics"]

            # 全体の中央生存時間
            if "median_survival" in metrics:
                survival_metrics["median_survival"] = metrics["median_survival"]

            # グループごとの中央生存時間
            if "group_median_survival" in metrics:
                survival_metrics["group_median_survival"] = metrics["group_median_survival"]

        # 総合サマリー
        return {
            "model_type": model_type,
            "sample_info": sample_info,
            "group_info": group_info,
            "test_results": test_results,
            "cox_model": cox_model,
            "survival_metrics": survival_metrics
        }

    def _create_confidence_band(self, times: List[float], lower_ci: List[float], upper_ci: List[float]) -> List[Dict[str, float]]:
        """信頼区間バンドのデータを作成する"""
        band_data = []

        # 上昇曲線のポイント（時間の昇順）
        for i in range(len(times)):
            band_data.append({"x": times[i], "y": lower_ci[i]})

        # 下降曲線のポイント（時間の降順）
        for i in range(len(times) - 1, -1, -1):
            band_data.append({"x": times[i], "y": upper_ci[i]})

        return band_data

    def _get_color_for_group(self, group_name: str, options: Dict[str, Any]) -> str:
        """グループ名に基づいて色を返す"""
        group_colors = options.get("group_colors", {})
        if group_name in group_colors:
            return group_colors[group_name]

        # グループ名に基づいてハッシュ値を計算し、色のインデックスを決定
        group_hash = sum(ord(c) for c in group_name)
        color_index = group_hash % len(self._get_color_palette())

        return self._get_color_palette()[color_index]

    def _get_color_palette(self) -> List[str]:
        """色パレットを返す"""
        return [
            "rgba(66, 133, 244, 0.7)",   # Google Blue
            "rgba(219, 68, 55, 0.7)",    # Google Red
            "rgba(244, 180, 0, 0.7)",    # Google Yellow
            "rgba(15, 157, 88, 0.7)",    # Google Green
            "rgba(171, 71, 188, 0.7)",   # Purple
            "rgba(255, 112, 67, 0.7)",   # Deep Orange
            "rgba(0, 172, 193, 0.7)",    # Cyan
            "rgba(124, 179, 66, 0.7)",   # Light Green
        ]

    def _get_transparent_color(self, color: str, alpha: float) -> str:
        """色の透明度を変更する"""
        if "rgba" in color:
            # 既存のRGBA値から透明度だけを変更
            rgba_parts = color.replace("rgba(", "").replace(")", "").split(",")
            if len(rgba_parts) >= 4:
                rgba_parts[3] = str(alpha)
                return f"rgba({','.join(rgba_parts)})"
        elif "rgb" in color:
            # RGBからRGBAへ変換
            rgb_parts = color.replace("rgb(", "").replace(")", "").split(",")
            return f"rgba({','.join(rgb_parts)}, {alpha})"

        # デフォルト値を返す
        return f"rgba(66, 133, 244, {alpha})"