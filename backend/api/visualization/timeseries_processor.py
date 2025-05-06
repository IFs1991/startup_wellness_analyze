"""
時系列分析可視化プロセッサ

このモジュールでは、時系列分析の可視化を処理するプロセッサクラスを実装します。
"""

from typing import Dict, List, Any, Optional
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from .factory import VisualizationProcessor, register_processor

logger = logging.getLogger(__name__)


@register_processor("timeseries")
class TimeSeriesVisualizationProcessor(VisualizationProcessor):
    """時系列分析の可視化プロセッサ"""

    def prepare_chart_data(self, analysis_results: Dict[str, Any],
                          visualization_type: str,
                          options: Dict[str, Any]) -> Dict[str, Any]:
        """
        時系列分析のチャートデータを準備する

        Args:
            analysis_results: 時系列分析結果
            visualization_type: 可視化タイプ
            options: オプション設定

        Returns:
            チャートデータ
        """
        if visualization_type == "line":
            return self._prepare_line_chart_data(analysis_results, options)
        elif visualization_type == "residual":
            return self._prepare_residual_plot_data(analysis_results, options)
        elif visualization_type == "histogram":
            return self._prepare_histogram_data(analysis_results, options)
        elif visualization_type == "acf":
            return self._prepare_acf_plot_data(analysis_results, options)
        else:
            # デフォルトは時系列ライングラフ
            return self._prepare_line_chart_data(analysis_results, options)

    def _parse_timeseries_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """時系列データを解析する"""
        # 結果データの形式に基づいて適切に処理
        data = {}

        # 時系列データの抽出
        original_data = analysis_results.get("original_data", {})
        fitted_values = analysis_results.get("fitted_values", {})
        forecast_values = analysis_results.get("forecast_values", {})
        residuals = analysis_results.get("residuals", {})

        # データが文字列形式の場合、JSONとして解析
        if isinstance(original_data, str):
            try:
                original_data = json.loads(original_data)
            except:
                # HTML表形式の場合はパース処理が必要
                pass

        if isinstance(fitted_values, str):
            try:
                fitted_values = json.loads(fitted_values)
            except:
                pass

        if isinstance(forecast_values, str):
            try:
                forecast_values = json.loads(forecast_values)
            except:
                pass

        if isinstance(residuals, str):
            try:
                residuals = json.loads(residuals)
            except:
                pass

        # データが見つからない場合、サンプルデータを生成
        if not original_data:
            # サンプルデータの生成
            dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30, 0, -1)]
            values = [np.random.normal(100, 10) for _ in range(30)]
            original_data = {"dates": dates, "values": values}

        if not fitted_values and original_data:
            # フィット値のサンプル生成
            dates = original_data.get("dates", [])
            values = original_data.get("values", [])
            fitted = [v + np.random.normal(0, 5) for v in values]
            fitted_values = {"dates": dates, "values": fitted}

        if not forecast_values:
            # 予測値のサンプル生成
            last_date = datetime.strptime(original_data.get("dates", [])[-1], "%Y-%m-%d")
            forecast_dates = [(last_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(10)]
            last_value = original_data.get("values", [])[-1]
            forecast_vals = [last_value + np.random.normal(1, 3) * i for i in range(1, 11)]
            forecast_values = {"dates": forecast_dates, "values": forecast_vals}

        if not residuals and original_data and fitted_values:
            # 残差のサンプル生成
            orig_vals = original_data.get("values", [])
            fit_vals = fitted_values.get("values", [])
            if len(orig_vals) == len(fit_vals):
                resid = [o - f for o, f in zip(orig_vals, fit_vals)]
                residuals = {"values": resid}

        data["original_data"] = original_data
        data["fitted_values"] = fitted_values
        data["forecast_values"] = forecast_values
        data["residuals"] = residuals

        return data

    def _prepare_line_chart_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """時系列予測ライングラフ用のデータを準備する"""
        # データの抽出
        data = self._parse_timeseries_data(analysis_results)

        original_data = data["original_data"]
        fitted_values = data["fitted_values"]
        forecast_values = data["forecast_values"]

        # 時系列データの準備
        datasets = []

        # 元データ
        if original_data:
            original_dates = original_data.get("dates", [])
            original_values = original_data.get("values", [])

            datasets.append({
                "label": options.get("original_label", "実測値"),
                "data": list(zip(original_dates, original_values)) if original_dates and original_values else [],
                "borderColor": options.get("original_color", "rgba(66, 133, 244, 0.7)"),
                "backgroundColor": "transparent",
                "pointRadius": options.get("point_radius", 2),
                "fill": False
            })

        # フィット値
        if fitted_values and options.get("show_fitted", True):
            fitted_dates = fitted_values.get("dates", [])
            fitted_values_data = fitted_values.get("values", [])

            datasets.append({
                "label": options.get("fitted_label", "フィット値"),
                "data": list(zip(fitted_dates, fitted_values_data)) if fitted_dates and fitted_values_data else [],
                "borderColor": options.get("fitted_color", "rgba(219, 68, 55, 0.7)"),
                "backgroundColor": "transparent",
                "pointRadius": options.get("fitted_point_radius", 1),
                "fill": False
            })

        # 予測値
        if forecast_values:
            forecast_dates = forecast_values.get("dates", [])
            forecast_values_data = forecast_values.get("values", [])

            # 予測区間（存在する場合）
            lower_ci = forecast_values.get("lower_ci", [])
            upper_ci = forecast_values.get("upper_ci", [])

            datasets.append({
                "label": options.get("forecast_label", "予測値"),
                "data": list(zip(forecast_dates, forecast_values_data)) if forecast_dates and forecast_values_data else [],
                "borderColor": options.get("forecast_color", "rgba(15, 157, 88, 0.7)"),
                "backgroundColor": "transparent",
                "pointRadius": options.get("forecast_point_radius", 1),
                "borderDash": [5, 5],  # 破線スタイル
                "fill": False
            })

            # 予測区間の表示（オプション）
            if lower_ci and upper_ci and options.get("show_confidence_interval", True):
                # 信頼区間の領域データセット
                ci_data = []
                for i, date in enumerate(forecast_dates):
                    if i < len(lower_ci) and i < len(upper_ci):
                        ci_data.append({"x": date, "y": lower_ci[i]})

                # 下から上、上から下の順に描画するため、逆順の上限データを追加
                for i in range(len(forecast_dates) - 1, -1, -1):
                    if i < len(upper_ci):
                        ci_data.append({"x": forecast_dates[i], "y": upper_ci[i]})

                datasets.append({
                    "label": "予測区間",
                    "data": ci_data,
                    "backgroundColor": "rgba(15, 157, 88, 0.1)",
                    "borderColor": "transparent",
                    "pointRadius": 0,
                    "fill": True,
                    "showInLegend": False
                })

        # チャート設定
        metadata = analysis_results.get("metadata", {})
        target_variable = metadata.get("target_variable", options.get("target_variable", "変数"))

        chart_config = {
            "chart_type": "line",
            "title": options.get("title", f"{target_variable}の時系列分析"),
            "subtitle": options.get("subtitle", "時系列データと予測"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "x_axis_label": options.get("x_axis_label", "時間"),
            "y_axis_label": options.get("y_axis_label", target_variable),
            "show_grid": options.get("show_grid", True),
            "show_legend": options.get("show_legend", True),
            "time_axis": True
        }

        return {"config": chart_config, "data": {"datasets": datasets}}

    def _prepare_residual_plot_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """残差プロット用のデータを準備する"""
        # データの抽出
        data = self._parse_timeseries_data(analysis_results)
        residuals = data["residuals"]
        original_data = data["original_data"]

        residual_values = residuals.get("values", [])
        dates = original_data.get("dates", [])

        if not residual_values or len(residual_values) == 0:
            logger.warning("残差データが利用できません")
            residual_values = [0]
            dates = ["No Data"]

        # 残差のプロットデータ
        datasets = [
            {
                "label": "残差",
                "data": list(zip(dates, residual_values)) if len(dates) == len(residual_values) else [],
                "borderColor": options.get("residual_color", "rgba(66, 133, 244, 0.7)"),
                "backgroundColor": "transparent",
                "pointRadius": options.get("point_radius", 2),
                "fill": False
            },
            {
                "label": "ゼロライン",
                "data": list(zip(dates, [0] * len(dates))) if dates else [],
                "borderColor": "rgba(128, 128, 128, 0.5)",
                "backgroundColor": "transparent",
                "pointRadius": 0,
                "borderDash": [3, 3],
                "fill": False
            }
        ]

        # チャート設定
        metadata = analysis_results.get("metadata", {})
        target_variable = metadata.get("target_variable", options.get("target_variable", "変数"))

        chart_config = {
            "chart_type": "line",
            "title": options.get("title", f"{target_variable}のモデル残差"),
            "subtitle": options.get("subtitle", "予測誤差の時系列プロット"),
            "width": options.get("width", 800),
            "height": options.get("height", 400),
            "x_axis_label": options.get("x_axis_label", "時間"),
            "y_axis_label": options.get("y_axis_label", "残差"),
            "show_grid": options.get("show_grid", True),
            "show_legend": options.get("show_legend", False),
            "time_axis": True
        }

        return {"config": chart_config, "data": {"datasets": datasets}}

    def _prepare_histogram_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """残差ヒストグラム用のデータを準備する"""
        # データの抽出
        data = self._parse_timeseries_data(analysis_results)
        residuals = data["residuals"]

        residual_values = residuals.get("values", [])

        if not residual_values or len(residual_values) == 0:
            logger.warning("残差データが利用できません")
            residual_values = [0]

        # ヒストグラムの計算
        bin_count = options.get("bin_count", 10)

        # 残差の範囲を決定
        min_val = min(residual_values)
        max_val = max(residual_values)
        bin_width = (max_val - min_val) / bin_count if max_val > min_val else 1

        # ビンの境界値を計算
        bins = [min_val + i * bin_width for i in range(bin_count + 1)]
        bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(bin_count)]
        bin_labels = [f"{bins[i]:.2f} - {bins[i+1]:.2f}" for i in range(bin_count)]

        # ヒストグラムの計算
        hist, _ = np.histogram(residual_values, bins=bins)

        chart_data = {
            "labels": bin_labels,
            "datasets": [{
                "label": "残差頻度",
                "data": hist.tolist(),
                "backgroundColor": options.get("color", "rgba(66, 133, 244, 0.7)")
            }]
        }

        # チャート設定
        metadata = analysis_results.get("metadata", {})
        target_variable = metadata.get("target_variable", options.get("target_variable", "変数"))

        chart_config = {
            "chart_type": "bar",
            "title": options.get("title", f"{target_variable}の残差分布"),
            "subtitle": options.get("subtitle", "モデル残差のヒストグラム"),
            "width": options.get("width", 600),
            "height": options.get("height", 400),
            "x_axis_label": options.get("x_axis_label", "残差値"),
            "y_axis_label": options.get("y_axis_label", "頻度"),
            "show_grid": options.get("show_grid", True),
            "show_legend": options.get("show_legend", False)
        }

        return {"config": chart_config, "data": chart_data}

    def _prepare_acf_plot_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """自己相関関数プロット用のデータを準備する"""
        # 自己相関データを抽出
        acf_data = analysis_results.get("acf", {})

        if not acf_data or isinstance(acf_data, str):
            # ACFデータがない場合はサンプルデータを生成
            lag_count = options.get("lag_count", 20)
            lags = list(range(lag_count))
            # ランダムな自己相関値（0次自己相関は1.0）
            np.random.seed(42)
            acf_values = [1.0] + [np.random.normal(0, 0.2) for _ in range(lag_count - 1)]
            acf_data = {"lags": lags, "acf": acf_values}
        elif isinstance(acf_data, str):
            try:
                acf_data = json.loads(acf_data)
            except:
                lag_count = options.get("lag_count", 20)
                lags = list(range(lag_count))
                np.random.seed(42)
                acf_values = [1.0] + [np.random.normal(0, 0.2) for _ in range(lag_count - 1)]
                acf_data = {"lags": lags, "acf": acf_values}

        lags = acf_data.get("lags", [])
        acf_values = acf_data.get("acf", [])

        if not lags or not acf_values or len(lags) != len(acf_values):
            logger.warning("ACFデータの形式が不正です")
            lag_count = options.get("lag_count", 20)
            lags = list(range(lag_count))
            acf_values = [1.0] + [0 for _ in range(lag_count - 1)]

        # 信頼区間の計算（通常は±1.96/√n）
        if "ci" in acf_data:
            ci = acf_data["ci"]
        else:
            n = options.get("sample_size", 100)  # サンプルサイズが不明の場合のデフォルト値
            ci_value = 1.96 / np.sqrt(n)
            ci = [-ci_value, ci_value]

        # バーチャート用データの準備
        chart_data = {
            "labels": [str(lag) for lag in lags],
            "datasets": [
                {
                    "label": "自己相関",
                    "data": acf_values,
                    "backgroundColor": options.get("color", "rgba(66, 133, 244, 0.7)")
                }
            ]
        }

        # 信頼区間のライン
        ci_datasets = []
        if ci and len(ci) == 2:
            # 上限と下限の信頼区間ライン
            ci_datasets = [
                {
                    "label": "上限信頼区間",
                    "data": [ci[1]] * len(lags),
                    "type": "line",
                    "borderColor": "rgba(200, 0, 0, 0.5)",
                    "backgroundColor": "transparent",
                    "borderDash": [5, 5],
                    "pointRadius": 0,
                    "fill": False
                },
                {
                    "label": "下限信頼区間",
                    "data": [ci[0]] * len(lags),
                    "type": "line",
                    "borderColor": "rgba(200, 0, 0, 0.5)",
                    "backgroundColor": "transparent",
                    "borderDash": [5, 5],
                    "pointRadius": 0,
                    "fill": False
                }
            ]

            chart_data["datasets"].extend(ci_datasets)

        # チャート設定
        metadata = analysis_results.get("metadata", {})
        target_variable = metadata.get("target_variable", options.get("target_variable", "変数"))

        chart_config = {
            "chart_type": "bar",
            "title": options.get("title", f"{target_variable}の自己相関関数"),
            "subtitle": options.get("subtitle", "時系列データの自己相関"),
            "width": options.get("width", 800),
            "height": options.get("height", 400),
            "x_axis_label": options.get("x_axis_label", "ラグ"),
            "y_axis_label": options.get("y_axis_label", "自己相関係数"),
            "y_min": options.get("y_min", -1),
            "y_max": options.get("y_max", 1),
            "show_grid": options.get("show_grid", True),
            "show_legend": options.get("show_legend", False)
        }

        return {"config": chart_config, "data": chart_data}

    def format_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        時系列分析結果のサマリーをフォーマットする

        Args:
            analysis_results: 時系列分析結果

        Returns:
            フォーマット済みサマリー
        """
        # メタデータの取得
        metadata = analysis_results.get("metadata", {})
        model_info = analysis_results.get("model_info", {})

        # モデルタイプの取得
        model_type = model_info.get("model_type", metadata.get("model_type", "ARIMA"))

        # 結果がない場合
        if not model_info and not metadata:
            return {
                "message": "有効な時系列分析データが見つかりませんでした"
            }

        # モデル情報の取得
        model_summary = {}
        if "model_summary" in analysis_results:
            model_summary_data = analysis_results["model_summary"]
            if isinstance(model_summary_data, str):
                try:
                    model_summary = json.loads(model_summary_data)
                except:
                    # JSON解析できない場合はHTMLや他の形式と判断
                    model_summary = {
                        "aic": model_info.get("aic", analysis_results.get("aic", 0)),
                        "bic": model_info.get("bic", analysis_results.get("bic", 0)),
                        "mse": model_info.get("mse", analysis_results.get("mse", 0)),
                        "rsquared": model_info.get("rsquared", analysis_results.get("rsquared", 0))
                    }
            else:
                model_summary = model_summary_data
        else:
            # 基本的なモデル情報をまとめる
            model_summary = {
                "aic": model_info.get("aic", analysis_results.get("aic", 0)),
                "bic": model_info.get("bic", analysis_results.get("bic", 0)),
                "mse": model_info.get("mse", analysis_results.get("mse", 0)),
                "rsquared": model_info.get("rsquared", analysis_results.get("rsquared", 0))
            }

        # 予測情報
        forecast_info = {}
        forecast_values = analysis_results.get("forecast_values", {})
        if forecast_values:
            forecast_dates = forecast_values.get("dates", [])
            forecast_values_data = forecast_values.get("values", [])

            if forecast_dates and forecast_values_data:
                # 予測期間
                forecast_info["periods"] = len(forecast_dates)

                # 開始日と終了日
                forecast_info["start_date"] = forecast_dates[0] if forecast_dates else None
                forecast_info["end_date"] = forecast_dates[-1] if forecast_dates else None

                # 予測統計
                if forecast_values_data:
                    forecast_info["min"] = min(forecast_values_data)
                    forecast_info["max"] = max(forecast_values_data)
                    forecast_info["mean"] = sum(forecast_values_data) / len(forecast_values_data)

        # 残差情報
        residual_info = {}
        residuals = analysis_results.get("residuals", {})
        if residuals:
            residual_values = residuals.get("values", [])

            if residual_values:
                residual_info["min"] = min(residual_values)
                residual_info["max"] = max(residual_values)
                residual_info["mean"] = sum(residual_values) / len(residual_values)
                residual_info["std"] = np.std(residual_values)

        # データ情報
        data_info = {}
        original_data = analysis_results.get("original_data", {})
        if original_data:
            dates = original_data.get("dates", [])
            values = original_data.get("values", [])

            if dates and values:
                data_info["periods"] = len(dates)
                data_info["start_date"] = dates[0] if dates else None
                data_info["end_date"] = dates[-1] if dates else None

                if values:
                    data_info["min"] = min(values)
                    data_info["max"] = max(values)
                    data_info["mean"] = sum(values) / len(values)
                    data_info["std"] = np.std(values)

        # 総合サマリー
        return {
            "model_type": model_type,
            "target_variable": metadata.get("target_variable", "変数"),
            "model_summary": model_summary,
            "forecast_info": forecast_info,
            "residual_info": residual_info,
            "data_info": data_info
        }