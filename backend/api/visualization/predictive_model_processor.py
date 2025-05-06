"""
予測モデル可視化プロセッサ

このモジュールでは、予測モデルの可視化を処理するプロセッサクラスを実装します。
"""

from typing import Dict, List, Any, Optional
import logging
import numpy as np
from .factory import VisualizationProcessor, register_processor

logger = logging.getLogger(__name__)


@register_processor("predictive_model")
class PredictiveModelVisualizationProcessor(VisualizationProcessor):
    """予測モデルの可視化プロセッサ"""

    def prepare_chart_data(self, analysis_results: Dict[str, Any],
                          visualization_type: str,
                          options: Dict[str, Any]) -> Dict[str, Any]:
        """
        予測モデルのチャートデータを準備する

        Args:
            analysis_results: 予測モデル分析結果
            visualization_type: 可視化タイプ
            options: オプション設定

        Returns:
            チャートデータ
        """
        if visualization_type == "feature_importance":
            return self._prepare_feature_importance_data(analysis_results, options)
        elif visualization_type == "roc_curve":
            return self._prepare_roc_curve_data(analysis_results, options)
        elif visualization_type == "confusion_matrix":
            return self._prepare_confusion_matrix_data(analysis_results, options)
        elif visualization_type == "learning_curve":
            return self._prepare_learning_curve_data(analysis_results, options)
        elif visualization_type == "precision_recall_curve":
            return self._prepare_precision_recall_curve_data(analysis_results, options)
        else:
            # デフォルトは特徴量重要度
            return self._prepare_feature_importance_data(analysis_results, options)

    def _prepare_feature_importance_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """特徴量重要度データを準備する"""
        if "feature_importance" not in analysis_results:
            logger.warning("特徴量重要度データが分析結果に含まれていません")
            return {
                "config": {
                    "chart_type": "bar",
                    "title": options.get("title", "特徴量重要度"),
                    "width": options.get("width", 800),
                    "height": options.get("height", 500),
                    "x_axis_label": "特徴量",
                    "y_axis_label": "重要度"
                },
                "data": {
                    "labels": [],
                    "datasets": [{
                        "label": "重要度",
                        "data": []
                    }]
                }
            }

        feature_importance = analysis_results["feature_importance"]

        # 重要度でソート
        features = feature_importance.get("features", [])
        importance = feature_importance.get("importance", [])

        if not features or not importance or len(features) != len(importance):
            logger.warning("特徴量と重要度のデータが不正です")
            return {
                "config": {
                    "chart_type": "bar",
                    "title": options.get("title", "特徴量重要度"),
                    "width": options.get("width", 800),
                    "height": options.get("height", 500),
                    "x_axis_label": "特徴量",
                    "y_axis_label": "重要度"
                },
                "data": {
                    "labels": [],
                    "datasets": [{
                        "label": "重要度",
                        "data": []
                    }]
                }
            }

        # 重要度順にソート
        if options.get("sort", True):
            sorted_pairs = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
            features, importance = zip(*sorted_pairs) if sorted_pairs else ([], [])

        # 上位N件のみ表示
        top_n = options.get("top_n")
        if top_n and isinstance(top_n, int) and 0 < top_n < len(features):
            features = features[:top_n]
            importance = importance[:top_n]

        chart_data = {
            "labels": features,
            "datasets": [{
                "label": "重要度",
                "data": importance,
                "backgroundColor": options.get("color", "rgba(66, 133, 244, 0.7)")
            }]
        }

        chart_config = {
            "chart_type": "bar",
            "title": options.get("title", "特徴量重要度"),
            "subtitle": options.get("subtitle", "モデル予測に対する各特徴量の影響度"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "x_axis_label": options.get("x_axis_label", "特徴量"),
            "y_axis_label": options.get("y_axis_label", "重要度"),
            "horizontal": options.get("horizontal", True),  # 横向きバーチャートがデフォルト
            "show_grid": options.get("show_grid", True),
            "show_legend": options.get("show_legend", False)
        }

        return {"config": chart_config, "data": chart_data}

    def _prepare_roc_curve_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """ROC曲線データを準備する"""
        if "roc_curve" not in analysis_results:
            logger.warning("ROC曲線データが分析結果に含まれていません")
            return {
                "config": {
                    "chart_type": "line",
                    "title": options.get("title", "ROC曲線"),
                    "width": options.get("width", 600),
                    "height": options.get("height", 600),
                    "x_axis_label": "偽陽性率 (FPR)",
                    "y_axis_label": "真陽性率 (TPR)"
                },
                "data": {
                    "labels": [0, 1],
                    "datasets": [
                        {
                            "label": "ROC曲線",
                            "data": [0, 1],
                            "fill": False,
                            "borderColor": "rgba(66, 133, 244, 0.7)"
                        },
                        {
                            "label": "ランダム予測",
                            "data": [0, 1],
                            "fill": False,
                            "borderColor": "rgba(200, 200, 200, 0.7)",
                            "borderDash": [5, 5]
                        }
                    ]
                }
            }

        roc_data = analysis_results["roc_curve"]
        fpr = roc_data.get("fpr", [0, 1])
        tpr = roc_data.get("tpr", [0, 1])
        auc = roc_data.get("auc", 0.5)

        # xy形式のデータポイントに変換
        roc_points = []
        for i in range(len(fpr)):
            roc_points.append(fpr[i])
            roc_points.append(tpr[i])

        # ランダム予測の対角線
        diagonal = [0, 0, 1, 1]  # [x1, y1, x2, y2]

        chart_data = {
            "labels": [f"{round(fpr[i], 2)}" for i in range(len(fpr))],
            "datasets": [
                {
                    "label": f"ROC曲線 (AUC: {auc:.3f})",
                    "data": list(zip(fpr, tpr)),
                    "fill": options.get("fill", False),
                    "backgroundColor": "rgba(66, 133, 244, 0.1)",
                    "borderColor": "rgba(66, 133, 244, 0.7)",
                    "pointRadius": options.get("point_radius", 0)
                },
                {
                    "label": "ランダム予測 (AUC: 0.5)",
                    "data": [(0, 0), (1, 1)],
                    "fill": False,
                    "borderColor": "rgba(200, 200, 200, 0.7)",
                    "borderDash": [5, 5],
                    "pointRadius": 0
                }
            ]
        }

        chart_config = {
            "chart_type": "scatter",
            "title": options.get("title", f"ROC曲線 (AUC: {auc:.3f})"),
            "subtitle": options.get("subtitle", "モデルの分類性能評価"),
            "width": options.get("width", 600),
            "height": options.get("height", 600),
            "x_axis_label": options.get("x_axis_label", "偽陽性率 (FPR)"),
            "y_axis_label": options.get("y_axis_label", "真陽性率 (TPR)"),
            "x_min": 0,
            "x_max": 1,
            "y_min": 0,
            "y_max": 1,
            "show_grid": options.get("show_grid", True),
            "show_legend": options.get("show_legend", True),
            "line_shape": "spline"
        }

        return {"config": chart_config, "data": chart_data}

    def _prepare_confusion_matrix_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """混同行列データを準備する"""
        if "confusion_matrix" not in analysis_results:
            logger.warning("混同行列データが分析結果に含まれていません")
            return {
                "config": {
                    "chart_type": "heatmap",
                    "title": options.get("title", "混同行列"),
                    "width": options.get("width", 600),
                    "height": options.get("height", 600)
                },
                "data": {
                    "matrix": [[0, 0], [0, 0]],
                    "x_labels": ["予測: 陰性", "予測: 陽性"],
                    "y_labels": ["実際: 陰性", "実際: 陽性"]
                }
            }

        conf_matrix = analysis_results["confusion_matrix"]
        matrix_data = conf_matrix.get("matrix", [[0, 0], [0, 0]])

        # ラベル名の取得
        class_names = conf_matrix.get("class_names", ["陰性", "陽性"])
        if len(class_names) != len(matrix_data):
            class_names = [f"クラス{i}" for i in range(len(matrix_data))]

        x_labels = [f"予測: {name}" for name in class_names]
        y_labels = [f"実際: {name}" for name in class_names]

        # パーセンテージ計算（オプション）
        if options.get("normalize", False):
            row_sums = [sum(row) for row in matrix_data]
            matrix_percent = []
            for i, row in enumerate(matrix_data):
                if row_sums[i] > 0:
                    matrix_percent.append([val / row_sums[i] * 100 for val in row])
                else:
                    matrix_percent.append([0] * len(row))
            display_matrix = matrix_percent
            value_suffix = "%"
        else:
            display_matrix = matrix_data
            value_suffix = ""

        chart_data = {
            "matrix": display_matrix,
            "x_labels": x_labels,
            "y_labels": y_labels,
            "value_suffix": value_suffix
        }

        chart_config = {
            "chart_type": "heatmap",
            "title": options.get("title", "混同行列"),
            "subtitle": options.get("subtitle", "モデルの予測と実際の分類の比較"),
            "width": options.get("width", 600),
            "height": options.get("height", 600),
            "color_scheme": options.get("color_scheme", "blues"),
            "show_values": options.get("show_values", True),
            "show_scale": options.get("show_scale", True)
        }

        return {"config": chart_config, "data": chart_data}

    def _prepare_learning_curve_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """学習曲線データを準備する"""
        if "learning_curve" not in analysis_results:
            logger.warning("学習曲線データが分析結果に含まれていません")
            return {
                "config": {
                    "chart_type": "line",
                    "title": options.get("title", "学習曲線"),
                    "width": options.get("width", 800),
                    "height": options.get("height", 500),
                    "x_axis_label": "訓練サンプル数",
                    "y_axis_label": "スコア"
                },
                "data": {
                    "labels": [],
                    "datasets": []
                }
            }

        learning_curve = analysis_results["learning_curve"]
        train_sizes = learning_curve.get("train_sizes", [])
        train_scores = learning_curve.get("train_scores", [])
        test_scores = learning_curve.get("test_scores", [])

        if not train_sizes or not train_scores or not test_scores:
            logger.warning("学習曲線の必要なデータが不足しています")
            return {
                "config": {
                    "chart_type": "line",
                    "title": options.get("title", "学習曲線"),
                    "width": options.get("width", 800),
                    "height": options.get("height", 500),
                    "x_axis_label": "訓練サンプル数",
                    "y_axis_label": "スコア"
                },
                "data": {
                    "labels": [],
                    "datasets": []
                }
            }

        chart_data = {
            "labels": [str(size) for size in train_sizes],
            "datasets": [
                {
                    "label": "訓練スコア",
                    "data": train_scores,
                    "fill": False,
                    "borderColor": "rgba(66, 133, 244, 0.7)",
                    "backgroundColor": "rgba(66, 133, 244, 0.1)"
                },
                {
                    "label": "検証スコア",
                    "data": test_scores,
                    "fill": False,
                    "borderColor": "rgba(219, 68, 55, 0.7)",
                    "backgroundColor": "rgba(219, 68, 55, 0.1)"
                }
            ]
        }

        chart_config = {
            "chart_type": "line",
            "title": options.get("title", "学習曲線"),
            "subtitle": options.get("subtitle", "訓練サンプル数とモデル性能の関係"),
            "width": options.get("width", 800),
            "height": options.get("height", 500),
            "x_axis_label": options.get("x_axis_label", "訓練サンプル数"),
            "y_axis_label": options.get("y_axis_label", "スコア"),
            "y_min": options.get("y_min", 0),
            "y_max": options.get("y_max", 1),
            "show_grid": options.get("show_grid", True),
            "show_legend": options.get("show_legend", True)
        }

        return {"config": chart_config, "data": chart_data}

    def _prepare_precision_recall_curve_data(self, analysis_results: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """精度-再現率曲線データを準備する"""
        if "precision_recall_curve" not in analysis_results:
            logger.warning("精度-再現率曲線データが分析結果に含まれていません")
            return {
                "config": {
                    "chart_type": "line",
                    "title": options.get("title", "精度-再現率曲線"),
                    "width": options.get("width", 600),
                    "height": options.get("height", 600),
                    "x_axis_label": "再現率",
                    "y_axis_label": "精度"
                },
                "data": {
                    "labels": [],
                    "datasets": []
                }
            }

        pr_curve = analysis_results["precision_recall_curve"]
        precision = pr_curve.get("precision", [])
        recall = pr_curve.get("recall", [])
        average_precision = pr_curve.get("average_precision", 0)

        if not precision or not recall:
            logger.warning("精度-再現率曲線の必要なデータが不足しています")
            return {
                "config": {
                    "chart_type": "line",
                    "title": options.get("title", "精度-再現率曲線"),
                    "width": options.get("width", 600),
                    "height": options.get("height", 600),
                    "x_axis_label": "再現率",
                    "y_axis_label": "精度"
                },
                "data": {
                    "labels": [],
                    "datasets": []
                }
            }

        chart_data = {
            "datasets": [
                {
                    "label": f"精度-再現率曲線 (AP: {average_precision:.3f})",
                    "data": list(zip(recall, precision)),
                    "fill": options.get("fill", True),
                    "backgroundColor": "rgba(66, 133, 244, 0.1)",
                    "borderColor": "rgba(66, 133, 244, 0.7)",
                    "pointRadius": options.get("point_radius", 0)
                }
            ]
        }

        chart_config = {
            "chart_type": "scatter",
            "title": options.get("title", f"精度-再現率曲線 (AP: {average_precision:.3f})"),
            "subtitle": options.get("subtitle", "分類閾値の異なる精度と再現率のトレードオフ"),
            "width": options.get("width", 600),
            "height": options.get("height", 600),
            "x_axis_label": options.get("x_axis_label", "再現率"),
            "y_axis_label": options.get("y_axis_label", "精度"),
            "x_min": 0,
            "x_max": 1,
            "y_min": 0,
            "y_max": 1,
            "show_grid": options.get("show_grid", True),
            "show_legend": options.get("show_legend", True),
            "line_shape": "spline"
        }

        return {"config": chart_config, "data": chart_data}

    def format_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        予測モデル分析結果のサマリーをフォーマットする

        Args:
            analysis_results: 予測モデル分析結果

        Returns:
            フォーマット済みサマリー
        """
        # モデル情報の取得
        model_info = analysis_results.get("model_info", {})
        model_type = model_info.get("model_type", "不明")

        # モデル評価指標の取得
        metrics = analysis_results.get("metrics", {})

        # 結果がない場合
        if not metrics:
            return {
                "message": "有効な予測モデル評価データが見つかりませんでした"
            }

        # 分類モデルの場合
        classification_metrics = {}
        if "accuracy" in metrics:
            classification_metrics = {
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1_score", 0),
                "auc": metrics.get("auc", 0),
                "confusion_matrix": analysis_results.get("confusion_matrix", {}).get("matrix", [])
            }

        # 回帰モデルの場合
        regression_metrics = {}
        if "mse" in metrics or "rmse" in metrics:
            regression_metrics = {
                "mse": metrics.get("mse", 0),
                "rmse": metrics.get("rmse", 0),
                "mae": metrics.get("mae", 0),
                "r2": metrics.get("r2", 0),
                "explained_variance": metrics.get("explained_variance", 0)
            }

        # 特徴量重要度の取得
        feature_importance = analysis_results.get("feature_importance", {})
        top_features = []

        if feature_importance and "features" in feature_importance and "importance" in feature_importance:
            features = feature_importance["features"]
            importance = feature_importance["importance"]

            # 重要度順にソート
            if features and importance and len(features) == len(importance):
                sorted_pairs = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
                # 上位5つの特徴量を取得
                top_features = [{"feature": f, "importance": i} for f, i in sorted_pairs[:5]]

        return {
            "model_type": model_type,
            "classification_metrics": classification_metrics,
            "regression_metrics": regression_metrics,
            "top_features": top_features,
            "training_info": {
                "train_samples": analysis_results.get("train_info", {}).get("train_samples", 0),
                "test_samples": analysis_results.get("train_info", {}).get("test_samples", 0),
                "features_count": analysis_results.get("train_info", {}).get("features_count", 0),
                "training_time": analysis_results.get("train_info", {}).get("training_time", 0)
            }
        }