"""
予測モデル可視化API

このモジュールは次のエンドポイントを提供します：
- POST /api/predictive/visualize - 予測モデル分析結果の可視化
- POST /api/predictive/analyze-and-visualize - データ分析と可視化を一度に実行
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import json

from api.auth import get_current_user
from api.models import User
from api.dependencies import get_visualization_service
from api.middleware import APIError, ValidationFailedError
from api.core.config import get_settings, Settings
from analysis.PredictiveModelAnalyzer import PredictiveModelAnalyzer
from service.bigquery.client import BigQueryService
from backend.repository.data_repository import DataRepository
from backend.analysis.predictive_model_analyzer import PredictiveModelAnalyzer as NewPredictiveModelAnalyzer
from backend.services.logging_service import LoggingService
from backend.common.exceptions import AnalysisError

logger = logging.getLogger(__name__)

# APIルーター定義
router = APIRouter(
    prefix="/api/predictive",
    tags=["predictive"],
    responses={404: {"description": "リソースが見つかりません"}},
)

# リクエスト・レスポンスモデル定義
class PredictiveModelParams(BaseModel):
    """予測モデルパラメータモデル"""
    target_column: str = Field(..., description="目標変数の列名")
    feature_columns: List[str] = Field(..., description="特徴量として使用する列名リスト")
    model_type: str = Field("random_forest", description="使用するモデルタイプ (random_forest, logistic_regression, etc.)")
    test_size: float = Field(0.2, description="テストデータの割合")
    random_state: Optional[int] = Field(None, description="乱数シード")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="モデルのハイパーパラメータ")

class PredictiveModelRequest(BaseModel):
    """予測モデル分析リクエストモデル"""
    dataset_id: str = Field(..., description="データセットID")
    params: PredictiveModelParams

class PredictiveVisualizationRequest(BaseModel):
    """予測モデル可視化リクエストモデル"""
    analysis_results: Dict[str, Any] = Field(..., description="予測モデル分析結果")
    visualization_type: str = Field(..., description="可視化タイプ (feature_importance, roc_curve, confusion_matrix, etc.)")
    chart_title: Optional[str] = Field(None, description="チャートタイトル")
    chart_description: Optional[str] = Field(None, description="チャート説明")

class PredictiveVisualizationResponse(BaseModel):
    """予測モデル可視化レスポンスモデル"""
    chart_data: Dict[str, Any] = Field(..., description="チャートデータ")
    chart_type: str = Field(..., description="チャートタイプ")
    summary: Dict[str, Any] = Field(..., description="分析結果の要約")

# カスタム例外定義
class PredictiveModelAnalysisError(APIError):
    """予測モデル分析エラー"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="PREDICTIVE_MODEL_ANALYSIS_ERROR",
            message=message,
            details=details
        )

class InvalidPredictiveModelDataError(ValidationFailedError):
    """無効な予測モデルデータエラー"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details)

# ヘルパー関数
def _prepare_chart_data_from_predictive(
    analysis_results: Dict[str, Any],
    visualization_type: str
) -> Dict[str, Any]:
    """
    予測モデル分析結果からチャートデータを準備する

    Args:
        analysis_results: 予測モデル分析結果
        visualization_type: 可視化タイプ

    Returns:
        チャートデータ
    """
    try:
        if visualization_type == "feature_importance":
            return _prepare_feature_importance_data(analysis_results)
        elif visualization_type == "roc_curve":
            return _prepare_roc_curve_data(analysis_results)
        elif visualization_type == "confusion_matrix":
            return _prepare_confusion_matrix_data(analysis_results)
        elif visualization_type == "learning_curve":
            return _prepare_learning_curve_data(analysis_results)
        elif visualization_type == "precision_recall_curve":
            return _prepare_precision_recall_curve_data(analysis_results)
        else:
            raise InvalidPredictiveModelDataError(
                message=f"サポートされていない可視化タイプ: {visualization_type}",
                details={"supported_types": ["feature_importance", "roc_curve", "confusion_matrix", "learning_curve", "precision_recall_curve"]}
            )
    except KeyError as e:
        raise InvalidPredictiveModelDataError(
            message=f"分析結果に必要なデータがありません: {str(e)}",
            details={"reason": "分析結果に必要なデータが見つかりません"}
        )
    except Exception as e:
        logger.error(f"チャートデータ準備中にエラーが発生しました: {str(e)}")
        raise InvalidPredictiveModelDataError(
            message=f"チャートデータ準備中にエラーが発生しました: {str(e)}",
            details={"reason": "チャートデータ準備中にエラーが発生しました"}
        )

def _prepare_feature_importance_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    特徴量重要度データの準備

    Args:
        analysis_results: 予測モデル分析結果

    Returns:
        特徴量重要度チャートデータ
    """
    if "feature_importance" not in analysis_results:
        raise InvalidPredictiveModelDataError(
            message="特徴量重要度データが分析結果に含まれていません",
            details={"reason": "特徴量重要度データが分析結果に含まれていません"}
        )

    feature_importance = analysis_results["feature_importance"]

    # 重要度でソート
    sorted_indices = sorted(range(len(feature_importance["importance"])),
                          key=lambda i: feature_importance["importance"][i])

    features = [feature_importance["features"][i] for i in sorted_indices]
    importance = [feature_importance["importance"][i] for i in sorted_indices]

    return {
        "chart": {
            "type": "bar"
        },
        "title": {
            "text": "特徴量重要度"
        },
        "xAxis": {
            "categories": features,
            "title": {
                "text": "特徴量"
            }
        },
        "yAxis": {
            "title": {
                "text": "重要度"
            }
        },
        "series": [{
            "name": "重要度",
            "data": importance
        }]
    }

def _prepare_roc_curve_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    ROC曲線データの準備

    Args:
        analysis_results: 予測モデル分析結果

    Returns:
        ROC曲線チャートデータ
    """
    if "roc_curve" not in analysis_results:
        raise InvalidPredictiveModelDataError(
            message="ROC曲線データが分析結果に含まれていません",
            details={"reason": "ROC曲線データが分析結果に含まれていません"}
        )

    roc_data = analysis_results["roc_curve"]

    # データポイントの作成
    roc_series_data = []
    for i in range(len(roc_data["fpr"])):
        roc_series_data.append({
            "x": roc_data["fpr"][i],
            "y": roc_data["tpr"][i]
        })

    # 対角線のデータ
    diagonal_data = [
        {"x": 0, "y": 0},
        {"x": 1, "y": 1}
    ]

    return {
        "chart": {
            "type": "line",
            "zoomType": "xy"
        },
        "title": {
            "text": f"ROC曲線 (AUC: {roc_data.get('auc', 0):.3f})"
        },
        "xAxis": {
            "title": {
                "text": "偽陽性率 (FPR)"
            },
            "min": 0,
            "max": 1
        },
        "yAxis": {
            "title": {
                "text": "真陽性率 (TPR)"
            },
            "min": 0,
            "max": 1
        },
        "tooltip": {
            "headerFormat": "",
            "pointFormat": "FPR: {point.x:.3f}, TPR: {point.y:.3f}"
        },
        "series": [
            {
                "name": "ROC曲線",
                "data": roc_series_data,
                "color": "#1f77b4"
            },
            {
                "name": "ランダム (AUC=0.5)",
                "data": diagonal_data,
                "dashStyle": "dash",
                "color": "#999999"
            }
        ]
    }

def _prepare_confusion_matrix_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    混同行列データの準備

    Args:
        analysis_results: 予測モデル分析結果

    Returns:
        混同行列チャートデータ
    """
    if "confusion_matrix" not in analysis_results:
        raise InvalidPredictiveModelDataError(
            message="混同行列データが分析結果に含まれていません",
            details={"reason": "混同行列データが分析結果に含まれていません"}
        )

    conf_matrix = analysis_results["confusion_matrix"]

    # クラスラベルの取得
    class_labels = conf_matrix.get("class_labels", ["クラス0", "クラス1"])

    # データポイントの作成
    matrix_data = []
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            matrix_data.append([j, i, conf_matrix["matrix"][i][j]])

    return {
        "chart": {
            "type": "heatmap"
        },
        "title": {
            "text": "混同行列"
        },
        "xAxis": {
            "categories": class_labels,
            "title": {
                "text": "予測クラス"
            }
        },
        "yAxis": {
            "categories": class_labels,
            "title": {
                "text": "実際のクラス"
            }
        },
        "colorAxis": {
            "min": 0
        },
        "tooltip": {
            "formatter": "function() { return '<b>予測:</b> ' + this.series.xAxis.categories[this.point.x] + '<br><b>実際:</b> ' + this.series.yAxis.categories[this.point.y] + '<br><b>件数:</b> ' + this.point.value; }"
        },
        "series": [{
            "name": "混同行列",
            "data": matrix_data,
            "dataLabels": {
                "enabled": True,
                "color": "#000000"
            }
        }]
    }

def _prepare_learning_curve_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    学習曲線データの準備

    Args:
        analysis_results: 予測モデル分析結果

    Returns:
        学習曲線チャートデータ
    """
    if "learning_curve" not in analysis_results:
        raise InvalidPredictiveModelDataError(
            message="学習曲線データが分析結果に含まれていません",
            details={"reason": "学習曲線データが分析結果に含まれていません"}
        )

    learning_curve = analysis_results["learning_curve"]

    # トレーニングスコアとテストスコアのデータ作成
    train_scores_data = []
    test_scores_data = []

    for i, size in enumerate(learning_curve["train_sizes"]):
        train_scores_data.append({
            "x": size,
            "y": learning_curve["train_scores"][i]
        })
        test_scores_data.append({
            "x": size,
            "y": learning_curve["test_scores"][i]
        })

    return {
        "chart": {
            "type": "line",
            "zoomType": "xy"
        },
        "title": {
            "text": "学習曲線"
        },
        "xAxis": {
            "title": {
                "text": "トレーニングサンプル数"
            }
        },
        "yAxis": {
            "title": {
                "text": "スコア"
            },
            "min": 0,
            "max": 1
        },
        "tooltip": {
            "valueDecimals": 3
        },
        "series": [
            {
                "name": "トレーニングスコア",
                "data": train_scores_data
            },
            {
                "name": "検証スコア",
                "data": test_scores_data
            }
        ]
    }

def _prepare_precision_recall_curve_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    適合率-再現率曲線データの準備

    Args:
        analysis_results: 予測モデル分析結果

    Returns:
        適合率-再現率曲線チャートデータ
    """
    if "precision_recall_curve" not in analysis_results:
        raise InvalidPredictiveModelDataError(
            message="適合率-再現率曲線データが分析結果に含まれていません",
            details={"reason": "適合率-再現率曲線データが分析結果に含まれていません"}
        )

    pr_data = analysis_results["precision_recall_curve"]

    # データポイントの作成
    pr_series_data = []
    for i in range(len(pr_data["precision"])):
        pr_series_data.append({
            "x": pr_data["recall"][i],
            "y": pr_data["precision"][i]
        })

    return {
        "chart": {
            "type": "line",
            "zoomType": "xy"
        },
        "title": {
            "text": f"適合率-再現率曲線 (AP: {pr_data.get('average_precision', 0):.3f})"
        },
        "xAxis": {
            "title": {
                "text": "再現率"
            },
            "min": 0,
            "max": 1
        },
        "yAxis": {
            "title": {
                "text": "適合率"
            },
            "min": 0,
            "max": 1
        },
        "tooltip": {
            "headerFormat": "",
            "pointFormat": "再現率: {point.x:.3f}, 適合率: {point.y:.3f}"
        },
        "series": [{
            "name": "適合率-再現率曲線",
            "data": pr_series_data
        }]
    }

def _generate_predictive_model_summary(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    予測モデル分析結果の要約を生成する

    Args:
        analysis_results: 予測モデル分析結果

    Returns:
        分析結果の要約
    """
    summary = {
        "model_info": {},
        "performance_metrics": {},
        "key_insights": []
    }

    # モデル情報
    if "model_info" in analysis_results:
        summary["model_info"] = analysis_results["model_info"]

    # 性能メトリクス
    if "metrics" in analysis_results:
        summary["performance_metrics"] = analysis_results["metrics"]

        # メトリクスからキー洞察を生成
        metrics = analysis_results["metrics"]
        if "accuracy" in metrics:
            summary["key_insights"].append(f"モデルの精度: {metrics['accuracy']:.3f}")

        if "f1_score" in metrics:
            summary["key_insights"].append(f"F1スコア: {metrics['f1_score']:.3f}")

        if "auc" in metrics:
            summary["key_insights"].append(f"ROC曲線下面積 (AUC): {metrics['auc']:.3f}")

    # 特徴量重要度
    if "feature_importance" in analysis_results:
        feature_importance = analysis_results["feature_importance"]
        if len(feature_importance["features"]) > 0:
            # 上位3つの重要な特徴量を特定
            sorted_indices = sorted(range(len(feature_importance["importance"])),
                                key=lambda i: feature_importance["importance"][i], reverse=True)

            top_features = []
            for i in range(min(3, len(sorted_indices))):
                idx = sorted_indices[i]
                top_features.append(f"{feature_importance['features'][idx]} ({feature_importance['importance'][idx]:.3f})")

            summary["key_insights"].append(f"最も重要な特徴量: {', '.join(top_features)}")

    return summary

# エンドポイント実装
@router.post("/visualize", response_model=PredictiveVisualizationResponse, status_code=status.HTTP_200_OK)
async def visualize_predictive_model(
    request: PredictiveVisualizationRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    予測モデル分析の結果を可視化します。

    既存の分析結果から指定された可視化タイプに基づいてチャートを生成します。
    """
    try:
        logger.info(f"予測モデル分析の可視化リクエスト受信: タイプ={request.visualization_type}")

        # 入力データの検証
        if not request.analysis_results:
            raise InvalidPredictiveModelDataError(
                message="無効な分析結果データです",
                details={"reason": "分析結果が空です"}
            )

        # チャートデータの準備
        chart_data = _prepare_chart_data_from_predictive(
            analysis_results=request.analysis_results,
            visualization_type=request.visualization_type
        )

        # チャート生成
        result = await visualization_service.generate_chart(
            config=chart_data["chart"],
            data=chart_data["data"],
            format=request.options.get("format", "png"),
            template_id=request.options.get("template_id"),
            user_id=str(current_user.id)
        )

        # 分析サマリーを追加
        result["summary"] = _generate_predictive_model_summary(request.analysis_results)

        return PredictiveVisualizationResponse(
            chart_data=result["chart"],
            chart_type=result["chart_type"],
            summary=result["summary"]
        )

    except InvalidPredictiveModelDataError as e:
        logger.error(f"予測モデルデータ検証エラー: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"予測モデル分析可視化中にエラー: {str(e)}")
        raise PredictiveModelAnalysisError(
            message=f"予測モデル分析の可視化中にエラーが発生しました: {str(e)}"
        )

@router.post("/analyze-and-visualize", response_model=PredictiveVisualizationResponse, status_code=status.HTTP_200_OK)
async def analyze_and_visualize_predictive_model(
    request: PredictiveModelRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """
    予測モデル分析と可視化を一度のリクエストで実行します。

    データのクエリから予測モデル分析を実行し、結果を可視化して返します。
    """
    try:
        logger.info("予測モデル分析と可視化のリクエスト受信")

        # パラメータの検証
        if not request.params.query:
            raise InvalidPredictiveModelDataError(
                message="クエリが指定されていません",
                details={"reason": "分析用のSQLクエリは必須です"}
            )

        if not request.params.target_column:
            raise InvalidPredictiveModelDataError(
                message="目標変数が指定されていません",
                details={"reason": "予測モデル分析には目標変数の指定が必須です"}
            )

        if not request.params.features:
            raise InvalidPredictiveModelDataError(
                message="特徴量が指定されていません",
                details={"reason": "予測モデル分析には少なくとも1つの特徴量が必要です"}
            )

        # BigQueryサービスとアナライザーの初期化
        bq_service = BigQueryService()
        analyzer = NewPredictiveModelAnalyzer(db=bq_service)

        try:
            # 分析対象データの取得
            query_result = await bq_service.execute_query(request.params.query)
            df = pd.DataFrame(query_result)

            if df.empty:
                raise InvalidPredictiveModelDataError(
                    message="クエリ結果が空です",
                    details={"reason": "指定されたクエリで取得したデータが空です"}
                )

            # 予測モデル分析の実行
            # 前処理
            X_train, X_test, y_train, y_test = analyzer.preprocess_data(
                data=df,
                target_column=request.params.target_column,
                numerical_features=request.params.features,
                test_size=request.params.test_size
            )

            # モデル学習
            model = analyzer.train_model(
                X_train=X_train,
                y_train=y_train,
                model_type=request.params.model_type
            )

            # 予測と評価
            y_pred = analyzer.predict(X_test)

            # 評価指標の計算
            metrics = analyzer.evaluate_model_performance(
                model=model,
                X_test=X_test,
                y_test=y_test,
                plot=False
            )

            # 特徴量重要度の取得
            feature_importance = analyzer.analyze_feature_importance(
                model=model,
                feature_names=request.params.features,
                plot=False
            )

            # 実測値と予測値をまとめる
            actual_vs_predicted = {
                "y_true": y_test.tolist() if hasattr(y_test, 'tolist') else y_test,
                "y_pred": y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred
            }

            # 分析結果をまとめる
            analysis_results = {
                "model": model,
                "metrics": metrics,
                "feature_importance": feature_importance.to_dict() if hasattr(feature_importance, 'to_dict') else feature_importance,
                "actual_vs_predicted": actual_vs_predicted,
                "metadata": {
                    "model_type": request.params.model_type,
                    "target_column": request.params.target_column,
                    "features": request.params.features,
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "cv_folds": request.params.cv_folds,
                    "task_type": "classification" if len(np.unique(y_train)) <= 10 else "regression",
                    "timestamp": datetime.now().isoformat()
                }
            }

            # 結果を保存（オプション）
            if request.dataset_id and request.table_id:
                # 結果のJSON化
                results_for_storage = {k: v for k, v in analysis_results.items() if k != 'model'}
                await bq_service.save_results(
                    results=results_for_storage,
                    dataset_id=request.dataset_id,
                    table_id=request.table_id
                )

            # 可視化タイプの決定
            visualization_type = request.visualization_type

            # チャートデータの準備
            chart_data = _prepare_chart_data_from_predictive(
                analysis_results=analysis_results,
                visualization_type=visualization_type
            )

            # チャート生成
            result = await visualization_service.generate_chart(
                config=chart_data["chart"],
                data=chart_data["data"],
                format=request.options.get("format", "png"),
                template_id=request.options.get("template_id"),
                user_id=str(current_user.id)
            )

            # 分析サマリーを追加
            result["summary"] = _generate_predictive_model_summary(analysis_results)

            return PredictiveVisualizationResponse(
                chart_data=result["chart"],
                chart_type=result["chart_type"],
                summary=result["summary"]
            )

        finally:
            # リソース解放
            analyzer.release_resources()

    except InvalidPredictiveModelDataError as e:
        logger.error(f"予測モデルデータ検証エラー: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"予測モデル分析と可視化中にエラー: {str(e)}")
        raise PredictiveModelAnalysisError(
            message=f"予測モデル分析と可視化の実行中にエラーが発生しました: {str(e)}"
        )