"""
クラスタ分析可視化API

提供するエンドポイント:
- POST /api/cluster/visualize: クラスタ分析結果の可視化
- POST /api/cluster/analyze-and-visualize: データ分析と可視化を一度に実行
"""

import json
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator

from backend.api.auth import get_current_user
from backend.api.models import User
from backend.repository.data_repository import DataRepository
from backend.analysis.ClusterAnalyzer import DataAnalyzer
from backend.services.logging_service import LoggingService
from backend.common.exceptions import AnalysisError
from service.bigquery.client import BigQueryService

router = APIRouter(prefix="/api/cluster", tags=["cluster"])
logger = LoggingService.get_logger(__name__)

# モデル定義
class ClusterAnalysisParams(BaseModel):
    """クラスタ分析のパラメータ"""
    algorithm: str = Field("kmeans", description="使用するアルゴリズム (kmeans, dbscan, hierarchical)")
    n_clusters: int = Field(3, description="クラスタ数 (KMeansとHierarchicalの場合)")
    target_columns: Optional[List[str]] = Field(None, description="分析対象の列名リスト（指定しない場合は数値列すべて）")
    random_state: Optional[int] = Field(42, description="乱数シード")
    eps: Optional[float] = Field(0.5, description="DBSCANのイプシロンパラメータ")
    min_samples: Optional[int] = Field(5, description="DBSCANの最小サンプル数")
    linkage: Optional[str] = Field("ward", description="階層的クラスタリングの連結方法")
    standardize: bool = Field(True, description="データを標準化するかどうか")
    optimize_memory: bool = Field(True, description="メモリ使用量を最適化するかどうか")

class ClusterAnalysisRequest(BaseModel):
    """クラスタ分析リクエスト"""
    dataset_id: str = Field(..., description="データセットID")
    params: ClusterAnalysisParams = Field(..., description="分析パラメータ")

class ClusterVisualizationRequest(BaseModel):
    """クラスタ分析可視化リクエスト"""
    analysis_results: Dict[str, Any] = Field(..., description="クラスタ分析結果")
    visualization_type: str = Field(..., description="可視化タイプ (scatter, heatmap, silhouette, distribution)")
    chart_title: Optional[str] = Field(None, description="チャートタイトル")
    chart_description: Optional[str] = Field(None, description="チャート説明")
    target_features: Optional[List[str]] = Field(None, description="可視化対象の特徴量（最大2つ）")

class ClusterVisualizationResponse(BaseModel):
    """クラスタ分析可視化レスポンス"""
    chart_data: Dict[str, Any] = Field(..., description="チャートデータ")
    chart_type: str = Field(..., description="チャートタイプ")
    summary: Dict[str, Any] = Field(..., description="分析結果の要約")

# 例外定義
class ClusterAnalysisException(Exception):
    """クラスタ分析に関連するエラー"""
    pass

class VisualizationError(Exception):
    """可視化処理に関連するエラー"""
    pass

# ヘルパー関数
def _prepare_chart_data_from_cluster(
    analysis_results: Dict[str, Any],
    visualization_type: str,
    target_features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    クラスタ分析結果からチャートデータを準備する

    Args:
        analysis_results: クラスタ分析結果
        visualization_type: 可視化タイプ
        target_features: 可視化対象の特徴量

    Returns:
        チャートデータ
    """
    try:
        if visualization_type == "scatter":
            return _prepare_cluster_scatter_data(analysis_results, target_features)
        elif visualization_type == "heatmap":
            return _prepare_cluster_heatmap_data(analysis_results)
        elif visualization_type == "silhouette":
            return _prepare_silhouette_plot_data(analysis_results)
        elif visualization_type == "distribution":
            return _prepare_feature_distribution_data(analysis_results, target_features)
        else:
            raise VisualizationError(f"未対応の可視化タイプ: {visualization_type}")
    except KeyError as e:
        raise VisualizationError(f"分析結果に必要なデータがありません: {str(e)}")
    except Exception as e:
        logger.error(f"チャートデータ準備中にエラーが発生しました: {str(e)}")
        raise VisualizationError(f"チャートデータ準備中にエラーが発生しました: {str(e)}")

def _prepare_cluster_scatter_data(analysis_results: Dict[str, Any], target_features: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    クラスタスキャッタープロットデータの準備

    Args:
        analysis_results: クラスタ分析結果
        target_features: 可視化対象の特徴量

    Returns:
        スキャッタープロットチャートデータ
    """
    if "pca_coordinates" not in analysis_results and "data" not in analysis_results:
        raise VisualizationError("クラスタ座標データが分析結果に含まれていません")

    # クラスタラベルの取得
    if "cluster_labels" not in analysis_results:
        raise VisualizationError("クラスタラベルが分析結果に含まれていません")

    cluster_labels = analysis_results["cluster_labels"]

    # 座標データの取得 (PCA座標優先、なければtarget_featuresを使用)
    if "pca_coordinates" in analysis_results:
        # PCA座標を使用
        coordinates = analysis_results["pca_coordinates"]
        x_label = "第一主成分"
        y_label = "第二主成分"
    else:
        # 元データから特徴量を選択
        if not target_features or len(target_features) < 2:
            raise VisualizationError("可視化対象の特徴量を2つ指定してください")

        data = analysis_results["data"]
        if target_features[0] not in data or target_features[1] not in data:
            raise VisualizationError(f"指定された特徴量がデータに含まれていません: {target_features}")

        coordinates = []
        for i in range(len(data[target_features[0]])):
            coordinates.append([
                data[target_features[0]][i],
                data[target_features[1]][i]
            ])

        x_label = target_features[0]
        y_label = target_features[1]

    # ユニーククラスタの取得
    unique_clusters = list(set(cluster_labels))

    # クラスタごとのデータポイント作成
    series_data = []

    for cluster_id in unique_clusters:
        cluster_points = []
        for i, label in enumerate(cluster_labels):
            if label == cluster_id:
                if i < len(coordinates):
                    cluster_points.append({
                        "x": coordinates[i][0],
                        "y": coordinates[i][1]
                    })

        series_data.append({
            "name": f"クラスタ {cluster_id}",
            "data": cluster_points
        })

    # クラスタ中心の追加（KMeansの場合）
    if "cluster_centers" in analysis_results:
        centers = analysis_results["cluster_centers"]
        center_points = []

        for i, center in enumerate(centers):
            center_points.append({
                "x": center[0],
                "y": center[1],
                "name": f"クラスタ{i}中心"
            })

        series_data.append({
            "name": "クラスタ中心",
            "data": center_points,
            "marker": {
                "symbol": "diamond",
                "radius": 10,
                "lineWidth": 2,
                "lineColor": "#000000",
                "fillColor": "#FF0000"
            }
        })

    return {
        "chart": {
            "type": "scatter",
            "zoomType": "xy"
        },
        "title": {
            "text": "クラスタ分析結果"
        },
        "xAxis": {
            "title": {
                "text": x_label
            },
            "gridLineWidth": 1
        },
        "yAxis": {
            "title": {
                "text": y_label
            },
            "gridLineWidth": 1
        },
        "legend": {
            "enabled": True
        },
        "plotOptions": {
            "scatter": {
                "marker": {
                    "radius": 5,
                    "symbol": "circle"
                },
                "tooltip": {
                    "headerFormat": "<b>{series.name}</b><br>",
                    "pointFormat": f"{x_label}: {{point.x:.2f}}<br>{y_label}: {{point.y:.2f}}"
                }
            }
        },
        "series": series_data
    }

def _prepare_cluster_heatmap_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    クラスタヒートマップデータの準備

    Args:
        analysis_results: クラスタ分析結果

    Returns:
        ヒートマップチャートデータ
    """
    if "cluster_stats" not in analysis_results:
        raise VisualizationError("クラスタ統計データが分析結果に含まれていません")

    cluster_stats = analysis_results["cluster_stats"]

    # 特徴量と各クラスタの平均値を取得
    features = []
    cluster_means = {}

    # 最初のクラスタから特徴量リストを取得
    first_cluster = list(cluster_stats.keys())[0]
    if "mean" in cluster_stats[first_cluster]:
        features = list(cluster_stats[first_cluster]["mean"].keys())
    else:
        raise VisualizationError("クラスタの平均値データが含まれていません")

    # 各クラスタの平均値を取得
    for cluster_id, stats in cluster_stats.items():
        if "mean" in stats:
            cluster_means[cluster_id] = stats["mean"]

    # ヒートマップデータの作成
    heatmap_data = []

    for i, feature in enumerate(features):
        for j, cluster_id in enumerate(cluster_means.keys()):
            mean_value = cluster_means[cluster_id].get(feature, 0)
            heatmap_data.append([j, i, mean_value])

    return {
        "chart": {
            "type": "heatmap"
        },
        "title": {
            "text": "クラスタ特徴量ヒートマップ"
        },
        "xAxis": {
            "categories": [f"クラスタ {i}" for i in cluster_means.keys()],
            "title": {
                "text": "クラスタ"
            }
        },
        "yAxis": {
            "categories": features,
            "title": {
                "text": "特徴量"
            }
        },
        "colorAxis": {
            "min": -2,
            "max": 2,
            "stops": [
                [0, "#3060cf"],
                [0.5, "#ffffff"],
                [1, "#c4463a"]
            ]
        },
        "tooltip": {
            "formatter": "function() { return '<b>' + this.series.yAxis.categories[this.point.y] + '</b><br>クラスタ: ' + this.series.xAxis.categories[this.point.x] + '<br>値: ' + this.point.value.toFixed(2); }"
        },
        "series": [{
            "name": "特徴量平均値",
            "data": heatmap_data,
            "dataLabels": {
                "enabled": True,
                "color": "#000000",
                "format": "{point.value:.2f}"
            }
        }]
    }

def _prepare_silhouette_plot_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    シルエット分析プロットデータの準備

    Args:
        analysis_results: クラスタ分析結果

    Returns:
        シルエットプロットチャートデータ
    """
    if "silhouette_scores" not in analysis_results:
        raise VisualizationError("シルエットスコアデータが分析結果に含まれていません")

    silhouette_data = analysis_results["silhouette_scores"]

    # クラスタラベルの取得
    unique_clusters = list(sorted(silhouette_data.keys()))

    # クラスタごとのシルエットスコア
    series_data = []
    for cluster_id in unique_clusters:
        data = silhouette_data[cluster_id]

        # ソート
        sorted_data = sorted(data)

        series_data.append({
            "name": f"クラスタ {cluster_id}",
            "data": sorted_data,
            "pointPlacement": "on"
        })

    # 全体の平均シルエットスコア
    avg_silhouette = analysis_results.get("avg_silhouette_score", 0)

    return {
        "chart": {
            "type": "column"
        },
        "title": {
            "text": f"シルエット分析 (平均: {avg_silhouette:.3f})"
        },
        "xAxis": {
            "title": {
                "text": "サンプル"
            },
            "visible": False
        },
        "yAxis": {
            "title": {
                "text": "シルエットスコア"
            },
            "min": -1,
            "max": 1,
            "plotLines": [{
                "value": avg_silhouette,
                "color": "red",
                "dashStyle": "dash",
                "width": 1,
                "label": {
                    "text": f"平均: {avg_silhouette:.3f}"
                }
            }]
        },
        "tooltip": {
            "shared": False,
            "formatter": "function() { return this.series.name + '<br>シルエットスコア: ' + this.point.y.toFixed(3); }"
        },
        "plotOptions": {
            "column": {
                "grouping": False,
                "shadow": False,
                "pointPadding": 0,
                "borderWidth": 0
            }
        },
        "series": series_data
    }

def _prepare_feature_distribution_data(analysis_results: Dict[str, Any], target_features: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    特徴量分布データの準備

    Args:
        analysis_results: クラスタ分析結果
        target_features: 可視化対象の特徴量

    Returns:
        特徴量分布チャートデータ
    """
    if "cluster_stats" not in analysis_results:
        raise VisualizationError("クラスタ統計データが分析結果に含まれていません")

    cluster_stats = analysis_results["cluster_stats"]

    # 特徴量の取得
    if not target_features or len(target_features) == 0:
        # 最初のクラスタから特徴量リストを取得
        first_cluster = list(cluster_stats.keys())[0]
        if "mean" in cluster_stats[first_cluster]:
            all_features = list(cluster_stats[first_cluster]["mean"].keys())
            # 最初の特徴量を選択
            target_features = [all_features[0]] if all_features else []
        else:
            raise VisualizationError("クラスタの平均値データが含まれていません")

    if not target_features:
        raise VisualizationError("可視化対象の特徴量が指定されていません")

    feature = target_features[0]  # 最初の特徴量を使用

    # クラスタごとの平均値と標準偏差を取得
    clusters = []
    means = []
    errors = []

    for cluster_id, stats in cluster_stats.items():
        if "mean" in stats and "std" in stats:
            if feature in stats["mean"] and feature in stats["std"]:
                clusters.append(f"クラスタ {cluster_id}")
                means.append(stats["mean"][feature])
                errors.append(stats["std"][feature])

    # エラーバー付きの棒グラフデータを作成
    return {
        "chart": {
            "type": "column",
            "zoomType": "xy"
        },
        "title": {
            "text": f"{feature} のクラスタ別分布"
        },
        "xAxis": {
            "categories": clusters,
            "title": {
                "text": "クラスタ"
            }
        },
        "yAxis": {
            "title": {
                "text": feature
            }
        },
        "tooltip": {
            "formatter": "function() { return '<b>' + this.x + '</b><br>' + this.series.name + ': ' + this.y.toFixed(2) + (this.series.name === '平均値' ? '<br>標準偏差: ' + this.point.errorBar.toFixed(2) : ''); }"
        },
        "plotOptions": {
            "column": {
                "pointPadding": 0.2,
                "borderWidth": 0
            }
        },
        "series": [
            {
                "name": "平均値",
                "data": means,
                "tooltip": {
                    "pointFormat": "{point.y:.2f}"
                }
            },
            {
                "name": "標準偏差",
                "type": "errorbar",
                "data": [[mean - std, mean + std] for mean, std in zip(means, errors)],
                "tooltip": {
                    "pointFormat": "標準偏差: {point.low:.2f} - {point.high:.2f}"
                }
            }
        ]
    }

def _generate_cluster_summary(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    クラスタ分析結果の要約を生成する

    Args:
        analysis_results: クラスタ分析結果

    Returns:
        分析結果の要約
    """
    summary = {
        "model_info": {},
        "key_statistics": {},
        "key_insights": []
    }

    # モデル情報
    summary["model_info"] = {
        "algorithm": analysis_results.get("algorithm", "unknown"),
        "n_clusters": len(analysis_results.get("cluster_stats", {}))
    }

    # クラスタサイズの取得
    cluster_stats = analysis_results.get("cluster_stats", {})
    cluster_sizes = {}

    for cluster_id, stats in cluster_stats.items():
        cluster_sizes[f"クラスタ{cluster_id}"] = stats.get("count", 0)

    # キー統計情報の設定
    summary["key_statistics"] = {
        "cluster_sizes": cluster_sizes,
        "avg_silhouette_score": analysis_results.get("avg_silhouette_score", None),
        "inertia": analysis_results.get("inertia", None)
    }

    # キー洞察の生成

    # 1. クラスタサイズ分布
    if cluster_sizes:
        max_cluster = max(cluster_sizes, key=cluster_sizes.get)
        min_cluster = min(cluster_sizes, key=cluster_sizes.get)

        summary["key_insights"].append(f"最大のクラスタは{max_cluster}で、{cluster_sizes[max_cluster]}個のサンプルを含みます")
        summary["key_insights"].append(f"最小のクラスタは{min_cluster}で、{cluster_sizes[min_cluster]}個のサンプルを含みます")

    # 2. クラスタの特徴抽出
    for cluster_id, stats in cluster_stats.items():
        if "mean" in stats:
            means = stats["mean"]
            # 最も値が高い特徴と低い特徴を見つける
            if means:
                max_feature = max(means, key=means.get)
                min_feature = min(means, key=means.get)

                summary["key_insights"].append(f"クラスタ{cluster_id}の特徴: '{max_feature}'が最も高く、'{min_feature}'が最も低い値です")

    # 3. シルエットスコアからの洞察
    avg_silhouette = analysis_results.get("avg_silhouette_score")
    if avg_silhouette is not None:
        if avg_silhouette > 0.7:
            quality = "非常に良好"
        elif avg_silhouette > 0.5:
            quality = "良好"
        elif avg_silhouette > 0.3:
            quality = "一応の"
        else:
            quality = "弱い"

        summary["key_insights"].append(f"クラスタ品質: 平均シルエットスコア {avg_silhouette:.3f} は{quality}クラスタ構造を示しています")

    # 4. イナーシャ（KMeansの場合）
    inertia = analysis_results.get("inertia")
    if inertia is not None:
        summary["key_insights"].append(f"クラスタ内の分散（イナーシャ）: {inertia:.2f}")

    return summary

# エンドポイント
@router.post("/visualize", response_model=ClusterVisualizationResponse)
async def visualize_cluster(
    request: ClusterVisualizationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    クラスタ分析結果を可視化する

    Args:
        request: 可視化リクエスト
        current_user: 現在のユーザー

    Returns:
        可視化レスポンス
    """
    try:
        logger.info(f"クラスタ分析可視化リクエスト: type={request.visualization_type}")

        chart_data = _prepare_chart_data_from_cluster(
            request.analysis_results,
            request.visualization_type,
            request.target_features
        )

        if request.chart_title:
            chart_data["title"]["text"] = request.chart_title

        summary = _generate_cluster_summary(request.analysis_results)

        return ClusterVisualizationResponse(
            chart_data=chart_data,
            chart_type="cluster_analysis",
            summary=summary
        )
    except VisualizationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"クラスタ分析可視化中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail=f"クラスタ分析可視化中にエラーが発生しました: {str(e)}")

@router.post("/analyze-and-visualize", response_model=ClusterVisualizationResponse)
async def analyze_and_visualize_cluster(
    request: ClusterAnalysisRequest,
    visualization_type: str,
    target_features: Optional[List[str]] = None,
    chart_title: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    クラスタ分析を実行し、結果を可視化する

    Args:
        request: 分析リクエスト
        visualization_type: 可視化タイプ
        target_features: 可視化対象の特徴量
        chart_title: チャートタイトル
        current_user: 現在のユーザー

    Returns:
        可視化レスポンス
    """
    try:
        logger.info(f"クラスタ分析実行と可視化リクエスト: dataset_id={request.dataset_id}, algorithm={request.params.algorithm}")

        # リポジトリからデータを取得
        data_repo = DataRepository()
        df = await data_repo.get_dataset(request.dataset_id, current_user.id)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"データセット {request.dataset_id} が見つかりませんでした")

        # BigQueryサービスの初期化
        bq_service = BigQueryService()

        # クラスタ分析を実行
        analyzer = DataAnalyzer(bq_service)

        # 分析対象列の選択
        if request.params.target_columns:
            # SQLクエリを作成して対象列を取得
            columns_str = ", ".join([f"`{col}`" for col in request.params.target_columns])
            query = f"SELECT {columns_str} FROM `{request.dataset_id}`"
        else:
            # 全列を使用するクエリ
            query = f"SELECT * FROM `{request.dataset_id}`"

        # 分析パラメータの設定
        algorithm = request.params.algorithm
        n_clusters = request.params.n_clusters

        # アルゴリズム別のパラメータを設定
        algorithm_params = {}
        if algorithm == "kmeans":
            algorithm_params = {
                "random_state": request.params.random_state,
                "max_iter": 300
            }
        elif algorithm == "dbscan":
            algorithm_params = {
                "eps": request.params.eps,
                "min_samples": request.params.min_samples
            }
        elif algorithm == "hierarchical":
            algorithm_params = {
                "linkage": request.params.linkage
            }

        # クラスタ分析実行
        results, analysis_results = await analyzer.analyze(
            query=query,
            save_results=True,
            dataset_id=f"{request.dataset_id}_clustered",
            table_id="clusters",
            algorithm=algorithm,
            n_clusters=n_clusters,
            optimize_memory=request.params.optimize_memory,
            **algorithm_params
        )

        # 分析結果を保存
        analysis_id = await data_repo.save_analysis_result(
            user_id=current_user.id,
            dataset_id=request.dataset_id,
            analysis_type="cluster",
            analysis_params=request.params.dict(),
            analysis_result=analysis_results
        )

        logger.info(f"クラスタ分析結果が保存されました: analysis_id={analysis_id}")

        # 結果を可視化
        chart_data = _prepare_chart_data_from_cluster(
            analysis_results,
            visualization_type,
            target_features
        )

        if chart_title:
            chart_data["title"]["text"] = chart_title

        summary = _generate_cluster_summary(analysis_results)

        # リソース解放
        analyzer.release_resources()

        return ClusterVisualizationResponse(
            chart_data=chart_data,
            chart_type="cluster_analysis",
            summary=summary
        )
    except AnalysisError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"クラスタ分析と可視化中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail=f"クラスタ分析と可視化中にエラーが発生しました: {str(e)}")