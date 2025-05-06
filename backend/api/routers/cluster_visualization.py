"""
クラスタ分析可視化API

提供するエンドポイント:
- POST /api/cluster/visualize: クラスタ分析結果の可視化
- POST /api/cluster/analyze-and-visualize: データ分析と可視化を一度に実行
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, validator

from api.auth import get_current_user
from api.models import User
from api.dependencies import get_visualization_service
from api.core.config import get_settings, Settings
from analysis.ClusterAnalyzer import ClusterAnalyzer

# 共通可視化コンポーネントのインポート
from api.visualization.models import (
    BaseVisualizationRequest,
    BaseVisualizationResponse,
    ClusterVisualizationRequest
)
from api.visualization.errors import (
    handle_visualization_error,
    InvalidAnalysisResultError
)
from api.routers.visualization_helpers import (
    prepare_chart_data_by_analysis_type,
    create_visualization_response
)

router = APIRouter(
    prefix="/api/cluster",
    tags=["cluster"],
    responses={404: {"description": "リソースが見つかりません"}}
)
logger = logging.getLogger(__name__)

# リクエスト・レスポンスモデル定義
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
    data: List[Dict[str, Any]] = Field(..., description="分析対象データ")
    params: ClusterAnalysisParams = Field(default_factory=ClusterAnalysisParams, description="分析パラメータ")
    visualization_options: Dict[str, Any] = Field(default_factory=dict, description="可視化オプション")

# ヘルパー関数
def _format_analysis_summary(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    クラスタ分析結果のサマリーをフォーマットする

    Args:
        analysis_results: クラスタ分析結果

    Returns:
        フォーマット済みサマリー
    """
    # クラスタごとのサイズを計算
    if "cluster_labels" in analysis_results:
        labels = analysis_results["cluster_labels"]
        unique_clusters = set(labels)
        cluster_sizes = {f"クラスタ{cluster}": labels.count(cluster) for cluster in unique_clusters}
        total_points = len(labels)
    else:
        cluster_sizes = {}
        total_points = 0

    # 使用アルゴリズムの情報
    algorithm = analysis_results.get("algorithm", "不明")
    n_clusters = analysis_results.get("n_clusters", len(cluster_sizes))

    # クラスタ中心（存在する場合）
    cluster_centers = analysis_results.get("cluster_centers", [])
    center_info = {}
    feature_names = analysis_results.get("feature_names", [])

    if cluster_centers and feature_names:
        for i, center in enumerate(cluster_centers):
            if i < len(center) and i < len(feature_names):
                center_info[f"クラスタ{i}"] = {
                    feature_names[j]: round(center[j], 3)
                    for j in range(min(len(center), len(feature_names)))
                }

    return {
        "algorithm": algorithm,
        "n_clusters": n_clusters,
        "total_points": total_points,
        "cluster_sizes": cluster_sizes,
        "cluster_centers": center_info,
        "description": f"{algorithm} アルゴリズムにより {n_clusters} 個のクラスタが検出されました。"
    }

@router.post("/visualize", response_model=BaseVisualizationResponse, status_code=status.HTTP_200_OK)
async def visualize_cluster_analysis(
    request: ClusterVisualizationRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """クラスタ分析結果を可視化する"""
    try:
        logger.info(f"クラスタ分析可視化リクエスト: type={request.visualization_type}")

        # 共通関数を使用してチャートデータを準備
        chart_data = prepare_chart_data_by_analysis_type(
            analysis_type="cluster",
            analysis_results=request.analysis_results,
            visualization_type=request.visualization_type,
            options=request.options
        )

        # チャート生成
        chart_result = await visualization_service.generate_chart(
            config=chart_data["config"],
            data=chart_data["data"],
            format=request.options.get("format", "png"),
            template_id=request.options.get("template_id"),
            user_id=str(current_user.id)
        )

        # 分析サマリーを作成
        analysis_summary = _format_analysis_summary(request.analysis_results)

        # 統一されたレスポンスを作成して返す
        return create_visualization_response(chart_result, analysis_summary)

    except Exception as e:
        logger.exception(f"クラスタ分析可視化中にエラー: {str(e)}")
        raise handle_visualization_error(e)

@router.post("/analyze-and-visualize", response_model=BaseVisualizationResponse, status_code=status.HTTP_200_OK)
async def analyze_and_visualize_cluster(
    request: ClusterAnalysisRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """クラスタ分析と可視化を一度に実行する"""
    try:
        logger.info("クラスタ分析および可視化リクエスト")

        # クラスタ分析実行
        analyzer = ClusterAnalyzer()
        analysis_params = {
            "algorithm": request.params.algorithm,
            "n_clusters": request.params.n_clusters,
            "target_columns": request.params.target_columns,
            "random_state": request.params.random_state,
            "eps": request.params.eps,
            "min_samples": request.params.min_samples,
            "linkage": request.params.linkage,
            "standardize": request.params.standardize,
            "optimize_memory": request.params.optimize_memory
        }

        analysis_results = analyzer.analyze(request.data, analysis_params)

        if not analysis_results or "cluster_labels" not in analysis_results:
            raise InvalidAnalysisResultError("クラスタ分析結果が無効です。入力データを確認してください。")

        # 可視化タイプとオプションの設定
        visualization_type = request.visualization_options.get("visualization_type", "scatter")
        visualization_options = request.visualization_options.copy()

        # 共通関数を使用してチャートデータを準備
        chart_data = prepare_chart_data_by_analysis_type(
            analysis_type="cluster",
            analysis_results=analysis_results,
            visualization_type=visualization_type,
            options=visualization_options
        )

        # チャート生成
        chart_result = await visualization_service.generate_chart(
            config=chart_data["config"],
            data=chart_data["data"],
            format=visualization_options.get("format", "png"),
            template_id=visualization_options.get("template_id"),
            user_id=str(current_user.id)
        )

        # 分析サマリーを作成
        analysis_summary = _format_analysis_summary(analysis_results)

        # 統一されたレスポンスを作成して返す
        return create_visualization_response(chart_result, analysis_summary)

    except Exception as e:
        logger.exception(f"クラスタ分析および可視化中にエラー: {str(e)}")
        raise handle_visualization_error(e)