"""
主成分分析可視化API

提供するエンドポイント:
- POST /api/pca/visualize: 主成分分析結果の可視化
- POST /api/pca/analyze-and-visualize: データ分析と可視化を一度に実行
"""

import json
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator

from backend.api.auth import get_current_user
from backend.api.models import User
from backend.repository.data_repository import DataRepository
from backend.analysis.PCAAnalyzer import PCAAnalyzer
from backend.services.logging_service import LoggingService
from backend.common.exceptions import AnalysisError

router = APIRouter(prefix="/api/pca", tags=["pca"])
logger = LoggingService.get_logger(__name__)

# モデル定義
class PCAParams(BaseModel):
    """主成分分析のパラメータ"""
    n_components: int = Field(..., description="抽出する主成分の数")
    target_columns: Optional[List[str]] = Field(None, description="分析対象の列名リスト（指定しない場合は数値列すべて）")
    standardize: bool = Field(True, description="データを標準化するかどうか")
    random_state: Optional[int] = Field(None, description="乱数シード")

class PCARequest(BaseModel):
    """主成分分析リクエスト"""
    dataset_id: str = Field(..., description="データセットID")
    params: PCAParams = Field(..., description="分析パラメータ")

class PCAVisualizationRequest(BaseModel):
    """主成分分析可視化リクエスト"""
    analysis_results: Dict[str, Any] = Field(..., description="主成分分析結果")
    visualization_type: str = Field(..., description="可視化タイプ (scree_plot, loading_plot, biplot, variance_explained)")
    chart_title: Optional[str] = Field(None, description="チャートタイトル")
    chart_description: Optional[str] = Field(None, description="チャート説明")
    components: Optional[List[int]] = Field(None, description="表示する主成分のインデックス（デフォルトは最初の2つ）")

class PCAVisualizationResponse(BaseModel):
    """主成分分析可視化レスポンス"""
    chart_data: Dict[str, Any] = Field(..., description="チャートデータ")
    chart_type: str = Field(..., description="チャートタイプ")
    summary: Dict[str, Any] = Field(..., description="分析結果の要約")

# 例外定義
class PCAAnalysisException(Exception):
    """主成分分析に関連するエラー"""
    pass

class VisualizationError(Exception):
    """可視化処理に関連するエラー"""
    pass

# ヘルパー関数
def _prepare_chart_data_from_pca(
    analysis_results: Dict[str, Any],
    visualization_type: str,
    components: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    主成分分析結果からチャートデータを準備する

    Args:
        analysis_results: 主成分分析結果
        visualization_type: 可視化タイプ
        components: 表示する主成分のインデックス

    Returns:
        チャートデータ
    """
    try:
        if visualization_type == "scree_plot":
            return _prepare_scree_plot_data(analysis_results)
        elif visualization_type == "loading_plot":
            return _prepare_loading_plot_data(analysis_results, components)
        elif visualization_type == "biplot":
            return _prepare_biplot_data(analysis_results, components)
        elif visualization_type == "variance_explained":
            return _prepare_variance_explained_data(analysis_results)
        else:
            raise VisualizationError(f"未対応の可視化タイプ: {visualization_type}")
    except KeyError as e:
        raise VisualizationError(f"分析結果に必要なデータがありません: {str(e)}")
    except Exception as e:
        logger.error(f"チャートデータ準備中にエラーが発生しました: {str(e)}")
        raise VisualizationError(f"チャートデータ準備中にエラーが発生しました: {str(e)}")

def _prepare_scree_plot_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    スクリープロットデータの準備

    Args:
        analysis_results: 主成分分析結果

    Returns:
        スクリープロットチャートデータ
    """
    if "explained_variance_ratio" not in analysis_results:
        raise VisualizationError("分散説明率データが分析結果に含まれていません")

    explained_variance = analysis_results["explained_variance_ratio"]

    # 主成分番号を生成
    components = [f"PC{i+1}" for i in range(len(explained_variance))]

    # データポイントの作成
    individual_data = []
    for i, variance in enumerate(explained_variance):
        individual_data.append({
            "x": i + 1,
            "y": variance * 100  # パーセンテージ表示
        })

    # 累積分散説明率のデータ
    cumulative_data = []
    cumulative_sum = 0
    for i, variance in enumerate(explained_variance):
        cumulative_sum += variance
        cumulative_data.append({
            "x": i + 1,
            "y": cumulative_sum * 100  # パーセンテージ表示
        })

    return {
        "chart": {
            "type": "column",
            "zoomType": "xy"
        },
        "title": {
            "text": "スクリープロット"
        },
        "xAxis": {
            "categories": components,
            "title": {
                "text": "主成分"
            },
            "crosshair": True
        },
        "yAxis": [
            {
                "title": {
                    "text": "分散説明率 (%)"
                },
                "min": 0,
                "max": 100
            },
            {
                "title": {
                    "text": "累積分散説明率 (%)"
                },
                "opposite": True,
                "min": 0,
                "max": 100
            }
        ],
        "tooltip": {
            "shared": True
        },
        "plotOptions": {
            "column": {
                "pointPadding": 0.2,
                "borderWidth": 0
            }
        },
        "series": [
            {
                "name": "分散説明率",
                "type": "column",
                "data": [variance * 100 for variance in explained_variance],
                "tooltip": {
                    "valueSuffix": "%"
                }
            },
            {
                "name": "累積分散説明率",
                "type": "spline",
                "data": [cumulative_data[i]["y"] for i in range(len(cumulative_data))],
                "tooltip": {
                    "valueSuffix": "%"
                },
                "yAxis": 1,
                "marker": {
                    "enabled": True,
                    "radius": 3
                }
            }
        ]
    }

def _prepare_loading_plot_data(analysis_results: Dict[str, Any], components: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    ローディングプロットデータの準備

    Args:
        analysis_results: 主成分分析結果
        components: 表示する主成分のインデックス

    Returns:
        ローディングプロットチャートデータ
    """
    if "components" not in analysis_results or "original_columns" not in analysis_results:
        raise VisualizationError("主成分ローディングデータが分析結果に含まれていません")

    components_data = analysis_results["components"]
    feature_names = analysis_results["original_columns"]

    # デフォルトは最初の2つの主成分
    pc1_idx = 0
    pc2_idx = 1

    if components and len(components) >= 2:
        if max(components) <= len(components_data):
            pc1_idx = components[0] - 1  # 1-indexedから0-indexedに変換
            pc2_idx = components[1] - 1
        else:
            raise VisualizationError(f"指定された主成分インデックスが範囲外です: {components}")

    # データポイントの作成
    series_data = []
    for i, feature in enumerate(feature_names):
        series_data.append({
            "x": components_data[pc1_idx][i],
            "y": components_data[pc2_idx][i],
            "name": feature
        })

    return {
        "chart": {
            "type": "scatter",
            "zoomType": "xy"
        },
        "title": {
            "text": f"ローディングプロット (PC{pc1_idx+1} vs PC{pc2_idx+1})"
        },
        "xAxis": {
            "title": {
                "text": f"PC{pc1_idx+1} ローディング"
            },
            "gridLineWidth": 1,
            "plotLines": [{
                "color": "#FF0000",
                "width": 1,
                "value": 0
            }]
        },
        "yAxis": {
            "title": {
                "text": f"PC{pc2_idx+1} ローディング"
            },
            "plotLines": [{
                "color": "#FF0000",
                "width": 1,
                "value": 0
            }]
        },
        "tooltip": {
            "formatter": "function() { return '<b>' + this.point.name + '</b><br/>PC" + (pc1_idx+1) + ": ' + this.x.toFixed(3) + '<br/>PC" + (pc2_idx+1) + ": ' + this.y.toFixed(3); }"
        },
        "plotOptions": {
            "scatter": {
                "marker": {
                    "radius": 5,
                    "symbol": "circle"
                }
            }
        },
        "series": [{
            "name": "特徴量",
            "data": series_data,
            "dataLabels": {
                "enabled": True,
                "format": "{point.name}"
            }
        }]
    }

def _prepare_biplot_data(analysis_results: Dict[str, Any], components: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    バイプロットデータの準備

    Args:
        analysis_results: 主成分分析結果
        components: 表示する主成分のインデックス

    Returns:
        バイプロットチャートデータ
    """
    if "components" not in analysis_results or "principal_components" not in analysis_results:
        raise VisualizationError("バイプロットに必要なデータが分析結果に含まれていません")

    components_data = analysis_results["components"]
    principal_components = analysis_results["principal_components"]
    feature_names = analysis_results["original_columns"]

    # デフォルトは最初の2つの主成分
    pc1_idx = 0
    pc2_idx = 1

    if components and len(components) >= 2:
        if max(components) <= len(components_data):
            pc1_idx = components[0] - 1  # 1-indexedから0-indexedに変換
            pc2_idx = components[1] - 1
        else:
            raise VisualizationError(f"指定された主成分インデックスが範囲外です: {components}")

    # サンプルデータポイントの作成（最初の100点に制限）
    sample_data = []
    max_samples = min(100, len(principal_components))
    for i in range(max_samples):
        sample_data.append({
            "x": principal_components[i][pc1_idx],
            "y": principal_components[i][pc2_idx]
        })

    # ローディングベクトルデータの作成
    loading_data = []
    max_loading = 0

    # 最大ローディング値を見つける（スケーリング用）
    for i, feature in enumerate(feature_names):
        loading_x = components_data[pc1_idx][i]
        loading_y = components_data[pc2_idx][i]
        max_loading = max(max_loading, abs(loading_x), abs(loading_y))

    # スケーリング係数（バイプロットのベクトル長調整）
    scaling_factor = 1.0 / max_loading * 3

    for i, feature in enumerate(feature_names):
        loading_x = components_data[pc1_idx][i]
        loading_y = components_data[pc2_idx][i]

        loading_data.append({
            "x": 0,
            "y": 0,
            "dx": loading_x * scaling_factor,
            "dy": loading_y * scaling_factor,
            "name": feature
        })

    return {
        "chart": {
            "type": "scatter",
            "zoomType": "xy"
        },
        "title": {
            "text": f"バイプロット (PC{pc1_idx+1} vs PC{pc2_idx+1})"
        },
        "xAxis": {
            "title": {
                "text": f"PC{pc1_idx+1}"
            },
            "gridLineWidth": 1,
            "plotLines": [{
                "color": "#FF0000",
                "width": 1,
                "value": 0
            }]
        },
        "yAxis": {
            "title": {
                "text": f"PC{pc2_idx+1}"
            },
            "plotLines": [{
                "color": "#FF0000",
                "width": 1,
                "value": 0
            }]
        },
        "tooltip": {
            "formatter": "function() { return this.series.name === 'サンプル' ? 'サンプルID: ' + this.point.index : '<b>' + this.point.name + '</b><br/>PC" + (pc1_idx+1) + ": ' + this.point.dx.toFixed(3) + '<br/>PC" + (pc2_idx+1) + ": ' + this.point.dy.toFixed(3); }"
        },
        "plotOptions": {
            "scatter": {
                "marker": {
                    "radius": 4,
                    "symbol": "circle"
                }
            }
        },
        "series": [
            {
                "name": "サンプル",
                "data": sample_data,
                "color": "rgba(119, 152, 191, 0.5)",
                "marker": {
                    "radius": 3
                }
            },
            {
                "name": "特徴量ローディング",
                "type": "vector",
                "data": loading_data,
                "color": "rgba(223, 83, 83, 0.8)",
                "dataLabels": {
                    "enabled": True,
                    "format": "{point.name}",
                    "allowOverlap": False,
                    "x": 10,
                    "y": -5
                },
                "tooltip": {
                    "headerFormat": "",
                    "pointFormat": "<b>{point.name}</b><br>PC{pc1_idx+1}: {point.dx:.3f}<br>PC{pc2_idx+1}: {point.dy:.3f}"
                }
            }
        ]
    }

def _prepare_variance_explained_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    分散説明率データの準備

    Args:
        analysis_results: 主成分分析結果

    Returns:
        分散説明率チャートデータ
    """
    if "explained_variance_ratio" not in analysis_results or "cumulative_variance_ratio" not in analysis_results:
        raise VisualizationError("分散説明率データが分析結果に含まれていません")

    explained_variance = analysis_results["explained_variance_ratio"]
    cumulative_variance = analysis_results["cumulative_variance_ratio"]

    # 主成分番号を生成
    components = [f"PC{i+1}" for i in range(len(explained_variance))]

    return {
        "chart": {
            "type": "column",
            "zoomType": "xy"
        },
        "title": {
            "text": "累積分散説明率"
        },
        "xAxis": {
            "categories": components,
            "title": {
                "text": "主成分数"
            }
        },
        "yAxis": {
            "title": {
                "text": "累積分散説明率 (%)"
            },
            "min": 0,
            "max": 100,
            "plotLines": [
                {
                    "value": 80,
                    "color": "red",
                    "dashStyle": "dash",
                    "width": 1,
                    "label": {
                        "text": "80%",
                        "align": "right"
                    }
                },
                {
                    "value": 90,
                    "color": "green",
                    "dashStyle": "dash",
                    "width": 1,
                    "label": {
                        "text": "90%",
                        "align": "right"
                    }
                }
            ]
        },
        "tooltip": {
            "formatter": "function() { return '<b>' + this.x + '</b><br/>' + this.series.name + ': ' + this.y.toFixed(2) + '%'; }"
        },
        "series": [
            {
                "name": "累積分散説明率",
                "data": [v * 100 for v in cumulative_variance]
            }
        ]
    }

def _generate_pca_summary(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    主成分分析結果の要約を生成する

    Args:
        analysis_results: 主成分分析結果

    Returns:
        分析結果の要約
    """
    summary = {
        "model_info": {},
        "key_statistics": {},
        "key_insights": []
    }

    # モデル情報
    n_components = len(analysis_results.get("explained_variance_ratio", []))
    summary["model_info"] = {
        "n_components": n_components,
        "n_features": len(analysis_results.get("original_columns", [])),
        "n_samples": analysis_results.get("row_count", 0)
    }

    # 主要統計
    if "explained_variance_ratio" in analysis_results and "cumulative_variance_ratio" in analysis_results:
        explained_variance = analysis_results["explained_variance_ratio"]
        cumulative_variance = analysis_results["cumulative_variance_ratio"]

        summary["key_statistics"] = {
            "explained_variance_ratio": explained_variance,
            "cumulative_variance_ratio": cumulative_variance
        }

        # キー洞察の生成
        # 1. 第一主成分と第二主成分の説明率
        if len(explained_variance) > 1:
            pc1_variance = explained_variance[0] * 100
            pc2_variance = explained_variance[1] * 100
            summary["key_insights"].append(f"第一主成分は全分散の{pc1_variance:.2f}%を説明しています")
            summary["key_insights"].append(f"第一・第二主成分の合計で全分散の{pc1_variance + pc2_variance:.2f}%を説明しています")

        # 2. 80%の分散を説明するのに必要な主成分数
        components_for_80 = next((i+1 for i, cv in enumerate(cumulative_variance) if cv >= 0.8), n_components)
        summary["key_insights"].append(f"全分散の80%を説明するには{components_for_80}個の主成分が必要です")

        # 3. 90%の分散を説明するのに必要な主成分数
        components_for_90 = next((i+1 for i, cv in enumerate(cumulative_variance) if cv >= 0.9), n_components)
        if components_for_90 < n_components:
            summary["key_insights"].append(f"全分散の90%を説明するには{components_for_90}個の主成分が必要です")

    # 特徴量の寄与
    if "components" in analysis_results and "original_columns" in analysis_results:
        components = analysis_results["components"]
        features = analysis_results["original_columns"]

        if len(components) > 0 and len(components[0]) == len(features):
            # 第一主成分に最も寄与する特徴量を特定
            pc1_loadings = components[0]
            pc1_abs_loadings = [abs(l) for l in pc1_loadings]
            pc1_max_idx = pc1_abs_loadings.index(max(pc1_abs_loadings))
            pc1_max_feature = features[pc1_max_idx]
            pc1_max_loading = pc1_loadings[pc1_max_idx]

            loading_direction = "正" if pc1_max_loading > 0 else "負"
            summary["key_insights"].append(f"第一主成分に最も寄与する特徴量は「{pc1_max_feature}」で、{loading_direction}の方向に{abs(pc1_max_loading):.3f}の寄与度があります")

            # 第二主成分についても同様に
            if len(components) > 1:
                pc2_loadings = components[1]
                pc2_abs_loadings = [abs(l) for l in pc2_loadings]
                pc2_max_idx = pc2_abs_loadings.index(max(pc2_abs_loadings))
                pc2_max_feature = features[pc2_max_idx]
                pc2_max_loading = pc2_loadings[pc2_max_idx]

                loading_direction = "正" if pc2_max_loading > 0 else "負"
                summary["key_insights"].append(f"第二主成分に最も寄与する特徴量は「{pc2_max_feature}」で、{loading_direction}の方向に{abs(pc2_max_loading):.3f}の寄与度があります")

    return summary

# エンドポイント
@router.post("/visualize", response_model=PCAVisualizationResponse)
async def visualize_pca(
    request: PCAVisualizationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    主成分分析結果を可視化する

    Args:
        request: 可視化リクエスト
        current_user: 現在のユーザー

    Returns:
        可視化レスポンス
    """
    try:
        logger.info(f"主成分分析可視化リクエスト: type={request.visualization_type}")

        chart_data = _prepare_chart_data_from_pca(
            request.analysis_results,
            request.visualization_type,
            request.components
        )

        if request.chart_title:
            chart_data["title"]["text"] = request.chart_title

        summary = _generate_pca_summary(request.analysis_results)

        return PCAVisualizationResponse(
            chart_data=chart_data,
            chart_type="pca_analysis",
            summary=summary
        )
    except VisualizationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"主成分分析可視化中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail=f"主成分分析可視化中にエラーが発生しました: {str(e)}")

@router.post("/analyze-and-visualize", response_model=PCAVisualizationResponse)
async def analyze_and_visualize_pca(
    request: PCARequest,
    visualization_type: str,
    components: Optional[List[int]] = None,
    chart_title: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    主成分分析を実行し、結果を可視化する

    Args:
        request: 分析リクエスト
        visualization_type: 可視化タイプ
        components: 表示する主成分のインデックス
        chart_title: チャートタイトル
        current_user: 現在のユーザー

    Returns:
        可視化レスポンス
    """
    try:
        logger.info(f"主成分分析実行と可視化リクエスト: dataset_id={request.dataset_id}, visualization_type={visualization_type}")

        # リポジトリからデータを取得
        data_repo = DataRepository()
        df = await data_repo.get_dataset(request.dataset_id, current_user.id)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"データセット {request.dataset_id} が見つかりませんでした")

        # 主成分分析を実行
        from service.bigquery.client import BigQueryService
        bq_service = BigQueryService()

        analyzer = PCAAnalyzer(bq_service)

        # データを直接指定して分析実行
        n_components = min(request.params.n_components, len(df.columns))

        # 分析対象列の選択
        if request.params.target_columns:
            target_df = df[request.params.target_columns]
        else:
            # 数値型の列のみを選択
            target_df = df.select_dtypes(include=['number'])

        if target_df.empty:
            raise HTTPException(status_code=400, detail="分析対象となる数値データが見つかりませんでした")

        # 欠損値の処理
        target_df = target_df.fillna(target_df.mean())

        # 主成分分析実行
        analysis_results = analyzer.fit_transform(
            data=target_df,
            n_components=n_components,
            standardize=request.params.standardize,
            random_state=request.params.random_state
        )

        # 分析結果を保存
        analysis_id = await data_repo.save_analysis_result(
            user_id=current_user.id,
            dataset_id=request.dataset_id,
            analysis_type="pca",
            analysis_params=request.params.dict(),
            analysis_result=analysis_results
        )

        logger.info(f"主成分分析結果が保存されました: analysis_id={analysis_id}")

        # 結果を可視化
        chart_data = _prepare_chart_data_from_pca(
            analysis_results,
            visualization_type,
            components
        )

        if chart_title:
            chart_data["title"]["text"] = chart_title

        summary = _generate_pca_summary(analysis_results)

        # リソース解放
        analyzer.release_resources()

        return PCAVisualizationResponse(
            chart_data=chart_data,
            chart_type="pca_analysis",
            summary=summary
        )
    except AnalysisError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"主成分分析と可視化中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail=f"主成分分析と可視化中にエラーが発生しました: {str(e)}")