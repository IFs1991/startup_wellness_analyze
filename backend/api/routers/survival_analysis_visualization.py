"""
生存分析可視化API

提供するエンドポイント:
- POST /api/survival/visualize: 生存分析結果の可視化
- POST /api/survival/analyze-and-visualize: データ分析と可視化を一度に実行
"""

import json
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator

from backend.api.auth import get_current_user
from backend.api.models import User
from backend.repository.data_repository import DataRepository
from backend.analysis.survival_analyzer import SurvivalAnalyzer
from backend.services.logging_service import LoggingService
from backend.common.exceptions import AnalysisError

router = APIRouter(prefix="/api/survival", tags=["survival"])
logger = LoggingService.get_logger(__name__)

# モデル定義
class SurvivalAnalysisParams(BaseModel):
    """生存分析のパラメータ"""
    time_column: str = Field(..., description="時間を表す列名")
    event_column: str = Field(..., description="イベント発生（0=生存、1=イベント発生）を表す列名")
    group_column: Optional[str] = Field(None, description="グループ分けに使用する列名")
    model_type: str = Field("kaplan_meier", description="使用するモデルタイプ (kaplan_meier, cox_ph, etc.)")
    confidence_interval: float = Field(0.95, description="信頼区間のレベル")
    covariates: Optional[List[str]] = Field(None, description="共変量として使用する列名リスト (Coxモデルの場合)")

class SurvivalAnalysisRequest(BaseModel):
    """生存分析リクエスト"""
    dataset_id: str = Field(..., description="データセットID")
    params: SurvivalAnalysisParams = Field(..., description="分析パラメータ")

class SurvivalVisualizationRequest(BaseModel):
    """生存分析可視化リクエスト"""
    analysis_results: Dict[str, Any] = Field(..., description="生存分析結果")
    visualization_type: str = Field(..., description="可視化タイプ (survival_curve, cumulative_hazard, log_log, hazard_ratio)")
    chart_title: Optional[str] = Field(None, description="チャートタイトル")
    chart_description: Optional[str] = Field(None, description="チャート説明")

class SurvivalVisualizationResponse(BaseModel):
    """生存分析可視化レスポンス"""
    chart_data: Dict[str, Any] = Field(..., description="チャートデータ")
    chart_type: str = Field(..., description="チャートタイプ")
    summary: Dict[str, Any] = Field(..., description="分析結果の要約")

# 例外定義
class SurvivalAnalysisException(Exception):
    """生存分析に関連するエラー"""
    pass

class VisualizationError(Exception):
    """可視化処理に関連するエラー"""
    pass

# ヘルパー関数
def _prepare_chart_data_from_survival(
    analysis_results: Dict[str, Any],
    visualization_type: str
) -> Dict[str, Any]:
    """
    生存分析結果からチャートデータを準備する

    Args:
        analysis_results: 生存分析結果
        visualization_type: 可視化タイプ

    Returns:
        チャートデータ
    """
    try:
        if visualization_type == "survival_curve":
            return _prepare_survival_curve_data(analysis_results)
        elif visualization_type == "cumulative_hazard":
            return _prepare_cumulative_hazard_data(analysis_results)
        elif visualization_type == "log_log":
            return _prepare_log_log_data(analysis_results)
        elif visualization_type == "hazard_ratio":
            return _prepare_hazard_ratio_data(analysis_results)
        else:
            raise VisualizationError(f"未対応の可視化タイプ: {visualization_type}")
    except KeyError as e:
        raise VisualizationError(f"分析結果に必要なデータがありません: {str(e)}")
    except Exception as e:
        logger.error(f"チャートデータ準備中にエラーが発生しました: {str(e)}")
        raise VisualizationError(f"チャートデータ準備中にエラーが発生しました: {str(e)}")

def _prepare_survival_curve_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    生存曲線データの準備

    Args:
        analysis_results: 生存分析結果

    Returns:
        生存曲線チャートデータ
    """
    if "survival_curve" not in analysis_results:
        raise VisualizationError("生存曲線データが分析結果に含まれていません")

    survival_data = analysis_results["survival_curve"]
    series_data = []

    # グループが存在する場合
    if "groups" in survival_data:
        for group_name, group_data in survival_data["groups"].items():
            # 生存率データ
            survival_series = {
                "name": f"{group_name}",
                "data": [],
                "type": "line"
            }

            # 信頼区間データ（存在する場合）
            if "lower_ci" in group_data and "upper_ci" in group_data:
                lower_ci_series = {
                    "name": f"{group_name} (下限CI)",
                    "data": [],
                    "type": "line",
                    "dashStyle": "dash",
                    "opacity": 0.5,
                    "showInLegend": False
                }

                upper_ci_series = {
                    "name": f"{group_name} (上限CI)",
                    "data": [],
                    "type": "line",
                    "dashStyle": "dash",
                    "opacity": 0.5,
                    "showInLegend": False
                }

                # データポイントの作成
                for i, time in enumerate(group_data["time"]):
                    survival_series["data"].append({
                        "x": time,
                        "y": group_data["survival"][i]
                    })

                    lower_ci_series["data"].append({
                        "x": time,
                        "y": group_data["lower_ci"][i]
                    })

                    upper_ci_series["data"].append({
                        "x": time,
                        "y": group_data["upper_ci"][i]
                    })

                series_data.append(survival_series)
                series_data.append(lower_ci_series)
                series_data.append(upper_ci_series)
            else:
                # 信頼区間なしの場合
                for i, time in enumerate(group_data["time"]):
                    survival_series["data"].append({
                        "x": time,
                        "y": group_data["survival"][i]
                    })

                series_data.append(survival_series)
    else:
        # グループなしの場合（単一の生存曲線）
        survival_series = {
            "name": "生存率",
            "data": [],
            "type": "line"
        }

        # 信頼区間データ（存在する場合）
        if "lower_ci" in survival_data and "upper_ci" in survival_data:
            lower_ci_series = {
                "name": "下限CI",
                "data": [],
                "type": "line",
                "dashStyle": "dash",
                "opacity": 0.5,
                "showInLegend": False
            }

            upper_ci_series = {
                "name": "上限CI",
                "data": [],
                "type": "line",
                "dashStyle": "dash",
                "opacity": 0.5,
                "showInLegend": False
            }

            # データポイントの作成
            for i, time in enumerate(survival_data["time"]):
                survival_series["data"].append({
                    "x": time,
                    "y": survival_data["survival"][i]
                })

                lower_ci_series["data"].append({
                    "x": time,
                    "y": survival_data["lower_ci"][i]
                })

                upper_ci_series["data"].append({
                    "x": time,
                    "y": survival_data["upper_ci"][i]
                })

            series_data.append(survival_series)
            series_data.append(lower_ci_series)
            series_data.append(upper_ci_series)
        else:
            # 信頼区間なしの場合
            for i, time in enumerate(survival_data["time"]):
                survival_series["data"].append({
                    "x": time,
                    "y": survival_data["survival"][i]
                })

            series_data.append(survival_series)

    return {
        "chart": {
            "type": "line",
            "zoomType": "xy"
        },
        "title": {
            "text": "生存曲線"
        },
        "xAxis": {
            "title": {
                "text": "時間"
            },
            "min": 0
        },
        "yAxis": {
            "title": {
                "text": "生存率"
            },
            "min": 0,
            "max": 1
        },
        "tooltip": {
            "valueDecimals": 3
        },
        "plotOptions": {
            "line": {
                "marker": {
                    "enabled": False
                }
            }
        },
        "series": series_data
    }

def _prepare_cumulative_hazard_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    累積ハザード関数データの準備

    Args:
        analysis_results: 生存分析結果

    Returns:
        累積ハザード関数チャートデータ
    """
    if "cumulative_hazard" not in analysis_results:
        raise VisualizationError("累積ハザード関数データが分析結果に含まれていません")

    hazard_data = analysis_results["cumulative_hazard"]
    series_data = []

    # グループが存在する場合
    if "groups" in hazard_data:
        for group_name, group_data in hazard_data["groups"].items():
            # 累積ハザードデータ
            hazard_series = {
                "name": f"{group_name}",
                "data": [],
                "type": "line"
            }

            # 信頼区間データ（存在する場合）
            if "lower_ci" in group_data and "upper_ci" in group_data:
                lower_ci_series = {
                    "name": f"{group_name} (下限CI)",
                    "data": [],
                    "type": "line",
                    "dashStyle": "dash",
                    "opacity": 0.5,
                    "showInLegend": False
                }

                upper_ci_series = {
                    "name": f"{group_name} (上限CI)",
                    "data": [],
                    "type": "line",
                    "dashStyle": "dash",
                    "opacity": 0.5,
                    "showInLegend": False
                }

                # データポイントの作成
                for i, time in enumerate(group_data["time"]):
                    hazard_series["data"].append({
                        "x": time,
                        "y": group_data["hazard"][i]
                    })

                    lower_ci_series["data"].append({
                        "x": time,
                        "y": group_data["lower_ci"][i]
                    })

                    upper_ci_series["data"].append({
                        "x": time,
                        "y": group_data["upper_ci"][i]
                    })

                series_data.append(hazard_series)
                series_data.append(lower_ci_series)
                series_data.append(upper_ci_series)
            else:
                # 信頼区間なしの場合
                for i, time in enumerate(group_data["time"]):
                    hazard_series["data"].append({
                        "x": time,
                        "y": group_data["hazard"][i]
                    })

                series_data.append(hazard_series)
    else:
        # グループなしの場合（単一の累積ハザード曲線）
        hazard_series = {
            "name": "累積ハザード",
            "data": [],
            "type": "line"
        }

        # 信頼区間データ（存在する場合）
        if "lower_ci" in hazard_data and "upper_ci" in hazard_data:
            lower_ci_series = {
                "name": "下限CI",
                "data": [],
                "type": "line",
                "dashStyle": "dash",
                "opacity": 0.5,
                "showInLegend": False
            }

            upper_ci_series = {
                "name": "上限CI",
                "data": [],
                "type": "line",
                "dashStyle": "dash",
                "opacity": 0.5,
                "showInLegend": False
            }

            # データポイントの作成
            for i, time in enumerate(hazard_data["time"]):
                hazard_series["data"].append({
                    "x": time,
                    "y": hazard_data["hazard"][i]
                })

                lower_ci_series["data"].append({
                    "x": time,
                    "y": hazard_data["lower_ci"][i]
                })

                upper_ci_series["data"].append({
                    "x": time,
                    "y": hazard_data["upper_ci"][i]
                })

            series_data.append(hazard_series)
            series_data.append(lower_ci_series)
            series_data.append(upper_ci_series)
        else:
            # 信頼区間なしの場合
            for i, time in enumerate(hazard_data["time"]):
                hazard_series["data"].append({
                    "x": time,
                    "y": hazard_data["hazard"][i]
                })

            series_data.append(hazard_series)

    return {
        "chart": {
            "type": "line",
            "zoomType": "xy"
        },
        "title": {
            "text": "累積ハザード関数"
        },
        "xAxis": {
            "title": {
                "text": "時間"
            },
            "min": 0
        },
        "yAxis": {
            "title": {
                "text": "累積ハザード"
            },
            "min": 0
        },
        "tooltip": {
            "valueDecimals": 3
        },
        "plotOptions": {
            "line": {
                "marker": {
                    "enabled": False
                }
            }
        },
        "series": series_data
    }

def _prepare_log_log_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Log-Log曲線データの準備

    Args:
        analysis_results: 生存分析結果

    Returns:
        Log-Log曲線チャートデータ
    """
    if "log_log" not in analysis_results:
        raise VisualizationError("Log-Log曲線データが分析結果に含まれていません")

    log_log_data = analysis_results["log_log"]
    series_data = []

    # グループが存在する場合
    if "groups" in log_log_data:
        for group_name, group_data in log_log_data["groups"].items():
            log_log_series = {
                "name": f"{group_name}",
                "data": [],
                "type": "line"
            }

            # データポイントの作成
            for i, time in enumerate(group_data["time"]):
                if group_data["log_log"][i] is not None:  # Noneや無限値の場合はスキップ
                    log_log_series["data"].append({
                        "x": time,
                        "y": group_data["log_log"][i]
                    })

            series_data.append(log_log_series)
    else:
        # グループなしの場合（単一のLog-Log曲線）
        log_log_series = {
            "name": "Log-Log曲線",
            "data": [],
            "type": "line"
        }

        # データポイントの作成
        for i, time in enumerate(log_log_data["time"]):
            if log_log_data["log_log"][i] is not None:  # Noneや無限値の場合はスキップ
                log_log_series["data"].append({
                    "x": time,
                    "y": log_log_data["log_log"][i]
                })

        series_data.append(log_log_series)

    return {
        "chart": {
            "type": "line",
            "zoomType": "xy"
        },
        "title": {
            "text": "Log-Log曲線"
        },
        "xAxis": {
            "title": {
                "text": "Log(時間)"
            },
            "type": "logarithmic"
        },
        "yAxis": {
            "title": {
                "text": "Log(-Log(生存率))"
            }
        },
        "tooltip": {
            "valueDecimals": 3
        },
        "plotOptions": {
            "line": {
                "marker": {
                    "enabled": False
                }
            }
        },
        "series": series_data
    }

def _prepare_hazard_ratio_data(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    ハザード比データの準備

    Args:
        analysis_results: 生存分析結果

    Returns:
        ハザード比チャートデータ
    """
    if "hazard_ratio" not in analysis_results:
        raise VisualizationError("ハザード比データが分析結果に含まれていません")

    hazard_ratio_data = analysis_results["hazard_ratio"]

    # 変数名と係数を取得
    variables = hazard_ratio_data.get("variables", [])
    hazard_ratios = hazard_ratio_data.get("hazard_ratio", [])
    lower_ci = hazard_ratio_data.get("lower_ci", [])
    upper_ci = hazard_ratio_data.get("upper_ci", [])
    p_values = hazard_ratio_data.get("p_value", [])

    # データポイントを作成
    data = []
    for i, variable in enumerate(variables):
        data.append({
            "name": variable,
            "y": hazard_ratios[i],
            "low": lower_ci[i] if i < len(lower_ci) else None,
            "high": upper_ci[i] if i < len(upper_ci) else None,
            "p_value": p_values[i] if i < len(p_values) else None
        })

    # 昇順にソート
    data.sort(key=lambda x: x["y"])

    # グラフデータの準備
    variables = [item["name"] for item in data]
    ratios = [item["y"] for item in data]
    error_data = []

    for i, item in enumerate(data):
        if "low" in item and "high" in item and item["low"] is not None and item["high"] is not None:
            error_data.append([item["low"], item["high"]])
        else:
            error_data.append(None)

    series_data = [{
        "name": "ハザード比",
        "data": ratios,
        "type": "scatter"
    }]

    # 信頼区間データがある場合
    if any(error is not None for error in error_data):
        error_bars = []
        for i, item in enumerate(data):
            if error_data[i] is not None:
                error_bars.append({
                    "x": i,
                    "low": error_data[i][0],
                    "high": error_data[i][1]
                })
            else:
                error_bars.append(None)

        series_data.append({
            "name": "95%信頼区間",
            "data": error_bars,
            "type": "errorbar"
        })

    return {
        "chart": {
            "type": "scatter",
            "zoomType": "xy"
        },
        "title": {
            "text": "ハザード比"
        },
        "xAxis": {
            "categories": variables,
            "title": {
                "text": "変数"
            }
        },
        "yAxis": {
            "title": {
                "text": "ハザード比 (対数スケール)"
            },
            "type": "logarithmic",
            "plotLines": [{
                "value": 1,
                "color": "red",
                "width": 1,
                "dashStyle": "dash",
                "label": {
                    "text": "HR=1",
                    "align": "right"
                }
            }]
        },
        "tooltip": {
            "formatter": "function() { return '<b>' + this.x + '</b><br>ハザード比: ' + this.y.toFixed(3) + (this.point.p_value !== undefined ? '<br>p値: ' + this.point.p_value.toFixed(4) : ''); }"
        },
        "plotOptions": {
            "scatter": {
                "marker": {
                    "symbol": "circle",
                    "radius": 5
                }
            }
        },
        "series": series_data
    }

def _generate_survival_analysis_summary(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    生存分析結果の要約を生成する

    Args:
        analysis_results: 生存分析結果

    Returns:
        分析結果の要約
    """
    summary = {
        "model_info": {},
        "key_statistics": {},
        "key_insights": []
    }

    # モデル情報
    if "model_info" in analysis_results:
        summary["model_info"] = analysis_results["model_info"]

    # 主要統計
    if "statistics" in analysis_results:
        summary["key_statistics"] = analysis_results["statistics"]

        # 統計情報からキー洞察を生成
        stats = analysis_results["statistics"]

        # ログランク検定の結果がある場合
        if "logrank_test" in stats:
            logrank = stats["logrank_test"]
            p_value = logrank.get("p_value")
            if p_value is not None:
                sig_text = "有意" if p_value < 0.05 else "有意でない"
                summary["key_insights"].append(f"グループ間の差は統計的に{sig_text}です (p={p_value:.4f})")

        # 中央生存期間がある場合
        if "median_survival" in stats:
            median_data = stats["median_survival"]

            if isinstance(median_data, dict) and "groups" in median_data:
                # グループごとの中央生存期間
                median_insights = []
                for group, value in median_data["groups"].items():
                    if value is not None:
                        median_insights.append(f"{group}: {value:.2f}")

                if median_insights:
                    summary["key_insights"].append(f"中央生存期間: {', '.join(median_insights)}")
            elif isinstance(median_data, (int, float)) and median_data is not None:
                # 単一の中央生存期間
                summary["key_insights"].append(f"中央生存期間: {median_data:.2f}")

    # Coxモデルの結果がある場合
    if "cox_model" in analysis_results:
        cox_data = analysis_results["cox_model"]

        # 有意な変数を特定
        sig_vars = []
        if "variables" in cox_data and "p_value" in cox_data:
            for i, var in enumerate(cox_data["variables"]):
                if i < len(cox_data["p_value"]) and cox_data["p_value"][i] < 0.05:
                    hr = cox_data["hazard_ratio"][i] if "hazard_ratio" in cox_data and i < len(cox_data["hazard_ratio"]) else None
                    if hr is not None:
                        sig_vars.append(f"{var} (HR={hr:.2f}, p={cox_data['p_value'][i]:.4f})")

        if sig_vars:
            summary["key_insights"].append(f"有意な予測因子: {', '.join(sig_vars)}")

        # コンコーダンス指数がある場合
        if "concordance" in cox_data:
            summary["key_insights"].append(f"コンコーダンス指数: {cox_data['concordance']:.3f}")

    return summary

# エンドポイント
@router.post("/visualize", response_model=SurvivalVisualizationResponse)
async def visualize_survival_analysis(
    request: SurvivalVisualizationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    生存分析結果を可視化する

    Args:
        request: 可視化リクエスト
        current_user: 現在のユーザー

    Returns:
        可視化レスポンス
    """
    try:
        logger.info(f"生存分析可視化リクエスト: type={request.visualization_type}")

        chart_data = _prepare_chart_data_from_survival(
            request.analysis_results,
            request.visualization_type
        )

        if request.chart_title:
            chart_data["title"]["text"] = request.chart_title

        summary = _generate_survival_analysis_summary(request.analysis_results)

        return SurvivalVisualizationResponse(
            chart_data=chart_data,
            chart_type="survival_analysis",
            summary=summary
        )
    except VisualizationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"生存分析可視化中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生存分析可視化中にエラーが発生しました: {str(e)}")

@router.post("/analyze-and-visualize", response_model=SurvivalVisualizationResponse)
async def analyze_and_visualize_survival(
    request: SurvivalAnalysisRequest,
    visualization_type: str,
    chart_title: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    生存分析を実行し、結果を可視化する

    Args:
        request: 分析リクエスト
        visualization_type: 可視化タイプ
        chart_title: チャートタイトル
        current_user: 現在のユーザー

    Returns:
        可視化レスポンス
    """
    try:
        logger.info(f"生存分析実行と可視化リクエスト: dataset_id={request.dataset_id}, visualization_type={visualization_type}")

        # リポジトリからデータを取得
        data_repo = DataRepository()
        df = await data_repo.get_dataset(request.dataset_id, current_user.id)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"データセット {request.dataset_id} が見つかりませんでした")

        # 生存分析を実行
        analyzer = SurvivalAnalyzer()

        analysis_results = analyzer.analyze(
            df=df,
            time_column=request.params.time_column,
            event_column=request.params.event_column,
            group_column=request.params.group_column,
            model_type=request.params.model_type,
            confidence_interval=request.params.confidence_interval,
            covariates=request.params.covariates
        )

        # 分析結果を保存
        analysis_id = await data_repo.save_analysis_result(
            user_id=current_user.id,
            dataset_id=request.dataset_id,
            analysis_type="survival_analysis",
            analysis_params=request.params.dict(),
            analysis_result=analysis_results
        )

        logger.info(f"生存分析結果が保存されました: analysis_id={analysis_id}")

        # 結果を可視化
        chart_data = _prepare_chart_data_from_survival(
            analysis_results,
            visualization_type
        )

        if chart_title:
            chart_data["title"]["text"] = chart_title

        summary = _generate_survival_analysis_summary(analysis_results)

        return SurvivalVisualizationResponse(
            chart_data=chart_data,
            chart_type="survival_analysis",
            summary=summary
        )
    except AnalysisError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"生存分析と可視化中にエラーが発生しました: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生存分析と可視化中にエラーが発生しました: {str(e)}")