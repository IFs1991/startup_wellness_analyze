"""
アソシエーション分析可視化API

このモジュールは次のエンドポイントを提供します：
- POST /api/association/visualize - アソシエーション分析結果の可視化
- POST /api/association/analyze-and-visualize - データ分析と可視化を一度に実行
"""

import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.auth import get_current_user
from api.models import User
from api.dependencies import get_visualization_service
from api.middleware import APIError, ValidationFailedError
from api.core.config import get_settings, Settings
from analysis.AssociationAnalyzer import AssociationAnalyzer

# 共通可視化コンポーネントのインポート
from api.visualization.models import (
    BaseVisualizationRequest,
    BaseVisualizationResponse,
    AssociationVisualizationRequest
)
from api.visualization.errors import (
    handle_visualization_error,
    InvalidAnalysisResultError
)
from api.routers.visualization_helpers import (
    prepare_chart_data_by_analysis_type,
    create_visualization_response
)

logger = logging.getLogger(__name__)

# APIルーター定義
router = APIRouter(
    prefix="/api/association",
    tags=["association"],
    responses={404: {"description": "リソースが見つかりません"}},
)

# リクエスト・レスポンスモデル定義
class AssociationAnalysisParams(BaseModel):
    """アソシエーション分析パラメータモデル"""
    min_support: float = Field(0.01, description="最小サポート値 (0-1)")
    min_confidence: float = Field(0.5, description="最小確信度 (0-1)")
    min_lift: float = Field(1.0, description="最小リフト値")

class AssociationAnalysisRequest(BaseModel):
    """アソシエーション分析リクエストモデル"""
    data: List[Dict[str, Any]] = Field(..., description="分析対象データ")
    params: AssociationAnalysisParams = Field(default_factory=AssociationAnalysisParams)
    visualization_options: Dict[str, Any] = Field(default_factory=dict, description="可視化オプション")

class AssociationVisualizationRequest(BaseModel):
    """アソシエーション分析可視化リクエストモデル"""
    analysis_results: Dict[str, Any] = Field(..., description="アソシエーション分析結果")
    visualization_type: str = Field("network", description="可視化タイプ (network, heatmap, scatter)")
    options: Dict[str, Any] = Field(default_factory=dict, description="可視化設定オプション")

class AssociationVisualizationResponse(BaseModel):
    """アソシエーション分析可視化レスポンスモデル"""
    chart_id: str = Field(..., description="チャートID")
    url: str = Field(..., description="チャートURL")
    format: str = Field(..., description="フォーマット")
    thumbnail_url: Optional[str] = Field(None, description="サムネイルURL")
    metadata: Dict[str, Any] = Field(..., description="メタデータ")
    analysis_summary: Dict[str, Any] = Field(..., description="分析サマリー")

# カスタム例外定義
class AssociationAnalysisError(APIError):
    """アソシエーション分析エラー"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="ASSOCIATION_ANALYSIS_ERROR",
            message=message,
            details=details
        )

class InvalidAssociationDataError(ValidationFailedError):
    """無効なアソシエーションデータエラー"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details)

# ヘルパー関数
def _prepare_chart_data_from_rules(analysis_results: Dict[str, Any],
                                  visualization_type: str,
                                  options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    アソシエーション分析結果からチャートデータを準備する

    Args:
        analysis_results: アソシエーション分析結果
        visualization_type: 可視化タイプ
        options: 可視化オプション

    Returns:
        チャート設定とデータ
    """
    options = options or {}
    rules = analysis_results.get('rules', [])

    if not rules:
        raise InvalidAssociationDataError(
            message="可視化するルールが見つかりません",
            details={"reason": "分析結果にルールが含まれていません"}
        )

    # ルール数の制限（デフォルトは上位20個）
    max_rules = options.get('max_rules', 20)
    if len(rules) > max_rules:
        # リフト値で並べ替え
        rules = sorted(rules, key=lambda x: x.get('lift', 0), reverse=True)[:max_rules]

    # 可視化タイプに基づいてデータを準備
    if visualization_type == "network":
        return _prepare_network_chart_data(rules, options)
    elif visualization_type == "heatmap":
        return _prepare_heatmap_chart_data(rules, options)
    elif visualization_type == "scatter":
        return _prepare_scatter_chart_data(rules, options)
    else:
        raise InvalidAssociationDataError(
            message=f"サポートされていない可視化タイプ: {visualization_type}",
            details={"supported_types": ["network", "heatmap", "scatter"]}
        )

def _prepare_network_chart_data(rules: List[Dict[str, Any]], options: Dict[str, Any]) -> Dict[str, Any]:
    """ネットワークチャート用のデータを準備する"""
    # ノードとエッジのデータ準備
    nodes = set()
    edges = []

    for rule in rules:
        antecedents = rule.get('antecedents', set())
        consequents = rule.get('consequents', set())

        # フロズンセットを文字列にして追加
        for item in antecedents:
            nodes.add(str(item))
        for item in consequents:
            nodes.add(str(item))

        # エッジデータの作成
        for a_item in antecedents:
            for c_item in consequents:
                edges.append({
                    'source': str(a_item),
                    'target': str(c_item),
                    'value': rule.get('lift', 1.0),
                    'confidence': rule.get('confidence', 0),
                    'support': rule.get('support', 0)
                })

    # ネットワークチャート用のデータ構造
    chart_config = {
        "chart_type": "network",
        "title": options.get("title", "アソシエーションルールネットワーク"),
        "width": options.get("width", 800),
        "height": options.get("height", 600),
        "show_legend": True,
        "color_scheme": options.get("color_scheme", "category10")
    }

    chart_data = {
        "labels": list(nodes),
        "datasets": [{
            "label": "アソシエーションルール",
            "data": edges,
            "color": None
        }]
    }

    return {"config": chart_config, "data": chart_data}

def _prepare_heatmap_chart_data(rules: List[Dict[str, Any]], options: Dict[str, Any]) -> Dict[str, Any]:
    """ヒートマップ用のデータを準備する"""
    # アイテムの一覧を取得
    all_items = set()
    for rule in rules:
        for item in rule.get('antecedents', set()):
            all_items.add(str(item))
        for item in rule.get('consequents', set()):
            all_items.add(str(item))

    items_list = list(all_items)

    # マトリックスデータの準備
    matrix_data = []
    for i, item1 in enumerate(items_list):
        row = []
        for j, item2 in enumerate(items_list):
            # 同じアイテム同士の関連は0
            if i == j:
                row.append(0)
            else:
                # 関連の強さを探す
                lift_value = 0
                for rule in rules:
                    antecedents = set(str(x) for x in rule.get('antecedents', set()))
                    consequents = set(str(x) for x in rule.get('consequents', set()))

                    if item1 in antecedents and item2 in consequents:
                        lift_value = max(lift_value, rule.get('lift', 0))

                row.append(lift_value)
        matrix_data.append(row)

    # ヒートマップ用のデータ構造
    chart_config = {
        "chart_type": "heatmap",
        "title": options.get("title", "アソシエーションルールヒートマップ"),
        "width": options.get("width", 800),
        "height": options.get("height", 700),
        "show_legend": True,
        "color_scheme": options.get("color_scheme", "viridis")
    }

    chart_data = {
        "labels": items_list,
        "datasets": [{
            "label": "リフト値",
            "data": matrix_data,
            "color": None
        }]
    }

    return {"config": chart_config, "data": chart_data}

def _prepare_scatter_chart_data(rules: List[Dict[str, Any]], options: Dict[str, Any]) -> Dict[str, Any]:
    """散布図用のデータを準備する"""
    # サポート、確信度、リフト値のデータポイント作成
    data_points = []
    labels = []

    for i, rule in enumerate(rules):
        data_points.append({
            "x": rule.get('support', 0),
            "y": rule.get('confidence', 0),
            "r": rule.get('lift', 1) * 10  # リフト値をバブルサイズに
        })

        # ルールのラベル作成
        antecedents = [str(x) for x in rule.get('antecedents', set())]
        consequents = [str(x) for x in rule.get('consequents', set())]
        label = f"{', '.join(antecedents)} → {', '.join(consequents)}"
        labels.append(label)

    # 散布図用のデータ構造
    chart_config = {
        "chart_type": "bubble",
        "title": options.get("title", "アソシエーションルール散布図"),
        "x_axis_label": "サポート",
        "y_axis_label": "確信度",
        "width": options.get("width", 800),
        "height": options.get("height", 600),
        "show_legend": True,
        "color_scheme": options.get("color_scheme", "viridis")
    }

    chart_data = {
        "labels": labels,
        "datasets": [{
            "label": "アソシエーションルール",
            "data": data_points,
            "color": None
        }]
    }

    return {"config": chart_config, "data": chart_data}

def _format_analysis_summary(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    アソシエーション分析結果のサマリーをフォーマットする

    Args:
        analysis_results: アソシエーション分析結果

    Returns:
        フォーマット済みサマリー
    """
    total_rules = len(analysis_results.get("rules", []))
    top_rules = _get_top_rules(analysis_results.get("rules", []))

    return {
        "total_rules": total_rules,
        "top_rules": top_rules,
        "description": f"アソシエーション分析により{total_rules}のルールが発見されました。"
    }

def _get_top_rules(rules: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """リフト値に基づくトップルールを取得する"""
    if not rules:
        return []

    # リフト値に基づいてソート
    sorted_rules = sorted(rules, key=lambda x: x.get('lift', 0), reverse=True)[:limit]

    # 表示用にフォーマット
    formatted_rules = []
    for rule in sorted_rules:
        antecedents = [str(x) for x in rule.get('antecedents', set())]
        consequents = [str(x) for x in rule.get('consequents', set())]

        formatted_rules.append({
            "rule": f"{', '.join(antecedents)} → {', '.join(consequents)}",
            "support": round(rule.get('support', 0), 3),
            "confidence": round(rule.get('confidence', 0), 3),
            "lift": round(rule.get('lift', 0), 3)
        })

    return formatted_rules

# エンドポイント実装
@router.post("/visualize", response_model=BaseVisualizationResponse, status_code=status.HTTP_200_OK)
async def visualize_association_analysis(
    request: AssociationVisualizationRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """アソシエーション分析結果を可視化する"""
    try:
        logger.info(f"アソシエーション分析可視化リクエスト: type={request.visualization_type}")

        # 共通関数を使用してチャートデータを準備
        chart_data = prepare_chart_data_by_analysis_type(
            analysis_type="association",
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
        logger.exception(f"アソシエーション分析可視化中にエラー: {str(e)}")
        raise handle_visualization_error(e)

@router.post("/analyze-and-visualize", response_model=BaseVisualizationResponse, status_code=status.HTTP_200_OK)
async def analyze_and_visualize(
    request: AssociationAnalysisRequest,
    current_user: User = Depends(get_current_user),
    visualization_service = Depends(get_visualization_service),
    settings: Settings = Depends(get_settings)
):
    """アソシエーション分析と可視化を一度に実行する"""
    try:
        logger.info("アソシエーション分析および可視化リクエスト")

        # アソシエーション分析実行
        analyzer = AssociationAnalyzer()
        analysis_params = {
            "min_support": request.params.min_support,
            "min_confidence": request.params.min_confidence,
            "min_lift": request.params.min_lift
        }

        analysis_results = analyzer.analyze(request.data, analysis_params)

        if not analysis_results or "rules" not in analysis_results or not analysis_results["rules"]:
            raise InvalidAnalysisResultError("ルールが見つかりませんでした。パラメータを調整してください。")

        # 可視化タイプとオプションの設定
        visualization_type = request.visualization_options.get("visualization_type", "network")
        visualization_options = request.visualization_options.copy()

        # 共通関数を使用してチャートデータを準備
        chart_data = prepare_chart_data_by_analysis_type(
            analysis_type="association",
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
        logger.exception(f"アソシエーション分析および可視化中にエラー: {str(e)}")
        raise handle_visualization_error(e)