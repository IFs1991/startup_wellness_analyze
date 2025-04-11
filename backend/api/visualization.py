"""
旧版の可視化モジュール (visualization.py)
-----------------------------------------

このモジュールは非推奨です。以下の新しいモジュールに移行してください：
- api.routers.visualization - 標準の可視化API (/api/visualization)
- api.routers.gemini_visualization - Gemini特化の可視化API (/visualization/gemini)

非推奨の理由:
------------
このモジュールは、APIリファクタリングプロジェクトの一環として非推奨となりました。
routes/ディレクトリとrouters/ディレクトリの統合により、より一貫した
API構造を提供する新しいモジュールへの移行を推奨しています。

後方互換性:
----------
このモジュールは後方互換性のために残されており、自動的に新しい
モジュールにリダイレクトします。既存のインポートや関数呼び出しは
引き続き動作しますが、警告が表示されます。

移行ガイド:
----------
1. import文を更新する:
   変更前: `from api.visualization import generate_chart`
   変更後: `from api.routers.visualization import generate_chart`

2. URLパスを更新する:
   変更前: `/api/v1/visualizations/...`
   変更後: `/api/visualizations/...`

3. 依存関係を更新する:
   すべての依存関係も新しいモジュールからインポートすること。
"""

import logging
import warnings
from typing import Optional, Dict, Any, List

# 構造化ロギング用のロガーを設定
from api.logging_utils import get_logger

logger = get_logger(__name__)

# 非推奨警告を発行
warnings.warn(
    "このモジュールは非推奨です。代わりに api.routers.visualization または "
    "api.routers.gemini_visualization を使用してください。",
    DeprecationWarning,
    stacklevel=2
)

# 後方互換性のためのリダイレクト
# 優先して新しいvisualizationルーターへリダイレクト
from api.routers.visualization import (
    router,
    generate_chart,
    generate_multiple_charts,
    generate_dashboard,
    generate_chart_background,
    get_chart_status
)

# 依存関係も含めてリダイレクト
try:
    from api.routers.visualization import (
        ChartRequest,
        MultipleChartRequest,
        DashboardRequest,
        ChartResponse,
        DashboardResponse,
        get_chart_generator
    )

    logger.info(
        "visualization.py: 新しいvisualizationモジュールからの依存関係をインポートしました",
        extra={"context": {"module": "visualization", "action": "import", "status": "success"}}
    )
except ImportError as e:
    logger.warning(
        "新しい可視化APIの依存関係のインポートに失敗しました。互換性に問題が発生する可能性があります。",
        extra={
            "context": {
                "module": "visualization",
                "action": "import",
                "status": "failed",
                "error": str(e)
            }
        }
    )

# Gemini可視化へのリダイレクト（必要に応じて）
try:
    from api.routers.gemini_visualization import (
        generate_chart as generate_gemini_chart,
        generate_multiple_charts as generate_gemini_multiple_charts,
        generate_dashboard as generate_gemini_dashboard
    )

    logger.info(
        "visualization.py: Gemini可視化モジュールからの関数をインポートしました",
        extra={"context": {"module": "gemini_visualization", "action": "import", "status": "success"}}
    )
except ImportError as e:
    logger.warning(
        "Gemini可視化モジュールが見つかりません。関連機能は利用できません。",
        extra={
            "context": {
                "module": "gemini_visualization",
                "action": "import",
                "status": "failed",
                "error": str(e)
            }
        }
    )

# ドキュメント拡張のための関数アノテーション
def _document_redirect_function(
    original_func,
    target_module: str,
    target_func: str,
    description: str
) -> Any:
    """
    関数のドキュメント文字列を拡張するヘルパー関数

    Args:
        original_func: 元の関数
        target_module: ターゲットモジュールの名前
        target_func: ターゲット関数の名前
        description: 関数の説明

    Returns:
        拡張されたドキュメント文字列を持つ関数
    """
    if original_func.__doc__:
        original_doc = original_func.__doc__
    else:
        original_doc = ""

    redirect_doc = f"""
    {description}

    注意:
        この関数は非推奨です。代わりに `{target_module}.{target_func}` を使用してください。
        このリダイレクト関数は後方互換性のために提供されています。

    元の関数のドキュメント:
    {original_doc}
    """

    original_func.__doc__ = redirect_doc
    return original_func

# 主要な関数のドキュメントを拡張
generate_chart = _document_redirect_function(
    generate_chart,
    "api.routers.visualization",
    "generate_chart",
    "指定された設定とデータに基づいてチャートを生成します。"
)

generate_multiple_charts = _document_redirect_function(
    generate_multiple_charts,
    "api.routers.visualization",
    "generate_multiple_charts",
    "複数のチャートを一度に生成します。"
)

generate_dashboard = _document_redirect_function(
    generate_dashboard,
    "api.routers.visualization",
    "generate_dashboard",
    "複数のチャートを含むダッシュボードを生成します。"
)

generate_chart_background = _document_redirect_function(
    generate_chart_background,
    "api.routers.visualization",
    "generate_chart_background",
    "バックグラウンドジョブとしてチャートを生成します。"
)

get_chart_status = _document_redirect_function(
    get_chart_status,
    "api.routers.visualization",
    "get_chart_status",
    "バックグラウンドチャート生成ジョブのステータスを取得します。"
)