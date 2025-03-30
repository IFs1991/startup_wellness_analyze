"""
旧版の可視化モジュール

このモジュールは非推奨です。代わりに以下のモジュールを使用してください：
- api.routers.visualization - Firestoreベースの可視化
- api.routers.gemini_visualization - Geminiベースの可視化
"""

import logging
import warnings

logger = logging.getLogger(__name__)
warnings.warn(
    "このモジュールは非推奨です。代わりに api.routers.visualization または "
    "api.routers.gemini_visualization を使用してください。",
    DeprecationWarning,
    stacklevel=2
)

# 後方互換性のためのリダイレクト
from api.routers.visualization import (
    router,
    create_dashboard,
    generate_graph,
    get_visualizations,
    get_visualization,
    update_visualization,
    delete_visualization
)

# Gemini可視化へのリダイレクト
try:
    from api.routers.gemini_visualization import (
        generate_chart,
        generate_multiple_charts,
        generate_dashboard as generate_gemini_dashboard
    )
except ImportError:
    logger.warning("Gemini可視化モジュールが見つかりません。関連機能は利用できません。")