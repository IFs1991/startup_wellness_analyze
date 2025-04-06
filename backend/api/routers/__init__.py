# -*- coding: utf-8 -*-
"""
API ルーターモジュール
----------------------
APIルーターを集約して提供します。
このモジュールは、各種APIエンドポイントを機能別に分類し、
FastAPIルーターとして提供します。
"""

# 各ルーターをインポート
from . import auth
from . import analysis
from . import prediction
from . import compliance
from . import reports
from . import gemini_visualization
# from . import federated  # 連合学習ルーターを追加 (一時的に無効化)
from . import financial  # 財務分析ルーターを追加
from . import market     # 市場分析ルーターを追加
from . import team       # チーム分析ルーターを追加
from . import companies  # 企業情報ルーターを追加

# エクスポートするルーター一覧
all_routers = [
    auth.router,
    analysis.router,
    prediction.router,
    compliance.router,
    reports.router,
    gemini_visualization.router,
    # federated.router,  # 連合学習ルーターを追加 (一時的に無効化)
    financial.router,  # 財務分析ルーターを追加
    market.router,     # 市場分析ルーターを追加
    team.router,       # チーム分析ルーターを追加
    companies.router   # 企業情報ルーターを追加
]

__all__ = [
    "auth",
    "analysis",
    "prediction",
    "compliance",
    "reports",
    "gemini_visualization",
    # "federated",  # 連合学習ルーターを追加 (一時的に無効化)
    "financial",  # 財務分析ルーターを追加
    "market",     # 市場分析ルーターを追加
    "team",       # チーム分析ルーターを追加
    "companies",  # 企業情報ルーターを追加
    "all_routers"
]
