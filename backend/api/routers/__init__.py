"""
API ルーターモジュール
APIルーターの集中管理と初期化を行う
"""
from fastapi import APIRouter
from typing import Dict, Any

# 型付きルーターオブジェクトの作成
class RouterModule:
    """ルーターモジュールのラッパークラス"""
    def __init__(self, router: APIRouter):
        """
        Parameters:
            router (APIRouter): FastAPIのルーターインスタンス
        """
        self._router = router

    def __call__(self) -> APIRouter:
        """
        ルーターインスタンスを返す
        Returns:
            APIRouter: FastAPIのルーターインスタンス
        """
        return self._router

# 各機能別ルーターのインポート
from .auth import router as auth_router
from .data_input import router as data_input_router
from .analysis import router as analysis_router
from .visualization import router as visualization_router
from .data_processing import router as data_processing_router
from .prediction import router as prediction_router
from .report_generation import router as report_generation_router

# APIルーターのエクスポート
# 各モジュールのルーターをパブリックインターフェースとして提供
auth = RouterModule(auth_router)
data_input = RouterModule(data_input_router)
analysis = RouterModule(analysis_router)
visualization = RouterModule(visualization_router)
data_processing = RouterModule(data_processing_router)
prediction = RouterModule(prediction_router)
report_generation = RouterModule(report_generation_router)

# バージョン情報
__version__ = "1.0.0"
