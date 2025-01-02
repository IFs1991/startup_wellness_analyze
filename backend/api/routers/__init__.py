"""
API ルーターモジュール
"""

from .auth import router as auth_router
from .data_input import router as data_input_router
from .analysis import router as analysis_router
from .visualization import router as visualization_router
from .data_processing import router as data_processing_router
from .prediction import router as prediction_router
from .report_generation import router as report_generation_router

# エクスポートするルーター
auth = auth_router
data_input = data_input_router
analysis = analysis_router
visualization = visualization_router
data_processing = data_processing_router
prediction = prediction_router
report_generation = report_generation_router
