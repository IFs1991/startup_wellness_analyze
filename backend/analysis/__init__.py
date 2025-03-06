from .base import BaseAnalyzer, AnalysisError
from .association_analyzer import AssociationAnalyzer
from .text_miner import TextMiner
from .cluster_analyzer import ClusterAnalyzer
from .pca_analyzer import PCAAnalyzer
from .survival_analyzer import SurvivalAnalyzer
from .time_series_analyzer import TimeSeriesAnalyzer
from .bayesian_analyzer import BayesianAnalyzer
from .correlation_analysis import CorrelationAnalyzer
from .calculate_descriptive_stats import DescriptiveStatsCalculator
from typing import List, Optional, Dict, Any, Union, Protocol
from datetime import datetime
from dataclasses import dataclass
import logging
import pandas as pd
from enum import Enum

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@dataclass
class AnalysisConfig:
    """分析設定を表すデータクラス"""
    collection_name: str
    target_fields: List[str]
    filters: Optional[List[tuple]] = None
    order_by: Optional[tuple] = None
    limit: Optional[int] = None

class AnalysisType(Enum):
    """分析タイプの定義"""
    CORRELATION = "correlation"
    TIME_SERIES = "time_series"
    REGRESSION = "regression"
    CLUSTER = "cluster"
    SURVIVAL = "survival"
    TEXT_MINING = "text_mining"
    BAYESIAN = "bayesian"
    PCA = "pca"
    ASSOCIATION = "association"

class AnalysisResult:
    """分析結果の基本クラス"""
    def __init__(self, analysis_type: AnalysisType, result_data: Any, metadata: Dict):
        self.analysis_type = analysis_type
        self.result_data = result_data
        self.metadata = metadata
        self.created_at = datetime.utcnow()

    def to_dict(self) -> Dict:
        """分析結果を辞書形式に変換"""
        return {
            "analysis_type": self.analysis_type.value,
            "result_data": self.result_data,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

class AnalysisService(Protocol):
    """分析サービスの基本インターフェース"""
    def analyze(self, data: Any, params: Dict = None) -> AnalysisResult:
        """データを分析し、結果を返す"""
        ...

    def get_analysis_type(self) -> AnalysisType:
        """分析タイプを返す"""
        ...

    def explain_result(self, result: AnalysisResult) -> str:
        """分析結果を自然言語で説明"""
        ...

class AnalysisRegistry:
    """分析サービスのレジストリ"""
    _services: Dict[AnalysisType, AnalysisService] = {}

    @classmethod
    def register(cls, service: AnalysisService):
        """分析サービスを登録"""
        cls._services[service.get_analysis_type()] = service

    @classmethod
    def get_service(cls, analysis_type: AnalysisType) -> AnalysisService:
        """分析サービスを取得"""
        return cls._services.get(analysis_type)

    @classmethod
    def get_all_services(cls) -> List[AnalysisService]:
        """全ての分析サービスを取得"""
        return list(cls._services.values())

__all__ = [
    'BaseAnalyzer',
    'AnalysisError',
    'AssociationAnalyzer',
    'TextMiner',
    'ClusterAnalyzer',
    'PCAAnalyzer',
    'SurvivalAnalyzer',
    'TimeSeriesAnalyzer',
    'BayesianAnalyzer',
    'CorrelationAnalyzer',
    'DescriptiveStatsCalculator'
]
