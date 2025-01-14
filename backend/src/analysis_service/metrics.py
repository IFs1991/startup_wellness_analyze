from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel
import numpy as np
from scipy import stats

class KPIMetric(BaseModel):
    name: str
    value: float
    target: float
    achievement_rate: float
    trend: str
    period: str
    created_at: datetime

class TrendAnalysis(BaseModel):
    metric_name: str
    values: List[float]
    trend_direction: str
    growth_rate: float
    seasonality: Optional[bool]
    created_at: datetime

class GrowthMetric(BaseModel):
    metric_name: str
    current_value: float
    previous_value: float
    growth_rate: float
    annualized_growth: float
    created_at: datetime

class MetricsProcessor:
    def __init__(self, database):
        self.db = database

    async def calculate_kpis(
        self,
        company_id: str,
        metrics: List[str],
        period: str
    ) -> List[KPIMetric]:
        """KPIを計算する"""
        kpi_results = []

        for metric in metrics:
            # TODO: データベースからメトリクスデータを取得する実装
            metric_data = await self._get_metric_data(company_id, metric, period)
            target = await self._get_metric_target(company_id, metric, period)

            kpi = KPIMetric(
                name=metric,
                value=metric_data["value"],
                target=target,
                achievement_rate=metric_data["value"] / target if target != 0 else 0,
                trend=self._calculate_trend(metric_data["history"]),
                period=period,
                created_at=datetime.utcnow()
            )
            kpi_results.append(kpi)

        return kpi_results

    async def analyze_trends(
        self,
        company_id: str,
        metrics: List[str],
        period: str,
        window_size: int = 12
    ) -> List[TrendAnalysis]:
        """トレンド分析を実行する"""
        trend_results = []

        for metric in metrics:
            # TODO: データベースから時系列データを取得する実装
            time_series = await self._get_time_series_data(company_id, metric, period)

            trend_analysis = TrendAnalysis(
                metric_name=metric,
                values=time_series,
                trend_direction=self._determine_trend_direction(time_series),
                growth_rate=self._calculate_growth_rate(time_series),
                seasonality=self._detect_seasonality(time_series),
                created_at=datetime.utcnow()
            )
            trend_results.append(trend_analysis)

        return trend_results

    async def calculate_growth_metrics(
        self,
        company_id: str,
        metrics: List[str],
        period: str
    ) -> List[GrowthMetric]:
        """成長指標を計算する"""
        growth_results = []

        for metric in metrics:
            current_data = await self._get_current_metric(company_id, metric, period)
            previous_data = await self._get_previous_metric(company_id, metric, period)

            growth = GrowthMetric(
                metric_name=metric,
                current_value=current_data,
                previous_value=previous_data,
                growth_rate=self._calculate_period_growth_rate(current_data, previous_data),
                annualized_growth=self._calculate_annualized_growth(current_data, previous_data, period),
                created_at=datetime.utcnow()
            )
            growth_results.append(growth)

        return growth_results

    def _calculate_trend(self, history: List[float]) -> str:
        """トレンドを計算する"""
        if len(history) < 2:
            return "insufficient_data"

        slope, _ = np.polyfit(range(len(history)), history, 1)
        if slope > 0:
            return "upward"
        elif slope < 0:
            return "downward"
        return "stable"

    def _determine_trend_direction(self, values: List[float]) -> str:
        """トレンドの方向を判定する"""
        if len(values) < 2:
            return "insufficient_data"

        # 移動平均を使用してトレンドを判定
        ma = np.convolve(values, np.ones(3)/3, mode='valid')
        if ma[-1] > ma[0]:
            return "upward"
        elif ma[-1] < ma[0]:
            return "downward"
        return "stable"

    def _calculate_growth_rate(self, values: List[float]) -> float:
        """成長率を計算する"""
        if len(values) < 2:
            return 0.0
        return (values[-1] - values[0]) / values[0] if values[0] != 0 else 0.0

    def _detect_seasonality(self, values: List[float]) -> Optional[bool]:
        """季節性を検出する"""
        if len(values) < 12:  # 最低1年分のデータが必要
            return None

        # 自己相関を使用して季節性を検出
        acf = np.correlate(values, values, mode='full')
        acf = acf[len(acf)//2:]

        # 季節性の閾値を設定
        threshold = 0.7
        return any(correlation > threshold for correlation in acf[1:])

    def _calculate_period_growth_rate(self, current: float, previous: float) -> float:
        """期間成長率を計算する"""
        return (current - previous) / previous if previous != 0 else 0.0

    def _calculate_annualized_growth(self, current: float, previous: float, period: str) -> float:
        """年率換算成長率を計算する"""
        growth_rate = self._calculate_period_growth_rate(current, previous)

        # 期間に応じて年率換算
        period_multipliers = {
            "monthly": 12,
            "quarterly": 4,
            "semi_annual": 2,
            "annual": 1
        }

        multiplier = period_multipliers.get(period, 1)
        return (1 + growth_rate) ** multiplier - 1

    async def _get_metric_data(self, company_id: str, metric: str, period: str) -> Dict:
        """メトリクスデータを取得する"""
        # TODO: データベースからメトリクスデータを取得する実装
        return {"value": 0.0, "history": []}

    async def _get_metric_target(self, company_id: str, metric: str, period: str) -> float:
        """メトリクスの目標値を取得する"""
        # TODO: データベースから目��値を取得する実装
        return 0.0

    async def _get_time_series_data(self, company_id: str, metric: str, period: str) -> List[float]:
        """時系列データを取得する"""
        # TODO: データベースから時系列データを取得する実装
        return []

    async def _get_current_metric(self, company_id: str, metric: str, period: str) -> float:
        """現在のメトリクス値を取得する"""
        # TODO: データベースから現在の値を取得する実装
        return 0.0

    async def _get_previous_metric(self, company_id: str, metric: str, period: str) -> float:
        """前期のメトリクス値を取得する"""
        # TODO: データベースから前期の値を取得する実装
        return 0.0