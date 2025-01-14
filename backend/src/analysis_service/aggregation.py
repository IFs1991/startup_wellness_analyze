from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel
import pandas as pd
import numpy as np

class AggregationResult(BaseModel):
    entity_id: str
    entity_type: str
    metrics: Dict[str, float]
    period: str
    aggregation_method: str
    created_at: datetime

class DataAggregator:
    def __init__(self, database):
        self.db = database

    async def aggregate_company_data(
        self,
        company_id: str,
        metrics: List[str],
        period: str,
        aggregation_method: str = "sum"
    ) -> AggregationResult:
        """企業データを集計する"""
        aggregated_metrics = {}

        for metric in metrics:
            # TODO: データベースから企業データを取得する実装
            data = await self._get_company_data(company_id, metric, period)
            aggregated_metrics[metric] = self._apply_aggregation(data, aggregation_method)

        return AggregationResult(
            entity_id=company_id,
            entity_type="company",
            metrics=aggregated_metrics,
            period=period,
            aggregation_method=aggregation_method,
            created_at=datetime.utcnow()
        )

    async def aggregate_group_data(
        self,
        group_id: str,
        metrics: List[str],
        period: str,
        aggregation_method: str = "sum"
    ) -> AggregationResult:
        """グループデータを集計する"""
        # TODO: グループに属する企業のIDリストを取得する実装
        company_ids = await self._get_group_companies(group_id)

        group_metrics = {}
        for metric in metrics:
            metric_values = []
            for company_id in company_ids:
                data = await self._get_company_data(company_id, metric, period)
                metric_values.extend(data)

            group_metrics[metric] = self._apply_aggregation(metric_values, aggregation_method)

        return AggregationResult(
            entity_id=group_id,
            entity_type="group",
            metrics=group_metrics,
            period=period,
            aggregation_method=aggregation_method,
            created_at=datetime.utcnow()
        )

    async def aggregate_portfolio_data(
        self,
        portfolio_id: str,
        metrics: List[str],
        period: str,
        aggregation_method: str = "sum",
        weights: Optional[Dict[str, float]] = None
    ) -> AggregationResult:
        """ポートフォリオデータを集計する"""
        # TODO: ポートフォリオに属する企業のIDリストとウェイトを取得する実装
        portfolio_companies = await self._get_portfolio_companies(portfolio_id)

        if weights is None:
            weights = {company_id: 1.0 for company_id in portfolio_companies}

        portfolio_metrics = {}
        for metric in metrics:
            weighted_values = []
            for company_id in portfolio_companies:
                data = await self._get_company_data(company_id, metric, period)
                weight = weights.get(company_id, 1.0)
                weighted_values.extend([value * weight for value in data])

            portfolio_metrics[metric] = self._apply_aggregation(weighted_values, aggregation_method)

        return AggregationResult(
            entity_id=portfolio_id,
            entity_type="portfolio",
            metrics=portfolio_metrics,
            period=period,
            aggregation_method=aggregation_method,
            created_at=datetime.utcnow()
        )

    def _apply_aggregation(self, data: List[float], method: str) -> float:
        """集計メソッドを適用する"""
        if not data:
            return 0.0

        aggregation_methods = {
            "sum": np.sum,
            "mean": np.mean,
            "median": np.median,
            "min": np.min,
            "max": np.max,
            "std": np.std
        }

        aggregator = aggregation_methods.get(method, np.sum)
        return float(aggregator(data))

    async def _get_company_data(
        self,
        company_id: str,
        metric: str,
        period: str
    ) -> List[float]:
        """企業データを取得する"""
        # TODO: データベースから企業データを取得する実装
        return []

    async def _get_group_companies(self, group_id: str) -> List[str]:
        """グループに属する企業のIDリストを取得する"""
        # TODO: データベースからグループ企業リストを取得する実装
        return []

    async def _get_portfolio_companies(self, portfolio_id: str) -> List[str]:
        """ポートフォリオに属する企業のIDリストを取得する"""
        # TODO: データベースからポートフォリオ企業リストを取得する実装
        return []