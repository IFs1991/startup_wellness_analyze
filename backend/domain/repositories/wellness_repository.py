"""
ウェルネスリポジトリインターフェース
ウェルネスデータへのアクセスを抽象化します。
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from domain.entities.wellness import WellnessScore, WellnessMetric, WellnessRecommendation, WellnessDimension


class WellnessRepositoryInterface(ABC):
    """
    ウェルネスデータへのアクセスを抽象化するリポジトリインターフェース
    """

    @abstractmethod
    async def get_score_by_id(self, score_id: str) -> Optional[WellnessScore]:
        """
        IDによりウェルネススコアを取得

        Args:
            score_id: 取得するスコアのID

        Returns:
            WellnessScoreエンティティ、存在しない場合はNone
        """
        pass

    @abstractmethod
    async def get_latest_score_for_user(self, user_id: str) -> Optional[WellnessScore]:
        """
        ユーザーの最新ウェルネススコアを取得

        Args:
            user_id: ユーザーID

        Returns:
            最新のWellnessScoreエンティティ、存在しない場合はNone
        """
        pass

    @abstractmethod
    async def get_latest_score_for_company(self, company_id: str) -> Optional[WellnessScore]:
        """
        企業の最新ウェルネススコアを取得

        Args:
            company_id: 企業ID

        Returns:
            最新のWellnessScoreエンティティ、存在しない場合はNone
        """
        pass

    @abstractmethod
    async def save_score(self, score: WellnessScore) -> WellnessScore:
        """
        ウェルネススコアを保存

        Args:
            score: 保存するWellnessScoreエンティティ

        Returns:
            保存されたWellnessScoreエンティティ

        Raises:
            WellnessRepositoryError: 保存に失敗した場合
        """
        pass

    @abstractmethod
    async def get_score_history(self,
                               user_id: Optional[str] = None,
                               company_id: Optional[str] = None,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               limit: int = 100) -> List[WellnessScore]:
        """
        ウェルネススコアの履歴を取得

        Args:
            user_id: ユーザーID（指定された場合）
            company_id: 企業ID（指定された場合）
            start_date: 開始日時（指定された場合）
            end_date: 終了日時（指定された場合）
            limit: 取得する最大件数

        Returns:
            WellnessScoreエンティティのリスト
        """
        pass

    @abstractmethod
    async def save_metric(self, metric: WellnessMetric) -> WellnessMetric:
        """
        ウェルネスメトリックを保存

        Args:
            metric: 保存するWellnessMetricエンティティ

        Returns:
            保存されたWellnessMetricエンティティ

        Raises:
            WellnessRepositoryError: 保存に失敗した場合
        """
        pass

    @abstractmethod
    async def get_metrics_by_dimension(self,
                                      dimension: WellnessDimension,
                                      user_id: Optional[str] = None,
                                      company_id: Optional[str] = None,
                                      limit: int = 100) -> List[WellnessMetric]:
        """
        特定のディメンションのメトリックを取得

        Args:
            dimension: ウェルネスディメンション
            user_id: ユーザーID（指定された場合）
            company_id: 企業ID（指定された場合）
            limit: 取得する最大件数

        Returns:
            WellnessMetricエンティティのリスト
        """
        pass

    @abstractmethod
    async def get_recommendations(self,
                                user_id: Optional[str] = None,
                                company_id: Optional[str] = None,
                                dimensions: Optional[List[WellnessDimension]] = None,
                                limit: int = 10) -> List[WellnessRecommendation]:
        """
        ウェルネス改善のための推奨事項を取得

        Args:
            user_id: ユーザーID（指定された場合）
            company_id: 企業ID（指定された場合）
            dimensions: 対象とするディメンションのリスト（指定された場合）
            limit: 取得する最大件数

        Returns:
            WellnessRecommendationエンティティのリスト
        """
        pass

    @abstractmethod
    async def save_recommendation(self, recommendation: WellnessRecommendation) -> WellnessRecommendation:
        """
        ウェルネス推奨事項を保存

        Args:
            recommendation: 保存するWellnessRecommendationエンティティ

        Returns:
            保存されたWellnessRecommendationエンティティ

        Raises:
            WellnessRepositoryError: 保存に失敗した場合
        """
        pass

    @abstractmethod
    async def get_average_scores_by_dimension(self,
                                             company_id: str,
                                             start_date: Optional[datetime] = None,
                                             end_date: Optional[datetime] = None) -> Dict[WellnessDimension, float]:
        """
        企業のディメンション別平均スコアを取得

        Args:
            company_id: 企業ID
            start_date: 開始日時（指定された場合）
            end_date: 終了日時（指定された場合）

        Returns:
            ディメンションごとの平均スコアの辞書
        """
        pass