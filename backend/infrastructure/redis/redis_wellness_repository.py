"""
Redisウェルネスリポジトリ

ウェルネススコア情報をRedisにキャッシュするリポジトリの実装。
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from backend.domain.models.wellness import (
    RecommendationAction,
    RecommendationPlan,
    ScoreCategory,
    ScoreHistory,
    ScoreMetric,
    WellnessScore,
)
from backend.domain.repositories.wellness_repository import WellnessRepositoryInterface
from backend.core.exceptions import DataNotFoundError
from backend.infrastructure.redis.redis_service import RedisService

logger = logging.getLogger(__name__)


class RedisWellnessRepository(WellnessRepositoryInterface):
    """
    Redisを使用したウェルネスデータのキャッシュリポジトリ実装。
    デコレータパターンを使用して、メインリポジトリの前にキャッシュ層として機能します。
    """

    # キャッシュキーのプレフィックス
    SCORE_KEY_PREFIX = "wellness:score:"
    COMPANY_LATEST_SCORE_KEY_PREFIX = "wellness:company:latest:"
    COMPANY_SCORES_KEY_PREFIX = "wellness:company:scores:"
    COMPANY_HISTORY_KEY_PREFIX = "wellness:company:history:"
    METRIC_KEY_PREFIX = "wellness:metric:"
    COMPANY_METRICS_KEY_PREFIX = "wellness:company:metrics:"
    TREND_KEY_PREFIX = "wellness:trend:"
    PLAN_KEY_PREFIX = "wellness:plan:"
    BENCHMARK_KEY_PREFIX = "wellness:benchmark:"

    def __init__(
        self,
        redis_service: RedisService,
        main_repository: WellnessRepositoryInterface,
        ttl_seconds: int = 3600  # デフォルト: 1時間
    ):
        """
        初期化メソッド

        Args:
            redis_service: Redisサービスインスタンス
            main_repository: メインのウェルネスリポジトリ実装
            ttl_seconds: キャッシュの有効期限（秒）
        """
        self.redis = redis_service
        self.main_repository = main_repository
        self.ttl = ttl_seconds

    async def save_score(self, score: WellnessScore) -> WellnessScore:
        """
        ウェルネススコアの保存。メインリポジトリに保存し、キャッシュを更新します。

        Args:
            score: 保存するスコア

        Returns:
            保存されたスコア
        """
        # メインリポジトリに保存
        saved_score = await self.main_repository.save_score(score)
        logger.info(f"ウェルネススコア(ID: {saved_score.id})を保存しました")

        # スコアをキャッシュに保存
        score_data = self._serialize_score(saved_score)
        await self.redis.set_json(self._get_score_key(saved_score.id), score_data, self.ttl)

        # 最新スコアキャッシュを更新
        await self.redis.set_json(
            self._get_company_latest_score_key(saved_score.company_id),
            score_data,
            self.ttl
        )

        # 関連するキャッシュを無効化
        company_scores_key = self._get_company_scores_key(saved_score.company_id)
        company_history_key = self._get_company_history_key(saved_score.company_id)
        await self.redis.delete_key(company_scores_key)
        await self.redis.delete_key(company_history_key)

        return saved_score

    async def get_score_by_id(self, score_id: str) -> Optional[WellnessScore]:
        """
        IDによるスコア取得。キャッシュにあればそこから、なければメインリポジトリから取得します。

        Args:
            score_id: スコアID

        Returns:
            スコアが存在する場合はWellnessScoreオブジェクト、存在しない場合はNone
        """
        # キャッシュからの取得を試みる
        cache_key = self._get_score_key(score_id)
        cached_data = await self.redis.get_json(cache_key)

        if cached_data:
            # キャッシュヒット
            logger.debug(f"ウェルネススコア(ID: {score_id})のキャッシュヒット")
            return self._deserialize_score(cached_data)

        # キャッシュミス - メインリポジトリから取得
        try:
            logger.debug(f"ウェルネススコア(ID: {score_id})のキャッシュミス、メインリポジトリから取得")
            score = await self.main_repository.get_score_by_id(score_id)
            if score:
                # キャッシュに保存
                score_data = self._serialize_score(score)
                await self.redis.set_json(cache_key, score_data, self.ttl)
            return score
        except Exception as e:
            logger.error(f"ウェルネススコア(ID: {score_id})の取得エラー: {str(e)}")
            raise

    async def get_latest_score(self, company_id: str) -> Optional[WellnessScore]:
        """
        最新のスコア取得。キャッシュにあればそこから、なければメインリポジトリから取得します。

        Args:
            company_id: 会社ID

        Returns:
            最新のスコア
        """
        # キャッシュからの取得を試みる
        cache_key = self._get_company_latest_score_key(company_id)
        cached_data = await self.redis.get_json(cache_key)

        if cached_data:
            # キャッシュヒット
            logger.debug(f"会社(ID: {company_id})の最新ウェルネススコアのキャッシュヒット")
            return self._deserialize_score(cached_data)

        # キャッシュミス - メインリポジトリから取得
        try:
            logger.debug(f"会社(ID: {company_id})の最新ウェルネススコアのキャッシュミス、メインリポジトリから取得")
            score = await self.main_repository.get_latest_score(company_id)
            if score:
                # キャッシュに保存
                score_data = self._serialize_score(score)
                await self.redis.set_json(cache_key, score_data, self.ttl)
            return score
        except Exception as e:
            logger.error(f"会社(ID: {company_id})の最新ウェルネススコアの取得エラー: {str(e)}")
            raise

    async def get_scores_by_company(
        self,
        company_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10
    ) -> List[WellnessScore]:
        """
        会社IDによるスコア一覧の取得。パラメータが同一の場合はキャッシュを使用します。

        Args:
            company_id: 会社ID
            start_date: 開始日時
            end_date: 終了日時
            limit: 取得する最大件数

        Returns:
            スコアのリスト
        """
        # キャッシュキーを生成（日付範囲とリミットを含む）
        cache_params = {
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "limit": limit
        }
        cache_key = f"{self._get_company_scores_key(company_id)}:{json.dumps(cache_params)}"

        # キャッシュからの取得を試みる
        cached_data = await self.redis.get_json(cache_key)
        if cached_data:
            # キャッシュヒット
            logger.debug(f"会社(ID: {company_id})のスコア一覧のキャッシュヒット")
            return [self._deserialize_score(item) for item in cached_data]

        # キャッシュミス - メインリポジトリから取得
        try:
            logger.debug(f"会社(ID: {company_id})のスコア一覧のキャッシュミス、メインリポジトリから取得")
            scores = await self.main_repository.get_scores_by_company(company_id, start_date, end_date, limit)

            # キャッシュに保存
            if scores:
                scores_data = [self._serialize_score(score) for score in scores]
                await self.redis.set_json(cache_key, scores_data, self.ttl)

            return scores
        except Exception as e:
            logger.error(f"会社(ID: {company_id})のスコア一覧の取得エラー: {str(e)}")
            raise

    async def get_score_history(
        self,
        company_id: str,
        time_period: str = "monthly",
        limit: int = 12
    ) -> ScoreHistory:
        """
        スコア履歴の取得。パラメータが同一の場合はキャッシュを使用します。

        Args:
            company_id: 会社ID
            time_period: 期間種別 (monthly, quarterly, yearly)
            limit: 取得する履歴の数

        Returns:
            スコア履歴
        """
        # キャッシュキーを生成
        cache_params = {
            "time_period": time_period,
            "limit": limit
        }
        cache_key = f"{self._get_company_history_key(company_id)}:{json.dumps(cache_params)}"

        # キャッシュからの取得を試みる
        cached_data = await self.redis.get_json(cache_key)
        if cached_data:
            # キャッシュヒット
            logger.debug(f"会社(ID: {company_id})のスコア履歴のキャッシュヒット")
            return self._deserialize_score_history(cached_data)

        # キャッシュミス - メインリポジトリから取得
        try:
            logger.debug(f"会社(ID: {company_id})のスコア履歴のキャッシュミス、メインリポジトリから取得")
            history = await self.main_repository.get_score_history(company_id, time_period, limit)

            # キャッシュに保存
            history_data = self._serialize_score_history(history)
            await self.redis.set_json(cache_key, history_data, self.ttl)

            return history
        except Exception as e:
            logger.error(f"会社(ID: {company_id})のスコア履歴の取得エラー: {str(e)}")
            raise

    async def save_metric(self, metric: ScoreMetric, company_id: str) -> ScoreMetric:
        """
        メトリックの保存。メインリポジトリに保存し、関連するキャッシュを無効化します。

        Args:
            metric: 保存するメトリック
            company_id: 会社ID

        Returns:
            保存されたメトリック
        """
        # メインリポジトリに保存
        saved_metric = await self.main_repository.save_metric(metric, company_id)
        logger.info(f"会社(ID: {company_id})のメトリック({saved_metric.name})を保存しました")

        # 関連するキャッシュを無効化（該当会社のメトリックキャッシュすべて）
        company_metrics_key_prefix = f"{self.COMPANY_METRICS_KEY_PREFIX}{company_id}"
        keys = await self.redis.keys(f"{company_metrics_key_prefix}*")
        for key in keys:
            await self.redis.delete_key(key)

        # トレンドキャッシュも無効化
        trend_key_prefix = f"{self.TREND_KEY_PREFIX}{company_id}"
        trend_keys = await self.redis.keys(f"{trend_key_prefix}*")
        for key in trend_keys:
            await self.redis.delete_key(key)

        return saved_metric

    async def get_metrics_by_company(
        self,
        company_id: str,
        category: Optional[ScoreCategory] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[ScoreMetric]:
        """
        会社IDによるメトリック一覧の取得。パラメータが同一の場合はキャッシュを使用します。

        Args:
            company_id: 会社ID
            category: スコアカテゴリ（指定された場合）
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            メトリックのリスト
        """
        # キャッシュキーを生成
        cache_params = {
            "category": category.value if category else None,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None
        }
        cache_key = f"{self._get_company_metrics_key(company_id)}:{json.dumps(cache_params)}"

        # キャッシュからの取得を試みる
        cached_data = await self.redis.get_json(cache_key)
        if cached_data:
            # キャッシュヒット
            logger.debug(f"会社(ID: {company_id})のメトリック一覧のキャッシュヒット")
            return [self._deserialize_metric(item) for item in cached_data]

        # キャッシュミス - メインリポジトリから取得
        try:
            logger.debug(f"会社(ID: {company_id})のメトリック一覧のキャッシュミス、メインリポジトリから取得")
            metrics = await self.main_repository.get_metrics_by_company(company_id, category, start_date, end_date)

            # キャッシュに保存
            if metrics:
                metrics_data = [self._serialize_metric(metric) for metric in metrics]
                await self.redis.set_json(cache_key, metrics_data, self.ttl)

            return metrics
        except Exception as e:
            logger.error(f"会社(ID: {company_id})のメトリック一覧の取得エラー: {str(e)}")
            raise

    async def get_category_trend(
        self,
        company_id: str,
        category: ScoreCategory,
        periods: int = 6
    ) -> List[Tuple[datetime, float]]:
        """
        カテゴリごとのトレンド取得。パラメータが同一の場合はキャッシュを使用します。

        Args:
            company_id: 会社ID
            category: スコアカテゴリ
            periods: 取得する期間数

        Returns:
            (日時, スコア)のリスト
        """
        # キャッシュキーを生成
        cache_params = {
            "category": category.value,
            "periods": periods
        }
        cache_key = f"{self._get_trend_key(company_id)}:{json.dumps(cache_params)}"

        # キャッシュからの取得を試みる
        cached_data = await self.redis.get_json(cache_key)
        if cached_data:
            # キャッシュヒット
            logger.debug(f"会社(ID: {company_id})のカテゴリ({category.value})トレンドのキャッシュヒット")
            return [(datetime.fromisoformat(item[0]), item[1]) for item in cached_data]

        # キャッシュミス - メインリポジトリから取得
        try:
            logger.debug(f"会社(ID: {company_id})のカテゴリ({category.value})トレンドのキャッシュミス、メインリポジトリから取得")
            trend_data = await self.main_repository.get_category_trend(company_id, category, periods)

            # キャッシュに保存
            if trend_data:
                # datetimeオブジェクトをシリアライズ可能な形式に変換
                serializable_data = [(dt.isoformat(), score) for dt, score in trend_data]
                await self.redis.set_json(cache_key, serializable_data, self.ttl)

            return trend_data
        except Exception as e:
            logger.error(f"会社(ID: {company_id})のカテゴリ({category.value})トレンドの取得エラー: {str(e)}")
            raise

    async def save_recommendation_plan(self, plan: RecommendationPlan) -> RecommendationPlan:
        """
        推奨プランの保存。メインリポジトリに保存し、キャッシュを更新します。

        Args:
            plan: 保存する推奨プラン

        Returns:
            保存された推奨プラン
        """
        # メインリポジトリに保存
        saved_plan = await self.main_repository.save_recommendation_plan(plan)
        logger.info(f"会社(ID: {saved_plan.company_id})のスコア(ID: {saved_plan.score_id})の推奨プランを保存しました")

        # キャッシュに保存
        plan_data = self._serialize_recommendation_plan(saved_plan)

        # スコアIDあり/なしの両方のキャッシュを更新
        base_key = self._get_plan_key(saved_plan.company_id)
        score_key = f"{base_key}:{saved_plan.score_id}"

        await self.redis.set_json(base_key, plan_data, self.ttl)  # スコアIDなし
        await self.redis.set_json(score_key, plan_data, self.ttl)  # スコアIDあり

        return saved_plan

    async def get_recommendation_plan(
        self,
        company_id: str,
        score_id: Optional[str] = None
    ) -> Optional[RecommendationPlan]:
        """
        推奨プランの取得。キャッシュにあればそこから、なければメインリポジトリから取得します。

        Args:
            company_id: 会社ID
            score_id: スコアID（指定された場合）

        Returns:
            推奨プラン
        """
        # キャッシュキーを生成
        cache_key = self._get_plan_key(company_id)
        if score_id:
            cache_key = f"{cache_key}:{score_id}"

        # キャッシュからの取得を試みる
        cached_data = await self.redis.get_json(cache_key)
        if cached_data:
            # キャッシュヒット
            logger.debug(f"会社(ID: {company_id})の推奨プランのキャッシュヒット")
            return self._deserialize_recommendation_plan(cached_data)

        # キャッシュミス - メインリポジトリから取得
        try:
            logger.debug(f"会社(ID: {company_id})の推奨プランのキャッシュミス、メインリポジトリから取得")
            plan = await self.main_repository.get_recommendation_plan(company_id, score_id)

            # キャッシュに保存
            if plan:
                plan_data = self._serialize_recommendation_plan(plan)
                await self.redis.set_json(cache_key, plan_data, self.ttl)

            return plan
        except Exception as e:
            logger.error(f"会社(ID: {company_id})の推奨プランの取得エラー: {str(e)}")
            raise

    async def get_company_benchmarks(
        self,
        company_id: str,
        category: Optional[ScoreCategory] = None
    ) -> Dict[str, float]:
        """
        業界ベンチマークの取得。キャッシュにあればそこから、なければメインリポジトリから取得します。

        Args:
            company_id: 会社ID
            category: スコアカテゴリ（指定された場合）

        Returns:
            ベンチマークスコアのマップ
        """
        # キャッシュキーを生成
        cache_key = self._get_benchmark_key(company_id)
        if category:
            cache_key = f"{cache_key}:{category.value}"

        # キャッシュからの取得を試みる
        cached_data = await self.redis.get_json(cache_key)
        if cached_data:
            # キャッシュヒット
            logger.debug(f"会社(ID: {company_id})のベンチマークのキャッシュヒット")
            return cached_data

        # キャッシュミス - メインリポジトリから取得
        try:
            logger.debug(f"会社(ID: {company_id})のベンチマークのキャッシュミス、メインリポジトリから取得")
            benchmarks = await self.main_repository.get_company_benchmarks(company_id, category)

            # キャッシュに保存
            if benchmarks:
                await self.redis.set_json(cache_key, benchmarks, self.ttl)

            return benchmarks
        except Exception as e:
            logger.error(f"会社(ID: {company_id})のベンチマークの取得エラー: {str(e)}")
            raise

    # プライベートメソッド: キー生成関数
    def _get_score_key(self, score_id: str) -> str:
        """スコアIDからRedisキーを生成"""
        return f"{self.SCORE_KEY_PREFIX}{score_id}"

    def _get_company_latest_score_key(self, company_id: str) -> str:
        """会社IDから最新スコアのRedisキーを生成"""
        return f"{self.COMPANY_LATEST_SCORE_KEY_PREFIX}{company_id}"

    def _get_company_scores_key(self, company_id: str) -> str:
        """会社IDからスコア一覧のRedisキーを生成"""
        return f"{self.COMPANY_SCORES_KEY_PREFIX}{company_id}"

    def _get_company_history_key(self, company_id: str) -> str:
        """会社IDからスコア履歴のRedisキーを生成"""
        return f"{self.COMPANY_HISTORY_KEY_PREFIX}{company_id}"

    def _get_company_metrics_key(self, company_id: str) -> str:
        """会社IDからメトリック一覧のRedisキーを生成"""
        return f"{self.COMPANY_METRICS_KEY_PREFIX}{company_id}"

    def _get_trend_key(self, company_id: str) -> str:
        """会社IDからトレンドのRedisキーを生成"""
        return f"{self.TREND_KEY_PREFIX}{company_id}"

    def _get_plan_key(self, company_id: str) -> str:
        """会社IDから推奨プランのRedisキーを生成"""
        return f"{self.PLAN_KEY_PREFIX}{company_id}"

    def _get_benchmark_key(self, company_id: str) -> str:
        """会社IDからベンチマークのRedisキーを生成"""
        return f"{self.BENCHMARK_KEY_PREFIX}{company_id}"

    # シリアライゼーション/デシリアライゼーション関数
    def _serialize_score(self, score: WellnessScore) -> Dict[str, Any]:
        """WellnessScoreオブジェクトを辞書に変換"""
        return {
            "id": score.id,
            "company_id": score.company_id,
            "total_score": score.total_score,
            "category_scores": {k.value: v for k, v in score.category_scores.items()},
            "metrics": [self._serialize_metric(m) for m in score.metrics],
            "timestamp": score.timestamp.isoformat() if score.timestamp else None,
            "created_by": score.created_by
        }

    def _deserialize_score(self, data: Dict[str, Any]) -> WellnessScore:
        """辞書からWellnessScoreオブジェクトを復元"""
        # カテゴリスコアの変換
        category_scores = {}
        for k, v in data.get("category_scores", {}).items():
            try:
                category = ScoreCategory(k)
                category_scores[category] = v
            except ValueError:
                # 不明なカテゴリの場合はスキップ
                logger.warning(f"不明なスコアカテゴリ: {k}")

        # メトリックの変換
        metrics = []
        for m_data in data.get("metrics", []):
            try:
                metrics.append(self._deserialize_metric(m_data))
            except Exception as e:
                logger.warning(f"メトリックの変換エラー: {str(e)}")

        # タイムスタンプの変換
        timestamp = None
        if data.get("timestamp"):
            try:
                timestamp = datetime.fromisoformat(data["timestamp"])
            except ValueError:
                logger.warning(f"タイムスタンプの変換エラー: {data['timestamp']}")

        return WellnessScore(
            id=data.get("id", ""),
            company_id=data.get("company_id", ""),
            total_score=data.get("total_score", 0.0),
            category_scores=category_scores,
            metrics=metrics,
            timestamp=timestamp,
            created_by=data.get("created_by")
        )

    def _serialize_metric(self, metric: ScoreMetric) -> Dict[str, Any]:
        """ScoreMetricオブジェクトを辞書に変換"""
        return {
            "name": metric.name,
            "value": metric.value,
            "weight": metric.weight,
            "category": metric.category.value,
            "timestamp": metric.timestamp.isoformat() if metric.timestamp else None,
            "raw_data": metric.raw_data
        }

    def _deserialize_metric(self, data: Dict[str, Any]) -> ScoreMetric:
        """辞書からScoreMetricオブジェクトを復元"""
        # カテゴリの変換
        category = ScoreCategory.FINANCIAL  # デフォルト
        if data.get("category"):
            try:
                category = ScoreCategory(data["category"])
            except ValueError:
                logger.warning(f"不明なメトリックカテゴリ: {data['category']}")

        # タイムスタンプの変換
        timestamp = None
        if data.get("timestamp"):
            try:
                timestamp = datetime.fromisoformat(data["timestamp"])
            except ValueError:
                logger.warning(f"タイムスタンプの変換エラー: {data['timestamp']}")

        return ScoreMetric(
            name=data.get("name", ""),
            value=data.get("value", 0.0),
            weight=data.get("weight", 1.0),
            category=category,
            timestamp=timestamp,
            raw_data=data.get("raw_data", {})
        )

    def _serialize_score_history(self, history: ScoreHistory) -> Dict[str, Any]:
        """ScoreHistoryオブジェクトを辞書に変換"""
        return {
            "company_id": history.company_id,
            "scores": [self._serialize_score(s) for s in history.scores],
            "time_period": history.time_period
        }

    def _deserialize_score_history(self, data: Dict[str, Any]) -> ScoreHistory:
        """辞書からScoreHistoryオブジェクトを復元"""
        scores = []
        for s_data in data.get("scores", []):
            try:
                scores.append(self._deserialize_score(s_data))
            except Exception as e:
                logger.warning(f"スコアの変換エラー: {str(e)}")

        return ScoreHistory(
            company_id=data.get("company_id", ""),
            scores=scores,
            time_period=data.get("time_period", "monthly")
        )

    def _serialize_recommendation_action(self, action: RecommendationAction) -> Dict[str, Any]:
        """RecommendationActionオブジェクトを辞書に変換"""
        return {
            "id": action.id,
            "title": action.title,
            "description": action.description,
            "category": action.category.value,
            "impact_level": action.impact_level,
            "effort_level": action.effort_level,
            "time_frame": action.time_frame,
            "resources": action.resources
        }

    def _deserialize_recommendation_action(self, data: Dict[str, Any]) -> RecommendationAction:
        """辞書からRecommendationActionオブジェクトを復元"""
        # カテゴリの変換
        category = ScoreCategory.FINANCIAL  # デフォルト
        if data.get("category"):
            try:
                category = ScoreCategory(data["category"])
            except ValueError:
                logger.warning(f"不明な推奨アクションカテゴリ: {data['category']}")

        return RecommendationAction(
            id=data.get("id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            category=category,
            impact_level=data.get("impact_level", 0),
            effort_level=data.get("effort_level", 0),
            time_frame=data.get("time_frame", ""),
            resources=data.get("resources", [])
        )

    def _serialize_recommendation_plan(self, plan: RecommendationPlan) -> Dict[str, Any]:
        """RecommendationPlanオブジェクトを辞書に変換"""
        return {
            "company_id": plan.company_id,
            "score_id": plan.score_id,
            "actions": [self._serialize_recommendation_action(a) for a in plan.actions],
            "generated_at": plan.generated_at.isoformat() if plan.generated_at else None,
            "generated_by": plan.generated_by
        }

    def _deserialize_recommendation_plan(self, data: Dict[str, Any]) -> RecommendationPlan:
        """辞書からRecommendationPlanオブジェクトを復元"""
        # アクションの変換
        actions = []
        for a_data in data.get("actions", []):
            try:
                actions.append(self._deserialize_recommendation_action(a_data))
            except Exception as e:
                logger.warning(f"推奨アクションの変換エラー: {str(e)}")

        # 生成日時の変換
        generated_at = None
        if data.get("generated_at"):
            try:
                generated_at = datetime.fromisoformat(data["generated_at"])
            except ValueError:
                logger.warning(f"生成日時の変換エラー: {data['generated_at']}")

        return RecommendationPlan(
            company_id=data.get("company_id", ""),
            score_id=data.get("score_id", ""),
            actions=actions,
            generated_at=generated_at,
            generated_by=data.get("generated_by")
        )


def create_redis_wellness_repository(
    redis_service: RedisService,
    main_repository: WellnessRepositoryInterface,
    ttl: int = 3600
) -> RedisWellnessRepository:
    """
    RedisWellnessRepositoryインスタンスを作成するファクトリ関数

    Args:
        redis_service: Redisサービスのインスタンス
        main_repository: メインのウェルネスリポジトリ実装
        ttl: キャッシュの有効期限（秒）

    Returns:
        設定されたRedisWellnessRepositoryインスタンス
    """
    return RedisWellnessRepository(
        redis_service=redis_service,
        main_repository=main_repository,
        ttl_seconds=ttl
    )