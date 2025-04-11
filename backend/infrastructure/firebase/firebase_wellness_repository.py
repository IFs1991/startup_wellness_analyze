"""
Firebase ウェルネスリポジトリ
Firestore を使用したウェルネススコアデータの保存と取得
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, cast
import uuid
import logging
from firebase_admin import firestore

from core.common_logger import get_logger
from core.exceptions import DataNotFoundError, DatabaseError, FirestoreError
from core.firebase_client import FirebaseClientInterface
from domain.models.wellness import (
    RecommendationAction,
    RecommendationPlan,
    ScoreCategory,
    ScoreHistory,
    ScoreMetric,
    WellnessScore,
    WellnessMetric,
    WellnessRecommendation,
    WellnessDimension
)
from domain.repositories.wellness_repository import WellnessRepositoryInterface

logger = get_logger(__name__)


class WellnessRepositoryError(Exception):
    """ウェルネスリポジトリの操作に関するエラー"""
    pass


class FirebaseWellnessRepository(WellnessRepositoryInterface):
    """
    Firebase/Firestoreを使用したウェルネスリポジトリの実装
    """

    def __init__(self, firebase_client: FirebaseClientInterface):
        """
        初期化

        Args:
            firebase_client: Firebaseクライアント
        """
        self.logger = get_logger(__name__)
        self.firebase_client = firebase_client
        self.db = firebase_client.firestore_client
        self.scores_collection = "wellness_scores"
        self.metrics_collection = "wellness_metrics"
        self.recommendations_collection = "recommendations"
        self.benchmarks_collection = "benchmarks"

    async def save_score(self, score: WellnessScore) -> WellnessScore:
        """
        ウェルネススコアの保存

        Args:
            score: 保存するスコア

        Returns:
            保存されたスコア

        Raises:
            WellnessRepositoryError: 保存に失敗した場合
        """
        try:
            score_dict = self._score_to_dict(score)

            if not score.id:
                # 新規作成
                doc_ref = self.db.collection(self.scores_collection).document()
                score.id = doc_ref.id
                score_dict["id"] = doc_ref.id
                doc_ref.set(score_dict)
            else:
                # 更新
                doc_ref = self.db.collection(self.scores_collection).document(score.id)
                doc_ref.set(score_dict, merge=True)

            logger.info(f"ウェルネススコア {score.id} を保存しました")
            return score
        except Exception as e:
            self.logger.error(f"スコア保存中にエラーが発生しました: {e}")
            raise WellnessRepositoryError(f"スコア保存エラー: {str(e)}")

    async def get_score_by_id(self, score_id: str) -> Optional[WellnessScore]:
        """
        IDによるスコア取得

        Args:
            score_id: スコアID

        Returns:
            スコアが存在する場合はWellnessScoreオブジェクト、存在しない場合はNone
        """
        try:
            doc_ref = self.db.collection(self.scores_collection).document(score_id)
            doc = doc_ref.get()

            if not doc.exists:
                return None

            return self._dict_to_score(doc.to_dict())
        except Exception as e:
            self.logger.error(f"スコア取得中にエラーが発生しました (ID: {score_id}): {e}")
            raise WellnessRepositoryError(f"スコア取得エラー: {str(e)}")

    async def get_latest_score(self, company_id: str) -> Optional[WellnessScore]:
        """
        最新のスコア取得

        Args:
            company_id: 会社ID

        Returns:
            最新のスコア
        """
        try:
            query = (
                self.db.collection(self.scores_collection)
                .where("company_id", "==", company_id)
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(1)
            )

            results = query.stream()
            for doc in results:
                return self._dict_to_score(doc.to_dict())

            return None
        except Exception as e:
            self.logger.error(f"会社の最新スコア取得中にエラーが発生しました (会社ID: {company_id}): {e}")
            raise WellnessRepositoryError(f"会社スコア取得エラー: {str(e)}")

    async def get_scores_by_company(
        self,
        company_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10
    ) -> List[WellnessScore]:
        """
        会社IDによるスコア一覧の取得

        Args:
            company_id: 会社ID
            start_date: 開始日時
            end_date: 終了日時
            limit: 取得する最大件数

        Returns:
            スコアのリスト
        """
        try:
            query = self.db.collection(self.scores_collection)

            if company_id:
                query = query.where("company_id", "==", company_id)

            if start_date:
                query = query.where("timestamp", ">=", start_date)
            if end_date:
                query = query.where("timestamp", "<=", end_date)

            query = query.order_by("timestamp", direction=firestore.Query.DESCENDING)

            query = query.limit(limit)

            results = []
            for doc in query.stream():
                results.append(self._dict_to_score(doc.to_dict()))

            return results
        except Exception as e:
            self.logger.error(f"会社スコア一覧取得中にエラーが発生しました (会社ID: {company_id}): {e}")
            raise WellnessRepositoryError(f"会社スコア一覧取得エラー: {str(e)}")

    async def get_score_history(
        self,
        company_id: str,
        time_period: str = "monthly",
        limit: int = 12
    ) -> ScoreHistory:
        """
        スコア履歴の取得

        Args:
            company_id: 会社ID
            time_period: 期間種別 (monthly, quarterly, yearly)
            limit: 取得する履歴の数

        Returns:
            スコア履歴
        """
        try:
            # 期間に基づいて日付範囲を決定
            now = datetime.now()
            if time_period == "yearly":
                start_date = now - timedelta(days=365 * limit)
            elif time_period == "quarterly":
                start_date = now - timedelta(days=90 * limit)
            else:  # monthly
                start_date = now - timedelta(days=30 * limit)

            # スコアを取得
            scores = await self.get_scores_by_company(
                company_id, start_date, now, limit
            )

            # スコア履歴を作成
            history = ScoreHistory(
                company_id=company_id,
                scores=scores,
                time_period=time_period
            )

            return history

        except Exception as e:
            self.logger.error(f"会社スコア履歴取得中にエラーが発生しました (会社ID: {company_id}): {e}")
            raise WellnessRepositoryError(f"会社スコア履歴取得エラー: {str(e)}")

    async def save_metric(self, metric: WellnessMetric, company_id: str) -> WellnessMetric:
        """
        メトリックの保存

        Args:
            metric: 保存するメトリック
            company_id: 会社ID

        Returns:
            保存されたメトリック
        """
        try:
            metric_dict = self._metric_to_dict(metric)
            metric_dict["company_id"] = company_id

            # ドキュメントIDを生成（IDがない場合）
            metric_id = str(uuid.uuid4())

            # Firestoreに保存
            await self.db.collection(self.metrics_collection).document(metric_id).set(metric_dict)

            logger.info(f"メトリック {metric.name} を会社 {company_id} に保存しました")
            return metric

        except Exception as e:
            self.logger.error(f"メトリック保存中にエラーが発生しました: {e}")
            raise WellnessRepositoryError(f"メトリック保存エラー: {str(e)}")

    async def get_metrics_by_company(
        self,
        company_id: str,
        category: Optional[ScoreCategory] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[WellnessMetric]:
        """
        会社IDによるメトリック一覧の取得

        Args:
            company_id: 会社ID
            category: スコアカテゴリ（指定された場合）
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            メトリックのリスト
        """
        try:
            query = self.db.collection(self.metrics_collection)

            if company_id:
                query = query.where("company_id", "==", company_id)

            if category:
                query = query.where("category", "==", category.value)

            if start_date:
                query = query.where("timestamp", ">=", start_date)
            if end_date:
                query = query.where("timestamp", "<=", end_date)

            query = query.order_by("timestamp", direction=firestore.Query.DESCENDING)

            results = []
            for doc in query.stream():
                results.append(self._dict_to_metric(doc.to_dict()))

            return results
        except Exception as e:
            self.logger.error(f"会社メトリック一覧取得中にエラーが発生しました (会社ID: {company_id}): {e}")
            raise WellnessRepositoryError(f"会社メトリック一覧取得エラー: {str(e)}")

    async def get_category_trend(
        self,
        company_id: str,
        category: ScoreCategory,
        periods: int = 6
    ) -> List[Tuple[datetime, float]]:
        """
        カテゴリごとのトレンド取得

        Args:
            company_id: 会社ID
            category: スコアカテゴリ
            periods: 取得する期間数

        Returns:
            (日時, スコア)のリスト
        """
        try:
            # スコア一覧を取得
            scores = await self.get_scores_by_company(company_id, limit=periods)

            # カテゴリスコアと日時のタプルを作成
            trend_data: List[Tuple[datetime, float]] = []
            for score in scores:
                if category in score.category_scores:
                    trend_data.append((score.timestamp, score.category_scores[category]))

            # 日時の昇順でソート
            trend_data.sort(key=lambda x: x[0])

            return trend_data

        except Exception as e:
            self.logger.error(f"会社カテゴリトレンド取得中にエラーが発生しました (会社ID: {company_id}): {e}")
            raise WellnessRepositoryError(f"会社カテゴリトレンド取得エラー: {str(e)}")

    async def save_recommendation_plan(self, plan: RecommendationPlan) -> RecommendationPlan:
        """
        推奨プランの保存

        Args:
            plan: 保存する推奨プラン

        Returns:
            保存された推奨プラン
        """
        try:
            plan_dict = self._recommendation_plan_to_dict(plan)

            # ドキュメントID（会社ID + スコアID）
            doc_id = f"{plan.company_id}_{plan.score_id}"

            # Firestoreに保存
            await self.db.collection(self.recommendations_collection).document(doc_id).set(plan_dict)

            logger.info(f"推奨プラン {doc_id} を保存しました")
            return plan

        except Exception as e:
            self.logger.error(f"推奨プラン保存中にエラーが発生しました: {e}")
            raise WellnessRepositoryError(f"推奨プラン保存エラー: {str(e)}")

    async def get_recommendation_plan(
        self,
        company_id: str,
        score_id: Optional[str] = None
    ) -> Optional[RecommendationPlan]:
        """
        推奨プランの取得

        Args:
            company_id: 会社ID
            score_id: スコアID（指定された場合）

        Returns:
            推奨プラン
        """
        try:
            if score_id:
                # 特定のスコアIDに関連する推奨プランを取得
                doc_id = f"{company_id}_{score_id}"
                doc = await self.db.collection(self.recommendations_collection).document(doc_id).get()
                if doc.exists:
                    return self._dict_to_recommendation_plan(doc.to_dict())
                return None
            else:
                # 最新の推奨プランを取得
                query = self.db.collection(self.recommendations_collection)
                query = query.where("company_id", "==", company_id)
                query = query.order_by("generated_at", direction=firestore.Query.DESCENDING)
                query = query.limit(1)

                results = query.stream()
                for doc in results:
                    return self._dict_to_recommendation_plan(doc.to_dict())
                return None

        except Exception as e:
            self.logger.error(f"会社推奨プラン取得中にエラーが発生しました (会社ID: {company_id}): {e}")
            raise WellnessRepositoryError(f"会社推奨プラン取得エラー: {str(e)}")

    async def get_company_benchmarks(
        self,
        company_id: str,
        category: Optional[ScoreCategory] = None
    ) -> Dict[str, float]:
        """
        業界ベンチマークの取得

        Args:
            company_id: 会社ID
            category: スコアカテゴリ（指定された場合）

        Returns:
            ベンチマークスコアのマップ
        """
        try:
            # 会社の最新スコアを取得
            latest_score = await self.get_latest_score(company_id)
            if not latest_score:
                return {
                    "company": 0.0,
                    "industry_avg": 0.0,
                    "industry_top": 0.0,
                    "all_avg": 0.0
                }

            # 会社のスコア
            if category:
                company_score = latest_score.category_scores.get(category, 0.0)
            else:
                company_score = latest_score.total_score

            # TODO: 実際のベンチマークデータを取得するロジックを実装
            # ここではモックデータを返す
            return {
                "company": company_score,
                "industry_avg": 65.0,
                "industry_top": 85.0,
                "all_avg": 60.0
            }

        except Exception as e:
            self.logger.error(f"会社ベンチマーク取得中にエラーが発生しました (会社ID: {company_id}): {e}")
            raise WellnessRepositoryError(f"会社ベンチマーク取得エラー: {str(e)}")

    # プライベートヘルパーメソッド
    def _score_to_dict(self, score: WellnessScore) -> Dict[str, Any]:
        """WellnessScoreオブジェクトをディクショナリに変換"""
        # カテゴリスコアを文字列キーに変換
        category_scores = {
            cat.value: score for cat, score in score.category_scores.items()
        }

        # メトリックをディクショナリに変換
        metrics = [self._metric_to_dict(metric) for metric in score.metrics]

        return {
            "id": score.id,
            "company_id": score.company_id,
            "total_score": score.total_score,
            "category_scores": category_scores,
            "metrics": metrics,
            "timestamp": score.timestamp,
            "created_by": score.created_by
        }

    def _dict_to_score(self, data: Dict[str, Any]) -> WellnessScore:
        """ディクショナリからWellnessScoreオブジェクトを作成"""
        # カテゴリスコアを列挙型キーに変換
        category_scores = {}
        for cat_str, score_val in data.get("category_scores", {}).items():
            try:
                category = ScoreCategory(cat_str)
                category_scores[category] = float(score_val)
            except (ValueError, TypeError):
                logger.warning(f"無効なカテゴリ {cat_str} をスキップします")

        # メトリックを変換
        metrics = []
        for metric_data in data.get("metrics", []):
            try:
                metrics.append(self._dict_to_metric(metric_data))
            except Exception as e:
                logger.warning(f"メトリックの変換に失敗しました: {str(e)}")

        # タイムスタンプを処理
        timestamp = data.get("timestamp")
        if isinstance(timestamp, Dict):
            # Firestoreのタイムスタンプ型の場合
            timestamp = datetime.fromtimestamp(timestamp.get("seconds", 0))
        elif not isinstance(timestamp, datetime):
            # デフォルトの日時
            timestamp = datetime.now()

        return WellnessScore(
            id=data.get("id", ""),
            company_id=data.get("company_id", ""),
            total_score=float(data.get("total_score", 0.0)),
            category_scores=category_scores,
            metrics=metrics,
            timestamp=timestamp,
            created_by=data.get("created_by")
        )

    def _metric_to_dict(self, metric: WellnessMetric) -> Dict[str, Any]:
        """WellnessMetricオブジェクトをディクショナリに変換"""
        return {
            "name": metric.name,
            "value": metric.value,
            "weight": metric.weight,
            "category": metric.category.value,
            "timestamp": metric.timestamp,
            "raw_data": metric.raw_data
        }

    def _dict_to_metric(self, data: Dict[str, Any]) -> WellnessMetric:
        """ディクショナリからWellnessMetricオブジェクトを作成"""
        # カテゴリを変換
        try:
            category = ScoreCategory(data.get("category", ScoreCategory.FINANCIAL.value))
        except ValueError:
            category = ScoreCategory.FINANCIAL
            logger.warning(f"無効なカテゴリ {data.get('category')} をデフォルトに置き換えます")

        # タイムスタンプを処理
        timestamp = data.get("timestamp")
        if isinstance(timestamp, Dict):
            # Firestoreのタイムスタンプ型の場合
            timestamp = datetime.fromtimestamp(timestamp.get("seconds", 0))
        elif not isinstance(timestamp, datetime):
            # デフォルトの日時
            timestamp = datetime.now()

        return WellnessMetric(
            id=data.get("id", ""),
            name=data.get("name", ""),
            value=float(data.get("value", 0.0)),
            dimension=WellnessDimension(data.get("dimension", "physical")),
            timestamp=timestamp,
            source=data.get("source", "manual"),
            metadata=data.get("metadata", {})
        )

    def _recommendation_plan_to_dict(self, plan: RecommendationPlan) -> Dict[str, Any]:
        """RecommendationPlanオブジェクトをディクショナリに変換"""
        return {
            "company_id": plan.company_id,
            "score_id": plan.score_id,
            "actions": [self._action_to_dict(action) for action in plan.actions],
            "generated_at": plan.generated_at,
            "generated_by": plan.generated_by
        }

    def _dict_to_recommendation_plan(self, data: Dict[str, Any]) -> RecommendationPlan:
        """ディクショナリからRecommendationPlanオブジェクトを作成"""
        # アクションを変換
        actions = []
        for action_data in data.get("actions", []):
            try:
                actions.append(self._dict_to_action(action_data))
            except Exception as e:
                logger.warning(f"アクションの変換に失敗しました: {str(e)}")

        # タイムスタンプを処理
        generated_at = data.get("generated_at")
        if isinstance(generated_at, Dict):
            # Firestoreのタイムスタンプ型の場合
            generated_at = datetime.fromtimestamp(generated_at.get("seconds", 0))
        elif not isinstance(generated_at, datetime):
            # デフォルトの日時
            generated_at = datetime.now()

        return RecommendationPlan(
            company_id=data.get("company_id", ""),
            score_id=data.get("score_id", ""),
            actions=actions,
            generated_at=generated_at,
            generated_by=data.get("generated_by")
        )

    def _action_to_dict(self, action: RecommendationAction) -> Dict[str, Any]:
        """RecommendationActionオブジェクトをディクショナリに変換"""
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

    def _dict_to_action(self, data: Dict[str, Any]) -> RecommendationAction:
        """ディクショナリからRecommendationActionオブジェクトを作成"""
        # カテゴリを変換
        try:
            category = ScoreCategory(data.get("category", ScoreCategory.FINANCIAL.value))
        except ValueError:
            category = ScoreCategory.FINANCIAL
            logger.warning(f"無効なカテゴリ {data.get('category')} をデフォルトに置き換えます")

        return RecommendationAction(
            id=data.get("id", str(uuid.uuid4())),
            title=data.get("title", ""),
            description=data.get("description", ""),
            category=category,
            impact_level=int(data.get("impact_level", 3)),
            effort_level=int(data.get("effort_level", 3)),
            time_frame=data.get("time_frame", "medium"),
            resources=data.get("resources", [])
        )