"""
Firebase ウェルネスリポジトリ
Firestore を使用したウェルネススコアデータの保存と取得
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, cast
import uuid

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
)
from domain.repositories.wellness_repository import WellnessRepositoryInterface

logger = get_logger(__name__)


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
        self.firebase_client = firebase_client
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
            DatabaseError: データベース操作に失敗した場合
        """
        try:
            # スコアをディクショナリに変換
            score_dict = self._score_to_dict(score)

            # Firestoreに保存
            if score.id:
                # 既存のドキュメントを更新
                await self.firebase_client.set_document(
                    self.scores_collection, score.id, score_dict
                )
            else:
                # 新しいドキュメントを作成
                score.id = str(uuid.uuid4())
                score_dict["id"] = score.id
                await self.firebase_client.set_document(
                    self.scores_collection, score.id, score_dict
                )

            logger.info(f"ウェルネススコア {score.id} を保存しました")
            return score

        except FirestoreError as e:
            error_msg = f"ウェルネススコアの保存に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    async def get_score_by_id(self, score_id: str) -> Optional[WellnessScore]:
        """
        IDによるスコア取得

        Args:
            score_id: スコアID

        Returns:
            スコアが存在する場合はWellnessScoreオブジェクト、存在しない場合はNone
        """
        try:
            # Firestoreからドキュメントを取得
            doc = await self.firebase_client.get_document(self.scores_collection, score_id)
            if not doc:
                return None

            # ドキュメントをWellnessScoreオブジェクトに変換
            score = self._dict_to_score(doc)
            return score

        except FirestoreError as e:
            error_msg = f"スコア {score_id} の取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    async def get_latest_score(self, company_id: str) -> Optional[WellnessScore]:
        """
        最新のスコア取得

        Args:
            company_id: 会社ID

        Returns:
            最新のスコア
        """
        try:
            # 会社IDと日付でフィルタリングしてクエリを実行
            filters = [
                {"field": "company_id", "op": "==", "value": company_id}
            ]
            order_by = [{"field": "timestamp", "direction": "desc"}]

            docs = await self.firebase_client.query_documents(
                self.scores_collection, filters, order_by, limit=1
            )

            if not docs:
                return None

            # 最新のドキュメントをWellnessScoreオブジェクトに変換
            latest_score = self._dict_to_score(docs[0])
            return latest_score

        except FirestoreError as e:
            error_msg = f"会社 {company_id} の最新スコア取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

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
            # フィルタの作成
            filters = [
                {"field": "company_id", "op": "==", "value": company_id}
            ]

            # 日付フィルタの追加
            if start_date:
                filters.append({"field": "timestamp", "op": ">=", "value": start_date})
            if end_date:
                filters.append({"field": "timestamp", "op": "<=", "value": end_date})

            # 日付の降順でソート
            order_by = [{"field": "timestamp", "direction": "desc"}]

            # クエリの実行
            docs = await self.firebase_client.query_documents(
                self.scores_collection, filters, order_by, limit=limit
            )

            # ドキュメントをWellnessScoreオブジェクトのリストに変換
            scores = [self._dict_to_score(doc) for doc in docs]
            return scores

        except FirestoreError as e:
            error_msg = f"会社 {company_id} のスコア一覧取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

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

        except FirestoreError as e:
            error_msg = f"会社 {company_id} のスコア履歴取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    async def save_metric(self, metric: ScoreMetric, company_id: str) -> ScoreMetric:
        """
        メトリックの保存

        Args:
            metric: 保存するメトリック
            company_id: 会社ID

        Returns:
            保存されたメトリック
        """
        try:
            # メトリックをディクショナリに変換
            metric_dict = self._metric_to_dict(metric)
            metric_dict["company_id"] = company_id

            # ドキュメントIDを生成（IDがない場合）
            metric_id = str(uuid.uuid4())

            # Firestoreに保存
            await self.firebase_client.set_document(
                self.metrics_collection, metric_id, metric_dict
            )

            logger.info(f"メトリック {metric.name} を会社 {company_id} に保存しました")
            return metric

        except FirestoreError as e:
            error_msg = f"メトリックの保存に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    async def get_metrics_by_company(
        self,
        company_id: str,
        category: Optional[ScoreCategory] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[ScoreMetric]:
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
            # フィルタの作成
            filters = [
                {"field": "company_id", "op": "==", "value": company_id}
            ]

            # カテゴリフィルタの追加
            if category:
                filters.append({"field": "category", "op": "==", "value": category.value})

            # 日付フィルタの追加
            if start_date:
                filters.append({"field": "timestamp", "op": ">=", "value": start_date})
            if end_date:
                filters.append({"field": "timestamp", "op": "<=", "value": end_date})

            # 日付の降順でソート
            order_by = [{"field": "timestamp", "direction": "desc"}]

            # クエリの実行
            docs = await self.firebase_client.query_documents(
                self.metrics_collection, filters, order_by
            )

            # ドキュメントをScoreMetricオブジェクトのリストに変換
            metrics = [self._dict_to_metric(doc) for doc in docs]
            return metrics

        except FirestoreError as e:
            error_msg = f"会社 {company_id} のメトリック一覧取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

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

        except FirestoreError as e:
            error_msg = f"会社 {company_id} のカテゴリ {category} トレンド取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    async def save_recommendation_plan(self, plan: RecommendationPlan) -> RecommendationPlan:
        """
        推奨プランの保存

        Args:
            plan: 保存する推奨プラン

        Returns:
            保存された推奨プラン
        """
        try:
            # プランをディクショナリに変換
            plan_dict = self._recommendation_plan_to_dict(plan)

            # ドキュメントID（会社ID + スコアID）
            doc_id = f"{plan.company_id}_{plan.score_id}"

            # Firestoreに保存
            await self.firebase_client.set_document(
                self.recommendations_collection, doc_id, plan_dict
            )

            logger.info(f"推奨プラン {doc_id} を保存しました")
            return plan

        except FirestoreError as e:
            error_msg = f"推奨プランの保存に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

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
                doc = await self.firebase_client.get_document(self.recommendations_collection, doc_id)
                if doc:
                    return self._dict_to_recommendation_plan(doc)
                return None
            else:
                # 最新の推奨プランを取得
                filters = [
                    {"field": "company_id", "op": "==", "value": company_id}
                ]
                order_by = [{"field": "generated_at", "direction": "desc"}]

                docs = await self.firebase_client.query_documents(
                    self.recommendations_collection, filters, order_by, limit=1
                )

                if docs:
                    return self._dict_to_recommendation_plan(docs[0])
                return None

        except FirestoreError as e:
            error_msg = f"会社 {company_id} の推奨プラン取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

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

        except FirestoreError as e:
            error_msg = f"会社 {company_id} のベンチマーク取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

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

    def _metric_to_dict(self, metric: ScoreMetric) -> Dict[str, Any]:
        """ScoreMetricオブジェクトをディクショナリに変換"""
        return {
            "name": metric.name,
            "value": metric.value,
            "weight": metric.weight,
            "category": metric.category.value,
            "timestamp": metric.timestamp,
            "raw_data": metric.raw_data
        }

    def _dict_to_metric(self, data: Dict[str, Any]) -> ScoreMetric:
        """ディクショナリからScoreMetricオブジェクトを作成"""
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

        return ScoreMetric(
            name=data.get("name", ""),
            value=float(data.get("value", 0.0)),
            weight=float(data.get("weight", 1.0)),
            category=category,
            timestamp=timestamp,
            raw_data=data.get("raw_data", {})
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