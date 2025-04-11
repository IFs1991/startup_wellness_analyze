"""
ウェルネス関連のユースケース
ウェルネススコア計算や分析に関するビジネスロジックを提供します。
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from domain.entities.user import User
from domain.entities.wellness import WellnessScore, WellnessMetric, WellnessRecommendation, WellnessDimension
from domain.repositories.wellness_repository import WellnessRepositoryInterface
from domain.repositories.user_repository import UserRepositoryInterface
from core.common_logger import get_logger


class WellnessUseCase:
    """ウェルネス関連のユースケースクラス"""

    def __init__(
        self,
        wellness_repository: WellnessRepositoryInterface,
        user_repository: UserRepositoryInterface
    ):
        """
        初期化

        Args:
            wellness_repository: ウェルネスリポジトリインターフェース
            user_repository: ユーザーリポジトリインターフェース
        """
        self.logger = get_logger(__name__)
        self.wellness_repository = wellness_repository
        self.user_repository = user_repository

    async def calculate_user_wellness_score(
        self,
        user_id: str,
        metrics: List[Dict[str, Any]]
    ) -> WellnessScore:
        """
        ユーザーのウェルネススコアを計算

        Args:
            user_id: ユーザーID
            metrics: 計算に使用するメトリックのリスト
                各メトリックは以下のキーを持つ辞書:
                - name: メトリック名
                - value: メトリック値
                - dimension: WellnessDimension文字列

        Returns:
            計算されたWellnessScoreエンティティ

        Raises:
            ValueError: ユーザーが存在しない場合やメトリックが不正な場合
        """
        # ユーザーの存在確認
        user = await self.user_repository.get_by_id(user_id)
        if not user:
            self.logger.warning(f"存在しないユーザーID: {user_id}")
            raise ValueError(f"ユーザーID {user_id} は存在しません")

        # メトリックの妥当性確認
        if not metrics:
            self.logger.warning(f"ユーザー {user_id} の計算に使用するメトリックが空です")
            raise ValueError("少なくとも1つのメトリックが必要です")

        # WellnessMetricオブジェクトのリストを作成
        wellness_metrics = []

        for metric_data in metrics:
            try:
                name = metric_data.get("name")
                value = float(metric_data.get("value", 0))
                dimension_str = metric_data.get("dimension", "physical")

                # 値の範囲チェック
                if value < 0 or value > 100:
                    self.logger.warning(f"無効なメトリック値: {value}。0-100の範囲に調整します。")
                    value = max(0, min(100, value))

                # ディメンションの妥当性チェック
                try:
                    dimension = WellnessDimension(dimension_str)
                except ValueError:
                    self.logger.warning(f"無効なディメンション: {dimension_str}。PHYSICALに設定します。")
                    dimension = WellnessDimension.PHYSICAL

                # メトリックオブジェクトを作成
                metric = WellnessMetric(
                    name=name,
                    value=value,
                    dimension=dimension,
                    timestamp=datetime.now(),
                    source="user_input",
                    metadata={"raw_data": metric_data}
                )

                wellness_metrics.append(metric)

            except Exception as e:
                self.logger.error(f"メトリック変換エラー: {str(e)}")
                continue

        if not wellness_metrics:
            self.logger.warning(f"ユーザー {user_id} の有効なメトリックがありません")
            raise ValueError("有効なメトリックがありません")

        # スコアオブジェクトを作成
        score = WellnessScore(
            user_id=user_id,
            company_id=user.company_id,
            metrics=wellness_metrics
        )

        # スコアを計算
        score._recalculate_scores()

        # スコアを保存
        saved_score = await self.wellness_repository.save_score(score)

        self.logger.info(f"ユーザー {user_id} のウェルネススコアを計算しました: {saved_score.score}")
        return saved_score

    async def get_user_wellness_history(
        self,
        user_id: str,
        days: int = 30
    ) -> List[WellnessScore]:
        """
        ユーザーのウェルネス履歴を取得

        Args:
            user_id: ユーザーID
            days: 取得する履歴の日数

        Returns:
            WellnessScoreエンティティのリスト
        """
        # 日付範囲を設定
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # リポジトリから履歴を取得
        scores = await self.wellness_repository.get_score_history(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )

        self.logger.info(f"ユーザー {user_id} のウェルネス履歴 {len(scores)} 件を取得しました")
        return scores

    async def get_recommendations_for_user(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[WellnessRecommendation]:
        """
        ユーザーのウェルネス改善のための推奨事項を取得

        Args:
            user_id: ユーザーID
            limit: 取得する推奨事項の最大数

        Returns:
            WellnessRecommendationエンティティのリスト
        """
        # 最新のスコアを取得
        latest_score = await self.wellness_repository.get_latest_score_for_user(user_id)

        # 弱いディメンションを特定
        weak_dimensions = []
        if latest_score:
            # スコアが60未満のディメンションを「弱い」と判断
            for dimension, score in latest_score.dimension_scores.items():
                if score < 60:
                    weak_dimensions.append(dimension)

        # 推奨事項を取得
        recommendations = await self.wellness_repository.get_recommendations(
            user_id=user_id,
            dimensions=weak_dimensions if weak_dimensions else None,
            limit=limit
        )

        self.logger.info(f"ユーザー {user_id} の推奨事項 {len(recommendations)} 件を取得しました")
        return recommendations

    async def get_company_wellness_overview(self, company_id: str) -> Dict[str, Any]:
        """
        企業のウェルネス概要を取得

        Args:
            company_id: 企業ID

        Returns:
            企業のウェルネス概要を含む辞書
        """
        # 最新の企業スコアを取得
        latest_score = await self.wellness_repository.get_latest_score_for_company(company_id)

        # ディメンション別の平均スコアを取得
        start_date = datetime.now() - timedelta(days=90)  # 90日間のデータ
        dimension_averages = await self.wellness_repository.get_average_scores_by_dimension(
            company_id=company_id,
            start_date=start_date
        )

        # 企業に所属するユーザーを取得
        users = await self.user_repository.list_by_company(company_id)

        # 結果を組み立て
        result = {
            "company_id": company_id,
            "current_score": latest_score.score if latest_score else 0,
            "user_count": len(users),
            "dimension_averages": {d.value: score for d, score in dimension_averages.items()},
            "timestamp": datetime.now().isoformat()
        }

        self.logger.info(f"企業 {company_id} のウェルネス概要を取得しました")
        return result

    async def create_recommendation(
        self,
        title: str,
        description: str,
        target_dimensions: List[str],
        priority: int = 5,
        impact_estimate: float = 0.5
    ) -> WellnessRecommendation:
        """
        ウェルネス改善のための推奨事項を作成

        Args:
            title: 推奨事項のタイトル
            description: 推奨事項の詳細説明
            target_dimensions: 対象とするディメンションのリスト（文字列）
            priority: 優先度（1-10）
            impact_estimate: 推定される影響度（0-1）

        Returns:
            作成されたWellnessRecommendationエンティティ
        """
        # パラメータの妥当性チェック
        if not title or not description:
            raise ValueError("タイトルと説明は必須です")

        if not target_dimensions:
            raise ValueError("少なくとも1つのディメンションを指定してください")

        # 優先度の範囲チェック
        priority = max(1, min(10, priority))

        # 影響度の範囲チェック
        impact_estimate = max(0.0, min(1.0, impact_estimate))

        # ディメンションの変換
        dimensions = []
        for dim_str in target_dimensions:
            try:
                dimensions.append(WellnessDimension(dim_str))
            except ValueError:
                self.logger.warning(f"無効なディメンション: {dim_str} - スキップします")

        if not dimensions:
            raise ValueError("有効なディメンションがありません")

        # 推奨事項の作成
        recommendation = WellnessRecommendation(
            title=title,
            description=description,
            priority=priority,
            target_dimensions=dimensions,
            impact_estimate=impact_estimate,
            created_at=datetime.now()
        )

        # 推奨事項を保存
        saved_recommendation = await self.wellness_repository.save_recommendation(recommendation)

        self.logger.info(f"新しい推奨事項を作成しました: {saved_recommendation.id}")
        return saved_recommendation