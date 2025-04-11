"""
ウェルネススコアユースケース
ウェルネススコアの計算と分析に関するビジネスロジックを実装
"""
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import MinMaxScaler

from core.common_logger import get_logger
from core.di import inject
from core.exceptions import DataNotFoundError, CalculationError
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


class WellnessScoreUseCase:
    """
    ウェルネススコアユースケース
    スコアの計算、保存、分析に関するビジネスロジックを実装
    """

    def __init__(self, wellness_repository: WellnessRepositoryInterface, firebase_client: FirebaseClientInterface):
        """
        初期化

        Args:
            wellness_repository: ウェルネスリポジトリ
            firebase_client: Firebaseクライアント
        """
        self.wellness_repository = wellness_repository
        self.firebase_client = firebase_client

        # データスケーラーの初期化
        self.scaler = MinMaxScaler(feature_range=(0, 100))

        # カテゴリ重み設定
        self.category_weights = {
            "physical": 0.25,
            "mental": 0.30,
            "social": 0.20,
            "productivity": 0.25
        }

        # 業種と成長段階に基づく調整要素
        self.industry_adjustments = {
            "tech": 1.05,
            "healthcare": 1.10,
            "finance": 0.95,
            "retail": 0.90,
            "manufacturing": 0.92,
            "education": 1.02,
            "other": 1.00
        }

        self.stage_adjustments = {
            "seed": 1.10,
            "early": 1.05,
            "growth": 1.00,
            "expansion": 0.95,
            "mature": 0.90,
            "other": 1.00
        }

        logger.info("WellnessScoreUseCaseが初期化されました")

    async def calculate_wellness_score(
        self,
        company_id: str,
        industry: str,
        stage: str,
        calculation_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        ウェルネススコアを計算する

        Args:
            company_id: 企業ID
            industry: 業界
            stage: 企業ステージ
            calculation_date: 計算基準日（Noneの場合は現在日時）

        Returns:
            Dict[str, Any]: 総合スコアと各カテゴリスコア
        """
        try:
            if calculation_date is None:
                calculation_date = datetime.now()

            # データ取得と前処理
            vas_data = await self._fetch_vas_data(company_id)
            financial_data = await self._fetch_financial_data(company_id)

            if vas_data.empty:
                raise CalculationError(f"VASデータが存在しません: company_id={company_id}")

            # カテゴリごとのスコア計算
            category_scores = await self._calculate_category_scores(vas_data)

            # 財務相関の調整
            financial_adjustment = 0.0
            if not financial_data.empty:
                financial_adjustment = await self._calculate_financial_correlation(vas_data, financial_data)

            # 業界・ステージによる調整
            industry_stage_adjustment = self._apply_industry_stage_adjustment(industry, stage)

            # 時系列トレンドによる調整
            trend_adjustment = 0.0
            if len(vas_data) >= 3:  # 最低3つのデータポイントが必要
                trend_adjustment = await self._calculate_trend_adjustment(vas_data)

            # 総合スコア計算
            base_score = self._calculate_base_score(category_scores)
            total_score = self._apply_adjustments(
                base_score,
                financial_adjustment,
                industry_stage_adjustment,
                trend_adjustment
            )

            # 最終スコアの範囲を0-10に制限
            final_score = min(max(total_score, 0), 10)

            # 結果のまとめ
            result = {
                "company_id": company_id,
                "industry": industry,
                "stage": stage,
                "calculation_date": calculation_date,
                "total_score": final_score,
                "base_score": base_score,
                "financial_adjustment": financial_adjustment,
                "industry_stage_adjustment": industry_stage_adjustment,
                "trend_adjustment": trend_adjustment,
                "category_scores": category_scores,
            }

            # WellnessScoreエンティティの作成と保存
            score_entity = self._create_wellness_score_entity(result)
            saved_score = await self.wellness_repository.save_score(score_entity)

            # 結果に保存されたエンティティのIDを追加
            result["id"] = saved_score.id

            return result

        except Exception as e:
            logger.error(f"ウェルネススコア計算エラー: {e}")
            raise CalculationError(f"ウェルネススコア計算中にエラーが発生しました: {e}")

    def _create_wellness_score_entity(self, result: Dict[str, Any]) -> WellnessScore:
        """
        計算結果からWellnessScoreエンティティを作成する

        Args:
            result: 計算結果の辞書

        Returns:
            WellnessScore: 作成されたエンティティ
        """
        # カテゴリスコアをドメインモデルの形式に変換
        category_scores = {}
        for category_name, score in result["category_scores"].items():
            try:
                category = ScoreCategory(category_name)
                category_scores[category] = score
            except ValueError:
                # 不明なカテゴリは無視
                logger.warning(f"不明なカテゴリ名: {category_name}")
                continue

        # エンティティの作成
        return WellnessScore(
            id="",  # 保存時に自動生成される
            company_id=result["company_id"],
            total_score=result["total_score"],
            category_scores=category_scores,
            timestamp=result["calculation_date"],
            created_by="system"
        )

    async def _fetch_vas_data(self, company_id: str) -> pd.DataFrame:
        """
        企業のVASデータを取得する

        Args:
            company_id: 企業ID

        Returns:
            pd.DataFrame: VASデータ
        """
        try:
            # Firestoreからデータを取得
            db = self.firebase_client.firestore_client
            vas_ref = db.collection("vas_assessments").where("company_id", "==", company_id)
            vas_docs = vas_ref.get()

            if not vas_docs:
                logger.warning(f"企業 {company_id} のVASデータが見つかりません")
                return pd.DataFrame()

            # DataFrameに変換
            vas_data = []
            for doc in vas_docs:
                data = doc.to_dict()
                data["id"] = doc.id
                vas_data.append(data)

            if not vas_data:
                return pd.DataFrame()

            return pd.DataFrame(vas_data)

        except Exception as e:
            logger.error(f"VASデータ取得エラー: {e}")
            raise CalculationError(f"VASデータの取得に失敗しました: {e}")

    async def _fetch_financial_data(self, company_id: str) -> pd.DataFrame:
        """
        企業の財務データを取得する

        Args:
            company_id: 企業ID

        Returns:
            pd.DataFrame: 財務データ
        """
        try:
            # Firestoreからデータを取得
            db = self.firebase_client.firestore_client
            financial_ref = db.collection("financial_data").where("company_id", "==", company_id)
            financial_docs = financial_ref.get()

            if not financial_docs:
                logger.warning(f"企業 {company_id} の財務データが見つかりません")
                return pd.DataFrame()

            # DataFrameに変換
            financial_data = []
            for doc in financial_docs:
                data = doc.to_dict()
                data["id"] = doc.id
                financial_data.append(data)

            if not financial_data:
                return pd.DataFrame()

            return pd.DataFrame(financial_data)

        except Exception as e:
            logger.error(f"財務データ取得エラー: {e}")
            raise CalculationError(f"財務データの取得に失敗しました: {e}")

    async def _calculate_category_scores(self, vas_data: pd.DataFrame) -> Dict[str, float]:
        """
        カテゴリごとのスコアを計算する

        Args:
            vas_data: VASデータ

        Returns:
            Dict[str, float]: カテゴリごとのスコア
        """
        # 最新のアセスメントデータを取得
        latest_date = vas_data["assessment_date"].max()
        latest_vas = vas_data[vas_data["assessment_date"] == latest_date]

        # カテゴリごとの集計
        categories = {
            "physical": ["energy", "sleep", "physical_activity"],
            "mental": ["stress", "anxiety", "motivation"],
            "social": ["team_connection", "communication", "support"],
            "productivity": ["focus", "efficiency", "work_quality"]
        }

        category_scores = {}
        for category, metrics in categories.items():
            # 各メトリックの値を取得（欠損値は0で補完）
            metric_values = []
            for metric in metrics:
                if metric in latest_vas.columns:
                    values = latest_vas[metric].fillna(0).values
                    if len(values) > 0:
                        metric_values.append(values[0])
                    else:
                        metric_values.append(0)
                else:
                    metric_values.append(0)

            # カテゴリスコアの計算（0-100のスケール）
            if metric_values:
                category_scores[category] = sum(metric_values) / len(metric_values) * 10
            else:
                category_scores[category] = 0

        return category_scores

    def _calculate_base_score(self, category_scores: Dict[str, float]) -> float:
        """
        基本スコアを計算する

        Args:
            category_scores: カテゴリごとのスコア

        Returns:
            float: 基本スコア（0-10のスケール）
        """
        total_weight = sum(self.category_weights.values())
        weighted_sum = 0

        for category, score in category_scores.items():
            weight = self.category_weights.get(category, 0)
            weighted_sum += score * weight

        # 正規化して0-10のスケールに変換
        if total_weight > 0:
            base_score = weighted_sum / total_weight / 10
        else:
            base_score = 0

        return min(max(base_score, 0), 10)  # 0-10の範囲に制限

    async def _calculate_financial_correlation(
        self,
        vas_data: pd.DataFrame,
        financial_data: pd.DataFrame
    ) -> float:
        """
        VASデータと財務データの相関に基づく調整値を計算する

        Args:
            vas_data: VASデータ
            financial_data: 財務データ

        Returns:
            float: 財務調整値（-1.0〜1.0）
        """
        try:
            # データの準備
            if vas_data.empty or financial_data.empty:
                return 0.0

            # 財務指標の抽出
            key_metrics = ["revenue_growth", "profit_margin", "cash_flow", "burn_rate"]
            financial_scores = []

            for metric in key_metrics:
                if metric in financial_data.columns:
                    value = financial_data[metric].mean()
                    if not pd.isna(value):
                        financial_scores.append(value)

            if not financial_scores:
                return 0.0

            # 財務スコアの平均（-1〜1の範囲に正規化）
            avg_financial_score = sum(financial_scores) / len(financial_scores)
            normalized_score = max(min(avg_financial_score, 1.0), -1.0)

            # 調整値の計算（最大で±1.0の調整）
            adjustment = normalized_score * 0.5  # 調整の影響を50%に制限

            return adjustment

        except Exception as e:
            logger.warning(f"財務相関計算中にエラーが発生しました: {e}")
            return 0.0  # エラー時は調整なし

    def _apply_industry_stage_adjustment(self, industry: str, stage: str) -> float:
        """
        業界とステージに基づく調整値を計算する

        Args:
            industry: 業界
            stage: 企業ステージ

        Returns:
            float: 業界・ステージ調整値（-0.5〜0.5）
        """
        # 業界調整の取得（デフォルトは1.0）
        industry_adj = self.industry_adjustments.get(industry.lower(), 1.0)

        # ステージ調整の取得（デフォルトは1.0）
        stage_adj = self.stage_adjustments.get(stage.lower(), 1.0)

        # 合成調整値（1.0を中心に±0.5の範囲）
        combined_adj = (industry_adj * stage_adj - 1.0) * 0.5

        return max(min(combined_adj, 0.5), -0.5)  # -0.5〜0.5の範囲に制限

    async def _calculate_trend_adjustment(self, vas_data: pd.DataFrame) -> float:
        """
        時系列トレンドに基づく調整値を計算する

        Args:
            vas_data: VASデータ

        Returns:
            float: トレンド調整値（-0.5〜0.5）
        """
        try:
            # 時系列データの準備
            if "assessment_date" not in vas_data.columns:
                return 0.0

            # 日付でソート
            vas_data = vas_data.sort_values("assessment_date")

            # 主要メトリックの抽出
            key_metrics = ["energy", "stress", "motivation", "focus"]
            trend_values = []

            for metric in key_metrics:
                if metric in vas_data.columns:
                    values = vas_data[metric].fillna(0).values
                    if len(values) >= 3:  # 少なくとも3ポイント必要
                        # 傾き（単純な差分）
                        slope = (values[-1] - values[0]) / len(values)
                        trend_values.append(slope)

            if not trend_values:
                return 0.0

            # 平均トレンド（-0.5〜0.5の範囲に正規化）
            avg_trend = sum(trend_values) / len(trend_values)
            adjustment = max(min(avg_trend, 0.5), -0.5)

            return adjustment

        except Exception as e:
            logger.warning(f"トレンド計算中にエラーが発生しました: {e}")
            return 0.0  # エラー時は調整なし

    def _apply_adjustments(
        self,
        base_score: float,
        financial_adjustment: float,
        industry_stage_adjustment: float,
        trend_adjustment: float
    ) -> float:
        """
        各種調整を適用して最終スコアを計算する

        Args:
            base_score: 基本スコア
            financial_adjustment: 財務調整値
            industry_stage_adjustment: 業界・ステージ調整値
            trend_adjustment: トレンド調整値

        Returns:
            float: 調整適用後の最終スコア
        """
        # 調整適用
        adjusted_score = base_score
        adjusted_score += financial_adjustment  # 財務調整
        adjusted_score += industry_stage_adjustment  # 業界・ステージ調整
        adjusted_score += trend_adjustment  # トレンド調整

        # 0-10の範囲に制限
        return min(max(adjusted_score, 0), 10)

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

        Raises:
            DataNotFoundError: データが見つからない場合
        """
        # リポジトリからスコア履歴を取得
        history = await self.wellness_repository.get_score_history(
            company_id, time_period, limit
        )

        if not history or not history.scores:
            logger.warning(f"会社 {company_id} のスコア履歴が見つかりません")
            raise DataNotFoundError(f"会社 {company_id} のスコア履歴が見つかりません")

        return history

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
        # リポジトリからカテゴリトレンドを取得
        trend_data = await self.wellness_repository.get_category_trend(
            company_id, category, periods
        )

        if not trend_data:
            logger.warning(f"会社 {company_id} のカテゴリ {category} のトレンドが見つかりません")
            # トレンドデータがない場合は空のリストを返す
            return []

        return trend_data

    async def compare_with_benchmarks(
        self,
        company_id: str,
        category: Optional[ScoreCategory] = None
    ) -> Dict[str, float]:
        """
        ベンチマークとの比較

        Args:
            company_id: 会社ID
            category: スコアカテゴリ（指定された場合）

        Returns:
            ベンチマークスコアのマップ
        """
        # リポジトリからベンチマークデータを取得
        benchmarks = await self.wellness_repository.get_company_benchmarks(
            company_id, category
        )

        return benchmarks

    async def generate_recommendations(
        self,
        company_id: str,
        score_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> RecommendationPlan:
        """
        改善推奨の生成

        Args:
            company_id: 会社ID
            score_id: スコアID（指定されない場合は最新のスコアを使用）
            user_id: 生成を実行したユーザーID（オプション）

        Returns:
            生成された推奨プラン

        Raises:
            DataNotFoundError: スコアが見つからない場合
        """
        # スコアIDが指定されていない場合は最新のスコアを使用
        score = None
        if score_id:
            score = await self.wellness_repository.get_score_by_id(score_id)
        else:
            score = await self.wellness_repository.get_latest_score(company_id)

        if not score:
            logger.warning(f"会社 {company_id} のスコアが見つかりません")
            raise DataNotFoundError(f"会社 {company_id} のスコアが見つかりません")

        # 推奨アクションのリストを作成
        actions: List[RecommendationAction] = []

        # 各カテゴリに対して推奨アクションを生成
        for category, category_score in score.category_scores.items():
            # 低スコアのカテゴリに対してより多くの推奨アクションを生成
            num_actions = 1
            if category_score < 40:
                num_actions = 3
            elif category_score < 60:
                num_actions = 2

            # カテゴリに基づいた推奨アクションを生成
            category_actions = self._generate_category_actions(
                category, category_score, num_actions
            )
            actions.extend(category_actions)

        # 推奨プランの作成
        plan = RecommendationPlan(
            company_id=company_id,
            score_id=score.id,
            actions=actions,
            generated_at=datetime.now(),
            generated_by=user_id
        )

        # 推奨プランを保存
        saved_plan = await self.wellness_repository.save_recommendation_plan(plan)
        logger.info(f"会社 {company_id} の推奨プランを生成しました: {len(actions)} アクション")

        return saved_plan

    def _generate_category_actions(
        self,
        category: ScoreCategory,
        score: float,
        num_actions: int
    ) -> List[RecommendationAction]:
        """
        カテゴリに基づいた推奨アクションの生成（内部メソッド）

        Args:
            category: スコアカテゴリ
            score: カテゴリスコア
            num_actions: 生成するアクション数

        Returns:
            推奨アクションのリスト
        """
        # カテゴリごとの推奨アクションテンプレート
        action_templates = {
            ScoreCategory.FINANCIAL: [
                {
                    "title": "キャッシュフロー管理の改善",
                    "description": "週次のキャッシュフロー予測を導入し、支出の優先順位付けを行う",
                    "impact_level": 4,
                    "effort_level": 2,
                    "time_frame": "short"
                },
                {
                    "title": "経費削減計画の実施",
                    "description": "不必要な支出を特定し、コスト削減目標を設定する",
                    "impact_level": 3,
                    "effort_level": 2,
                    "time_frame": "medium"
                },
                {
                    "title": "収益源の多様化",
                    "description": "新しい収益モデルや製品ラインの検討",
                    "impact_level": 5,
                    "effort_level": 4,
                    "time_frame": "long"
                }
            ],
            ScoreCategory.HEALTH: [
                {
                    "title": "従業員ウェルネスプログラムの導入",
                    "description": "ストレス管理、運動促進、健康的な食事の奨励などを含むプログラム",
                    "impact_level": 3,
                    "effort_level": 2,
                    "time_frame": "medium"
                },
                {
                    "title": "フレックスワークの導入",
                    "description": "柔軟な勤務時間と場所の選択肢を提供",
                    "impact_level": 4,
                    "effort_level": 3,
                    "time_frame": "short"
                },
                {
                    "title": "メンタルヘルスサポートの強化",
                    "description": "カウンセリングサービスへのアクセス提供とメンタルヘルスの啓発",
                    "impact_level": 4,
                    "effort_level": 2,
                    "time_frame": "medium"
                }
            ],
            # 他のカテゴリも同様に定義...
            ScoreCategory.WORK_LIFE_BALANCE: [
                {
                    "title": "ノー会議デーの設定",
                    "description": "週に1日、会議を入れない日を設定し、集中作業時間を確保",
                    "impact_level": 3,
                    "effort_level": 1,
                    "time_frame": "short"
                },
                {
                    "title": "休暇取得の促進",
                    "description": "未消化の有給休暇を可視化し、定期的な休暇取得を奨励",
                    "impact_level": 4,
                    "effort_level": 2,
                    "time_frame": "medium"
                }
            ]
        }

        # カテゴリに対応するアクションテンプレートを取得
        templates = action_templates.get(category, [])
        if not templates:
            # テンプレートがない場合は汎用アクションを作成
            templates = [
                {
                    "title": f"{category.value}の改善",
                    "description": f"{category.value}のスコアを向上させるための施策検討",
                    "impact_level": 3,
                    "effort_level": 2,
                    "time_frame": "medium"
                }
            ]

        # スコアに基づいてアクションを選択（低スコアなら優先度高のアクションを選択）
        selected_templates = templates[:num_actions]

        # 推奨アクションを作成
        actions = []
        for template in selected_templates:
            action = RecommendationAction(
                id=str(uuid.uuid4()),
                title=template["title"],
                description=template["description"],
                category=category,
                impact_level=template["impact_level"],
                effort_level=template["effort_level"],
                time_frame=template["time_frame"],
                resources=[]
            )
            actions.append(action)

        return actions


# DIコンテナからウェルネスリポジトリとFirebaseClientをインジェクションするファクトリ関数
@inject(WellnessRepositoryInterface)
def get_wellness_score_usecase(
    wellness_repository: WellnessRepositoryInterface,
    firebase_client: Optional[FirebaseClientInterface] = None
) -> WellnessScoreUseCase:
    """
    ウェルネススコアユースケースのファクトリ関数

    Args:
        wellness_repository: ウェルネスリポジトリ（DIによって自動的に注入される）
        firebase_client: Firebaseクライアント（オプション）

    Returns:
        WellnessScoreUseCaseのインスタンス
    """
    # Firebaseクライアントが指定されていない場合は取得
    if firebase_client is None:
        from core.firebase_client import get_firebase_client
        firebase_client = get_firebase_client()

    return WellnessScoreUseCase(wellness_repository, firebase_client)