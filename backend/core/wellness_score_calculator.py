# -*- coding: utf-8 -*-
"""
ウェルネススコア計算モジュール

VASデータ、財務データ、業界情報などから総合的なウェルネススコアを算出します。
複数の分析手法を組み合わせて、多角的な評価を行います。
"""
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from backend.core.data_preprocessor import DataPreprocessor
from backend.core.correlation_analyzer import CorrelationAnalyzer
from backend.core.time_series_analyzer import TimeSeriesAnalyzer
from backend.analysis.bayesian_analyzer import BayesianAnalyzer

# Firestoreクライアントのインポート
try:
    # 既存のFirestoreクライアント実装がある場合はそちらを使用
    from firebase_admin import firestore
except ImportError:
    # firebase_adminがインストールされていない場合は警告を表示
    import warnings
    warnings.warn("firebase_admin package is not installed. Using mock FirestoreClient.")

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class WellnessScoreError(Exception):
    """ウェルネススコア計算に関するエラー"""
    pass

# Firestoreクライアントのモッククラス（実際のFirestoreクライアントがない場合用）
class FirestoreClient:
    """
    Firestoreのシンプルなモッククライアント
    実際の実装に置き換える場合は同じインターフェースを維持してください
    """
    def __init__(self):
        self._collections = {}
        logger.info("Mock FirestoreClient initialized")

    async def query_documents(self, collection: str, filters=None, order_by=None, limit=None):
        """ドキュメントのクエリ"""
        logger.info(f"Mock query on {collection} with filters: {filters}")
        # 実際の実装ではここでFirestoreからデータを取得
        return []

    async def add_document(self, collection: str, data: Dict[str, Any]):
        """ドキュメントの追加"""
        logger.info(f"Mock adding document to {collection}: {data}")
        # 実際の実装ではここでFirestoreにデータを追加
        return {"id": "mock-doc-id"}

    async def update_document(self, collection: str, document_id: str, data: Dict[str, Any]):
        """ドキュメントの更新"""
        logger.info(f"Mock updating document {document_id} in {collection}: {data}")
        # 実際の実装ではここでFirestoreのドキュメントを更新
        return True

    async def get_document(self, collection: str, document_id: str):
        """ドキュメントの取得"""
        logger.info(f"Mock getting document {document_id} from {collection}")
        # 実際の実装ではここでFirestoreからドキュメントを取得
        return None

    async def set_document(self, collection: str, document_id: str, data: Dict[str, Any]):
        """ドキュメントの設定（存在しなければ作成）"""
        logger.info(f"Mock setting document {document_id} in {collection}: {data}")
        # 実際の実装ではここでFirestoreにドキュメントを設定
        return True

class WellnessScoreCalculator:
    """
    ウェルネススコアを計算するクラス
    VASデータ、財務データ、業界情報などから総合的なウェルネススコアを算出
    """
    def __init__(
        self,
        data_preprocessor: DataPreprocessor,
        correlation_analyzer: CorrelationAnalyzer,
        time_series_analyzer: TimeSeriesAnalyzer,
        firestore_client: FirestoreClient
    ):
        """
        初期化メソッド

        Args:
            data_preprocessor: データ前処理クラス
            correlation_analyzer: 相関分析クラス
            time_series_analyzer: 時系列分析クラス
            firestore_client: Firestoreクライアント
        """
        self.data_preprocessor = data_preprocessor
        self.correlation_analyzer = correlation_analyzer
        self.time_series_analyzer = time_series_analyzer
        self.firestore_client = firestore_client

        # VASカテゴリの重み設定
        self.category_weights = {
            'work_environment': 0.25,      # 労働環境
            'engagement': 0.30,            # エンゲージメント
            'health': 0.20,                # 健康状態
            'leadership': 0.15,            # リーダーシップ
            'communication': 0.10          # コミュニケーション
        }

        # 業界係数（業界によるスコア調整係数） - 科学的根拠に基づく
        self.industry_factors = {
            'IT': 1.0,             # 標準基準値
            'Healthcare': 1.05,    # ヘルスケア産業はウェルネスへの意識が高い傾向
            'Finance': 0.95,       # 金融業界は独自のストレス要因
            'Manufacturing': 0.90, # 製造業は物理的な労働環境の影響大
            'Retail': 0.85,        # 小売業は多様な勤務形態と顧客対応ストレス
            'Service': 0.95,       # サービス業は人的要素が大きい
            'Others': 1.0          # その他の業界は標準評価
        }

        # 企業ステージ係数（成長段階によるスコア調整係数） - 科学的根拠に基づく
        self.stage_factors = {
            'Seed': 1.10,          # シード期は高めに評価
            'Series A': 1.05,      # 初期成長段階
            'Series B': 1.00,      # 標準的な評価
            'Series C': 0.95,      # 成熟段階はやや厳しめに評価
            'Series D+': 0.90,     # より成熟した段階はさらに厳しく評価
            'IPO Ready': 0.85      # 上場準備段階は高い基準で評価
        }

        logger.info("WellnessScoreCalculator initialized successfully")

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
                raise WellnessScoreError(f"VASデータが存在しません: company_id={company_id}")

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

            # スコアの正規化（0-100の範囲に収める）
            total_score = max(0, min(100, total_score))

            # 結果の整形
            result = {
                "company_id": company_id,
                "total_score": round(total_score, 1),
                "category_scores": {k: round(v, 1) for k, v in category_scores.items()},
                "adjustments": {
                    "financial": round(financial_adjustment, 2),
                    "industry_stage": round(industry_stage_adjustment, 2),
                    "trend": round(trend_adjustment, 2)
                },
                "metadata": {
                    "industry": industry,
                    "stage": stage,
                    "calculation_date": calculation_date.isoformat(),
                    "data_points": len(vas_data)
                }
            }

            # 結果をFirestoreに保存
            await self._save_score_to_firestore(result)

            return result

        except Exception as e:
            error_msg = f"ウェルネススコア計算中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            raise WellnessScoreError(error_msg) from e

    async def _fetch_vas_data(self, company_id: str) -> pd.DataFrame:
        """VASデータをFirestoreから取得"""
        try:
            # 対象企業のVASデータを取得
            vas_docs = await self.firestore_client.query_documents(
                collection="vas_data",
                filters=[("company_id", "==", company_id)],
                order_by=("timestamp", "desc"),
                limit=100  # 直近100件まで
            )

            # DataFrameに変換
            if not vas_docs:
                return pd.DataFrame()

            vas_data = self.data_preprocessor.preprocess_firestore_data(
                vas_docs,
                data_type="vas_data"
            )

            return vas_data

        except Exception as e:
            logger.error(f"VASデータ取得中にエラーが発生: {str(e)}")
            raise WellnessScoreError(f"VASデータ取得エラー: {str(e)}") from e

    async def _fetch_financial_data(self, company_id: str) -> pd.DataFrame:
        """財務データをFirestoreから取得"""
        try:
            # 対象企業の財務データを取得
            financial_docs = await self.firestore_client.query_documents(
                collection="financial_data",
                filters=[("company_id", "==", company_id)],
                order_by=("timestamp", "desc"),
                limit=12  # 直近12か月分
            )

            # DataFrameに変換
            if not financial_docs:
                return pd.DataFrame()

            financial_data = self.data_preprocessor.preprocess_firestore_data(
                financial_docs,
                data_type="financial_data"
            )

            return financial_data

        except Exception as e:
            logger.error(f"財務データ取得中にエラーが発生: {str(e)}")
            raise WellnessScoreError(f"財務データ取得エラー: {str(e)}") from e

    async def _calculate_category_scores(self, vas_data: pd.DataFrame) -> Dict[str, float]:
        """VASデータから各カテゴリのスコアを計算"""
        try:
            # 各カテゴリごとのスコアを集計
            category_scores = {}

            for category in self.category_weights.keys():
                if category in vas_data.columns:
                    # カテゴリごとの平均スコアを算出（直近のデータほど重みを高く）
                    # 指数関数的な重み付け
                    weights = np.exp(np.linspace(0, 1, len(vas_data)))
                    weights = weights / weights.sum()  # 正規化

                    category_scores[category] = np.average(
                        vas_data[category],
                        weights=weights
                    )
                else:
                    # カテゴリが存在しない場合はデフォルト値
                    category_scores[category] = 70.0

            return category_scores

        except Exception as e:
            logger.error(f"カテゴリスコア計算中にエラーが発生: {str(e)}")
            raise WellnessScoreError(f"カテゴリスコア計算エラー: {str(e)}") from e

    def _calculate_base_score(self, category_scores: Dict[str, float]) -> float:
        """カテゴリスコアから基本スコアを計算"""
        try:
            # 加重平均によるベーススコアの計算
            base_score = 0.0

            for category, score in category_scores.items():
                if category in self.category_weights:
                    base_score += score * self.category_weights[category]

            return base_score

        except Exception as e:
            logger.error(f"基本スコア計算中にエラーが発生: {str(e)}")
            raise WellnessScoreError(f"基本スコア計算エラー: {str(e)}") from e

    async def _calculate_financial_correlation(
        self,
        vas_data: pd.DataFrame,
        financial_data: pd.DataFrame
    ) -> float:
        """VASスコアと財務指標の相関に基づく調整値を計算"""
        try:
            # 日付ベースでデータを結合
            merged_data = pd.merge(
                vas_data,
                financial_data,
                on="timestamp",
                how="inner",
                suffixes=('_vas', '_fin')
            )

            if len(merged_data) < 3:
                # 相関を計算するには少なくとも3つのデータポイントが必要
                return 0.0

            # 相関係数の計算
            correlation_result = await self.correlation_analyzer.analyze(
                merged_data,
                [c for c in vas_data.columns if c != 'timestamp'],
                [c for c in financial_data.columns if c != 'timestamp']
            )

            # 平均相関係数に基づく調整値
            # 正の相関が強いほど調整値は大きくなる
            avg_corr = correlation_result.get('average_correlation', 0)

            # 相関係数に基づいて-5から+5の範囲で調整
            adjustment = avg_corr * 5

            return adjustment

        except Exception as e:
            logger.error(f"財務相関計算中にエラーが発生: {str(e)}")
            # エラーの場合は調整しない
            return 0.0

    def _apply_industry_stage_adjustment(self, industry: str, stage: str) -> float:
        """業界とステージに基づく調整値を計算"""
        try:
            # 業界係数の取得
            industry_factor = self.industry_factors.get(industry, 1.0)

            # ステージ係数の取得
            stage_factor = self.stage_factors.get(stage, 1.0)

            # 調整値の計算 (-5から+5の範囲)
            adjustment = ((industry_factor * stage_factor) - 1.0) * 10

            return adjustment

        except Exception as e:
            logger.error(f"業界・ステージ調整中にエラーが発生: {str(e)}")
            # エラーの場合は調整しない
            return 0.0

    async def _calculate_trend_adjustment(self, vas_data: pd.DataFrame) -> float:
        """時系列トレンドに基づく調整値を計算"""
        try:
            # 時系列分析の実行
            trend_result = await self.time_series_analyzer.analyze_trend(
                vas_data,
                "overall_score" if "overall_score" in vas_data.columns else vas_data.columns[1]
            )

            # トレンド係数の取得
            trend_coefficient = trend_result.get('trend_coefficient', 0)

            # トレンドに基づく調整値 (-3から+3の範囲)
            # 上昇トレンドの場合はプラス、下降トレンドの場合はマイナス
            adjustment = trend_coefficient * 3

            return adjustment

        except Exception as e:
            logger.error(f"トレンド調整中にエラーが発生: {str(e)}")
            # エラーの場合は調整しない
            return 0.0

    def _apply_adjustments(
        self,
        base_score: float,
        financial_adjustment: float,
        industry_stage_adjustment: float,
        trend_adjustment: float
    ) -> float:
        """各種調整を適用して最終スコアを計算"""
        try:
            # 調整の適用
            adjusted_score = base_score

            # 財務相関による調整
            adjusted_score += financial_adjustment

            # 業界・ステージによる調整
            adjusted_score += industry_stage_adjustment

            # トレンドによる調整
            adjusted_score += trend_adjustment

            return adjusted_score

        except Exception as e:
            logger.error(f"調整適用中にエラーが発生: {str(e)}")
            # エラーの場合はベーススコアを返す
            return base_score

    async def _save_score_to_firestore(self, score_data: Dict[str, Any]) -> None:
        """計算したスコアをFirestoreに保存"""
        try:
            # スコア履歴コレクションに保存
            await self.firestore_client.add_document(
                collection="wellness_scores",
                data=score_data
            )

            # 企業ドキュメントを更新
            await self.firestore_client.update_document(
                collection="companies",
                document_id=score_data["company_id"],
                data={
                    "wellnessScore": score_data["total_score"],
                    "lastScoreUpdate": score_data["metadata"]["calculation_date"],
                    "categoryScores": score_data["category_scores"]
                }
            )

            logger.info(f"ウェルネススコアをFirestoreに保存しました: company_id={score_data['company_id']}")

        except Exception as e:
            logger.error(f"スコア保存中にエラーが発生: {str(e)}")
            raise WellnessScoreError(f"スコア保存エラー: {str(e)}") from e


def create_wellness_score_calculator() -> WellnessScoreCalculator:
    """WellnessScoreCalculatorのインスタンスを作成するファクトリ関数"""
    from backend.core.data_preprocessor import DataPreprocessor
    from backend.core.correlation_analyzer import CorrelationAnalyzer
    from backend.core.time_series_analyzer import TimeSeriesAnalyzer

    data_preprocessor = DataPreprocessor()
    correlation_analyzer = CorrelationAnalyzer()
    time_series_analyzer = TimeSeriesAnalyzer()
    firestore_client = FirestoreClient()  # モックFirestoreClientのインスタンスを作成

    return WellnessScoreCalculator(
        data_preprocessor,
        correlation_analyzer,
        time_series_analyzer,
        firestore_client
    )

# テスト用関数
def test_industry_stage_adjustment():
    """業界・ステージ調整のテスト関数"""
    calculator = create_wellness_score_calculator()

    # テストケース
    test_cases = [
        # (業界, ステージ, 期待値)
        ("IT", "Series B", 0.0),  # 基準値: 1.0 * 1.0 = 1.0 -> (1.0 - 1.0) * 10 = 0.0
        ("Retail", "Seed", -0.65),  # 0.85 * 1.10 = 0.935 -> (0.935 - 1.0) * 10 = -0.65
        ("Healthcare", "IPO Ready", -1.075),  # 1.05 * 0.85 = 0.8925 -> (0.8925 - 1.0) * 10 = -1.075
    ]

    for industry, stage, expected in test_cases:
        adjustment = calculator._apply_industry_stage_adjustment(industry, stage)
        error_margin = abs(adjustment - expected)
        assert error_margin < 0.001, f"業界 '{industry}', ステージ '{stage}' の調整値が期待値と異なります: {adjustment} != {expected}"

    print("業界・ステージ調整のテスト成功!")