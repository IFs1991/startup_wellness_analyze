# -*- coding: utf-8 -*-
"""
企業ウェルネス スコア計算モジュール
組織のVASと財務データを分析し、総合的なウェルネススコアを計算します。

注意: このモジュールはレガシーコードとの互換性のために維持されています。
新しい実装はusecases.wellness_score_usecaseを使用してください。
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging
import importlib
from sklearn.preprocessing import MinMaxScaler
# 循環インポートを避けるため、パターンモジュールの遅延インポートを使用
from .patterns import LazyImport
from analysis.correlation_analysis import CorrelationAnalyzer
from analysis.TimeSeriesAnalyzer import TimeSeriesAnalyzer
from .common_logger import get_logger
from .exceptions import WellnessScoreError
from .firebase_client import FirebaseClientInterface, get_firebase_client
from .di_config import get_wellness_score_usecase_from_di

# 遅延インポートの設定
DataPreprocessor = LazyImport('core.data_preprocessor', 'DataPreprocessor')

# 連合学習モジュールのインポート - エラーを回避するためにフラグのみ設定
FEDERATED_LEARNING_AVAILABLE = False
import warnings
warnings.warn("Federated learning module is not available. Using standard calculation only.")

# ロギングの設定
logger = get_logger(__name__)

# Firestoreクライアントのモッククラスは削除し、新しいモジュールのものを使用

class WellnessScoreCalculator:
    """
    ウェルネススコアを計算するクラス
    VASデータ、財務データ、業界情報などから総合的なウェルネススコアを算出

    注意: このクラスはレガシーコードとの互換性のために維持されています。
    新しい実装はusecases.wellness_score_usecaseを使用してください。
    """
    def __init__(
        self,
        correlation_analyzer: CorrelationAnalyzer,
        time_series_analyzer: TimeSeriesAnalyzer,
        firebase_client: FirebaseClientInterface,
        use_federated_learning: bool = False
    ):
        """
        WellnessScoreCalculatorクラスのコンストラクタ

        Args:
            correlation_analyzer: 相関分析クラス
            time_series_analyzer: 時系列分析クラス
            firebase_client: Firebaseクライアント
            use_federated_learning: 連合学習を使用するかどうか
        """
        self._data_preprocessor = None  # 遅延ロード用
        self.correlation_analyzer = correlation_analyzer
        self.time_series_analyzer = time_series_analyzer
        self.firebase_client = firebase_client

        # 連合学習の設定 - 無効化して標準計算のみを使用
        self.use_federated_learning = False
        self.federated_integration = None
        logger.info("連合学習モジュールは使用しません")

        # データスケーラーの初期化
        self.scaler = MinMaxScaler(feature_range=(0, 100))
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

        # 新しいユースケースを取得
        self._wellness_score_usecase = get_wellness_score_usecase_from_di()

        logger.info("WellnessScoreCalculatorが初期化されました（互換モード）")

    def _get_data_preprocessor(self):
        """DataPreprocessorインスタンスを遅延ロードして返す"""
        if self._data_preprocessor is None:
            self._data_preprocessor = DataPreprocessor()
            logger.info("DataPreprocessorを初期化しました")
        return self._data_preprocessor

    async def calculate_wellness_score(
        self,
        company_id: str,
        industry: str,
        stage: str,
        calculation_date: Optional[datetime] = None,
        use_federated: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        ウェルネススコアを計算する（新しいユースケースに委譲）

        Args:
            company_id: 企業ID
            industry: 業界
            stage: 企業ステージ
            calculation_date: 計算基準日（Noneの場合は現在日時）
            use_federated: 連合学習を使用するかどうか（Noneの場合はインスタンス設定を使用）

        Returns:
            Dict[str, Any]: 総合スコアと各カテゴリスコア
        """
        try:
            # 新しいユースケースに処理を委譲
            logger.info(f"ウェルネススコア計算リクエストを新実装に委譲します: company_id={company_id}")

            # 新しいユースケース実装を呼び出し
            result = await self._wellness_score_usecase.calculate_wellness_score(
                company_id=company_id,
                industry=industry,
                stage=stage,
                calculation_date=calculation_date
            )

            # 必要に応じて互換性のためのデータ変換
            if 'federated_learning' not in result:
                result['federated_learning'] = {"federated_used": False}

            if 'federated_adjustment' not in result:
                result['federated_adjustment'] = 0.0

            logger.info(f"ウェルネススコア計算が完了しました: company_id={company_id}, score={result['total_score']}")
            return result

        except Exception as e:
            logger.error(f"ウェルネススコア計算エラー: {e}")
            raise WellnessScoreError(f"ウェルネススコア計算中にエラーが発生しました: {e}")

    def _prepare_federated_data(self, vas_data: pd.DataFrame, financial_data: pd.DataFrame) -> pd.DataFrame:
        """
        連合学習用のデータを準備する

        Args:
            vas_data: VASデータ
            financial_data: 財務データ

        Returns:
            pd.DataFrame: 連合学習用データフレーム
        """
        # 両方のデータが利用可能な場合は結合
        if not vas_data.empty and not financial_data.empty:
            # 基準日付の取得
            latest_date = max(vas_data['assessment_date'].max(), financial_data['report_date'].max())

            # 最新のVASデータを取得
            latest_vas = vas_data[vas_data['assessment_date'] == vas_data['assessment_date'].max()]

            # 最新の財務データを取得
            latest_financial = financial_data[financial_data['report_date'] == financial_data['report_date'].max()]

            # 両方のデータを結合するための共通カラム
            vas_features = {
                col: f'vas_{col}' for col in latest_vas.columns
                if col not in ['company_id', 'assessment_date', 'employee_id']
            }

            financial_features = {
                col: f'fin_{col}' for col in latest_financial.columns
                if col not in ['company_id', 'report_date']
            }

            # カラム名を変更して結合
            vas_renamed = latest_vas.rename(columns=vas_features)
            financial_renamed = latest_financial.rename(columns=financial_features)

            # 集約（VASデータは複数の従業員分があるため）
            vas_agg = vas_renamed.groupby('company_id').mean().reset_index()

            # マージ
            merged_data = pd.merge(
                vas_agg,
                financial_renamed,
                on='company_id',
                how='inner'
            )

            return merged_data

        # VASデータのみの場合
        elif not vas_data.empty:
            return vas_data.copy()

        # 財務データのみの場合
        elif not financial_data.empty:
            return financial_data.copy()

        # どちらのデータもない場合
        else:
            return pd.DataFrame()

    def calculate_scores(self, company_data: pd.DataFrame) -> Dict[str, float]:
        """
        連合学習統合用の簡易スコア計算メソッド
        CoreModelIntegrationから呼び出されるため、非同期でないバージョンを提供

        Args:
            company_data: 企業データ（VASと財務の結合データ）

        Returns:
            Dict[str, float]: 各カテゴリのスコア
        """
        # 簡易的なスコア計算（実際の実装ではもっと複雑）
        scores = {}

        # VAS関連の特徴量を取得
        vas_cols = [col for col in company_data.columns if col.startswith('vas_')]
        if vas_cols:
            # 各カテゴリの平均を計算
            for category in ['work_environment', 'engagement', 'health', 'leadership', 'communication']:
                category_cols = [col for col in vas_cols if category in col]
                if category_cols:
                    scores[category] = company_data[category_cols].mean().mean()

        # 財務関連の特徴量を取得
        fin_cols = [col for col in company_data.columns if col.startswith('fin_')]
        if fin_cols:
            scores['financial_health'] = company_data[fin_cols].mean().mean()

        # 値を0-10の範囲に正規化
        for key in scores:
            # 最小値0、最大値10として正規化（単純化のため）
            scores[key] = min(max(scores[key] * 2, 0), 10)

        return scores

    async def _fetch_vas_data(self, company_id: str) -> pd.DataFrame:
        """
        企業のVASデータをFirestoreから取得

        Args:
            company_id: 企業ID

        Returns:
            VASデータを含むDataFrame
        """
        try:
            logger.info(f"企業ID {company_id} のVASデータを取得中")

            # Firestoreからデータを取得
            filters = [
                {"field": "company_id", "op": "==", "value": company_id}
            ]

            vas_documents = await self.firebase_client.query_documents(
                collection="vas_responses",
                filters=filters,
                order_by=[{"field": "timestamp", "direction": "asc"}]
            )

            if not vas_documents:
                logger.warning(f"企業ID {company_id} のVASデータが見つかりません")
                return pd.DataFrame()

            # DataFrameに変換
            df = pd.DataFrame(vas_documents)

            # ソートと前処理
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            return df

        except Exception as e:
            error_msg = f"VASデータ取得中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            raise WellnessScoreError(error_msg)

    async def _fetch_financial_data(self, company_id: str) -> pd.DataFrame:
        """
        企業の財務データをFirestoreから取得

        Args:
            company_id: 企業ID

        Returns:
            財務データを含むDataFrame
        """
        try:
            logger.info(f"企業ID {company_id} の財務データを取得中")

            # Firestoreからデータを取得
            filters = [
                {"field": "company_id", "op": "==", "value": company_id}
            ]

            financial_documents = await self.firebase_client.query_documents(
                collection="financial_data",
                filters=filters,
                order_by=[{"field": "timestamp", "direction": "asc"}]
            )

            if not financial_documents:
                logger.warning(f"企業ID {company_id} の財務データが見つかりません")
                return pd.DataFrame()

            # DataFrameに変換
            df = pd.DataFrame(financial_documents)

            # ソートと前処理
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            return df

        except Exception as e:
            error_msg = f"財務データ取得中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            raise WellnessScoreError(error_msg)

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
            industry_factor = self.industry_adjustments.get(industry, 1.0)

            # ステージ係数の取得
            stage_factor = self.stage_adjustments.get(stage, 1.0)

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
            await self.firebase_client.add_document(
                collection="wellness_scores",
                data=score_data
            )

            # 企業ドキュメントを更新
            await self.firebase_client.update_document(
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
    """
    WellnessScoreCalculatorのファクトリ関数
    DIコンテナから取得したコンポーネントを使用してWellnessScoreCalculatorのインスタンスを作成

    Returns:
        WellnessScoreCalculator: 作成されたインスタンス
    """
    try:
        # 必要なコンポーネントの取得（必要に応じて遅延インポート）
        from analysis.correlation_analysis import CorrelationAnalyzer
        from analysis.TimeSeriesAnalyzer import TimeSeriesAnalyzer
        from core.firebase_client import get_firebase_client

        correlation_analyzer = CorrelationAnalyzer()
        time_series_analyzer = TimeSeriesAnalyzer()
        firebase_client = get_firebase_client()

        # 新しいWellnessScoreCalculatorインスタンスを作成して返す
        logger.info("WellnessScoreCalculatorのファクトリ関数が呼び出されました（互換モード）")
        return WellnessScoreCalculator(
            correlation_analyzer=correlation_analyzer,
            time_series_analyzer=time_series_analyzer,
            firebase_client=firebase_client
        )
    except Exception as e:
        logger.error(f"WellnessScoreCalculatorの作成に失敗しました: {e}")
        raise

# テスト用の関数（本来は別のモジュールに分離すべき）
def test_industry_stage_adjustment():
    """業界・ステージの調整値を確認するためのテスト関数"""
    calculator = create_wellness_score_calculator()
    industries = ["tech", "healthcare", "finance", "retail", "manufacturing", "education", "other"]
    stages = ["seed", "early", "growth", "expansion", "mature", "other"]

    for industry in industries:
        for stage in stages:
            adjustment = calculator._apply_industry_stage_adjustment(industry, stage)
            logger.info(f"Industry: {industry}, Stage: {stage}, Adjustment: {adjustment}")