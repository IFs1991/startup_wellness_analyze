# -*- coding: utf-8 -*-
"""
企業ウェルネス スコア計算モジュール
組織のVASと財務データを分析し、総合的なウェルネススコアを計算します。
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging
import importlib
from sklearn.preprocessing import MinMaxScaler
# 循環インポートを避けるため、クラス使用時に動的インポート
# from core.data_preprocessor import DataPreprocessor
from core.correlation_analyzer import CorrelationAnalyzer
from core.time_series_analyzer import TimeSeriesAnalyzer
from service.firestore.client import get_firestore_client

# 連合学習モジュールのインポート - エラーを回避するためにフラグのみ設定
FEDERATED_LEARNING_AVAILABLE = False
import warnings
warnings.warn("Federated learning module is not available. Using standard calculation only.")

# Firestoreクライアントのインポート
# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

try:
    # 既存のFirestoreクライアント実装がある場合はそちらを使用
    from firebase_admin import firestore
    from firebase_admin import credentials
    import firebase_admin

    # アプリが初期化されていない場合は初期化
    try:
        firebase_admin.get_app()
    except ValueError:
        # 環境に応じた初期化（本番環境ではサービスアカウント認証情報を使用）
        try:
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred)
            logger.info("Firebase initialized with default credentials")
        except Exception as e:
            logger.warning(f"Firebase initialization failed: {e}")
except ImportError:
    # firebase_adminがインストールされていない場合は警告を表示
    import warnings
    warnings.warn("firebase_admin package is not installed. Using mock FirestoreClient.")

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
        correlation_analyzer: CorrelationAnalyzer,
        time_series_analyzer: TimeSeriesAnalyzer,
        firestore_client: FirestoreClient,
        use_federated_learning: bool = False
    ):
        """
        WellnessScoreCalculatorクラスのコンストラクタ

        Args:
            correlation_analyzer: 相関分析クラス
            time_series_analyzer: 時系列分析クラス
            firestore_client: Firestoreクライアント
            use_federated_learning: 連合学習を使用するかどうか
        """
        self._data_preprocessor = None  # 遅延ロード用
        self.correlation_analyzer = correlation_analyzer
        self.time_series_analyzer = time_series_analyzer
        self.firestore_client = firestore_client

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

        logger.info("WellnessScoreCalculatorが初期化されました")

    def _get_data_preprocessor(self):
        """DataPreprocessorインスタンスを遅延ロードして返す"""
        if self._data_preprocessor is None:
            from core.data_preprocessor import DataPreprocessor
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
        ウェルネススコアを計算する

        Args:
            company_id: 企業ID
            industry: 業界
            stage: 企業ステージ
            calculation_date: 計算基準日（Noneの場合は現在日時）
            use_federated: 連合学習を使用するかどうか（Noneの場合はインスタンス設定を使用）

        Returns:
            Dict[str, Any]: 総合スコアと各カテゴリスコア
        """
        # 連合学習使用フラグの設定
        use_federated_learning = self.use_federated_learning if use_federated is None else use_federated

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

            # 連合学習によるスコア強化
            federated_adjustment = 0.0
            federated_data = {}
            if use_federated_learning and self.use_federated_learning:
                try:
                    # 連合学習データを作成
                    company_data = self._prepare_federated_data(vas_data, financial_data)

                    # 連合学習による予測を取得
                    fl_scores = self.federated_integration.enhance_wellness_score_calculation(
                        self, company_data, industry, True
                    )

                    # 連合学習の結果を保存
                    federated_data = {
                        "federated_scores": fl_scores,
                        "federated_used": True
                    }

                    # 連合学習によるスコア調整
                    if "prediction_confidence" in fl_scores:
                        confidence = fl_scores["prediction_confidence"] / 10.0  # 0-1に正規化
                        federated_adjustment = (fl_scores.get("financial_health", base_score) - base_score) * confidence
                        logger.info(f"Applied federated learning adjustment: {federated_adjustment}")
                except Exception as e:
                    logger.error(f"Failed to apply federated learning: {e}")
                    federated_data = {"federated_used": False, "error": str(e)}
            else:
                federated_data = {"federated_used": False}

            # 連合学習の調整を適用
            final_score = min(max(total_score + federated_adjustment, 0), 10)

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
                "federated_adjustment": federated_adjustment,
                "category_scores": category_scores,
                "federated_learning": federated_data
            }

            # スコアの保存
            await self._save_score_to_firestore(result)

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
        Firestoreから指定された企業のVASデータを取得します。
        """
        logger.info(f"企業 {company_id} のVASデータを取得中...")
        data_preprocessor = self._get_data_preprocessor()

        try:
            conditions = [
                {"field": "company_id", "operator": "==", "value": company_id}
            ]

            vas_data = await data_preprocessor.get_data(
                collection_name="vas_responses",
                conditions=conditions
            )

            if vas_data.empty:
                logger.warning(f"企業 {company_id} のVASデータが見つかりませんでした")
                return pd.DataFrame()

            # 前処理を適用
            vas_data = data_preprocessor.preprocess_firestore_data(
                vas_data.to_dict('records'),
                data_type='vas_data'
            )

            logger.info(f"企業 {company_id} のVASデータ取得と前処理が完了しました。行数: {len(vas_data)}")
            return vas_data

        except Exception as e:
            logger.error(f"VASデータ取得中にエラーが発生しました: {str(e)}")
            raise WellnessScoreError(f"VASデータ取得エラー: {str(e)}")

    async def _fetch_financial_data(self, company_id: str) -> pd.DataFrame:
        """
        Firestoreから指定された企業の財務データを取得します。
        """
        logger.info(f"企業 {company_id} の財務データを取得中...")
        data_preprocessor = self._get_data_preprocessor()

        try:
            conditions = [
                {"field": "company_id", "operator": "==", "value": company_id}
            ]

            financial_data = await data_preprocessor.get_data(
                collection_name="financial_data",
                conditions=conditions
            )

            if financial_data.empty:
                logger.warning(f"企業 {company_id} の財務データが見つかりませんでした")
                return pd.DataFrame()

            # 前処理を適用
            financial_data = data_preprocessor.preprocess_firestore_data(
                financial_data.to_dict('records'),
                data_type='financial_data'
            )

            logger.info(f"企業 {company_id} の財務データ取得と前処理が完了しました。行数: {len(financial_data)}")
            return financial_data

        except Exception as e:
            logger.error(f"財務データ取得中にエラーが発生しました: {str(e)}")
            raise WellnessScoreError(f"財務データ取得エラー: {str(e)}")

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
    from core.correlation_analyzer import CorrelationAnalyzer
    from core.time_series_analyzer import TimeSeriesAnalyzer

    correlation_analyzer = CorrelationAnalyzer()
    firestore_client = FirestoreClient()

    # FirestoreClientインスタンスをTimeSeriesAnalyzerに渡す
    db_client = get_firestore_client()
    time_series_analyzer = TimeSeriesAnalyzer(db=db_client)

    return WellnessScoreCalculator(
        correlation_analyzer,
        time_series_analyzer,
        firestore_client,
        use_federated_learning=False
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