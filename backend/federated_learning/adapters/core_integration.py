"""
コア統合モジュール

このモジュールは、連合学習システムとコアモジュールを統合するためのインターフェースを提供します。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
import os
import sys
from pathlib import Path

# 絶対パスでのインポートを確保するため
sys.path.append(str(Path(__file__).parents[3]))

from backend.federated_learning.client.federated_client import FederatedClient
from backend.federated_learning.models.financial_performance_predictor import FinancialPerformancePredictor, ModelFactory
from backend.federated_learning.adapters.health_impact_adapter import HealthImpactAdapter

logger = logging.getLogger(__name__)

class CoreModelIntegration:
    """コアモデル統合

    連合学習システムとコアモジュールのモデルを統合するためのクラス。
    """

    def __init__(self, client_id: str = "core_integration"):
        """初期化

        Args:
            client_id: 連合学習クライアントID
        """
        self.client_id = client_id
        self.fl_client = FederatedClient(client_id=client_id)
        self.health_adapter = HealthImpactAdapter()
        self._init_models()

        logger.info(f"コアモデル統合を初期化しました: client_id={client_id}")

    def _init_models(self) -> None:
        """モデルを初期化する"""
        # 財務パフォーマンス予測モデル - ModelFactoryを使用して自動フレームワーク検出
        try:
            # 最適なフレームワークを自動検出
            financial_model = ModelFactory.create_model(
                framework="auto",  # 利用可能なフレームワークを自動検出
                hidden_layers=[64, 32, 16],
                activation="relu",
                final_activation="linear"
            )
            logger.info(f"財務パフォーマンス予測モデルを初期化しました (フレームワーク: {financial_model.__class__.__name__})")
        except Exception as e:
            logger.error(f"モデル初期化でエラー発生: {e}")
            # フォールバックとして標準実装を使用
            financial_model = FinancialPerformancePredictor()
            logger.info("フォールバックとして標準モデルを使用します")

        self.fl_client.register_model("financial_performance", financial_model)

        # グローバルモデルを取得（存在する場合）
        try:
            self.fl_client.get_global_model("financial_performance")
            logger.info("グローバル財務パフォーマンスモデルを取得しました")
        except Exception as e:
            logger.warning(f"グローバルモデルの取得に失敗しました: {e}")

    def enhance_wellness_score_calculation(self, wellness_calculator, company_data: pd.DataFrame,
                                         industry_type: str, use_federated: bool = True) -> Dict[str, float]:
        """健康スコア計算を強化する

        Args:
            wellness_calculator: wellness_score_calculatorインスタンス
            company_data: 企業データ
            industry_type: 業界タイプ
            use_federated: 連合学習モデルを使用するかどうか

        Returns:
            拡張された健康スコア
        """
        # 既存の健康スコア計算
        base_scores = wellness_calculator.calculate_scores(company_data)

        if not use_federated:
            return base_scores

        try:
            # データの前処理
            processed_data = self._preprocess_for_federated(company_data)

            # 連合学習モデルを使用した予測
            fl_predictions = self.predict_with_federated_model(
                processed_data,
                industry_type=industry_type
            )

            # スコアの統合
            enhanced_scores = self._integrate_scores(base_scores, fl_predictions)

            logger.info(f"連合学習を使用して健康スコアを強化しました: 業界={industry_type}")
            return enhanced_scores

        except Exception as e:
            logger.error(f"連合学習による強化に失敗しました: {e}")
            return base_scores

    def _preprocess_for_federated(self, data: pd.DataFrame) -> np.ndarray:
        """連合学習用にデータを前処理する

        Args:
            data: 入力データ

        Returns:
            前処理されたデータ
        """
        # 必要な特徴量を選択
        features = data.select_dtypes(include=[np.number])

        # 欠損値を処理
        features = features.fillna(0)

        return features.values

    def predict_with_federated_model(self, data: Union[pd.DataFrame, np.ndarray],
                                   model_name: str = "financial_performance",
                                   industry_type: str = None,
                                   position_level: str = None) -> Dict[str, np.ndarray]:
        """連合学習モデルを使用して予測する

        Args:
            data: 入力データ
            model_name: モデル名
            industry_type: 業界タイプ（指定時は健康影響を考慮）
            position_level: 職位レベル（指定時は健康影響を考慮）

        Returns:
            予測結果
        """
        # データフレームをNumPy配列に変換
        if isinstance(data, pd.DataFrame):
            data = data.values

        # モデルの取得
        if model_name not in self.fl_client.models:
            raise ValueError(f"モデル '{model_name}' は登録されていません")

        model = self.fl_client.models[model_name]

        # 健康影響を考慮したモデル調整
        if industry_type:
            model = self.health_adapter.adapt_model_with_weights(
                model,
                industry_type=industry_type,
                position_level=position_level
            )

        # 予測実行
        predictions = model.predict(data)

        logger.info(f"連合学習モデル '{model_name}' を使用して予測しました")
        return predictions

    def _integrate_scores(self, base_scores: Dict[str, float],
                        fl_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """基本スコアと連合学習予測を統合する

        Args:
            base_scores: 基本スコア
            fl_predictions: 連合学習予測

        Returns:
            統合されたスコア
        """
        integrated_scores = base_scores.copy()

        # 予測平均値の取得
        pred_mean = fl_predictions.get("mean", None)
        if pred_mean is not None and len(pred_mean) > 0:
            # 例: 財務パフォーマンス予測を統合
            if "financial_health" in integrated_scores and pred_mean.shape[1] > 0:
                # 財務健全性スコアを調整（予測値の最初の次元を使用）
                base_score = integrated_scores["financial_health"]
                pred_factor = float(pred_mean[0][0])

                # スコアを調整（単純な例）
                adjusted_score = base_score * (1.0 + 0.1 * pred_factor)
                integrated_scores["financial_health"] = min(max(adjusted_score, 0.0), 10.0)  # 0-10に制限

            # 信頼性指標の追加
            if "prediction_confidence" not in integrated_scores and "std" in fl_predictions:
                pred_std = fl_predictions["std"]
                if pred_std.shape[1] > 0:
                    # 標準偏差から信頼性を計算（値が小さいほど信頼性が高い）
                    confidence = 1.0 / (1.0 + float(pred_std[0][0]))
                    integrated_scores["prediction_confidence"] = min(confidence * 10.0, 10.0)  # 0-10に制限

        return integrated_scores

    def train_with_local_data(self, data: pd.DataFrame, targets: pd.DataFrame,
                            model_name: str = "financial_performance",
                            submit_update: bool = True) -> Dict[str, float]:
        """ローカルデータでモデルを訓練する

        Args:
            data: 特徴量データ
            targets: 目標データ
            model_name: モデル名
            submit_update: 更新を提出するかどうか

        Returns:
            訓練メトリクス
        """
        # データの前処理
        X = self._preprocess_for_federated(data)
        y = targets.values

        # ローカルモデルの訓練
        metrics = self.fl_client.train_local_model(model_name, X, y)

        # 更新を提出
        if submit_update:
            success = self.fl_client.submit_model_update(model_name)
            if success:
                logger.info(f"モデル '{model_name}' の更新を提出しました")
            else:
                logger.warning(f"モデル '{model_name}' の更新提出に失敗しました")

        return metrics

    def integrate_with_performance_predictor(self, performance_predictor, data: pd.DataFrame,
                                          industry_type: str = None) -> Dict[str, Any]:
        """パフォーマンス予測器と統合する

        Args:
            performance_predictor: パフォーマンス予測器
            data: 入力データ
            industry_type: 業界タイプ

        Returns:
            拡張された予測結果
        """
        # 既存の予測
        base_prediction = performance_predictor.predict(data)

        try:
            # 連合学習を使用した予測
            fl_prediction = self.predict_with_federated_model(
                data,
                industry_type=industry_type
            )

            # 予測結果の統合
            integrated_prediction = self._integrate_predictions(base_prediction, fl_prediction)

            # 不確実性情報の追加
            if "std" in fl_prediction:
                uncertainty = np.mean(fl_prediction["std"])
                integrated_prediction["uncertainty"] = float(uncertainty)
                integrated_prediction["confidence_interval"] = [
                    float(integrated_prediction["prediction"] - 1.96 * uncertainty),
                    float(integrated_prediction["prediction"] + 1.96 * uncertainty)
                ]

            logger.info("パフォーマンス予測に連合学習予測を統合しました")
            return integrated_prediction

        except Exception as e:
            logger.error(f"連合学習の統合に失敗しました: {e}")
            return base_prediction

    def _integrate_predictions(self, base_prediction: Dict[str, Any],
                            fl_prediction: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """基本予測と連合学習予測を統合する

        Args:
            base_prediction: 基本予測
            fl_prediction: 連合学習予測

        Returns:
            統合された予測
        """
        integrated = base_prediction.copy()

        # 予測平均値の取得
        pred_mean = fl_prediction.get("mean", None)
        if pred_mean is not None and len(pred_mean) > 0:
            # 基本予測値と連合学習予測値の重み付き平均
            if "prediction" in integrated:
                base_value = integrated["prediction"]
                fl_value = float(pred_mean[0][0])

                # 重み付き平均（連合学習の方が少し重み大）
                integrated["prediction"] = 0.4 * base_value + 0.6 * fl_value
                integrated["federated_prediction"] = fl_value

        return integrated