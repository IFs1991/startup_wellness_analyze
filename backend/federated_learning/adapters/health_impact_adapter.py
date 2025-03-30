"""
健康影響指標アダプター

このモジュールは、連合学習モデルと健康影響指標データを統合するためのアダプターを提供します。
健康影響重みを使用して、モデルのパラメータを調整したり、特徴量を拡張したりします。
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

from backend.federated_learning.models.model_interface import ModelInterface
from backend.federated_learning.models.financial_performance_predictor import FinancialPerformancePredictor, ModelFactory
from backend.federated_learning.client.federated_client import FederatedClient

logger = logging.getLogger(__name__)

# モデルタイプの識別関数
def is_financial_performance_model(model: ModelInterface) -> bool:
    """モデルが金融パフォーマンスモデルかどうかを判定する

    異なるフレームワーク実装でも対応できるように、クラス名で判断するのではなく、
    モデルの特性に基づいて判定します。

    Args:
        model: 判定対象のモデル

    Returns:
        金融パフォーマンスモデルかどうか
    """
    # クラス名で判定（TensorFlowFinancialPredictor, PyTorchFinancialPredictorなど）
    if "financial" in model.__class__.__name__.lower():
        return True

    # メタデータが存在する場合はそれで判定
    if hasattr(model, "metrics") and hasattr(model, "hidden_layers"):
        return True

    # 直接インスタンスチェック（後方互換性のため）
    if isinstance(model, FinancialPerformancePredictor):
        return True

    return False

class HealthImpactAdapter:
    """健康影響指標アダプター

    連合学習モデルと健康影響指標データを統合するためのアダプター。
    """

    def __init__(self, db_connection=None):
        """初期化

        Args:
            db_connection: データベース接続オブジェクト（省略時は環境変数から接続）
        """
        self.db_connection = db_connection
        self._cache = {}
        logger.info("健康影響指標アダプターを初期化しました")

    def get_health_impact_weights(self, industry_type: str, position_level: Optional[str] = None) -> pd.DataFrame:
        """健康影響重みを取得する

        Args:
            industry_type: 業界タイプ
            position_level: 職位レベル（省略可）

        Returns:
            健康影響重みデータフレーム
        """
        # キャッシュキー
        cache_key = f"{industry_type}_{position_level or 'all'}"

        # キャッシュにある場合はキャッシュから返す
        if cache_key in self._cache:
            return self._cache[cache_key]

        # TODO: 実際のデータベース接続から健康影響重みを取得
        # 現在はダミーデータを生成
        weights_df = self._generate_dummy_weights(industry_type, position_level)

        # キャッシュに保存
        self._cache[cache_key] = weights_df

        return weights_df

    def _generate_dummy_weights(self, industry_type: str, position_level: Optional[str] = None) -> pd.DataFrame:
        """ダミーの健康影響重みを生成する（開発用）

        Args:
            industry_type: 業界タイプ
            position_level: 職位レベル（省略可）

        Returns:
            健康影響重みデータフレーム
        """
        # 業界ごとに特化した重み
        industry_weights = {
            "tech": {
                "work_life_balance": 0.82,
                "stress_level": 0.75,
                "physical_activity": 0.60,
                "sleep_quality": 0.85,
                "nutrition": 0.70,
                "mental_health": 0.90
            },
            "finance": {
                "work_life_balance": 0.70,
                "stress_level": 0.85,
                "physical_activity": 0.55,
                "sleep_quality": 0.75,
                "nutrition": 0.65,
                "mental_health": 0.80
            },
            "healthcare": {
                "work_life_balance": 0.75,
                "stress_level": 0.80,
                "physical_activity": 0.70,
                "sleep_quality": 0.75,
                "nutrition": 0.80,
                "mental_health": 0.85
            }
        }

        # デフォルト重み
        default_weights = {
            "work_life_balance": 0.75,
            "stress_level": 0.75,
            "physical_activity": 0.65,
            "sleep_quality": 0.75,
            "nutrition": 0.70,
            "mental_health": 0.80
        }

        # 業界に応じた重みを選択
        industry_weights = industry_weights.get(industry_type.lower(), default_weights)

        # 職位レベルに基づいて調整（存在する場合）
        if position_level:
            if position_level.lower() == "manager":
                # マネージャーはストレスとワークライフバランスの影響が大きい
                industry_weights["stress_level"] *= 1.1
                industry_weights["work_life_balance"] *= 1.15
            elif position_level.lower() == "individual_contributor":
                # 個人プレイヤーは身体活動と睡眠の影響が大きい
                industry_weights["physical_activity"] *= 1.1
                industry_weights["sleep_quality"] *= 1.1

        # データフレームに変換
        weights_df = pd.DataFrame([{
            "factor": factor,
            "base_weight": value,
            "adjustment_coefficient": 1.0,
            "final_weight": value
        } for factor, value in industry_weights.items()])

        return weights_df

    def adapt_model_with_weights(self,
                               model: ModelInterface,
                               industry_type: str,
                               position_level: Optional[str] = None) -> ModelInterface:
        """健康影響重みに基づいてモデルを調整する

        Args:
            model: 調整対象モデル
            industry_type: 業界タイプ
            position_level: 職位レベル（省略可）

        Returns:
            調整されたモデル
        """
        # 健康影響重みを取得
        weights = self.get_health_impact_weights(industry_type, position_level)

        # モデルの重みを取得
        model_weights = model.get_weights()

        # モデルタイプに応じた調整
        if is_financial_performance_model(model):
            adjusted_weights = self._adjust_financial_model(model_weights, weights)
            model.set_weights(adjusted_weights)
        else:
            logger.warning(f"未対応のモデルタイプ: {type(model).__name__}")

        logger.info(f"モデルを健康影響重みで調整しました: 業界={industry_type}, 職位={position_level or 'すべて'}")
        return model

    def _adjust_financial_model(self,
                              model_weights: List[np.ndarray],
                              health_weights: pd.DataFrame) -> List[np.ndarray]:
        """金融パフォーマンスモデルの重みを調整する

        Args:
            model_weights: モデルの重みリスト
            health_weights: 健康影響重みデータフレーム

        Returns:
            調整されたモデルの重みリスト
        """
        # 重みの調整は主にバイアス項に適用
        # これは簡略化された実装であり、実際にはもっと複雑な調整が必要
        adjusted_weights = model_weights.copy()

        # 出力層のバイアスが最後から2番目の要素と仮定
        if len(adjusted_weights) >= 2:
            # 健康影響要因の平均重みを計算
            avg_health_impact = health_weights['final_weight'].mean()

            # バイアス項を調整（出力を補正）
            bias_term = adjusted_weights[-2]
            adjustment_factor = 1.0 + (avg_health_impact - 0.7) / 0.7  # 0.7を基準値と仮定
            adjusted_weights[-2] = bias_term * adjustment_factor

        return adjusted_weights

    def create_health_weighted_features(self,
                                      data: pd.DataFrame,
                                      industry_type: str,
                                      position_level: Optional[str] = None) -> pd.DataFrame:
        """健康影響重みに基づいて特徴量を拡張する

        Args:
            data: 入力データフレーム
            industry_type: 業界タイプ
            position_level: 職位レベル（省略可）

        Returns:
            拡張された特徴量を含むデータフレーム
        """
        # 健康影響重みを取得
        weights = self.get_health_impact_weights(industry_type, position_level)

        # 入力データをコピー
        enhanced_data = data.copy()

        # 健康関連の特徴量があれば、重み付けした特徴量を追加
        health_factors = weights['factor'].unique()
        for factor in health_factors:
            if factor in data.columns:
                weight_value = weights[weights['factor'] == factor]['final_weight'].values[0]
                enhanced_data[f"{factor}_weighted"] = data[factor] * weight_value

        # 産業タイプをカテゴリ特徴量として追加
        enhanced_data['industry_type'] = industry_type

        if position_level:
            enhanced_data['position_level'] = position_level

        logger.info(f"健康影響重みに基づいて特徴量を拡張しました: 業界={industry_type}")
        return enhanced_data