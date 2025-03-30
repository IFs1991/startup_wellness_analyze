"""
差分プライバシー実装

このモジュールは、連合学習における差分プライバシーを実装します。
モデル更新にノイズを追加することで、個人データを保護します。
Flower連合学習フレームワークと互換性があり、複数のMLフレームワークをサポートします。
"""

import logging
import numpy as np
import math
from typing import List, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

class DifferentialPrivacy:
    """差分プライバシー

    連合学習における差分プライバシーを実装します。モデル更新にノイズを追加することで、
    クライアントのプライバシーを保護します。フレームワーク非依存の実装を提供します。
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, clip_norm: float = 1.0):
        """初期化

        Args:
            epsilon: プライバシー予算 (小さいほど強力)
            delta: 失敗確率 (小さいほど強力)
            clip_norm: 勾配クリッピングのためのL2ノルム
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.noise_multiplier = self._calculate_noise_multiplier()

        logger.info(f"差分プライバシー初期化: epsilon={epsilon}, delta={delta}, clip_norm={clip_norm}, "
                   f"noise_multiplier={self.noise_multiplier}")

    def _calculate_noise_multiplier(self) -> float:
        """ノイズ乗数を計算する

        Returns:
            ノイズ乗数
        """
        # ラプラス機構の場合の単純な式 - 実際の実装ではより複雑な計算が必要
        # この単純化された式は例示目的
        noise_multiplier = self.clip_norm / (2.0 * self.epsilon)
        return max(1.0, noise_multiplier)  # 最小値は1.0

    def _clip_weights(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """重みをクリッピングする

        Args:
            weights: 重みリスト

        Returns:
            クリッピングされた重み
        """
        # 全重みを一つのベクトルとみなしてL2ノルムを計算
        flattened = np.concatenate([w.flatten() for w in weights])
        l2_norm = np.linalg.norm(flattened)

        if l2_norm <= self.clip_norm:
            # ノルムが閾値以下ならそのまま返す
            return weights

        # ノルムが閾値を超える場合はスケーリング
        scaling_factor = self.clip_norm / l2_norm
        clipped_weights = [w * scaling_factor for w in weights]

        logger.debug(f"重みクリッピング: 元ノルム={l2_norm}, 縮小比={scaling_factor}")
        return clipped_weights

    def _add_noise(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """ガウスノイズを追加する

        Args:
            weights: 重みリスト

        Returns:
            ノイズが追加された重み
        """
        noised_weights = []

        for w in weights:
            # 各層の形状に合わせたガウスノイズを生成
            noise_stddev = self.clip_norm * self.noise_multiplier
            noise = np.random.normal(0, noise_stddev, w.shape)

            # ノイズを追加
            noised_w = w + noise
            noised_weights.append(noised_w)

        logger.debug(f"ノイズ追加: 標準偏差={noise_stddev}")
        return noised_weights

    def apply(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """差分プライバシーを適用する

        Args:
            weights: 元の重み

        Returns:
            プライバシー保護された重み
        """
        # 重みをクリッピング
        clipped_weights = self._clip_weights(weights)

        # ノイズを追加
        noised_weights = self._add_noise(clipped_weights)

        logger.info("差分プライバシーを適用しました")
        return noised_weights

    def get_privacy_spent(self, num_iterations: int, batch_rate: float) -> Dict[str, float]:
        """消費されたプライバシー予算を計算する

        Args:
            num_iterations: 繰り返し回数
            batch_rate: バッチ率 (バッチサイズ / データサイズ)

        Returns:
            プライバシー予算
        """
        # RDPアカウンタント計算の単純化バージョン
        effective_noise = self.noise_multiplier / math.sqrt(batch_rate)
        effective_epsilon = self.epsilon * math.sqrt(num_iterations * batch_rate)

        return {
            "epsilon": effective_epsilon,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "iterations": num_iterations
        }

    def get_flower_dp_config(self) -> Dict[str, Any]:
        """Flower用のDP設定を取得する

        Returns:
            Flower用DP設定
        """
        config = {
            "dp_strategy": "DPFedAvg",
            "noise_multiplier": self.noise_multiplier,
            "clip_norm": self.clip_norm,
            "target_epsilon": self.epsilon,
            "target_delta": self.delta
        }

        logger.info(f"Flower DP設定: {config}")
        return config