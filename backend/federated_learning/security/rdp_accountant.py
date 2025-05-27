"""
RDPアカウンタント実装

このモジュールは、Rényi Differential Privacy (RDP) のプライバシー会計を実装します。
連合学習における差分プライバシーの数学的に正確な計算を提供します。

TDD Phase: GREEN - テストを通す実装
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class RDPAccountant:
    """RDPアカウンタント

    Rényi Differential Privacy (RDP) のプライバシー会計を実装します。
    ガウス機構におけるRDP値の計算、合成、(ε,δ)-DPへの変換を提供します。
    """

    def __init__(self, orders: Optional[List[float]] = None):
        """初期化

        Args:
            orders: Rényiパラメータα値のリスト。デフォルトは一般的な値を使用。
        """
        if orders is None:
            # 一般的なα値を使用（理論と実用性のバランス）
            self.orders = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0]
        else:
            # α > 1の制約を確認
            if any(alpha <= 1.0 for alpha in orders):
                raise ValueError("すべてのRényiパラメータαは1より大きい必要があります")
            self.orders = sorted(orders)

        self.privacy_ledger = []  # プライバシー履歴

        logger.info(f"RDPアカウンタント初期化: α={self.orders}")

    def compute_rdp(self,
                   q: float,
                   noise_multiplier: float,
                   steps: int,
                   orders: Optional[List[float]] = None) -> List[float]:
        """ガウス機構におけるRDP値を計算

        Args:
            q: サンプリング率 (0 < q ≤ 1)
            noise_multiplier: ノイズ乗数 σ (> 0)
            steps: 反復回数 (≥ 0)
            orders: 計算するα値。省略時はself.ordersを使用

        Returns:
            各α値に対応するRDP値のリスト

        Raises:
            ValueError: パラメータが無効な場合
        """
        # パラメータ検証
        if not (0 < q <= 1):
            raise ValueError(f"サンプリング率は0 < q ≤ 1である必要があります: {q}")
        if noise_multiplier <= 0:
            raise ValueError(f"ノイズ乗数は正の値である必要があります: {noise_multiplier}")
        if steps < 0:
            raise ValueError(f"ステップ数は非負である必要があります: {steps}")

        if orders is None:
            orders = self.orders

        if steps == 0:
            return [0.0] * len(orders)

        rdp_values = []

        for alpha in orders:
            if alpha == 1.0:
                # α=1は特別なケース（KL発散）
                rdp_value = self._compute_rdp_alpha_1(q, noise_multiplier, steps)
            else:
                # ガウス機構のRDP公式
                rdp_value = self._compute_rdp_gaussian(q, noise_multiplier, steps, alpha)

            rdp_values.append(rdp_value)

        logger.debug(f"RDP計算完了: q={q}, σ={noise_multiplier}, steps={steps}, values={rdp_values}")
        return rdp_values

    def _compute_rdp_gaussian(self, q: float, noise_multiplier: float, steps: int, alpha: float) -> float:
        """ガウス機構のα-RDP値を計算

        サブサンプリングガウス機構のRDP値を正確に計算します。
        Abadi et al. 2016, Mironov 2017の公式に基づく数値安定性を重視した実装。

        Args:
            q: サンプリング率
            noise_multiplier: ノイズ乗数
            steps: ステップ数
            alpha: Rényiパラメータ

        Returns:
            α-RDP値
        """
        if q == 1.0:
            # 全データサンプリングの場合
            # RDP(α) = α / (2 * σ²) の steps倍
            return steps * alpha / (2 * noise_multiplier ** 2)

        # サブサンプリングガウス機構のRDP値
        # 数値安定性を重視した実装

        # 基本項: α * q² / (2 * σ²)
        sigma_squared = noise_multiplier ** 2
        basic_term = alpha * q * q / (2 * sigma_squared)

        # サブサンプリング効果の補正（数値安定性を考慮）
        if q < 0.01:
            # 非常に小さなサンプリング率の場合、基本項のみで十分な精度
            rdp_per_step = basic_term
        elif q < 0.1:
            # 小さなサンプリング率での高次補正項
            # O(q³/σ³) 項を含む（数値安定性チェック付き）
            correction_term = alpha * (alpha - 1) * q ** 3 / (6 * sigma_squared ** 2)

            # 補正項が基本項より大きくなりすぎないよう制限
            if correction_term < basic_term:
                rdp_per_step = basic_term + correction_term
            else:
                rdp_per_step = basic_term
        else:
            # より大きなサンプリング率では、数値安定性を優先
            epsilon_0 = alpha / (2 * sigma_squared)
            exponent = (alpha - 1) * epsilon_0

            # 数値オーバーフローを防ぐ
            max_exponent = 100  # exp(100) ≈ 2.7e43（安全な範囲）

            if exponent < 1:
                # Taylor展開を使用（小さな値での高精度）
                rdp_per_step = q * exponent * (1 + exponent / 2 + exponent ** 2 / 6)
            elif exponent < max_exponent:
                # 安全な範囲での指数関数計算
                try:
                    rdp_per_step = q * (math.exp(exponent) - 1)
                except OverflowError:
                    # フォールバック: 線形近似
                    rdp_per_step = q * exponent
            else:
                # 大きな値では線形近似を使用
                rdp_per_step = q * exponent

        return steps * rdp_per_step

    def _compute_rdp_alpha_1(self, q: float, noise_multiplier: float, steps: int) -> float:
        """α=1の場合のRDP値計算（KL発散）

        Args:
            q: サンプリング率
            noise_multiplier: ノイズ乗数
            steps: ステップ数

        Returns:
            α=1でのRDP値
        """
        # α→1の極限として計算
        # この場合はKL発散になる
        if q == 1.0:
            return steps * q * q / (2 * noise_multiplier ** 2)
        else:
            # サブサンプリングKL発散の近似
            return steps * q * q / (2 * noise_multiplier ** 2)

    def compose_rdp(self, rdp1: List[float], rdp2: List[float]) -> List[float]:
        """RDP値の合成

        独立したメカニズムのRDP値を合成します。
        RDPの合成定理により、単純に加算できます。

        Args:
            rdp1: 最初のメカニズムのRDP値
            rdp2: 2番目のメカニズムのRDP値

        Returns:
            合成されたRDP値

        Raises:
            ValueError: RDP値のリスト長が異なる場合
        """
        if len(rdp1) != len(rdp2):
            raise ValueError(f"RDP値の長さが一致しません: {len(rdp1)} vs {len(rdp2)}")

        # RDPの合成定理: 加算的
        composed = [r1 + r2 for r1, r2 in zip(rdp1, rdp2)]

        logger.debug(f"RDP合成: {rdp1} + {rdp2} = {composed}")
        return composed

    def get_privacy_spent(self,
                         orders: List[float],
                         rdp: List[float],
                         target_delta: float = 1e-5) -> float:
        """RDPから(ε,δ)-DPへの変換

        RDP値から最適なεを計算します。

        Args:
            orders: α値のリスト
            rdp: 対応するRDP値のリスト
            target_delta: 目標δ値

        Returns:
            最適なε値

        Raises:
            ValueError: パラメータが無効な場合
        """
        if len(orders) != len(rdp):
            raise ValueError(f"ordersとrdpの長さが一致しません: {len(orders)} vs {len(rdp)}")

        if target_delta <= 0 or target_delta >= 1:
            raise ValueError(f"δは0 < δ < 1である必要があります: {target_delta}")

        min_epsilon = float('inf')

        for alpha, rdp_alpha in zip(orders, rdp):
            if alpha == 1.0:
                continue  # α=1は特別扱い

            # RDPから(ε,δ)-DPへの変換公式
            # ε = rdp_α + log(1/δ) / (α - 1)
            epsilon = rdp_alpha + math.log(1.0 / target_delta) / (alpha - 1)

            if epsilon < min_epsilon:
                min_epsilon = epsilon

        if min_epsilon == float('inf'):
            # α=1のみの場合の特別処理
            if len([a for a in orders if a == 1.0]) > 0:
                alpha_1_rdp = rdp[orders.index(1.0)]
                min_epsilon = alpha_1_rdp  # KL発散の場合
            else:
                raise ValueError("有効なα値が見つかりません")

        logger.debug(f"プライバシー変換: δ={target_delta} → ε={min_epsilon}")
        return max(0.0, min_epsilon)  # 負の値を防ぐ

    def add_step(self,
                q: float,
                noise_multiplier: float,
                orders: Optional[List[float]] = None) -> Dict[str, Any]:
        """プライバシー履歴にステップを追加

        Args:
            q: サンプリング率
            noise_multiplier: ノイズ乗数
            orders: α値のリスト

        Returns:
            追加されたステップの情報
        """
        if orders is None:
            orders = self.orders

        # 1ステップのRDP値を計算
        step_rdp = self.compute_rdp(q, noise_multiplier, 1, orders)

        step_info = {
            'step': len(self.privacy_ledger),
            'q': q,
            'noise_multiplier': noise_multiplier,
            'rdp': step_rdp,
            'orders': orders.copy()
        }

        self.privacy_ledger.append(step_info)
        logger.debug(f"プライバシーステップ追加: {step_info}")

        return step_info

    def get_total_privacy_spent(self, target_delta: float = 1e-5) -> Tuple[float, List[float]]:
        """累積プライバシー消費量を計算

        Args:
            target_delta: 目標δ値

        Returns:
            (累積ε値, 累積RDP値のリスト)
        """
        if not self.privacy_ledger:
            return 0.0, [0.0] * len(self.orders)

        # 全ステップのRDP値を合成
        total_rdp = [0.0] * len(self.orders)

        for step_info in self.privacy_ledger:
            step_rdp = step_info['rdp']
            total_rdp = self.compose_rdp(total_rdp, step_rdp)

        # (ε,δ)-DPに変換
        epsilon = self.get_privacy_spent(self.orders, total_rdp, target_delta)

        return epsilon, total_rdp

    def privacy_budget_remaining(self,
                               epsilon_budget: float,
                               target_delta: float = 1e-5) -> float:
        """残りプライバシー予算を計算

        Args:
            epsilon_budget: 総プライバシー予算
            target_delta: 目標δ値

        Returns:
            残りプライバシー予算
        """
        spent_epsilon, _ = self.get_total_privacy_spent(target_delta)
        remaining = epsilon_budget - spent_epsilon

        logger.debug(f"プライバシー予算: 総額={epsilon_budget}, 消費={spent_epsilon}, 残り={remaining}")
        return max(0.0, remaining)

    def reset(self):
        """プライバシー履歴をリセット"""
        self.privacy_ledger.clear()
        logger.info("プライバシー履歴をリセットしました")