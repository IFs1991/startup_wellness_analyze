"""
適応的勾配クリッピング実装

このモジュールは、差分プライバシーにおける適応的勾配クリッピングを実装します。
勾配の分布に基づいて動的にクリッピング閾値を調整し、プライバシーと学習効率のバランスを最適化します。

TDD Phase: GREEN - テストを通す実装
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import logging
from collections import deque

logger = logging.getLogger(__name__)

class AdaptiveClipping:
    """適応的勾配クリッピング

    差分プライバシーにおける適応的勾配クリッピングを実装します。
    勾配の分布を動的に推定し、最適なクリッピング閾値を学習します。
    """

    def __init__(self,
                 initial_clipping_norm: float = 1.0,
                 noise_multiplier: float = 1.1,
                 target_delta: float = 1e-5,
                 learning_rate: float = 0.1,
                 adaptive: bool = True,
                 device: str = 'cpu',
                 window_size: int = 100,
                 privacy_mode: bool = False,
                 target_quantile: float = 0.5):
        """初期化

        Args:
            initial_clipping_norm: 初期クリッピング閾値
            noise_multiplier: ノイズ乗数
            target_delta: 目標δ値
            learning_rate: クリッピング閾値の学習率
            adaptive: 適応的クリッピングを有効にするか
            device: 計算デバイス ('cpu' or 'cuda')
            window_size: 勾配統計の移動平均ウィンドウサイズ
            privacy_mode: プライバシー保護モード（より保守的な適応）
            target_quantile: 目標分位点（適応的クリッピング用）

        Raises:
            ValueError: パラメータが無効な場合
        """
        # パラメータ検証
        if initial_clipping_norm <= 0:
            raise ValueError(f"クリッピング閾値は正の値である必要があります: {initial_clipping_norm}")
        if noise_multiplier <= 0:
            raise ValueError(f"ノイズ乗数は正の値である必要があります: {noise_multiplier}")
        if not (0 < target_delta < 1):
            raise ValueError(f"δは0 < δ < 1である必要があります: {target_delta}")
        if learning_rate < 0:
            raise ValueError(f"学習率は非負である必要があります: {learning_rate}")
        if not (0 < target_quantile < 1):
            raise ValueError(f"目標分位点は0 < quantile < 1である必要があります: {target_quantile}")

        self.initial_clipping_norm = initial_clipping_norm
        self.current_clipping_norm = initial_clipping_norm
        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta
        self.learning_rate = learning_rate
        self.adaptive = adaptive
        self.device = device
        self.window_size = window_size
        self.privacy_mode = privacy_mode
        self.target_quantile = target_quantile

        # 勾配統計の追跡
        self.gradient_norms_history = deque(maxlen=window_size)
        self.clipping_norms_history = deque(maxlen=window_size)
        self.step_count = 0

        # 適応的パラメータ
        self.momentum_beta = 0.9  # モメンタム係数
        self.gradient_norm_ema = None  # 勾配ノルムの指数移動平均
        self.clipping_rate_target = 0.5  # 目標クリッピング率

        # プライバシーモード用パラメータ
        if self.privacy_mode:
            # より保守的な設定
            self.learning_rate *= 0.5  # 学習率を半分に
            self.clipping_rate_target = 0.3  # より低い目標クリッピング率
            self.momentum_beta = 0.95  # より高いモメンタム

        logger.info(f"AdaptiveClipping初期化完了: norm={initial_clipping_norm}, σ={noise_multiplier}, adaptive={adaptive}, privacy_mode={privacy_mode}")

    def estimate_gradient_norm(self, gradients: List[torch.Tensor]) -> float:
        """勾配の総ノルムを推定

        Args:
            gradients: 勾配テンサーのリスト

        Returns:
            総勾配ノルム
        """
        if not gradients:
            return 0.0

        # 各勾配テンサーのノルムを計算
        squared_norms = []
        for grad in gradients:
            if grad is not None:
                # テンサーを適切なデバイスに移動
                if self.device == 'cuda' and torch.cuda.is_available():
                    grad = grad.cuda()
                else:
                    grad = grad.cpu()

                # L2ノルムの2乗を計算
                squared_norm = torch.sum(grad ** 2).item()
                squared_norms.append(squared_norm)

        # 総ノルムを計算
        total_squared_norm = sum(squared_norms)
        total_norm = math.sqrt(total_squared_norm) if total_squared_norm > 0 else 0.0

        logger.debug(f"勾配ノルム推定: {total_norm}")
        return total_norm

    def clip_gradients(self, gradients: Union[List[torch.Tensor], np.ndarray], clipping_norm: Optional[float] = None) -> Tuple[Union[List[torch.Tensor], np.ndarray], float]:
        """勾配をクリッピング

        Args:
            gradients: クリッピング対象の勾配（リストまたはnumpy配列）
            clipping_norm: 使用するクリッピング閾値（Noneの場合は現在の閾値を使用）

        Returns:
            (クリッピング後の勾配, 使用されたクリッピング閾値)
        """
        if gradients is None:
            return gradients, self.current_clipping_norm

        # クリッピング閾値の決定
        effective_clipping_norm = clipping_norm if clipping_norm is not None else self.current_clipping_norm

        # numpy配列の場合の処理
        if isinstance(gradients, np.ndarray):
            gradient_norm = np.linalg.norm(gradients)

            # 適応的クリッピング閾値の更新
            if self.adaptive and self.step_count > 0:
                self._update_clipping_norm(gradient_norm)
                effective_clipping_norm = self.current_clipping_norm

            # クリッピングの実行
            clipping_factor = min(1.0, effective_clipping_norm / max(gradient_norm, 1e-8))
            clipped_gradients = gradients * clipping_factor

            # 統計の更新
            self.gradient_norms_history.append(gradient_norm)
            self.clipping_norms_history.append(effective_clipping_norm)
            self.step_count += 1

            logger.debug(f"勾配クリッピング完了(numpy): norm={gradient_norm:.4f}, threshold={effective_clipping_norm:.4f}, factor={clipping_factor:.4f}")
            return clipped_gradients, effective_clipping_norm

        # Tensorリストの場合の処理（既存のロジック）
        if not gradients:
            return [], effective_clipping_norm

        # 勾配ノルムを計算
        gradient_norm = self.estimate_gradient_norm(gradients)

        # 適応的クリッピング閾値の更新
        if self.adaptive and self.step_count > 0:
            self._update_clipping_norm(gradient_norm)
            effective_clipping_norm = self.current_clipping_norm

        # クリッピングの実行
        clipped_gradients = []
        clipping_factor = min(1.0, effective_clipping_norm / max(gradient_norm, 1e-8))

        for grad in gradients:
            if grad is not None:
                # テンサーを適切なデバイスに移動
                if self.device == 'cuda' and torch.cuda.is_available():
                    grad = grad.cuda()
                else:
                    grad = grad.cpu()

                # クリッピングを適用
                clipped_grad = grad * clipping_factor
                clipped_gradients.append(clipped_grad)
            else:
                clipped_gradients.append(grad)

        # 統計の更新
        self.gradient_norms_history.append(gradient_norm)
        self.clipping_norms_history.append(effective_clipping_norm)
        self.step_count += 1

        logger.debug(f"勾配クリッピング完了: norm={gradient_norm:.4f}, threshold={effective_clipping_norm:.4f}, factor={clipping_factor:.4f}")

        return clipped_gradients, effective_clipping_norm

    def _update_clipping_norm(self, gradient_norm: float):
        """適応的クリッピング閾値の更新

        Args:
            gradient_norm: 現在の勾配ノルム
        """
        # 勾配ノルムの指数移動平均を更新
        if self.gradient_norm_ema is None:
            self.gradient_norm_ema = gradient_norm
        else:
            self.gradient_norm_ema = (
                self.momentum_beta * self.gradient_norm_ema +
                (1 - self.momentum_beta) * gradient_norm
            )

        # 現在のクリッピング率を計算
        current_clipping_rate = min(1.0, self.current_clipping_norm / max(gradient_norm, 1e-8))

        # 目標クリッピング率との差
        clipping_rate_error = current_clipping_rate - self.clipping_rate_target

        # 適応的更新（より安定化された更新ルール）
        # 学習率を動的に調整して収束を促進
        adaptive_lr = self.learning_rate * (1.0 / (1.0 + 0.01 * self.step_count))

        # クリッピング閾値の更新方向を決定
        # 勾配ノルムに基づく適応的な更新
        if abs(clipping_rate_error) > 0.1:  # 大きな誤差の場合
            # より積極的な調整
            if clipping_rate_error > 0:  # クリッピング率が目標より高い
                # 閾値を増加（クリッピングを少なくする）
                norm_update = adaptive_lr * gradient_norm * 0.1
            else:  # クリッピング率が目標より低い
                # 閾値を減少（クリッピングを多くする）
                norm_update = -adaptive_lr * gradient_norm * 0.1
        else:
            # 小さな誤差の場合は微調整
            norm_update = -adaptive_lr * clipping_rate_error * self.gradient_norm_ema * 0.1

        # クリッピング閾値の更新（正の値を保証し、変化量を制限）
        max_change = 0.2 * self.current_clipping_norm  # 変化量を現在値の20%に制限
        norm_update = max(-max_change, min(max_change, norm_update))

        self.current_clipping_norm = max(0.01, self.current_clipping_norm + norm_update)

        logger.debug(f"クリッピング閾値更新: {self.current_clipping_norm:.4f} (変化量: {norm_update:.4f})")

    def update_clipping_norm(self, gradient_norm: float):
        """適応的クリッピング閾値の更新（外部から呼び出し可能）

        Args:
            gradient_norm: 現在の勾配ノルム
        """
        self._update_clipping_norm(gradient_norm)

    def compute_clipping_loss(self,
                            original_gradients: List[torch.Tensor],
                            clipped_gradients: List[torch.Tensor]) -> float:
        """クリッピングによる損失を計算

        Args:
            original_gradients: 元の勾配
            clipped_gradients: クリッピング後の勾配

        Returns:
            クリッピング損失（L2距離）
        """
        if len(original_gradients) != len(clipped_gradients):
            raise ValueError("勾配リストの長さが一致しません")

        total_loss = 0.0
        total_elements = 0

        for orig, clipped in zip(original_gradients, clipped_gradients):
            if orig is not None and clipped is not None:
                # L2距離の2乗を計算
                diff_squared = torch.sum((orig - clipped) ** 2).item()
                total_loss += diff_squared
                total_elements += orig.numel()

        # 正規化された損失を返す
        if total_elements > 0:
            return total_loss / total_elements
        else:
            return 0.0

    def get_current_clipping_norm(self) -> float:
        """現在のクリッピング閾値を取得

        Returns:
            現在のクリッピング閾値
        """
        return self.current_clipping_norm

    def get_noise_multiplier(self) -> float:
        """ノイズ乗数を取得

        Returns:
            ノイズ乗数
        """
        return self.noise_multiplier

    def get_statistics(self) -> Dict[str, Any]:
        """クリッピング統計を取得

        Returns:
            統計情報の辞書
        """
        if not self.gradient_norms_history:
            return {
                'current_clipping_norm': self.current_clipping_norm,
                'avg_gradient_norm': 0.0,
                'clipping_rate': 0.0,
                'step_count': self.step_count
            }

        recent_gradient_norms = list(self.gradient_norms_history)
        recent_clipping_norms = list(self.clipping_norms_history)

        avg_gradient_norm = np.mean(recent_gradient_norms)
        avg_clipping_norm = np.mean(recent_clipping_norms)

        # クリッピング率の計算（平均的にどの程度クリッピングされているか）
        clipping_rates = [
            min(1.0, cn / max(gn, 1e-8))
            for gn, cn in zip(recent_gradient_norms, recent_clipping_norms)
        ]
        avg_clipping_rate = np.mean(clipping_rates)

        return {
            'current_clipping_norm': self.current_clipping_norm,
            'avg_gradient_norm': avg_gradient_norm,
            'avg_clipping_norm': avg_clipping_norm,
            'clipping_rate': avg_clipping_rate,
            'gradient_norm_std': np.std(recent_gradient_norms),
            'clipping_norm_std': np.std(recent_clipping_norms),
            'step_count': self.step_count,
            'gradient_norm_ema': self.gradient_norm_ema
        }

    def reset(self):
        """統計とパラメータをリセット"""
        self.current_clipping_norm = self.initial_clipping_norm
        self.gradient_norms_history.clear()
        self.clipping_norms_history.clear()
        self.step_count = 0
        self.gradient_norm_ema = None

        logger.info("AdaptiveClipping統計をリセットしました")

    def save_state(self) -> Dict[str, Any]:
        """現在の状態を保存

        Returns:
            状態辞書
        """
        return {
            'current_clipping_norm': self.current_clipping_norm,
            'gradient_norms_history': list(self.gradient_norms_history),
            'clipping_norms_history': list(self.clipping_norms_history),
            'step_count': self.step_count,
            'gradient_norm_ema': self.gradient_norm_ema
        }

    def load_state(self, state: Dict[str, Any]):
        """状態を復元

        Args:
            state: 保存された状態辞書
        """
        self.current_clipping_norm = state['current_clipping_norm']
        self.gradient_norms_history = deque(state['gradient_norms_history'], maxlen=self.window_size)
        self.clipping_norms_history = deque(state['clipping_norms_history'], maxlen=self.window_size)
        self.step_count = state['step_count']
        self.gradient_norm_ema = state['gradient_norm_ema']

        logger.info(f"AdaptiveClipping状態を復元しました: step={self.step_count}")

    def configure_target_clipping_rate(self, rate: float):
        """目標クリッピング率を設定

        Args:
            rate: 目標クリッピング率 (0.0 - 1.0)

        Raises:
            ValueError: 無効な率が指定された場合
        """
        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"クリッピング率は0.0-1.0の範囲である必要があります: {rate}")

        self.clipping_rate_target = rate
        logger.info(f"目標クリッピング率を{rate}に設定しました")

    def estimate_privacy_cost(self, steps: int, sample_rate: float) -> Tuple[float, float]:
        """プライバシーコストを推定

        Args:
            steps: 学習ステップ数
            sample_rate: サンプリング率

        Returns:
            (推定ε値, 使用δ値)
        """
        # RDPアカウンタントを使用してプライバシーコストを推定
        try:
            from .rdp_accountant import RDPAccountant

            accountant = RDPAccountant()
            rdp_values = accountant.compute_rdp(
                q=sample_rate,
                noise_multiplier=self.noise_multiplier,
                steps=steps,
                orders=accountant.orders
            )

            epsilon = accountant.get_privacy_spent(
                orders=accountant.orders,
                rdp=rdp_values,
                target_delta=self.target_delta
            )

            return epsilon, self.target_delta

        except ImportError:
            logger.warning("RDPAccountantが利用できません。近似計算を使用します")
            # 簡単な近似（ガウス機構の基本公式）
            epsilon_approx = steps * sample_rate * sample_rate / (2 * self.noise_multiplier ** 2)
            return epsilon_approx, self.target_delta