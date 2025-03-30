"""
モデルインターフェース

このモジュールは、連合学習で使用するモデルのインターフェースを定義します。
すべてのモデルはこのインターフェースを実装する必要があります。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union
import numpy as np

class ModelInterface(ABC):
    """モデルインターフェース

    連合学習で使用するモデルの共通インターフェース。
    すべてのモデルはこのクラスを継承して実装する必要があります。
    """

    @abstractmethod
    def build(self, input_dim: int, output_dim: int) -> None:
        """モデルを構築する

        Args:
            input_dim: 入力次元
            output_dim: 出力次元
        """
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 32) -> Dict[str, float]:
        """モデルを訓練する

        Args:
            X: 入力データ
            y: 目標データ
            epochs: エポック数
            batch_size: バッチサイズ

        Returns:
            訓練メトリクス
        """
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """モデルを評価する

        Args:
            X: 入力データ
            y: 目標データ

        Returns:
            評価メトリクス
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測を行う

        Args:
            X: 入力データ

        Returns:
            予測結果
        """
        pass

    @abstractmethod
    def get_weights(self) -> List[np.ndarray]:
        """モデルの重みを取得する

        Returns:
            モデルの重みリスト
        """
        pass

    @abstractmethod
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """モデルの重みを設定する

        Args:
            weights: モデルの重みリスト
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """モデルのメトリクスを取得する

        Returns:
            メトリクス
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """モデルを保存する

        Args:
            path: 保存先パス
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """モデルを読み込む

        Args:
            path: 読み込み元パス
        """
        pass