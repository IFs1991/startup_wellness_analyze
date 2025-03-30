"""
連合学習クライアント実装

このモジュールは、スタートアップ分析プラットフォームの連合学習クライアントを実装します。
クライアントはローカルデータでモデルを訓練し、更新をサーバーに送信します。
Flowerフレームワークを使用して実装されています。
"""

import os
import logging
import numpy as np
from pathlib import Path
import yaml
import requests
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

# Flowerのインポート
import flwr as fl
from flwr.common import Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes
from flwr.client import NumPyClient

from ..security.differential_privacy import DifferentialPrivacy
from ..models.model_interface import ModelInterface

logger = logging.getLogger(__name__)

class FederatedClient(NumPyClient):
    """連合学習クライアント

    連合学習のクライアント実装。ローカルデータでモデルを訓練し、更新を
    中央サーバーに送信します。Flowerのインターフェースを実装しています。
    """

    def __init__(self, client_id: str, model: ModelInterface, config_path: Optional[str] = None):
        """初期化

        Args:
            client_id: クライアントID
            model: モデルインスタンス
            config_path: 設定ファイルパス（省略時はデフォルト設定を使用）
        """
        self.client_id = client_id
        self.config = self._load_config(config_path)
        self.model = model
        self.current_round = 0
        self.dp = DifferentialPrivacy(
            epsilon=self.config['federated_learning']['differential_privacy']['client_level']['noise_multiplier'],
            delta=self.config['federated_learning']['differential_privacy']['client_level']['target_delta'],
            clip_norm=self.config['federated_learning']['differential_privacy']['client_level']['l2_norm_clip']
        )
        self.server_url = os.environ.get('FL_SERVER_URL', 'http://localhost:8080')
        self.train_data = None
        self.val_data = None

        logger.info(f"連合学習クライアント '{client_id}' を初期化しました")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """設定ファイルを読み込む

        Args:
            config_path: 設定ファイルパス

        Returns:
            設定辞書
        """
        if config_path is None:
            config_path = Path(__file__).parents[1] / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def set_train_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """トレーニングデータを設定する

        Args:
            X: 特徴量データ
            y: ターゲットデータ
        """
        self.train_data = (X, y)
        logger.info(f"トレーニングデータを設定しました: {X.shape}, {y.shape}")

    def set_val_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """検証データを設定する

        Args:
            X: 特徴量データ
            y: ターゲットデータ
        """
        self.val_data = (X, y)
        logger.info(f"検証データを設定しました: {X.shape}, {y.shape}")

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """現在のモデルパラメーターを取得する

        Args:
            config: 設定

        Returns:
            モデルパラメーター
        """
        return self.model.get_weights()

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """モデルをトレーニングする

        Args:
            parameters: モデルパラメーター
            config: トレーニング設定

        Returns:
            更新されたパラメーター、データサンプル数、メトリクス
        """
        if self.train_data is None:
            logger.warning("トレーニングデータが設定されていません")
            return parameters, 0, {}

        X, y = self.train_data
        self.current_round = config.get("epoch_global", 0)

        # パラメーターをモデルに設定
        self.model.set_weights(parameters)

        # トレーニング設定を取得
        epochs = config.get("epochs", 1)
        batch_size = config.get("batch_size", 32)

        # トレーニング実行
        metrics = self.model.train(X, y, epochs=epochs, batch_size=batch_size)

        # 新しいパラメーターを取得
        updated_parameters = self.model.get_weights()

        # 差分プライバシーを適用
        if self.config['federated_learning']['differential_privacy']['enabled']:
            updated_parameters = self.dp.apply(updated_parameters)

        logger.info(f"ラウンド {self.current_round} のトレーニング完了: {metrics}")

        return updated_parameters, len(X), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """モデルを評価する

        Args:
            parameters: モデルパラメーター
            config: 評価設定

        Returns:
            損失値、データサンプル数、メトリクス
        """
        if self.val_data is None:
            logger.warning("検証データが設定されていません")
            return 0.0, 0, {}

        X, y = self.val_data

        # パラメーターをモデルに設定
        self.model.set_weights(parameters)

        # 評価を実行
        metrics = self.model.evaluate(X, y)
        loss = metrics.get("loss", 0.0)

        logger.info(f"モデル評価完了: {metrics}")

        return loss, len(X), metrics

    def register_with_server(self) -> bool:
        """サーバーにクライアントを登録する

        Returns:
            成功したかどうか
        """
        try:
            url = f"{self.server_url}/api/v1/federated/clients/register"
            data = {
                "client_id": self.client_id,
                "client_name": f"Client-{self.client_id}",
                "industry_type": "unknown",  # 実際のユースケースに合わせて設定
                "data_size": len(self.train_data[0]) if self.train_data is not None else 0
            }

            response = requests.post(url, json=data)

            if response.status_code == 200:
                logger.info("サーバーへの登録が完了しました")
                return True
            else:
                logger.error(f"サーバー登録エラー: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.exception(f"サーバー登録中に例外が発生しました: {e}")
            return False

    def start_client(self, server_address: Optional[str] = None) -> None:
        """Flowerクライアントを起動する

        Args:
            server_address: サーバーアドレス (None の場合は環境変数から取得)
        """
        if server_address is None:
            server_address = self.server_url

        # サーバーに登録
        self.register_with_server()

        # クライアント設定
        fl_client = fl.client.start_numpy_client(
            server_address=server_address,
            client=self
        )

        logger.info(f"Flowerクライアントを起動しました: {server_address}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測を行う

        Args:
            X: 入力データ

        Returns:
            予測結果
        """
        return self.model.predict(X)