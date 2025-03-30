"""
金融パフォーマンス予測モデル

このモジュールは、スタートアップの金融パフォーマンスを予測するためのベイジアンニューラルネットワークを実装します。
複数のフレームワーク(TensorFlow, PyTorch)をサポートします。
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import importlib
from pathlib import Path

from .model_interface import ModelInterface

logger = logging.getLogger(__name__)

# 動的インポート関数
def _import_if_available(module_name: str) -> Optional[Any]:
    """モジュールが利用可能な場合にインポートする

    Args:
        module_name: モジュール名

    Returns:
        モジュールまたはNone
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None

# 利用可能なフレームワークを確認
tf = _import_if_available("tensorflow")
tfp = _import_if_available("tensorflow_probability")
torch = _import_if_available("torch")
pyro = _import_if_available("pyro")

# TensorFlowが利用可能な場合
if tf is not None and tfp is not None:
    tfd = tfp.distributions
    tfpl = tfp.layers
    tfk = tf.keras
    tfkl = tf.keras.layers

# モデルファクトリークラス
class ModelFactory:
    """モデルファクトリー

    適切なフレームワーク実装を返すファクトリークラス
    """

    @staticmethod
    def create_model(framework: str = "auto", **kwargs) -> "FinancialPerformancePredictor":
        """モデルを作成する

        Args:
            framework: フレームワーク名 ("tensorflow", "pytorch", "auto")
            **kwargs: モデルパラメータ

        Returns:
            モデルインスタンス

        Raises:
            ValueError: サポートされていないフレームワーク、または指定されたフレームワークが利用できない場合
        """
        # 自動検出の場合
        if framework == "auto":
            if tf is not None and tfp is not None:
                framework = "tensorflow"
                logger.info("TensorFlowベースのモデルを自動選択しました")
            elif torch is not None and pyro is not None:
                framework = "pytorch"
                logger.info("PyTorchベースのモデルを自動選択しました")
            else:
                raise ValueError("サポートされるフレームワークが見つかりません")

        # 指定されたフレームワークに基づいてモデルを作成
        if framework == "tensorflow":
            if tf is None or tfp is None:
                raise ValueError("TensorFlowまたはTensorFlow Probabilityがインストールされていません")
            return TensorFlowFinancialPredictor(**kwargs)
        elif framework == "pytorch":
            if torch is None or pyro is None:
                raise ValueError("PyTorchまたはPyroがインストールされていません")
            return PyTorchFinancialPredictor(**kwargs)
        else:
            raise ValueError(f"サポートされていないフレームワーク: {framework}")


# ベースクラス
class FinancialPerformancePredictor(ModelInterface):
    """金融パフォーマンス予測モデル

    スタートアップの金融パフォーマンスを予測するためのベイジアンニューラルネットワーク。
    不確実性を考慮した予測を提供します。

    注意: これは抽象基底クラスです。具体的な実装はサブクラスを使用してください。
    """

    def __init__(self, hidden_layers: List[int] = [64, 32, 16],
                 activation: str = "relu",
                 final_activation: str = "linear",
                 kl_weight: float = 1e-3):
        """初期化

        Args:
            hidden_layers: 隠れ層のユニット数リスト
            activation: 活性化関数
            final_activation: 出力層の活性化関数
            kl_weight: KLダイバージェンスの重み
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.final_activation = final_activation
        self.kl_weight = kl_weight
        self.model = None
        self.metrics = {}
        logger.info(f"金融パフォーマンス予測モデルを初期化: {hidden_layers}, {activation}, {final_activation}")

    def build(self, input_dim: int, output_dim: int) -> None:
        """モデルを構築する (サブクラスで実装)

        Args:
            input_dim: 入力次元
            output_dim: 出力次元
        """
        raise NotImplementedError("サブクラスで実装する必要があります")

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 32) -> Dict[str, float]:
        """モデルを訓練する (サブクラスで実装)

        Args:
            X: 入力データ
            y: 目標データ
            epochs: エポック数
            batch_size: バッチサイズ

        Returns:
            訓練メトリクス
        """
        raise NotImplementedError("サブクラスで実装する必要があります")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """モデルを評価する (サブクラスで実装)

        Args:
            X: 入力データ
            y: 目標データ

        Returns:
            評価メトリクス
        """
        raise NotImplementedError("サブクラスで実装する必要があります")

    def predict(self, X: np.ndarray, samples: int = 100) -> np.ndarray:
        """予測を行う (サブクラスで実装)

        Args:
            X: 入力データ
            samples: サンプル数

        Returns:
            予測結果
        """
        raise NotImplementedError("サブクラスで実装する必要があります")

    def get_weights(self) -> List[np.ndarray]:
        """モデルの重みを取得する (サブクラスで実装)

        Returns:
            モデルの重みリスト
        """
        raise NotImplementedError("サブクラスで実装する必要があります")

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """モデルの重みを設定する (サブクラスで実装)

        Args:
            weights: モデルの重みリスト
        """
        raise NotImplementedError("サブクラスで実装する必要があります")

    def get_metrics(self) -> Dict[str, float]:
        """モデルのメトリクスを取得する

        Returns:
            メトリクス
        """
        return self.metrics

    def save(self, path: str) -> None:
        """モデルを保存する (サブクラスで実装)

        Args:
            path: 保存先パス
        """
        raise NotImplementedError("サブクラスで実装する必要があります")

    def load(self, path: str) -> None:
        """モデルを読み込む (サブクラスで実装)

        Args:
            path: 読み込み元パス
        """
        raise NotImplementedError("サブクラスで実装する必要があります")


# TensorFlow実装
class TensorFlowFinancialPredictor(FinancialPerformancePredictor):
    """TensorFlowベースの金融パフォーマンス予測モデル"""

    def build(self, input_dim: int, output_dim: int) -> None:
        """モデルを構築する

        Args:
            input_dim: 入力次元
            output_dim: 出力次元
        """
        if tf is None or tfp is None:
            raise ImportError("TensorFlowまたはTensorFlow Probabilityが必要です")

        # 事後分布を定義するヘルパー関数
        def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            c = np.log(np.expm1(1.0))
            return tf.keras.Sequential([
                tfpl.VariableLayer(2 * n, dtype=dtype),
                tfpl.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :n],
                              scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                    reinterpreted_batch_ndims=1))
            ])

        # 事前分布を定義するヘルパー関数
        def prior_trainable(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            return tf.keras.Sequential([
                tfpl.VariableLayer(2 * n, dtype=dtype),
                tfpl.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :n],
                              scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                    reinterpreted_batch_ndims=1))
            ])

        # モデル構築
        model = tfk.Sequential()
        model.add(tfkl.InputLayer(input_shape=(input_dim,)))

        # 隠れ層
        for units in self.hidden_layers:
            model.add(tfpl.DenseVariational(
                units=units,
                make_posterior_fn=posterior_mean_field,
                make_prior_fn=prior_trainable,
                kl_weight=self.kl_weight,
                activation=self.activation
            ))

        # 出力層 (平均と標準偏差)
        model.add(tfpl.DenseVariational(
            units=tfpl.IndependentNormal.params_size(output_dim),
            make_posterior_fn=posterior_mean_field,
            make_prior_fn=prior_trainable,
            kl_weight=self.kl_weight
        ))
        model.add(tfpl.IndependentNormal(output_dim))

        # コンパイル
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=lambda y, p_y: -p_y.log_prob(y),
            metrics=['mse', 'mae']
        )

        self.model = model
        logger.info(f"TensorFlowモデルを構築しました: {input_dim} -> {self.hidden_layers} -> {output_dim}")

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
        if self.model is None:
            self.build(X.shape[1], y.shape[1] if len(y.shape) > 1 else 1)

        # 訓練
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        # メトリクスを保存
        self.metrics = {
            "loss": float(history.history["loss"][-1]),
            "mse": float(history.history["mse"][-1]) if "mse" in history.history else 0.0,
            "mae": float(history.history["mae"][-1]) if "mae" in history.history else 0.0
        }

        logger.info(f"TensorFlowモデルの訓練完了: loss={self.metrics['loss']:.4f}, mse={self.metrics.get('mse', 0):.4f}")
        return self.metrics

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """モデルを評価する

        Args:
            X: 入力データ
            y: 目標データ

        Returns:
            評価メトリクス
        """
        if self.model is None:
            raise ValueError("モデルが構築されていません")

        # 評価
        eval_results = self.model.evaluate(X, y, verbose=0)

        # メトリクスを保存
        metrics = {
            "loss": float(eval_results[0]),
            "mse": float(eval_results[1]) if len(eval_results) > 1 else 0.0,
            "mae": float(eval_results[2]) if len(eval_results) > 2 else 0.0
        }

        self.metrics.update(metrics)
        logger.info(f"TensorFlowモデルの評価: loss={metrics['loss']:.4f}, mse={metrics.get('mse', 0):.4f}")
        return metrics

    def predict(self, X: np.ndarray, samples: int = 100) -> np.ndarray:
        """予測を行う

        Args:
            X: 入力データ
            samples: サンプル数

        Returns:
            予測結果 (平均値)
        """
        if self.model is None:
            raise ValueError("モデルが構築されていません")

        # 予測サンプリング (実際のアプリケーションでは確率分布から複数サンプルを生成)
        predictions = []
        for _ in range(samples):
            pred = self.model(X)
            predictions.append(pred.mean().numpy())

        # 予測の平均を取る
        mean_prediction = np.mean(predictions, axis=0)
        return mean_prediction

    def get_weights(self) -> List[np.ndarray]:
        """モデルの重みを取得する

        Returns:
            モデルの重みリスト
        """
        if self.model is None:
            raise ValueError("モデルが構築されていません")

        return [w.numpy() for w in self.model.weights]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """モデルの重みを設定する

        Args:
            weights: モデルの重みリスト
        """
        if self.model is None:
            raise ValueError("モデルが構築されていません")

        self.model.set_weights(weights)
        logger.info("TensorFlowモデルの重みを設定しました")

    def save(self, path: str) -> None:
        """モデルを保存する

        Args:
            path: 保存先パス
        """
        if self.model is None:
            raise ValueError("モデルが構築されていません")

        # モデル構造とメタデータを保存
        model_info = {
            "type": "tensorflow",
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "final_activation": self.final_activation,
            "kl_weight": self.kl_weight,
            "metrics": self.metrics
        }

        # ディレクトリ作成
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # メタデータを保存
        with open(f"{path}_info.json", "w") as f:
            json.dump(model_info, f)

        # モデルを保存
        self.model.save(path)
        logger.info(f"TensorFlowモデルを保存しました: {path}")

    def load(self, path: str) -> None:
        """モデルを読み込む

        Args:
            path: 読み込み元パス
        """
        # メタデータを読み込み
        with open(f"{path}_info.json", "r") as f:
            model_info = json.load(f)

        # 属性を設定
        self.hidden_layers = model_info["hidden_layers"]
        self.activation = model_info["activation"]
        self.final_activation = model_info["final_activation"]
        self.kl_weight = model_info["kl_weight"]
        self.metrics = model_info["metrics"]

        # モデルを読み込み
        self.model = tf.keras.models.load_model(path)
        logger.info(f"TensorFlowモデルを読み込みました: {path}")


# PyTorch実装
class PyTorchFinancialPredictor(FinancialPerformancePredictor):
    """PyTorchベースの金融パフォーマンス予測モデル"""

    def build(self, input_dim: int, output_dim: int) -> None:
        """モデルを構築する

        Args:
            input_dim: 入力次元
            output_dim: 出力次元
        """
        if torch is None:
            raise ImportError("PyTorchが必要です")

        # PyTorchモデル定義
        class BayesianNN(torch.nn.Module):
            def __init__(self, input_dim, hidden_layers, output_dim, activation):
                super().__init__()
                self.layers = torch.nn.ModuleList()

                # 入力層から最初の隠れ層
                prev_dim = input_dim
                for h_dim in hidden_layers:
                    self.layers.append(torch.nn.Linear(prev_dim, h_dim))
                    prev_dim = h_dim

                # 出力層
                self.output_layer = torch.nn.Linear(prev_dim, output_dim * 2)  # 平均と標準偏差

                # 活性化関数
                if activation == "relu":
                    self.activation = torch.nn.ReLU()
                elif activation == "tanh":
                    self.activation = torch.nn.Tanh()
                else:
                    self.activation = torch.nn.ReLU()

            def forward(self, x):
                for layer in self.layers:
                    x = self.activation(layer(x))

                output = self.output_layer(x)
                mean, log_std = torch.chunk(output, 2, dim=1)

                return mean, torch.exp(log_std)

        # モデルとオプティマイザを初期化
        self.model = BayesianNN(input_dim, self.hidden_layers, output_dim, self.activation)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        logger.info(f"PyTorchモデルを構築しました: {input_dim} -> {self.hidden_layers} -> {output_dim}")

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
        if self.model is None:
            self.build(X.shape[1], y.shape[1] if len(y.shape) > 1 else 1)

        # NumPyからPyTorchテンソルへ変換
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # データセットとデータローダーを作成
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 訓練ループ
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        n_batches = 0

        for _ in range(epochs):
            for x_batch, y_batch in dataloader:
                self.optimizer.zero_grad()

                # 順伝播
                mean, std = self.model(x_batch)

                # ガウス分布の負の対数尤度（NLL）をロスとして計算
                loss = torch.nn.GaussianNLLLoss()(mean, y_batch, std.pow(2))

                # バックプロパゲーション
                loss.backward()
                self.optimizer.step()

                # メトリクス計算
                total_loss += loss.item()
                mse = torch.nn.MSELoss()(mean, y_batch).item()
                mae = torch.nn.L1Loss()(mean, y_batch).item()
                total_mse += mse
                total_mae += mae
                n_batches += 1

        # 平均メトリクスを計算
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        avg_mse = total_mse / n_batches if n_batches > 0 else 0
        avg_mae = total_mae / n_batches if n_batches > 0 else 0

        # メトリクスを保存
        self.metrics = {
            "loss": avg_loss,
            "mse": avg_mse,
            "mae": avg_mae
        }

        logger.info(f"PyTorchモデルの訓練完了: loss={avg_loss:.4f}, mse={avg_mse:.4f}")
        return self.metrics

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """モデルを評価する

        Args:
            X: 入力データ
            y: 目標データ

        Returns:
            評価メトリクス
        """
        if self.model is None:
            raise ValueError("モデルが構築されていません")

        # NumPyからPyTorchテンソルへ変換
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # 評価モード
        self.model.eval()

        # 勾配計算なし
        with torch.no_grad():
            # 順伝播
            mean, std = self.model(X_tensor)

            # メトリクス計算
            loss = torch.nn.GaussianNLLLoss()(mean, y_tensor, std.pow(2)).item()
            mse = torch.nn.MSELoss()(mean, y_tensor).item()
            mae = torch.nn.L1Loss()(mean, y_tensor).item()

        # メトリクスを保存
        metrics = {
            "loss": loss,
            "mse": mse,
            "mae": mae
        }

        self.metrics.update(metrics)
        logger.info(f"PyTorchモデルの評価: loss={loss:.4f}, mse={mse:.4f}")
        return metrics

    def predict(self, X: np.ndarray, samples: int = 100) -> np.ndarray:
        """予測を行う

        Args:
            X: 入力データ
            samples: サンプル数

        Returns:
            予測結果 (平均値)
        """
        if self.model is None:
            raise ValueError("モデルが構築されていません")

        # NumPyからPyTorchテンソルへ変換
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # 評価モード
        self.model.eval()

        # 予測サンプリング
        predictions = []
        with torch.no_grad():
            for _ in range(samples):
                mean, _ = self.model(X_tensor)
                predictions.append(mean.numpy())

        # 予測の平均を取る
        mean_prediction = np.mean(predictions, axis=0)
        return mean_prediction

    def get_weights(self) -> List[np.ndarray]:
        """モデルの重みを取得する

        Returns:
            モデルの重みリスト
        """
        if self.model is None:
            raise ValueError("モデルが構築されていません")

        return [p.detach().numpy() for p in self.model.parameters()]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """モデルの重みを設定する

        Args:
            weights: モデルの重みリスト
        """
        if self.model is None:
            raise ValueError("モデルが構築されていません")

        with torch.no_grad():
            for param, weight in zip(self.model.parameters(), weights):
                param.copy_(torch.tensor(weight))

        logger.info("PyTorchモデルの重みを設定しました")

    def save(self, path: str) -> None:
        """モデルを保存する

        Args:
            path: 保存先パス
        """
        if self.model is None:
            raise ValueError("モデルが構築されていません")

        # モデル構造とメタデータを保存
        model_info = {
            "type": "pytorch",
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "final_activation": self.final_activation,
            "kl_weight": self.kl_weight,
            "metrics": self.metrics
        }

        # ディレクトリ作成
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # メタデータを保存
        with open(f"{path}_info.json", "w") as f:
            json.dump(model_info, f)

        # モデルを保存
        torch.save(self.model.state_dict(), f"{path}.pt")
        logger.info(f"PyTorchモデルを保存しました: {path}")

    def load(self, path: str) -> None:
        """モデルを読み込む

        Args:
            path: 読み込み元パス
        """
        # メタデータを読み込み
        with open(f"{path}_info.json", "r") as f:
            model_info = json.load(f)

        # 属性を設定
        self.hidden_layers = model_info["hidden_layers"]
        self.activation = model_info["activation"]
        self.final_activation = model_info["final_activation"]
        self.kl_weight = model_info["kl_weight"]
        self.metrics = model_info["metrics"]

        # モデルを構築
        input_dim = None
        output_dim = None

        # パスからサイズ情報を抽出
        try:
            # path名からサイズ情報を取得する試み
            path_parts = Path(path).stem.split("_")
            for part in path_parts:
                if part.startswith("input"):
                    input_dim = int(part.replace("input", ""))
                elif part.startswith("output"):
                    output_dim = int(part.replace("output", ""))
        except:
            # サイズ情報が取得できない場合はデフォルト値を使用
            logger.warning("モデルサイズ情報を取得できません。デフォルト値を使用します。")
            input_dim = 10
            output_dim = 5

        # モデルを構築
        self.build(input_dim, output_dim)

        # モデルの重みを読み込み
        self.model.load_state_dict(torch.load(f"{path}.pt"))
        self.model.eval()
        logger.info(f"PyTorchモデルを読み込みました: {path}")


# 後方互換性のための別名
FinancialPerformancePredictor = ModelFactory.create_model