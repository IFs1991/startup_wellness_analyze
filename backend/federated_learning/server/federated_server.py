"""
連合学習サーバー実装

このモジュールは、スタートアップ分析プラットフォームの連合学習サーバーを実装します。
サーバーはクライアントからのモデル更新を集約し、グローバルモデルを更新します。
Flowerフレームワークを使用して実装されています。
"""

import os
import time
import logging
import numpy as np
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import asyncio
import yaml

# Flowerのインポート
import flwr as fl
from flwr.common import Parameters, Weights, FitIns, FitRes, EvaluateIns, EvaluateRes
from flwr.server.strategy import FedAvg, Strategy
from flwr.server.client_proxy import ClientProxy

from ..security.secure_aggregator import SecureAggregator
from ..models.model_interface import ModelInterface

logger = logging.getLogger(__name__)

class FederatedServer:
    """連合学習サーバー

    連合学習のサーバー実装。クライアントからのモデル更新を集約し、
    グローバルモデルを更新します。Flowerを使用して実装されています。
    """

    def __init__(self, config_path: Optional[str] = None):
        """初期化

        Args:
            config_path: 設定ファイルパス（省略時はデフォルト設定を使用）
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.model_storage_path = Path(os.environ.get('FL_MODEL_STORAGE', '/tmp/federated_models'))
        self.model_storage_path.mkdir(parents=True, exist_ok=True)

        self.secure_aggregator = SecureAggregator(
            protocol=self.config['federated_learning']['secure_aggregation']['protocol'],
            crypto_provider=self.config['federated_learning']['secure_aggregation']['crypto_provider']
        )

        self.current_round = {}
        self.client_registry = {}
        self.fl_server = None
        self.fl_strategies = {}

        self._init_db_connection()
        logger.info("連合学習サーバーを初期化しました")

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

    def _init_db_connection(self) -> None:
        """データベース接続を初期化する"""
        # 実際のデータベース接続実装はここに記述
        # この例では単純化のため省略
        pass

    def register_model(self, model_name: str, model: ModelInterface) -> None:
        """モデルを登録する

        Args:
            model_name: モデル名
            model: モデルインスタンス
        """
        self.models[model_name] = model
        self.current_round[model_name] = 0

        # モデルをデータベースに登録
        # 実際のデータベース操作はここに記述

        # Flower用の戦略を作成
        model_config = next(m for m in self.config['federated_learning']['models']
                           if m['name'] == model_name)

        # 戦略パラメータの設定
        strategy_params = {
            "fraction_fit": model_config['federated_settings'].get('sample_fraction', 0.8),
            "min_fit_clients": model_config['federated_settings'].get('min_available_clients', 3),
            "min_available_clients": model_config['federated_settings'].get('min_available_clients', 3),
            "min_evaluate_clients": model_config['federated_settings'].get('min_available_clients', 3),
            "on_fit_config_fn": self._get_fit_config_fn(model_name),
            "on_evaluate_config_fn": self._get_eval_config_fn(model_name),
            "initial_parameters": self._get_initial_parameters(model_name),
        }

        # 差分プライバシーが有効ならDPFedAvgを使用
        if self.config['federated_learning']['differential_privacy']['enabled']:
            # 実際の実装ではDPFedAvgの実装を追加
            strategy = FedAvg(**strategy_params)
        else:
            strategy = FedAvg(**strategy_params)

        self.fl_strategies[model_name] = strategy

        logger.info(f"モデル '{model_name}' を登録しました")

        # モデルを保存
        self._save_model(model_name)

    def _get_fit_config_fn(self, model_name: str) -> Callable:
        """トレーニング設定関数を取得

        Args:
            model_name: モデル名

        Returns:
            設定関数
        """
        model_config = next(m for m in self.config['federated_learning']['models']
                           if m['name'] == model_name)

        def fit_config_fn(rnd: int) -> Dict[str, Any]:
            """クライアントのFitのための設定を生成する"""
            config = {
                "epoch_global": rnd,
                "epochs": model_config['training']['local_epochs'],
                "batch_size": model_config['training']['batch_size'],
                "learning_rate": model_config['training']['learning_rate'],
                "timeout": model_config['federated_settings']['fit_config'].get('timeout', 600)
            }
            return config

        return fit_config_fn

    def _get_eval_config_fn(self, model_name: str) -> Callable:
        """評価設定関数を取得

        Args:
            model_name: モデル名

        Returns:
            設定関数
        """
        def eval_config_fn(rnd: int) -> Dict[str, Any]:
            """クライアントの評価のための設定を生成する"""
            return {"round": rnd}

        return eval_config_fn

    def _get_initial_parameters(self, model_name: str) -> Optional[Parameters]:
        """初期パラメーターを取得

        Args:
            model_name: モデル名

        Returns:
            初期パラメーター
        """
        if model_name not in self.models:
            return None

        model = self.models[model_name]
        weights = model.get_weights()
        return fl.common.weights_to_parameters(weights)

    def _save_model(self, model_name: str) -> None:
        """モデルを保存する

        Args:
            model_name: モデル名
        """
        if model_name not in self.models:
            logger.error(f"モデル '{model_name}' は登録されていません")
            return

        model = self.models[model_name]
        round_num = self.current_round[model_name]

        # モデルの重みを保存
        weights = model.get_weights()
        model_path = self.model_storage_path / f"{model_name}_round_{round_num}.json"

        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump({
                "model_name": model_name,
                "round": round_num,
                "weights": weights,
                "timestamp": datetime.now().isoformat()
            }, f)

        logger.info(f"モデル '{model_name}' (ラウンド {round_num}) を保存しました")

    def _load_model(self, model_name: str, round_num: Optional[int] = None) -> bool:
        """モデルを読み込む

        Args:
            model_name: モデル名
            round_num: ラウンド番号（省略時は最新）

        Returns:
            成功したかどうか
        """
        if model_name not in self.models:
            logger.error(f"モデル '{model_name}' は登録されていません")
            return False

        if round_num is None:
            round_num = self.current_round[model_name]

        model_path = self.model_storage_path / f"{model_name}_round_{round_num}.json"

        if not model_path.exists():
            logger.error(f"モデルファイル '{model_path}' が見つかりません")
            return False

        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)

            self.models[model_name].set_weights(model_data['weights'])
            logger.info(f"モデル '{model_name}' (ラウンド {round_num}) を読み込みました")
            return True

        except Exception as e:
            logger.exception(f"モデル読み込み中に例外が発生しました: {e}")
            return False

    def start_server(self, model_name: str, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Flowerサーバーを起動する

        Args:
            model_name: 使用するモデル名
            host: ホストアドレス
            port: ポート番号
        """
        if model_name not in self.models:
            raise ValueError(f"モデル '{model_name}' は登録されていません")

        if model_name not in self.fl_strategies:
            raise ValueError(f"モデル '{model_name}' の戦略が登録されていません")

        strategy = self.fl_strategies[model_name]

        # セキュア集約が有効な場合の設定
        secure_aggregation = self.config['federated_learning']['secure_aggregation']['enabled']
        if secure_aggregation:
            # 実際の実装ではSecureAggregationの設定を追加
            pass

        # 設定からラウンド数を取得
        model_config = next(m for m in self.config['federated_learning']['models']
                           if m['name'] == model_name)
        num_rounds = model_config['federated_settings'].get('num_rounds', 50)

        # サーバーの設定
        server_config = fl.server.ServerConfig(num_rounds=num_rounds)

        # サーバー起動
        logger.info(f"Flowerサーバーを起動しています: {host}:{port}, モデル: {model_name}")
        fl.server.start_server(
            server_address=f"{host}:{port}",
            config=server_config,
            strategy=strategy
        )

    def get_model_for_client(self, model_name: str, client_id: str) -> Dict[str, Any]:
        """クライアントにモデルを提供する

        Args:
            model_name: モデル名
            client_id: クライアントID

        Returns:
            モデルデータ
        """
        if model_name not in self.models:
            raise ValueError(f"モデル '{model_name}' は登録されていません")

        if client_id not in self.client_registry:
            raise ValueError(f"クライアント '{client_id}' は登録されていません")

        # クライアントのアクティビティを更新
        self.client_registry[client_id]["last_active"] = datetime.now().isoformat()

        # クライアントにモデル参加を記録
        if model_name not in self.client_registry[client_id]["models_participated"]:
            self.client_registry[client_id]["models_participated"][model_name] = {
                "first_round": self.current_round[model_name],
                "downloads": 0,
                "updates_submitted": 0
            }

        self.client_registry[client_id]["models_participated"][model_name]["downloads"] += 1

        # モデルの重みを取得
        model = self.models[model_name]
        weights = model.get_weights()

        return {
            "model_name": model_name,
            "round": self.current_round[model_name],
            "weights": weights,
            "timestamp": datetime.now().isoformat()
        }

    def register_client(self, client_id: str, client_info: Dict[str, Any]) -> bool:
        """クライアントを登録する

        Args:
            client_id: クライアントID
            client_info: クライアント情報

        Returns:
            成功したかどうか
        """
        try:
            self.client_registry[client_id] = {
                "client_name": client_info.get("client_name", f"Client-{client_id}"),
                "industry_type": client_info.get("industry_type", "unknown"),
                "data_size": client_info.get("data_size", 0),
                "last_active": datetime.now().isoformat(),
                "models_participated": {}
            }

            logger.info(f"クライアント '{client_id}' を登録しました")
            return True

        except Exception as e:
            logger.exception(f"クライアント登録中に例外が発生しました: {e}")
            return False

    def evaluate_global_model(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """グローバルモデルを評価する

        Args:
            model_name: モデル名
            X: 特徴量データ
            y: ターゲットデータ

        Returns:
            評価メトリクス
        """
        if model_name not in self.models:
            raise ValueError(f"モデル '{model_name}' は登録されていません")

        model = self.models[model_name]
        metrics = model.evaluate(X, y)

        logger.info(f"グローバルモデル '{model_name}' の評価結果: {metrics}")
        return metrics

    def get_training_status(self, model_name: str) -> Dict[str, Any]:
        """訓練ステータスを取得する

        Args:
            model_name: モデル名

        Returns:
            ステータス情報
        """
        if model_name not in self.models:
            raise ValueError(f"モデル '{model_name}' は登録されていません")

        return {
            "model_name": model_name,
            "current_round": self.current_round[model_name],
            "active_clients": len(self.client_registry),
            "total_clients": len(self.client_registry),
            "timestamp": datetime.now().isoformat()
        }