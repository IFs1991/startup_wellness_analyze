"""
Flowerフレームワーク統合テスト

このモジュールは、Flowerフレームワークとの統合および
マルチフレームワーク対応機能をテストします。
"""

import os
import pytest
import numpy as np
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock
import logging

# テスト対象のモジュール
from backend.federated_learning.models.financial_performance_predictor import (
    ModelFactory, TensorFlowFinancialPredictor, PyTorchFinancialPredictor
)
from backend.federated_learning.client.federated_client import FederatedClient
from backend.federated_learning.adapters.core_integration import CoreModelIntegration
from backend.federated_learning.adapters.health_impact_adapter import (
    HealthImpactAdapter, is_financial_performance_model
)

# テスト用のロガー
logger = logging.getLogger(__name__)

# フィクスチャ：テスト用データ
@pytest.fixture
def sample_data():
    """テスト用データセットを生成"""
    # 特徴量
    X = np.random.normal(0, 1, (100, 10))
    # ターゲット（1次元）
    y = np.random.normal(0, 1, (100, 1))
    return X, y

# フィクスチャ：テスト用データフレーム
@pytest.fixture
def sample_dataframe():
    """テスト用データフレームを生成"""
    df = pd.DataFrame({
        'revenue': np.random.normal(1000, 200, 100),
        'expenses': np.random.normal(800, 150, 100),
        'employees': np.random.randint(10, 100, 100),
        'work_life_balance': np.random.uniform(0, 10, 100),
        'stress_level': np.random.uniform(0, 10, 100),
        'sleep_quality': np.random.uniform(0, 10, 100),
        'mental_health': np.random.uniform(0, 10, 100)
    })
    return df

# テスト：ModelFactoryのフレームワーク自動検出
def test_model_factory_auto_detection():
    """ModelFactoryのフレームワーク自動検出機能をテスト"""
    # テスト実行環境によって利用可能なフレームワークが異なるため、
    # モックを使用して特定のフレームワークが利用可能な状況をシミュレート

    # TensorFlowのみが利用可能なケース
    with patch('backend.federated_learning.models.financial_performance_predictor.tf', MagicMock()):
        with patch('backend.federated_learning.models.financial_performance_predictor.tfp', MagicMock()):
            with patch('backend.federated_learning.models.financial_performance_predictor.torch', None):
                with patch('backend.federated_learning.models.financial_performance_predictor.pyro', None):
                    model = ModelFactory.create_model(framework="auto")
                    assert isinstance(model, TensorFlowFinancialPredictor)

    # PyTorchのみが利用可能なケース
    with patch('backend.federated_learning.models.financial_performance_predictor.tf', None):
        with patch('backend.federated_learning.models.financial_performance_predictor.tfp', None):
            with patch('backend.federated_learning.models.financial_performance_predictor.torch', MagicMock()):
                with patch('backend.federated_learning.models.financial_performance_predictor.pyro', MagicMock()):
                    model = ModelFactory.create_model(framework="auto")
                    assert isinstance(model, PyTorchFinancialPredictor)

# テスト：モデル実装の基本機能
def test_tensorflow_model_basic_functionality(sample_data):
    """TensorFlow実装の基本機能をテスト"""
    # このテストは、TensorFlowがインストールされている場合のみ実行
    try:
        # TensorFlowが利用可能な場合
        import tensorflow as tf
        import tensorflow_probability as tfp

        X, y = sample_data

        # モデルの作成
        model = ModelFactory.create_model(framework="tensorflow")

        # モデルの構築
        model.build(input_dim=X.shape[1], output_dim=y.shape[1])

        # モデル訓練
        metrics = model.train(X, y, epochs=1)

        # 基本的なメトリクスが存在することを確認
        assert "loss" in metrics
        assert "mse" in metrics

        # 予測
        predictions = model.predict(X)
        assert predictions.shape == (100, 1)

        # モデル保存と読み込み
        with tempfile.TemporaryDirectory() as tempdir:
            model_path = os.path.join(tempdir, "test_model")
            model.save(model_path)

            # 重みの取得
            weights = model.get_weights()

            # 新しいモデルに重みを設定
            new_model = ModelFactory.create_model(framework="tensorflow")
            new_model.build(input_dim=X.shape[1], output_dim=y.shape[1])
            new_model.set_weights(weights)

            # 予測が同じになることを確認
            new_predictions = new_model.predict(X)
            np.testing.assert_allclose(predictions, new_predictions, rtol=1e-5)

    except ImportError:
        pytest.skip("TensorFlowがインストールされていないため、このテストをスキップします")

# テスト：PyTorch実装の基本機能
def test_pytorch_model_basic_functionality(sample_data):
    """PyTorch実装の基本機能をテスト"""
    # このテストは、PyTorchがインストールされている場合のみ実行
    try:
        # PyTorchが利用可能な場合
        import torch
        import pyro

        X, y = sample_data

        # モデルの作成
        model = ModelFactory.create_model(framework="pytorch")

        # モデルの構築
        model.build(input_dim=X.shape[1], output_dim=y.shape[1])

        # モデル訓練
        metrics = model.train(X, y, epochs=1)

        # 基本的なメトリクスが存在することを確認
        assert "loss" in metrics
        assert "mse" in metrics

        # 予測
        predictions = model.predict(X)
        assert predictions.shape == (100, 1)

        # モデル保存と読み込み
        with tempfile.TemporaryDirectory() as tempdir:
            model_path = os.path.join(tempdir, "test_model")
            model.save(model_path)

            # 重みの取得
            weights = model.get_weights()

            # 新しいモデルに重みを設定
            new_model = ModelFactory.create_model(framework="pytorch")
            new_model.build(input_dim=X.shape[1], output_dim=y.shape[1])
            new_model.set_weights(weights)

            # 予測が類似していることを確認
            new_predictions = new_model.predict(X)
            # PyTorchでは確率的要素があるため、厳密な一致ではなく近似値をチェック
            np.testing.assert_allclose(predictions, new_predictions, rtol=1e-3)

    except ImportError:
        pytest.skip("PyTorchとPyroがインストールされていないため、このテストをスキップします")

# テスト：モデル識別関数
def test_is_financial_performance_model():
    """モデル識別関数のテスト"""
    # ModelFactoryを使ってモデルを作成
    # 利用可能なフレームワークによって結果が異なるため、両方のケースをモック

    # TensorFlow実装
    with patch('backend.federated_learning.models.financial_performance_predictor.tf', MagicMock()):
        with patch('backend.federated_learning.models.financial_performance_predictor.tfp', MagicMock()):
            model_tf = ModelFactory.create_model(framework="tensorflow")
            assert is_financial_performance_model(model_tf)

    # PyTorch実装
    with patch('backend.federated_learning.models.financial_performance_predictor.torch', MagicMock()):
        with patch('backend.federated_learning.models.financial_performance_predictor.pyro', MagicMock()):
            model_pt = ModelFactory.create_model(framework="pytorch")
            assert is_financial_performance_model(model_pt)

    # 非金融モデル（モック）
    class DummyModel:
        def __init__(self):
            self.metrics = {}

        def get_weights(self):
            return []

    dummy_model = DummyModel()
    assert not is_financial_performance_model(dummy_model)

# テスト：HealthImpactAdapter
def test_health_impact_adapter(sample_dataframe):
    """HealthImpactAdapterの機能テスト"""
    # アダプターを初期化
    adapter = HealthImpactAdapter()

    # 重みの取得
    weights = adapter.get_health_impact_weights("tech")
    assert not weights.empty
    assert "factor" in weights.columns
    assert "final_weight" in weights.columns

    # 特徴量拡張
    df = sample_dataframe
    enhanced_df = adapter.create_health_weighted_features(df, "tech")

    # 拡張された特徴量が存在することを確認
    assert "work_life_balance_weighted" in enhanced_df.columns
    assert "stress_level_weighted" in enhanced_df.columns
    assert "industry_type" in enhanced_df.columns

    # 拡張された特徴量の値が正しいことを確認
    work_life_weight = weights[weights["factor"] == "work_life_balance"]["final_weight"].values[0]
    assert np.allclose(
        enhanced_df["work_life_balance_weighted"].values,
        df["work_life_balance"].values * work_life_weight
    )

# テスト：CoreModelIntegration
def test_core_model_integration():
    """CoreModelIntegrationのテスト"""
    # FederatedClientとHealthImpactAdapterをモック
    with patch('backend.federated_learning.client.federated_client.FederatedClient') as MockClient:
        with patch('backend.federated_learning.adapters.health_impact_adapter.HealthImpactAdapter') as MockAdapter:
            # ModelFactoryもモック
            with patch('backend.federated_learning.models.financial_performance_predictor.ModelFactory') as MockFactory:
                # モックモデルの設定
                mock_model = MagicMock()
                MockFactory.create_model.return_value = mock_model

                # CoreModelIntegrationの初期化
                integration = CoreModelIntegration(client_id="test_client")

                # ModelFactoryが呼び出されたことを確認
                MockFactory.create_model.assert_called_once()

                # FederatedClientのregister_modelが呼び出されたことを確認
                client_instance = MockClient.return_value
                client_instance.register_model.assert_called_once_with(
                    "financial_performance", mock_model
                )

# メイン実行部分
if __name__ == "__main__":
    print("Flowerフレームワーク統合テストを実行中...")

    # 手動テスト実行の例
    # 自動検出のテスト
    try:
        model = ModelFactory.create_model(framework="auto")
        print(f"自動検出されたモデル: {model.__class__.__name__}")
    except Exception as e:
        print(f"モデル自動検出エラー: {e}")

    # 利用可能なモデルのテスト
    try:
        # TensorFlow実装
        tf_model = ModelFactory.create_model(framework="tensorflow")
        print("TensorFlow実装が利用可能です")
    except Exception as e:
        print(f"TensorFlow実装エラー: {e}")

    try:
        # PyTorch実装
        pt_model = ModelFactory.create_model(framework="pytorch")
        print("PyTorch実装が利用可能です")
    except Exception as e:
        print(f"PyTorch実装エラー: {e}")

    print("テスト完了")