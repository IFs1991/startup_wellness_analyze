# フェデレーテッド学習システム (Flower実装)

## 概要

本システムは、スタートアップの健康とパフォーマンスデータを安全にモデル学習するためのフェデレーテッド学習システムを実装しています。クライアント側のプライバシーを保護しながら、グローバルなモデルを学習することが可能です。

Flower（flwr）フレームワークを使用して実装されており、TensorFlowとPyTorch双方に対応したマルチフレームワーク機能を提供します。

## 主要コンポーネント

### サーバー側
- **FederatedServer**: Flowerベースのフェデレーテッド学習サーバー実装
- **SecureAggregator**: クライアントの更新を安全に集約するための暗号化機能
- **ModelFactory**: 適切なフレームワーク実装を自動的に選択するファクトリークラス

### クライアント側
- **FederatedClient**: Flowerベースのフェデレーテッド学習クライアント実装
- **DifferentialPrivacy**: 差分プライバシーによるデータ保護機能

### モデル
- **FinancialPerformancePredictor**: 金融パフォーマンス予測モデル（TensorFlow/PyTorch実装）
- **ModelInterface**: すべてのモデルが実装すべきインターフェース

### アダプター
- **CoreModelIntegration**: コアシステムとフェデレーテッド学習の統合
- **HealthImpactAdapter**: 健康関連データの統合と調整

## アーキテクチャ

```
+-------------------+       +----------------------+
|   フェデレーテッド    |       |   フェデレーテッド     |
|   クライアント      |<----->|   サーバー           |
+-------------------+       +----------------------+
       ^                             ^
       |                             |
+-------------------+       +----------------------+
|   ローカルモデル     |       |   グローバルモデル     |
|   (TF/PyTorch)    |       |   (TF/PyTorch)      |
+-------------------+       +----------------------+
       ^                             ^
       |                             |
+-------------------+       +----------------------+
|   ローカルデータ     |       |   モデル更新集約      |
+-------------------+       +----------------------+
```

## マルチフレームワーク対応

このシステムは、TensorFlowとPyTorchの両方のフレームワークに対応しており、実行環境に応じて最適なフレームワークを自動的に選択します。これにより、異なる環境や要件に対して柔軟に対応できます。

### 特徴
- **フレームワーク自動検出**: 利用可能なフレームワークを自動的に検出
- **共通インターフェース**: 異なるフレームワーク間で統一されたAPI
- **フォールバックメカニズム**: 優先フレームワークが使用できない場合の代替手段

## セキュリティと差分プライバシー

個人データのプライバシーを保護するために、以下の機能を実装しています：

- **セキュア集約**: 暗号化技術を用いた安全なモデル更新集約
- **差分プライバシー**: クライアントデータに対する差分プライバシー保護
- **通信の暗号化**: クライアント-サーバー間通信の暗号化

## 使用方法

### サーバーの起動

```python
from backend.federated_learning.server.federated_server import FederatedServer
from backend.federated_learning.models.financial_performance_predictor import FinancialPerformancePredictor, ModelFactory

# サーバーの初期化
server = FederatedServer()

# モデルの作成（TensorFlowとPyTorchに対応）
model = ModelFactory.create_model(framework="auto")

# モデルの登録
server.register_model("financial_performance", model)

# サーバーの起動
server.start_server("financial_performance", host="0.0.0.0", port=8080)
```

### クライアントの実行

```python
from backend.federated_learning.client.federated_client import FederatedClient
from backend.federated_learning.models.financial_performance_predictor import ModelFactory
import numpy as np

# クライアントの初期化
client = FederatedClient(client_id="client_1")

# モデルの作成（TensorFlowとPyTorchに対応）
model = ModelFactory.create_model(framework="auto")

# モデルの登録
client.register_model("financial_performance", model)

# サーバーへの接続
client.connect_to_server("localhost:8080")

# 学習用データの準備
X_train = np.random.normal(0, 1, (100, 10))
y_train = np.random.normal(0, 1, (100, 1))

# ローカルトレーニング
client.train_local_model("financial_performance", X_train, y_train)

# モデル更新の送信
client.submit_model_update("financial_performance")
```

## 設定

設定は `config.yaml` ファイルで管理されています。主な設定項目：

- **フェデレーテッド学習**:
  - 集約アルゴリズム
  - クライアントサンプリング率
  - 差分プライバシーパラメータ

- **モデル**:
  - モデルアーキテクチャ
  - ハイパーパラメータ
  - 評価メトリクス

- **セキュリティ**:
  - 暗号化アルゴリズム
  - プライバシー保護レベル

## 将来の開発計画

- **JAX対応**: マルチフレームワーク対応をJAXにも拡張
- **クロスデバイスフェデレーテッド学習**: モバイルおよびエッジデバイス対応
- **垂直分割フェデレーテッド学習**: 異なる特徴量セット間での学習