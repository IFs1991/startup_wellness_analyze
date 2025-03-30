"""
連合学習モジュール

このモジュールは、スタートアップ分析プラットフォームの連合学習機能を提供します。
連合学習を通じて、各クライアントのプライバシーを保護しながら、集合的な知識を活用した
高精度な予測モデルを構築します。

主要コンポーネント:
- クライアント: ローカルデータでモデルを訓練し、更新を送信 (Flower実装)
- サーバー: クライアントからの更新を集約し、グローバルモデルを更新 (Flower実装)
- モデル: ベイジアンニューラルネットワークなどの予測モデル (マルチフレームワーク対応)
- セキュリティ: 差分プライバシーや安全な集約などのプライバシー保護メカニズム (Flower対応)
"""

# Python 3.12互換性対応: 型ヒントの読み込みに失敗した場合の回避策
try:
    # サブモジュールのインポート
    from . import client
    from . import server
    from . import models
    from . import security
    from . import adapters
    from . import utils

    # 直接クラスのインポートは互換性の問題で保留
    # 必要な場合は個別モジュールからインポート

except ImportError as e:
    import sys
    import warnings
    warnings.warn(f"連合学習モジュールの一部のインポートに失敗しました: {e}")
    warnings.warn(f"現在のPythonバージョン: {sys.version}")

__version__ = "2.0.0"