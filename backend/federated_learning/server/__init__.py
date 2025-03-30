"""
連合学習サーバーモジュール

このモジュールは、連合学習サーバーの実装を提供します。
サーバーはクライアントからのモデル更新を集約し、グローバルモデルを更新します。
"""

from .federated_server import FederatedServer

__all__ = ['FederatedServer']