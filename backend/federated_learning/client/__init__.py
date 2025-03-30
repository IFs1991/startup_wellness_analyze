"""
連合学習クライアントモジュール

このモジュールは、連合学習クライアントの実装を提供します。
クライアントはローカルデータを使用してモデルを訓練し、更新をサーバーに送信します。
"""

from .federated_client import FederatedClient

__all__ = ['FederatedClient']