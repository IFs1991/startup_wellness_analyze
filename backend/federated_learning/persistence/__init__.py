# Phase 3: データ永続化層とスケーラビリティ
# 永続化層の実装パッケージ

"""
フェデレーテッド学習システムの永続化層

このモジュールは以下を提供します：
- SQLAlchemyベースのORM
- リポジトリパターン実装
- データベース接続管理
- トランザクション管理
- マイグレーション管理
"""

from .database import DatabaseManager
from .models import FLModel, ClientRegistration, TrainingSession
from .repositories import ModelRepository, ClientRegistryRepository, TrainingHistoryRepository

__all__ = [
    "DatabaseManager",
    "FLModel",
    "ClientRegistration",
    "TrainingSession",
    "ModelRepository",
    "ClientRegistryRepository",
    "TrainingHistoryRepository"
]