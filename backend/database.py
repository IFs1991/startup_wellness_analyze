# -*- coding: utf-8 -*-
"""
[非推奨] データベース設定モジュール

警告: このモジュールは非推奨であり、将来のバージョンで削除される予定です。
代わりに以下のモジュールを使用してください:
- データベース設定: backend.config.database_config
- データベース接続: backend.database.connection

後方互換性のために残されています。
"""

import warnings
import logging
from typing import Dict, Any

# 非推奨警告を表示
warnings.warn(
    "backend.database モジュールは非推奨です。代わりに backend.config.database_config と "
    "backend.database.connection を使用してください。",
    DeprecationWarning,
    stacklevel=2
)

# ロガーの設定
logger = logging.getLogger(__name__)
logger.warning(
    "backend.database モジュールは非推奨です。新しいコードでは "
    "backend.config.database_config と backend.database.connection を使用してください。"
)

# 新しいモジュールからの互換性インポート
try:
    from backend.config.database_config import (
        DatabaseConfig,
        DevelopmentDatabaseConfig,
        ProductionDatabaseConfig,
        TestingDatabaseConfig,
        current_database_config as current_config
    )

    class DatabaseManager:
        """
        [非推奨] DatabaseManagerクラス

        このクラスは後方互換性のために残されています。
        新しいコードでは backend.database.connection.DatabaseManager を使用してください。
        """
        def __new__(cls, *args, **kwargs):
            from backend.database.connection import DatabaseManager as NewDatabaseManager
            warnings.warn(
                "backend.database.DatabaseManager は非推奨です。"
                "代わりに backend.database.connection.DatabaseManager を使用してください。",
                DeprecationWarning,
                stacklevel=2
            )
            return NewDatabaseManager(*args, **kwargs)

    # その他の互換性関数
    def get_firestore_client():
        """[非推奨] Firestoreクライアントを取得する互換性関数"""
        from backend.database.connection import get_firestore_client as new_get_firestore_client
        return new_get_firestore_client()

    def get_db_session():
        """[非推奨] SQLセッションを取得する互換性関数"""
        from backend.database.connection import get_db_session as new_get_db_session
        return new_get_db_session()

except ImportError as e:
    logger.error(f"新しいデータベースモジュールのインポートに失敗しました: {str(e)}")
    # 失敗した場合は互換性のためのスタブを提供

    class DatabaseConfig:
        """DatabaseConfigスタブクラス"""
        DATABASE_URL = "sqlite:///./test.db"
        FIREBASE_ADMIN_CONFIG = None
        FIRESTORE_COLLECTIONS = {
            "USERS": "users",
            "CONSULTATIONS": "consultations",
            "TREATMENTS": "treatments",
            "ANALYTICS": "analytics"
        }

    class DevelopmentDatabaseConfig(DatabaseConfig):
        """開発環境用のデータベース設定"""
        pass

    class ProductionDatabaseConfig(DatabaseConfig):
        """本番環境用のデータベース設定"""
        pass

    class TestingDatabaseConfig(DatabaseConfig):
        """テスト環境用のデータベース設定"""
        DATABASE_URL = "sqlite:///./test.db"

    # 現在の環境に基づいて設定を選択
    import os

    config = {
        "development": DevelopmentDatabaseConfig,
        "production": ProductionDatabaseConfig,
        "testing": TestingDatabaseConfig
    }

    current_config = config[os.getenv("ENVIRONMENT", "development")]()

    class DatabaseManager:
        """DatabaseManagerスタブクラス"""
        @classmethod
        def get_firestore_client(cls):
            """Firestoreクライアントを取得するスタブメソッド"""
            logger.warning("スタブのDatabaseManager.get_firestore_clientが呼び出されました")
            return None