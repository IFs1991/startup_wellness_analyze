# -*- coding: utf-8 -*-
"""
[非推奨] データベース接続モジュール

警告: このモジュールは非推奨であり、将来のバージョンで削除される予定です。
代わりに backend.database.connection モジュールを使用してください。

後方互換性のために残されています。
"""

import warnings
import logging
from typing import Any, Generator

# 非推奨警告を表示
warnings.warn(
    "backend.database.database モジュールは非推奨です。代わりに backend.database.connection を使用してください。",
    DeprecationWarning,
    stacklevel=2
)

# ロガーの設定
logger = logging.getLogger(__name__)
logger.warning(
    "backend.database.database モジュールは非推奨です。新しいコードでは "
    "backend.database.connection を使用してください。"
)

# 新しいモジュールからの互換性インポート
try:
    from backend.database.connection import (
        DatabaseManager,
        get_firestore_client,
        get_db_session,
        get_db_for_category,
        get_db,
        Base,
        DatabaseType,
        DataCategory,
        init_db
    )

except ImportError as e:
    logger.error(f"新しいデータベースモジュールのインポートに失敗しました: {str(e)}")
    # 失敗した場合は互換性のためのスタブを提供
    from enum import Enum
    from contextlib import contextmanager

    class DatabaseType(Enum):
        """データベースタイプを定義する列挙型スタブ"""
        NOSQL = "firestore"
        SQL = "postgresql"

    class DataCategory(Enum):
        """データカテゴリを定義する列挙型スタブ"""
        STRUCTURED = "structured"
        # 他のカテゴリ...

    class DatabaseManager:
        """DatabaseManagerスタブクラス"""
        @classmethod
        def get_firestore_client(cls):
            """Firestoreクライアントを取得するスタブメソッド"""
            logger.warning("スタブのDatabaseManager.get_firestore_clientが呼び出されました")
            return None

        @classmethod
        @contextmanager
        def get_sql_session(cls):
            """SQLセッションを取得するスタブメソッド"""
            logger.warning("スタブのDatabaseManager.get_sql_sessionが呼び出されました")
            yield None

    def get_firestore_client():
        """Firestoreクライアントを取得するスタブ関数"""
        logger.warning("スタブのget_firestore_clientが呼び出されました")
        return None

    @contextmanager
    def get_db_session():
        """SQLセッションを取得するスタブ関数"""
        logger.warning("スタブのget_db_sessionが呼び出されました")
        yield None

    def get_db_for_category(category):
        """カテゴリに基づいてDBを取得するスタブ関数"""
        logger.warning("スタブのget_db_for_categoryが呼び出されました")
        return None

    # FastAPIのDependsで使用するための関数
    def get_db():
        """FastAPIのDependsで使用するスタブ関数"""
        logger.warning("スタブのget_dbが呼び出されました")
        yield None

    # SQLAlchemyのベースクラススタブ
    class Base:
        """SQLAlchemyベースクラススタブ"""
        pass

    def init_db():
        """データベース初期化スタブ関数"""
        logger.warning("スタブのinit_dbが呼び出されました")
        pass