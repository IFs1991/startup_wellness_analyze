# -*- coding: utf-8 -*-
"""
データベース接続管理モジュール
FirestoreとPostgreSQLのハイブリッド運用を実現するための機能を提供します。
"""
import os
from enum import Enum
from typing import Any, Dict, Optional, Union, Generator, List
from contextlib import contextmanager
import logging
# Firebase/Firestore
from firebase_admin import credentials, firestore, initialize_app
import firebase_admin
# SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# データベース設定をインポート
try:
    from config.database_config import current_database_config
except ImportError:
    # デフォルト設定を使用
    current_database_config = {
        'firestore': {
            'use_emulator': os.getenv('USE_FIRESTORE_EMULATOR', 'false').lower() == 'true',
            'emulator_host': os.getenv('FIRESTORE_EMULATOR_HOST', 'localhost:8080'),
            'project_id': os.getenv('FIRESTORE_PROJECT_ID', 'startup-wellness-dev')
        },
        'postgresql': {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
            'database': os.getenv('DB_NAME', 'startup_wellness'),
            'ssl_mode': os.getenv('DB_SSL_MODE', 'prefer')
        }
    }

# ロガーの設定
logger = logging.getLogger(__name__)

class DatabaseType(Enum):
    """データベースタイプを定義する列挙型"""
    NOSQL = "firestore"  # Firestore (NoSQL)
    SQL = "postgresql"   # PostgreSQL (リレーショナル)

class DataCategory(Enum):
    """データカテゴリを定義する列挙型"""
    # 構造化データ (PostgreSQL推奨)
    STRUCTURED = "structured"             # 構造化データ全般
    TRANSACTIONAL = "transactional"       # トランザクション要件の高いデータ
    USER_MASTER = "user_master"           # ユーザーマスタ
    COMPANY_MASTER = "company_master"     # 企業マスタ
    EMPLOYEE_MASTER = "employee_master"   # 従業員マスタ
    FINANCIAL = "financial"               # 損益計算書データ
    BILLING = "billing"                   # 請求情報
    AUDIT_LOG = "audit_log"               # 監査ログ
    # スケーラブルなデータ (Firestore推奨)
    REALTIME = "realtime"                 # リアルタイムデータ
    SCALABLE = "scalable"                 # スケーラブルなデータ
    CHAT = "chat"                         # チャットメッセージ
    ANALYTICS_CACHE = "analytics_cache"   # 分析結果のキャッシュ
    USER_SESSION = "user_session"         # ユーザーセッション情報
    SURVEY = "survey"                     # アンケート回答データ
    TREATMENT = "treatment"               # 施術記録
    REPORT = "report"                     # 分析レポート

class DatabaseManager:
    """
    ハイブリッドデータベース管理クラス
    FirestoreとPostgreSQLを適切に使い分けるための機能を提供します。
    データの種類に応じて最適なデータベースを自動選択します。
    """
    _instance = None
    _firestore_client = None
    _sql_engine = None
    _sql_session = None

    def __new__(cls):
        """シングルトンパターンの実装"""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._initialize_databases()
        return cls._instance

    @classmethod
    def _initialize_databases(cls):
        """データベース接続の初期化"""
        # Firestore初期化
        try:
            if not firebase_admin._apps:
                # Firebase Adminが未初期化の場合は初期化
                # 環境変数からクレデンシャル情報を取得
                cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                # 相対パスを絶対パスに変換
                if cred_path and not os.path.isabs(cred_path):
                    # Dockerコンテナ内の絶対パスに変換
                    cred_path = os.path.join('/app', cred_path) if os.path.exists('/app') else os.path.abspath(cred_path)
                    logger.info(f"認証情報の絶対パス: {cred_path}")

                if cred_path and os.path.exists(cred_path):
                    logger.info(f"Firebase認証情報ファイルが見つかりました: {cred_path}")
                    # 認証情報を読み込み
                    try:
                        # JSONファイルの内容を確認
                        with open(cred_path, 'r') as f:
                            cred_content = f.read()
                            logger.info(f"認証情報ファイルのサイズ: {len(cred_content)} バイト")
                            import json
                            try:
                                cred_dict = json.loads(cred_content)
                                required_keys = ["type", "project_id", "private_key_id", "private_key"]
                                missing_keys = [key for key in required_keys if key not in cred_dict]
                                if missing_keys:
                                    logger.error(f"認証情報ファイルに必要なキーがありません: {missing_keys}")
                                else:
                                    logger.info("認証情報ファイルの形式は正常です")
                                    cred = credentials.Certificate(cred_path)
                                    initialize_app(cred)
                            except json.JSONDecodeError as e:
                                logger.error(f"認証情報ファイルのJSONパースエラー: {str(e)}")
                    except Exception as e:
                        logger.error(f"認証情報ファイルの読み込みエラー: {str(e)}")
                        # 開発環境の場合はデフォルト認証情報を使用
                        if os.getenv("ENVIRONMENT") == "development":
                            logger.warning("開発環境では最低限の情報を使用してFirebaseを初期化します")
                            try:
                                initialize_app()
                            except Exception as e2:
                                logger.error(f"デフォルト認証情報でのFirebase初期化エラー: {str(e2)}")
                # 設定から直接認証情報を取得
                elif getattr(current_database_config, 'FIREBASE_ADMIN_CONFIG', None):
                    logger.info("設定から直接Firebase認証情報を使用します")
                    try:
                        cred = credentials.Certificate(current_database_config.FIREBASE_ADMIN_CONFIG)
                        initialize_app(cred)
                    except Exception as e:
                        logger.error(f"設定からの認証情報使用エラー: {str(e)}")
                else:
                    # 環境変数から直接JSONを取得
                    firebase_creds_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
                    if firebase_creds_json:
                        try:
                            import json
                            cred_dict = json.loads(firebase_creds_json)
                            logger.info("環境変数からFirebase認証情報を読み込みました")
                            cred = credentials.Certificate(cred_dict)
                            initialize_app(cred)
                        except Exception as e:
                            logger.error(f"環境変数からのFirebase認証情報の読み込みに失敗: {str(e)}")

                    # 開発環境ではデフォルト認証を試みる
                    elif os.getenv("ENVIRONMENT") == "development":
                        logger.warning("開発環境ではデフォルト認証情報を使用します")
                        try:
                            initialize_app()
                        except Exception as e:
                            logger.error(f"Firebase初期化に失敗しました: {str(e)}")
                            logger.warning("開発環境ではこのエラーを無視します。")
                    else:
                        logger.error("Firebase認証情報が見つかりません。アプリケーションは正しく動作しない可能性があります。")
            cls._firestore_client = firestore.client()
            logger.info("Firestoreクライアントの取得に成功しました。")
        except Exception as e:
            logger.error(f"Firestore初期化エラー: {str(e)}")
            cls._firestore_client = None
            if os.getenv("ENVIRONMENT") != "development":
                raise

        # PostgreSQL初期化
        try:
            database_url = current_database_config.DATABASE_URL
            cls._sql_engine = create_engine(database_url)
            cls._sql_session = sessionmaker(autocommit=False, autoflush=False, bind=cls._sql_engine)
            logger.info("PostgreSQLの初期化に成功しました。")
        except Exception as e:
            logger.error(f"PostgreSQL初期化エラー: {str(e)}")
            cls._sql_engine = None
            cls._sql_session = None

    @classmethod
    def get_firestore_client(cls) -> firestore.Client:
        """
        Firestoreクライアントを取得します
        Returns:
            firestore.Client: Firestoreクライアントインスタンス
        """
        if cls._instance is None:
            cls._instance = DatabaseManager()
        return cls._firestore_client

    @classmethod
    @contextmanager
    def get_sql_session(cls) -> Generator[Session, None, None]:
        """
        PostgreSQLセッションを取得します
        Yields:
            Session: SQLAlchemyセッションインスタンス
        Example:
            with DatabaseManager.get_sql_session() as session:
                users = session.query(User).all()
        """
        if cls._instance is None:
            cls._instance = DatabaseManager()
        session = cls._sql_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()

    @classmethod
    def get_db_by_type(cls, db_type: DatabaseType) -> Any:
        """
        指定されたタイプのデータベース接続を取得
        Args:
            db_type: データベースタイプ（NOSQL または SQL）
        Returns:
            データベース接続オブジェクト
        """
        if cls._instance is None:
            cls._instance = DatabaseManager()
        if db_type == DatabaseType.NOSQL:
            return cls._firestore_client
        elif db_type == DatabaseType.SQL:
            return cls._sql_session
        else:
            raise ValueError(f"不明なデータベースタイプ: {db_type}")

    @classmethod
    def get_db_for_data_category(cls, category: DataCategory) -> Any:
        """
        データカテゴリに基づいて適切なDBを返す
        Args:
            category: DataCategoryの列挙値
        Returns:
            適切なデータベース接続
        """
        if cls._instance is None:
            cls._instance = DatabaseManager()
        # 構造化データはPostgreSQLを使用
        sql_categories = [
            DataCategory.STRUCTURED,
            DataCategory.TRANSACTIONAL,
            DataCategory.USER_MASTER,
            DataCategory.COMPANY_MASTER,
            DataCategory.EMPLOYEE_MASTER,
            DataCategory.FINANCIAL,
            DataCategory.BILLING,
            DataCategory.AUDIT_LOG
        ]
        # リアルタイム/スケーラブルデータはFirestoreを使用
        nosql_categories = [
            DataCategory.REALTIME,
            DataCategory.SCALABLE,
            DataCategory.CHAT,
            DataCategory.ANALYTICS_CACHE,
            DataCategory.USER_SESSION,
            DataCategory.SURVEY,
            DataCategory.TREATMENT,
            DataCategory.REPORT
        ]
        if category in sql_categories:
            return cls.get_db_by_type(DatabaseType.SQL)
        elif category in nosql_categories:
            return cls.get_db_by_type(DatabaseType.NOSQL)
        else:
            # デフォルトはFirestore
            return cls.get_db_by_type(DatabaseType.NOSQL)

    @classmethod
    def get_collection_name(cls, category: DataCategory) -> str:
        """
        データカテゴリに対応するコレクション名を取得
        Args:
            category: DataCategoryの列挙値
        Returns:
            str: コレクション名
        """
        # カテゴリをコレクション名にマッピング
        collection_map = {
            DataCategory.USER_MASTER: current_database_config.FIRESTORE_COLLECTIONS["USERS"],
            DataCategory.SURVEY: current_database_config.FIRESTORE_COLLECTIONS["CONSULTATIONS"],
            DataCategory.TREATMENT: current_database_config.FIRESTORE_COLLECTIONS["TREATMENTS"],
            DataCategory.ANALYTICS_CACHE: current_database_config.FIRESTORE_COLLECTIONS["ANALYTICS"],
        }
        return collection_map.get(category, category.value)

# SQLAlchemyのベースクラス
Base = declarative_base()

# 外部から使用するためのヘルパー関数
def get_firestore_client() -> firestore.Client:
    """Firestoreクライアントを取得する簡易関数"""
    return DatabaseManager.get_firestore_client()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """PostgreSQLセッションを取得する簡易関数"""
    with DatabaseManager.get_sql_session() as session:
        yield session

def get_db_for_category(category: Union[str, DataCategory]) -> Any:
    """
    データカテゴリに基づいて適切なDBを返す簡易関数
    Args:
        category: カテゴリ名または DataCategory 列挙値
    Returns:
        適切なデータベース接続
    """
    if isinstance(category, str):
        try:
            category = DataCategory(category)
        except ValueError:
            # 不明なカテゴリの場合はデフォルトとしてFirestoreを使用
            return get_firestore_client()
    return DatabaseManager.get_db_for_data_category(category)

# migration.pyで使用される関数を追加
async def get_collection_data(collection_name: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Firestoreコレクションからデータを取得する関数

    Args:
        collection_name: 取得するコレクション名
        **kwargs: 追加のフィルタリング条件

    Returns:
        ドキュメントデータのリスト
    """
    # Firestoreクライアントを取得
    db = get_firestore_client()
    if not db:
        logger.error(f"Firestoreクライアントを取得できませんでした")
        return []

    collection_ref = db.collection(collection_name)

    # クエリ条件があれば適用
    query = collection_ref
    for key, value in kwargs.items():
        query = query.where(key, '==', value)

    # ドキュメントを取得
    try:
        docs = query.stream()
        results = []
        for doc in docs:
            data = doc.to_dict()
            # ドキュメントIDを追加
            if 'id' not in data:
                data['id'] = doc.id
            results.append(data)
        return results
    except Exception as e:
        logger.error(f"コレクション {collection_name} の取得中にエラーが発生しました: {e}")
        return []

def init_db():
    """
    全てのデータベースを初期化します。
    アプリケーション起動時に呼び出されます。
    """
    # PostgresSQL初期化
    try:
        from .models_sql import Base as SQLModels
        engine = DatabaseManager()._sql_engine
        if engine:
            SQLModels.metadata.create_all(bind=engine)
            logger.info("SQLデータベースのスキーマが初期化されました")
    except ImportError:
        logger.warning("models_sqlモジュールが見つかりません。SQLテーブル初期化をスキップします。")
    except Exception as e:
        logger.error(f"SQLデータベース初期化エラー: {str(e)}")

    # Firestore初期化 (必要に応じて追加のセットアップを行う)
    client = DatabaseManager.get_firestore_client()
    if client:
        logger.info("Firestoreクライアントが初期化されました")
    logger.info("データベース初期化完了")

# FastAPIのDependsで使用するための関数
def get_db():
    """
    FastAPIのDependsで使用するためのデータベース依存関数

    Example:
        @app.get("/items/")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = DatabaseManager._sql_session()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()