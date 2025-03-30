# -*- coding: utf-8 -*-
"""
データベース設定モジュール
Startup Wellness データ分析システムのデータベース接続情報を定義します。
GCP/Firebase環境に最適化されています。
"""
import os
from typing import Dict, List, Optional
from google.cloud import secretmanager
from google.auth import default
from dotenv import load_dotenv

# プロジェクトルートへのパスを取得
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# ルートの.envファイルのパスを設定
ENV_PATH = os.path.join(ROOT_DIR, '.env')

# 開発環境の場合のみ .env を読み込み
if os.getenv("ENVIRONMENT") != "production":
    # 優先的にルートの.envファイルを読み込む
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)
    else:
        # ルートの.envが見つからない場合は現在のディレクトリの.envを試す
        load_dotenv()

class DatabaseConfig:
    """データベース設定基本クラス"""
    # GCP Project 設定
    PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    REGION = os.getenv("GCP_REGION", "asia-northeast1")

    @staticmethod
    def get_secret(secret_id: str) -> str:
        """
        Secret Manager から機密情報を取得
        開発環境では環境変数から読み込み
        """
        # 環境変数に設定されている場合は、その値を優先して使用
        env_value = os.getenv(secret_id)
        if env_value:
            return env_value

        # 環境変数に設定されていない場合、Secret Managerから取得を試みる
        try:
            credentials, project = default()
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{DatabaseConfig.PROJECT_ID}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception:
            # 開発環境用のフォールバック
            return ""

    # 認証設定
    SECRET_KEY = get_secret.__func__("JWT_SECRET_KEY")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30

    # Google Forms設定
    FORMS_CONFIG: Dict[str, Optional[str]] = {
        "INITIAL_CONSULTATION": os.getenv("INITIAL_CONSULTATION_FORM_ID", None),
        "LATEST_CONSULTATION": os.getenv("LATEST_CONSULTATION_FORM_ID", None),
        "TREATMENT_EFFECT": os.getenv("TREATMENT_EFFECT_FORM_ID", None)
    }

    # Google Cloud Storage設定
    BUCKET_NAME = f"{PROJECT_ID}-storage" if PROJECT_ID else "local-storage"
    UPLOAD_FOLDER = "uploads"

    # Firebase Admin SDK設定
    FIREBASE_ADMIN_CREDENTIALS = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        os.path.join(ROOT_DIR, 'credentials', 'firebase-admin-sdk.json')
    )

    # Firestore設定
    FIRESTORE_COLLECTIONS = {
        "USERS": "users",
        "CONSULTATIONS": "consultations",
        "TREATMENTS": "treatments",
        "ANALYTICS": "analytics"
    }

    # Firebase Client設定
    FIREBASE_CONFIG = {
        "apiKey": os.getenv("FIREBASE_API_KEY", ""),
        "authDomain": f"{os.getenv('FIREBASE_PROJECT_ID', PROJECT_ID)}.firebaseapp.com",
        "projectId": os.getenv("FIREBASE_PROJECT_ID", PROJECT_ID),
        "storageBucket": f"{os.getenv('FIREBASE_PROJECT_ID', PROJECT_ID)}.appspot.com",
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", ""),
        "appId": os.getenv("FIREBASE_APP_ID", ""),
        "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID", "")
    }

    # Vertex AI (Gemini) 設定
    VERTEX_AI_LOCATION = REGION
    VERTEX_AI_MODEL_NAME = "gemini-pro"
    AI_MODELS: List[str] = [
        "gemini-pro",
        "text-bison",
        "chat-bison",
    ]

    # PostgreSQL接続設定
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/startup_wellness"
    )

    # フォームタイプの定義
    FORM_TYPES = {
        "INITIAL": "initial_consultation",
        "LATEST": "latest_consultation",
        "TREATMENT": "treatment_effect"
    }

class DevelopmentDatabaseConfig(DatabaseConfig):
    """開発環境用データベース設定"""
    DEBUG = True
    FIRESTORE_EMULATOR_HOST = os.getenv("FIRESTORE_EMULATOR_HOST", "localhost:8080")

class ProductionDatabaseConfig(DatabaseConfig):
    """本番環境用データベース設定"""
    DEBUG = False

class TestingDatabaseConfig(DatabaseConfig):
    """テスト環境用データベース設定"""
    TESTING = True
    DEBUG = True
    FIRESTORE_EMULATOR_HOST = "localhost:8080"
    DATABASE_URL = "postgresql://postgres:password@localhost:5432/test_wellness"

# 環境に応じた設定を選択
database_config_map = {
    "development": DevelopmentDatabaseConfig,
    "production": ProductionDatabaseConfig,
    "testing": TestingDatabaseConfig
}

# 現在の環境の設定を取得
current_database_config = database_config_map[os.getenv("ENVIRONMENT", "development")]()

# 旧コードとの互換性のため
Config = DatabaseConfig
current_config = current_database_config