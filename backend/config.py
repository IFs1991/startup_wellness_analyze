# -*- coding: utf-8 -*-
"""
設定ファイル
Startup Wellness データ分析システムの設定情報を定義します。
GCP/Firebase環境に最適化されています。
"""
import os
import sys
from typing import Dict, List, Optional
from google.cloud import secretmanager
from google.auth import default
from dotenv import load_dotenv

# プロジェクトルートへのパスを取得
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# backend/.envファイルのパスを設定
ENV_PATH = os.path.join(ROOT_DIR, '.env')

# 開発環境の場合のみ .env を読み込み
if os.getenv("ENVIRONMENT") != "production":
    # backend/.envファイルを読み込む
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)
    else:
        # ENVファイルが見つからない場合はログ出力
        print(f"Warning: .env file not found at {ENV_PATH}")

class Config:
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
            name = f"projects/{Config.PROJECT_ID}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception:
            # 開発環境用のフォールバック
            return ""

    # 認証設定
    SECRET_KEY = get_secret("JWT_SECRET_KEY")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30

    # Google Forms設定
    FORMS_CONFIG: Dict[str, Optional[str]] = {
        "INITIAL_CONSULTATION": os.getenv("INITIAL_CONSULTATION_FORM_ID", None),
        "LATEST_CONSULTATION": os.getenv("LATEST_CONSULTATION_FORM_ID", None),
        "TREATMENT_EFFECT": os.getenv("TREATMENT_EFFECT_FORM_ID", None)
    }

    # Google Cloud Storage設定
    BUCKET_NAME = f"{PROJECT_ID}-storage"
    UPLOAD_FOLDER = "uploads"

    # Firebase Admin SDK設定
    FIREBASE_ADMIN_CREDENTIALS = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        os.path.join(os.path.dirname(__file__), 'credentials', 'firebase-admin-sdk.json')
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

    # アプリケーション設定
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    API_V1_PREFIX = "/api/v1"
    PROJECT_NAME = "Startup Wellness Analytics"
    VERSION = "1.0.0"

    # Cloud Logging 設定
    ENABLE_CLOUD_LOGGING = os.getenv("ENVIRONMENT") == "production"

    # フォームタイプの定義
    FORM_TYPES = {
        "INITIAL": "initial_consultation",
        "LATEST": "latest_consultation",
        "TREATMENT": "treatment_effect"
    }

# 環境別の設定クラスを定義
class DevelopmentConfig(Config):
    DEBUG = True
    FIRESTORE_EMULATOR_HOST = os.getenv("FIRESTORE_EMULATOR_HOST", "localhost:8080")

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    FIRESTORE_EMULATOR_HOST = "localhost:8080"

# 環境に応じた設定を選択
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig
}

# 現在の環境の設定を取得
current_config = config[os.getenv("ENVIRONMENT", "development")]()