"""
設定ファイル
Startup Wellness データ分析システムの設定情報を定義します。
GCP/Firebase環境に最適化されています。
"""
import os
from typing import Dict, List, Optional
from google.cloud import secretmanager
from google.auth import default
from dotenv import load_dotenv

# 開発環境の場合のみ .env を読み込み
if os.getenv("ENVIRONMENT") != "production":
    load_dotenv()

class Config:
    # アプリケーション基本設定
    APP_NAME = os.getenv("APP_NAME", "startup_wellness_analyze")
    PROJECT_NAME = os.getenv("PROJECT_NAME", "Startup Wellness Data Analysis System")
    VERSION = os.getenv("VERSION", "0.1.0")

    # 環境設定
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    # GCP Project 設定
    PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    REGION = os.getenv("GCP_REGION", "asia-northeast1")

    # データベース基本設定
    DB_ENGINE = os.getenv("DB_ENGINE", "postgresql")
    DB_NAME = os.getenv("DB_NAME", "startup_wellness_analyze")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "1spel!2stack3win")
    DB_HOST = os.getenv("DB_HOST", "db")  # Docker Compose のサービス名
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_SCHEMA = os.getenv("DB_SCHEMA", "public")

    # データベース接続リトライ設定
    DB_RETRY_LIMIT = int(os.getenv("DB_RETRY_LIMIT", "5"))
    DB_RETRY_INTERVAL = int(os.getenv("DB_RETRY_INTERVAL", "10"))

    # SQLAlchemy設定
    SQLALCHEMY_CONFIG = {
        "pool_size": int(os.getenv("POSTGRES_POOL_SIZE", "5")),
        "max_overflow": int(os.getenv("POSTGRES_MAX_OVERFLOW", "10")),
        "pool_timeout": int(os.getenv("POSTGRES_POOL_TIMEOUT", "30")),
        "pool_recycle": int(os.getenv("POSTGRES_POOL_RECYCLE", "1800")),
        "pool_pre_ping": True,  # 接続前の生存確認
    }

    @property
    def DATABASE_URL(self) -> str:
        """データベースURLを生成"""
        return f"{self.DB_ENGINE}://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @staticmethod
    def get_secret(secret_id: str) -> str:
        """
        Secret Manager から機密情報を取得
        """
        try:
            credentials, project = default()
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{Config.PROJECT_ID}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception:
            # 開発環境用のフォールバック
            return os.getenv(secret_id, "")

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
        "apiKey": get_secret("FIREBASE_API_KEY"),
        "authDomain": f"{PROJECT_ID}.firebaseapp.com",
        "projectId": PROJECT_ID,
        "storageBucket": f"{PROJECT_ID}.appspot.com",
        "messagingSenderId": get_secret("FIREBASE_MESSAGING_SENDER_ID"),
        "appId": get_secret("FIREBASE_APP_ID"),
        "measurementId": get_secret("FIREBASE_MEASUREMENT_ID")
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
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    WORKERS = int(os.getenv("WORKERS", "4"))

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
    """開発環境の設定"""
    DEBUG: bool = True
    RELOAD: bool = True
    WORKERS: int = 1
    LOG_LEVEL: str = "DEBUG"

    # CORS設定
    CORS_ORIGINS: list = [
        "http://localhost:5173",  # フロントエンド開発サーバー
        "http://localhost:3000",
        "http://localhost:80",
        "http://localhost"
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list = ["*"]
    CORS_ALLOW_HEADERS: list = ["*"]

    ENVIRONMENT = "development"
    FIRESTORE_EMULATOR_HOST = os.getenv("FIRESTORE_EMULATOR_HOST", "localhost:8080")

    @property
    def DATABASE_URL(self) -> str:
        """開発環境用のデータベースURL"""
        return os.getenv("DATABASE_URL", super().DATABASE_URL)

class ProductionConfig(Config):
    DEBUG = False
    ENVIRONMENT = "production"

    @property
    def DATABASE_URL(self) -> str:
        """本番環境用のデータベースURL - Secret Managerから取得"""
        return self.get_secret("DATABASE_URL")

class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    ENVIRONMENT = "testing"
    FIRESTORE_EMULATOR_HOST = "localhost:8080"

    @property
    def DATABASE_URL(self) -> str:
        """テスト環境用のデータベースURL"""
        return "postgresql+asyncpg://postgres:test@localhost:5432/test_db"

# 環境に応じた設定を選択
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig
}

# 現在の環境の設定を取得してsettingsとしてエクスポート
settings = config[os.getenv("ENVIRONMENT", "development")]()