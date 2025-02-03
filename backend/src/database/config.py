from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import os
from dotenv import load_dotenv
import platform
from typing import Dict, Any, Optional
from enum import Enum

# .envファイルを読み込む
load_dotenv()

class EnvironmentType(str, Enum):
    """環境の種類を定義"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class AppConfig(BaseSettings):
    """アプリケーション全体の設定"""
    model_config = SettingsConfigDict(
        env_file='.env',
        case_sensitive=False,
        extra='ignore'
    )

    # アプリケーション基本設定
    PROJECT_NAME: str = Field(default="Startup Wellness Data Analysis System")
    VERSION: str = Field(default="0.1.0")
    DEBUG: bool = Field(default=False)
    ENVIRONMENT: EnvironmentType = Field(default=EnvironmentType.DEVELOPMENT)

    # APIエンドポイント設定
    API_V1_PREFIX: str = Field(default="/api/v1")
    DOCS_URL: Optional[str] = Field(default=None)
    REDOC_URL: Optional[str] = Field(default=None)

    # セキュリティ設定
    SECRET_KEY: str = Field(default=os.getenv("SECRET_KEY", "development-secret-key"))
    ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)

    # CORS設定
    CORS_ORIGINS: list = Field(default=["http://localhost:3000", "http://localhost:8000"])
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: list = Field(default=["*"])
    CORS_ALLOW_HEADERS: list = Field(default=["*"])

    @model_validator(mode='after')
    def set_docs_urls(self) -> 'AppConfig':
        """開発環境の場合のみドキュメントURLを設定"""
        if self.ENVIRONMENT != EnvironmentType.PRODUCTION:
            self.DOCS_URL = f"{self.API_V1_PREFIX}/docs"
            self.REDOC_URL = f"{self.API_V1_PREFIX}/redoc"
        return self

class FirestoreConfig(BaseSettings):
    """Firestore設定クラス"""
    project_id: str
    emulator_host: Optional[str] = None
    credentials_path: Optional[str] = None

    class Config:
        env_prefix = "FIREBASE_"
        env_file = ".env"
        env_file_encoding = "utf-8"

# シングルトンインスタンスの作成
app_config = AppConfig()

@lru_cache()
def get_app_config() -> AppConfig:
    """アプリケーション設定のシングルトンインスタンスを取得する"""
    return app_config

@lru_cache()
def get_firestore_config() -> FirestoreConfig:
    """Firestore設定を取得する

    Returns:
        FirestoreConfig: Firestore設定
    """
    return FirestoreConfig()