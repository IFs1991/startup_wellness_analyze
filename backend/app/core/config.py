"""
Startup Wellness データ分析システムの設定を管理します。
"""

import os
from typing import List, Dict, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv

# 開発環境の場合のみ .env を読み込み
if os.getenv("ENVIRONMENT") != "production":
    load_dotenv()

class Settings(BaseSettings):
    """アプリケーション設定"""
    # アプリケーション基本設定
    APP_NAME: str = "startup-wellness"
    PROJECT_NAME: str = "Startup Wellness Data Analysis System"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # API設定
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    ALGORITHM: str = "HS256"

    # Firebase設定
    FIREBASE_CREDENTIALS_PATH: str = os.getenv(
        "FIREBASE_CREDENTIALS_PATH",
        os.path.join(os.path.dirname(__file__), "../../../credentials/firebase-admin-sdk.json")
    )

    # CORS設定
    CORS_ORIGINS: List[str] = ["*"]

    # ログ設定
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # データベース設定
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = 'utf-8'

@lru_cache()
def get_settings() -> Settings:
    """設定のシングルトンインスタンスを取得"""
    return Settings()

# シングルトンインスタンスを作成
settings = get_settings()

# エクスポートする変数とクラス
__all__ = ['Settings', 'settings', 'get_settings']