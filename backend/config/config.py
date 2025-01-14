from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from functools import lru_cache
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # アプリケーション設定
    APP_NAME: str = "startup-wellness"
    PROJECT_NAME: str = "Startup Wellness Data Analysis System"
    VERSION: str = "0.1.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # ログ設定
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # CORS設定
    CORS_ORIGINS: List[str] = json.loads(os.getenv("CORS_ORIGINS", '["http://localhost:3000"]'))

    # PostgreSQL基本設定
    DB_ENGINE: str = os.getenv("DB_ENGINE", "postgresql")
    DB_NAME: str = os.getenv("DB_NAME", "startup_wellness_analyze")
    DB_USER: str = os.getenv("DB_USER", "startup_wellness_user")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "1spel!2stack3win")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_SCHEMA: str = os.getenv("DB_SCHEMA", "public")

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # 追加のフィールドを許可

    @property
    def DATABASE_URL(self) -> str:
        """SQLAlchemy用のデータベースURLを生成"""
        return f"{self.DB_ENGINE}+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def SQLALCHEMY_CONFIG(self) -> Dict[str, Any]:
        """SQLAlchemy用の設定を辞書形式で返す"""
        return {
            "pool_size": int(os.getenv("POSTGRES_POOL_SIZE", "5")),
            "max_overflow": int(os.getenv("POSTGRES_MAX_OVERFLOW", "10")),
            "pool_timeout": int(os.getenv("POSTGRES_POOL_TIMEOUT", "30")),
            "pool_recycle": int(os.getenv("POSTGRES_POOL_RECYCLE", "1800")),
            "connect_args": {
                "timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "60")),
                "server_settings": {"search_path": self.DB_SCHEMA}
            }
        }

@lru_cache()
def get_settings() -> Settings:
    """設定のシングルトンインスタンスを取得"""
    return Settings()