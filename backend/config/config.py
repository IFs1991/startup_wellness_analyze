from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from functools import lru_cache
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# .envファイルのパスを指定して読み込み
env_path = Path(__file__).parents[2] / '.env'
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    # アプリケーション設定
    APP_NAME: str = "startup-wellness"  # デフォルト値を設定
    PROJECT_NAME: str = "Startup Wellness Data Analysis System"  # デフォルト値を設定
    VERSION: str = os.getenv("VERSION", "0.1.0")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 環境変数が設定されている場合は、それを優先して使用
        self.APP_NAME = os.getenv("APP_NAME", self.APP_NAME)
        self.PROJECT_NAME = os.getenv("PROJECT_NAME", self.PROJECT_NAME)

    # ログ設定
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # CORS設定
    CORS_ORIGINS: List[str] = json.loads(os.getenv("CORS_ORIGINS", '["http://localhost:3000"]'))

    # PostgreSQL基本設定
    DB_ENGINE: str = os.getenv("DB_ENGINE", "postgresql")
    DB_NAME: str = os.getenv("DB_NAME")
    DB_USER: str = os.getenv("DB_USER")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD")
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
        required_vars = ["DB_NAME", "DB_USER", "DB_PASSWORD"]
        missing_vars = [var for var in required_vars if not getattr(self, var)]
        if missing_vars:
            raise ValueError(f"Required environment variables not set: {', '.join(missing_vars)}")
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