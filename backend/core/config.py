"""
アプリケーション設定モジュール
環境変数や設定ファイルから設定を読み込みます。
"""
import os
from pathlib import Path
from functools import lru_cache
from typing import List, Optional, Dict, Any, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

class Settings(BaseSettings):
    """アプリケーション設定"""
    # 基本設定
    app_name: str = "スタートアップウェルネス分析プラットフォーム"
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")

    # データベース設定
    db_host: str = Field(default="localhost", env="DB_HOST")
    db_port: int = Field(default=5432, env="DB_PORT")
    db_name: str = Field(default="wellness_db", env="DB_NAME")
    db_user: str = Field(default="postgres", env="DB_USER")
    db_password: str = Field(default="postgres", env="DB_PASSWORD")

    # フロントエンド設定
    frontend_url: str = Field(default="http://localhost:3000", env="FRONTEND_URL")

    # 認証設定
    jwt_secret_key: str = Field(default="your-secret-key", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")

    # 生成AI設定
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    gemini_cache_enabled: bool = Field(default=True, env="GEMINI_CACHE_ENABLED")
    gemini_cache_ttl: int = Field(default=86400, env="GEMINI_CACHE_TTL")  # 1日（秒）

    # レポート設定
    report_cache_enabled: bool = Field(default=True, env="REPORT_CACHE_ENABLED")
    report_cache_ttl: int = Field(default=86400, env="REPORT_CACHE_TTL")  # 1日（秒）

    # ストレージ設定
    storage_dir: Path = Field(default=Path("./storage"))
    reports_dir: Path = Field(default=Path("./storage/reports"))
    cache_dir: Path = Field(default=Path("./storage/cache"))

    # ログ設定
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")

    # CORS設定
    cors_origins: List[str] = Field(default=["http://localhost:3000"], env="CORS_ORIGINS")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """CORS_ORIGINSをカンマ区切りで解析"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("storage_dir", "reports_dir", "cache_dir")
    @classmethod
    def create_directory(cls, v):
        """必要なディレクトリを作成"""
        v.mkdir(parents=True, exist_ok=True)
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

@lru_cache()
def get_settings() -> Settings:
    """設定シングルトンを取得"""
    return Settings()