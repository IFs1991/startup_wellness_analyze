"""
アプリケーション設定モジュール
環境変数や設定ファイルから設定を読み込みます。
"""
import os
import json
from pathlib import Path
from functools import lru_cache
from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import yaml
from .common_logger import get_logger

# ロギングの設定
logger = get_logger(__name__)

# 環境変数
ENV_VAR_PREFIX = "WELLNESS_"
ENV_NAME = os.getenv(f"{ENV_VAR_PREFIX}ENVIRONMENT", "development")

# パス設定
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_DIR = os.path.join(BACKEND_DIR, 'config')
ENV_FILES = {
    "base": os.path.join(CONFIG_DIR, 'settings.base.env'),
    "environment": os.path.join(CONFIG_DIR, f'settings.{ENV_NAME}.env'),
    "local": os.path.join(BACKEND_DIR, '.env')
}

# 設定ファイルのロード
def load_env_files():
    """
    階層的な設定ファイルを読み込みます
    1. ベース設定 (settings.base.env)
    2. 環境固有設定 (settings.[environment].env)
    3. ローカルオーバーライド (.env)
    """
    # プロダクション環境では環境変数のみを使用
    if ENV_NAME == "production":
        logger.info("本番環境では環境変数のみを使用します")
        return

    # ベース設定ファイル（必須）
    if os.path.exists(ENV_FILES["base"]):
        load_dotenv(ENV_FILES["base"])
        logger.info(f"ベース設定ファイルを読み込みました: {ENV_FILES['base']}")
    else:
        logger.warning(f"ベース設定ファイルが見つかりません: {ENV_FILES['base']}")

    # 環境固有設定ファイル（オプション）
    if os.path.exists(ENV_FILES["environment"]):
        load_dotenv(ENV_FILES["environment"], override=True)
        logger.info(f"環境設定ファイルを読み込みました: {ENV_FILES['environment']}")
    else:
        logger.debug(f"環境設定ファイルが見つかりません: {ENV_FILES['environment']}")

    # ローカルオーバーライド（オプション）
    if os.path.exists(ENV_FILES["local"]):
        load_dotenv(ENV_FILES["local"], override=True)
        logger.info(f"ローカル設定ファイルを読み込みました: {ENV_FILES['local']}")
    else:
        logger.debug("ローカル設定ファイルが見つかりません")

# アプリケーション起動時に設定を読み込む
load_env_files()

class DatabaseSettings(BaseSettings):
    """データベース設定"""
    host: str = Field(default="localhost", env=f"{ENV_VAR_PREFIX}DB_HOST")
    port: int = Field(default=5432, env=f"{ENV_VAR_PREFIX}DB_PORT")
    name: str = Field(default="wellness_db", env=f"{ENV_VAR_PREFIX}DB_NAME")
    user: str = Field(default="postgres", env=f"{ENV_VAR_PREFIX}DB_USER")
    password: str = Field(default="postgres", env=f"{ENV_VAR_PREFIX}DB_PASSWORD")
    min_connections: int = Field(default=5, env=f"{ENV_VAR_PREFIX}DB_MIN_CONNECTIONS")
    max_connections: int = Field(default=20, env=f"{ENV_VAR_PREFIX}DB_MAX_CONNECTIONS")

    def get_connection_string(self) -> str:
        """データベース接続文字列を取得"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

class FirebaseSettings(BaseSettings):
    """Firebase設定"""
    project_id: Optional[str] = Field(default=None, env=f"{ENV_VAR_PREFIX}FIREBASE_PROJECT_ID")
    credentials_path: Optional[str] = Field(default=None, env=f"{ENV_VAR_PREFIX}FIREBASE_CREDENTIALS_PATH")
    storage_bucket: Optional[str] = Field(default=None, env=f"{ENV_VAR_PREFIX}FIREBASE_STORAGE_BUCKET")
    use_emulator: bool = Field(default=False, env=f"{ENV_VAR_PREFIX}FIREBASE_USE_EMULATOR")
    emulator_host: Optional[str] = Field(default="localhost", env=f"{ENV_VAR_PREFIX}FIREBASE_EMULATOR_HOST")
    emulator_port: Optional[int] = Field(default=8080, env=f"{ENV_VAR_PREFIX}FIREBASE_EMULATOR_PORT")

class CacheSettings(BaseSettings):
    """キャッシュ設定"""
    enabled: bool = Field(default=True, env=f"{ENV_VAR_PREFIX}CACHE_ENABLED")
    type: Literal["memory", "redis", "file"] = Field(default="memory", env=f"{ENV_VAR_PREFIX}CACHE_TYPE")
    ttl: int = Field(default=3600, env=f"{ENV_VAR_PREFIX}CACHE_TTL")  # 1時間（秒）
    redis_url: Optional[str] = Field(default=None, env=f"{ENV_VAR_PREFIX}REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env=f"{ENV_VAR_PREFIX}REDIS_PASSWORD")
    redis_db: int = Field(default=0, env=f"{ENV_VAR_PREFIX}REDIS_DB")
    file_path: Optional[Path] = Field(default=Path("./storage/cache"), env=f"{ENV_VAR_PREFIX}CACHE_FILE_PATH")

    @field_validator("file_path")
    @classmethod
    def create_directory(cls, v):
        """必要なディレクトリを作成"""
        if v:
            v.mkdir(parents=True, exist_ok=True)
        return v

class SecuritySettings(BaseSettings):
    """セキュリティ設定"""
    jwt_secret_key: str = Field(default="your-secret-key", env=f"{ENV_VAR_PREFIX}JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env=f"{ENV_VAR_PREFIX}JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, env=f"{ENV_VAR_PREFIX}JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    jwt_refresh_token_expire_days: int = Field(default=7, env=f"{ENV_VAR_PREFIX}JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    password_min_length: int = Field(default=8, env=f"{ENV_VAR_PREFIX}PASSWORD_MIN_LENGTH")
    cors_origins: List[str] = Field(default=["http://localhost:3000"], env=f"{ENV_VAR_PREFIX}CORS_ORIGINS")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """CORS_ORIGINSをカンマ区切りで解析"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

class AISettings(BaseSettings):
    """AI関連設定"""
    gemini_api_key: Optional[str] = Field(default=None, env=f"{ENV_VAR_PREFIX}GEMINI_API_KEY")
    model_cache_enabled: bool = Field(default=True, env=f"{ENV_VAR_PREFIX}MODEL_CACHE_ENABLED")
    model_cache_ttl: int = Field(default=86400, env=f"{ENV_VAR_PREFIX}MODEL_CACHE_TTL")  # 1日（秒）
    max_tokens: int = Field(default=2048, env=f"{ENV_VAR_PREFIX}MAX_TOKENS")
    temperature: float = Field(default=0.7, env=f"{ENV_VAR_PREFIX}TEMPERATURE")

class LogSettings(BaseSettings):
    """ログ設定"""
    level: str = Field(default="INFO", env=f"{ENV_VAR_PREFIX}LOG_LEVEL")
    file: Optional[str] = Field(default=None, env=f"{ENV_VAR_PREFIX}LOG_FILE")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env=f"{ENV_VAR_PREFIX}LOG_FORMAT"
    )
    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        env=f"{ENV_VAR_PREFIX}LOG_DATE_FORMAT"
    )

class StorageSettings(BaseSettings):
    """ストレージ設定"""
    base_dir: Path = Field(default=Path("./storage"), env=f"{ENV_VAR_PREFIX}STORAGE_BASE_DIR")
    reports_dir: Path = Field(default=Path("./storage/reports"), env=f"{ENV_VAR_PREFIX}REPORTS_DIR")
    uploads_dir: Path = Field(default=Path("./storage/uploads"), env=f"{ENV_VAR_PREFIX}UPLOADS_DIR")
    temp_dir: Path = Field(default=Path("./storage/temp"), env=f"{ENV_VAR_PREFIX}TEMP_DIR")

    @field_validator("base_dir", "reports_dir", "uploads_dir", "temp_dir")
    @classmethod
    def create_directory(cls, v):
        """必要なディレクトリを作成"""
        v.mkdir(parents=True, exist_ok=True)
        return v

class Settings(BaseSettings):
    """アプリケーション設定"""
    # 基本設定
    environment: str = Field(default=ENV_NAME, env=f"{ENV_VAR_PREFIX}ENVIRONMENT")
    app_name: str = Field(default="スタートアップウェルネス分析プラットフォーム", env=f"{ENV_VAR_PREFIX}APP_NAME")
    api_host: str = Field(default="0.0.0.0", env=f"{ENV_VAR_PREFIX}API_HOST")
    api_port: int = Field(default=8000, env=f"{ENV_VAR_PREFIX}API_PORT")
    debug: bool = Field(default=False, env=f"{ENV_VAR_PREFIX}DEBUG")
    version: str = Field(default="0.1.0", env=f"{ENV_VAR_PREFIX}VERSION")
    frontend_url: str = Field(default="http://localhost:3000", env=f"{ENV_VAR_PREFIX}FRONTEND_URL")

    # 機能フラグ
    enable_analytics: bool = Field(default=True, env=f"{ENV_VAR_PREFIX}ENABLE_ANALYTICS")
    enable_ai_features: bool = Field(default=True, env=f"{ENV_VAR_PREFIX}ENABLE_AI_FEATURES")
    enable_notifications: bool = Field(default=True, env=f"{ENV_VAR_PREFIX}ENABLE_NOTIFICATIONS")

    # 各サブ設定
    database: DatabaseSettings = DatabaseSettings()
    firebase: FirebaseSettings = FirebaseSettings()
    cache: CacheSettings = CacheSettings()
    security: SecuritySettings = SecuritySettings()
    ai: AISettings = AISettings()
    log: LogSettings = LogSettings()
    storage: StorageSettings = StorageSettings()

    def load_from_yaml(self, filepath: str) -> None:
        """YAMLファイルから設定を読み込む"""
        if not os.path.exists(filepath):
            logger.warning(f"YAMLファイルが見つかりません: {filepath}")
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
                for key, value in config_data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            logger.info(f"YAMLファイルから設定を読み込みました: {filepath}")
        except Exception as e:
            logger.error(f"YAMLファイルの読み込みに失敗しました: {filepath} - {str(e)}")

    def to_json(self) -> str:
        """設定をJSON文字列に変換"""
        return json.dumps(self.model_dump(), ensure_ascii=False, indent=2)

    def save_to_json(self, filepath: str) -> None:
        """設定をJSONファイルに保存"""
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(self.to_json())
            logger.info(f"設定をJSONファイルに保存しました: {filepath}")
        except Exception as e:
            logger.error(f"JSONファイルの保存に失敗しました: {filepath} - {str(e)}")

@lru_cache()
def get_settings() -> Settings:
    """設定シングルトンを取得"""
    settings = Settings()

    # 追加のYAML設定ファイル（オプション）
    yaml_config_path = os.path.join(CONFIG_DIR, f'config.{ENV_NAME}.yaml')
    if os.path.exists(yaml_config_path):
        settings.load_from_yaml(yaml_config_path)

    return settings