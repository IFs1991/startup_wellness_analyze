from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
import os
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

class PostgresConfig(BaseSettings):
    """PostgreSQL設定"""
    host: str = Field(default=os.getenv("POSTGRES_HOST", "localhost"))
    port: int = Field(default=int(os.getenv("POSTGRES_PORT", "5432")))
    database: str = Field(default=os.getenv("POSTGRES_DB", "startup_wellness"))
    username: str = Field(default=os.getenv("POSTGRES_USER", "postgres"))
    password: str = Field(default=os.getenv("POSTGRES_PASSWORD", "postgres"))
    pool_size: int = Field(default=int(os.getenv("POSTGRES_POOL_SIZE", "5")))
    max_overflow: int = Field(default=int(os.getenv("POSTGRES_MAX_OVERFLOW", "10")))
    pool_timeout: int = Field(default=int(os.getenv("POSTGRES_POOL_TIMEOUT", "30")))
    pool_recycle: int = Field(default=int(os.getenv("POSTGRES_POOL_RECYCLE", "1800")))
    echo: bool = Field(default=os.getenv("POSTGRES_ECHO", "False").lower() == "true")
    timezone: str = Field(default=os.getenv("POSTGRES_TIMEZONE", "UTC"))

    def get_database_url(self) -> str:
        """データベースURLを取得する"""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    def get_pool_settings(self) -> dict:
        """プール設定を取得する"""
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle
        }

    def get_engine_settings(self) -> dict:
        """エンジン設定を取得する"""
        return {
            "echo": self.echo,
            "connect_args": {
                "server_settings": {
                    "timezone": self.timezone
                }
            }
        }

    def get_all_settings(self) -> dict:
        """全ての設定を取得する"""
        return {
            "database_url": self.get_database_url(),
            "pool_settings": self.get_pool_settings(),
            "engine_settings": self.get_engine_settings()
        }

class PostgresTestConfig(PostgresConfig):
    """PostgreSQLテスト用設定"""
    host: str = Field(default=os.getenv("POSTGRES_TEST_HOST", "localhost"))
    port: int = Field(default=int(os.getenv("POSTGRES_TEST_PORT", "5432")))
    database: str = Field(default=os.getenv("POSTGRES_TEST_DB", "test_startup_wellness"))
    username: str = Field(default=os.getenv("POSTGRES_TEST_USER", "postgres"))
    password: str = Field(default=os.getenv("POSTGRES_TEST_PASSWORD", "postgres"))

@lru_cache()
def get_postgres_config() -> PostgresConfig:
    """PostgreSQL設定のシングルトンインスタンスを取得する"""
    return PostgresConfig()

@lru_cache()
def get_postgres_test_config() -> PostgresTestConfig:
    """PostgreSQLテスト用設定のシングルトンインスタンスを取得する"""
    return PostgresTestConfig()