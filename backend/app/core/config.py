from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """アプリケーション設定"""
    # アプリケーション設定
    PROJECT_NAME: str = "Startup Wellness API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Firebase設定
    FIREBASE_CREDENTIALS_PATH: str = "path/to/your/serviceAccount.json"

    # CORS設定
    CORS_ORIGINS: List[str] = ["*"]

    # ログ設定
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()