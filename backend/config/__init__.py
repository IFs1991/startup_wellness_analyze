# /app/backend/config/__init__.py
from .settings import *  # settings.pyからすべての設定をインポート

# 全設定を含む辞書をcurrent_configとして定義
current_config = {
    "DEBUG": DEBUG,
    "SECRET_KEY": SECRET_KEY,
    "ACCESS_TOKEN_EXPIRE_MINUTES": ACCESS_TOKEN_EXPIRE_MINUTES,
    "FIREBASE_PROJECT_ID": FIREBASE_PROJECT_ID,
    "GCP_PROJECT_ID": GCP_PROJECT_ID,
    "BIGQUERY_DATASET_ID": BIGQUERY_DATASET_ID,
    "LOG_LEVEL": LOG_LEVEL,
    "CORS_ORIGINS": CORS_ORIGINS
}