import os
from dotenv import load_dotenv

# 開発環境の場合のみ .env を読み込み
if os.getenv("ENVIRONMENT") != "production":
    load_dotenv()

DEBUG = os.getenv("DEBUG", "False").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable is not set")

ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# Firebase設定
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
if not FIREBASE_PROJECT_ID:
    raise ValueError("FIREBASE_PROJECT_ID environment variable is not set")

# Google Cloud設定
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
if not GCP_PROJECT_ID:
    raise ValueError("GCP_PROJECT_ID environment variable is not set")
BIGQUERY_DATASET_ID = os.getenv("BIGQUERY_DATASET_ID", "analysis_results")

# ロギング設定
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# CORS設定
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000,http://localhost:8080").split(",")