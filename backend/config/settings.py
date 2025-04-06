# -*- coding: utf-8 -*-
"""
アプリケーション設定モジュール
環境変数から設定を読み込み、適切なデフォルト値を提供します。
"""
import os
import json
import sys
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# backendディレクトリへのパスを取得
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# backend/.envファイルのパスを設定
ENV_PATH = os.path.join(BACKEND_DIR, '.env')

# 開発環境の場合のみ .env ファイルを読み込み
if os.getenv("ENVIRONMENT") != "production":
    # backend/.envファイルを読み込む
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)
    else:
        # ENVファイルが見つからない場合はログ出力
        print(f"Warning: .env file not found at {ENV_PATH}")

# 環境に応じた設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_DOCKER = os.path.exists('/.dockerenv')

# 環境設定
ENV = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# 基本設定
APP_ENV = os.getenv("APP_ENV", "production")
DEV_MODE = os.getenv("DEV_MODE", "true").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# Firebase設定
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "startupwellnessanalyze-445505")

# Google Cloud設定
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "startupwellnessanalyze-445505")
GCP_REGION = os.getenv("GCP_REGION", "asia-northeast1")
BIGQUERY_DATASET_ID = os.getenv("BIGQUERY_DATASET_ID", "analysis_results")

# Firebase Admin SDK 認証情報
FIREBASE_CREDS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# Firebase Admin認証情報パス
FIREBASE_ADMIN_CREDENTIALS = FIREBASE_CREDS_PATH

# Firebase認証情報
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY", "")
FIREBASE_AUTH_DOMAIN = f"{FIREBASE_PROJECT_ID}.firebaseapp.com"
FIREBASE_STORAGE_BUCKET = f"{FIREBASE_PROJECT_ID}.appspot.com"
FIREBASE_MESSAGING_SENDER_ID = os.getenv("FIREBASE_MESSAGING_SENDER_ID", "")
FIREBASE_APP_ID = os.getenv("FIREBASE_APP_ID", "")
FIREBASE_MEASUREMENT_ID = os.getenv("FIREBASE_MEASUREMENT_ID", "")

# Firebase Admin認証情報をファイルから読み込む
FIREBASE_ADMIN_CONFIG = {}
if FIREBASE_CREDS_PATH and os.path.exists(FIREBASE_CREDS_PATH):
    try:
        with open(FIREBASE_CREDS_PATH, 'r') as f:
            FIREBASE_ADMIN_CONFIG = json.load(f)
    except Exception as e:
        print(f"Firebase認証情報の読み込みエラー: {e}")
        # 開発環境では最低限の情報を提供
        if ENV == "development":
            FIREBASE_ADMIN_CONFIG = {
                "type": "service_account",
                "project_id": FIREBASE_PROJECT_ID
            }

# ロギング設定
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").lower()

# データベース設定
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:1spel!2stack3win@postgres:5432/startup_wellness_analyze"
)

# Redis設定
REDIS_URL = os.getenv("REDIS_URL", "redis://startup-wellness-redis:6379/0")
REDIS_HOST = "startup-wellness-redis" if IS_DOCKER else os.getenv("REDIS_HOST", "startup-wellness-redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# CORS設定
DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://localhost:8000,http://localhost:8080,http://backend:8000"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")

# ワーカー設定
WORKERS = int(os.getenv("WORKERS", "4"))
BACKLOG = int(os.getenv("BACKLOG", "2048"))
KEEP_ALIVE = int(os.getenv("KEEP_ALIVE", "5"))

# 割引設定
DISCOUNT_HOURS_START = int(os.getenv("DISCOUNT_HOURS_START", "22"))
DISCOUNT_HOURS_END = int(os.getenv("DISCOUNT_HOURS_END", "8"))
WEEKEND_DISCOUNT = os.getenv("WEEKEND_DISCOUNT", "true").lower() == "true"

# モニタリング設定
ENABLE_MEMORY_PROFILING = os.getenv("ENABLE_MEMORY_PROFILING", "false").lower() == "true"
ENABLE_PSUTIL_MONITORING = os.getenv("ENABLE_PSUTIL_MONITORING", "true").lower() == "true"

# Firestoreコレクション名マッピング
FIRESTORE_COLLECTIONS = {
    "USERS": "users",
    "STARTUPS": "startups",
    "CONSULTATIONS": "consultations",
    "TREATMENTS": "treatments",
    "ANALYTICS": "analytics",
    "FINANCIAL_DATA": "financial_data",
    "NOTES": "notes",
    "VAS_DATA": "vas_data"
}