DEBUG = False
SECRET_KEY = "your-secret-key-here"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Firebase設定
FIREBASE_PROJECT_ID = "your-firebase-project-id"

# Google Cloud設定
GCP_PROJECT_ID = "your-gcp-project-id"
BIGQUERY_DATASET_ID = "analysis_results"

# ロギング設定
LOG_LEVEL = "INFO"

# CORS設定
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8080"
]