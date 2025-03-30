"""
セキュリティ設定モジュール
システム全体のセキュリティ関連設定を一元管理します
"""
import os
from typing import Dict, Any, List, Optional
from passlib.context import CryptContext

# JWT設定
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-for-development-only")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7

# セッション設定
MAX_SESSIONS_PER_USER = 5
SESSION_TIMEOUT_MINUTES = 60
IDLE_TIMEOUT_MINUTES = 15

# パスワードポリシー
PASSWORD_MIN_LENGTH = 8
PASSWORD_REQUIRE_UPPERCASE = True
PASSWORD_REQUIRE_LOWERCASE = True
PASSWORD_REQUIRE_DIGIT = True
PASSWORD_REQUIRE_SPECIAL = True
PASSWORD_EXPIRY_DAYS = 90
PASSWORD_HISTORY_SIZE = 5

# レート制限設定
RATE_LIMIT_DEFAULT = "60/minute"
RATE_LIMIT_LOGIN = "5/minute"
RATE_LIMIT_REGISTER = "3/5minutes"
RATE_LIMIT_PASSWORD_RESET = "3/hour"
RATE_LIMIT_MFA = "5/5minutes"

# MFA設定
MFA_SETUP_EXPIRY_MINUTES = 10
MFA_TOKEN_EXPIRY_MINUTES = 5

# データ保持ポリシー
AUDIT_LOG_RETENTION_DAYS = 365
USER_DATA_RETENTION_DAYS = 730
INACTIVE_ACCOUNT_DAYS = 730

# コンプライアンス設定
COMPLIANCE_LOG_FORMAT = "json"
COMPLIANCE_LOG_PATH = "logs/compliance"

# パスワードハッシュコンテキスト
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_secret_key() -> str:
    """アプリケーションのシークレットキーを取得します"""
    return os.getenv("APP_SECRET_KEY", JWT_SECRET_KEY)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """パスワードを検証します"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """パスワードをハッシュ化します"""
    return pwd_context.hash(password)