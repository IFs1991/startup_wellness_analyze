# -*- coding: utf-8 -*-
"""
API 依存関係の定義
FastAPIエンドポイントで使用される依存関係を定義します。
レイヤー別に整理され、適切なスコープで依存関係を提供します。
"""

import logging
from typing import Optional, Annotated, Dict, Any
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import ValidationError

from database.models import UserModel
from service.firestore.client import get_firestore_client
from core.security_config import JWT_SECRET_KEY, JWT_ALGORITHM, verify_password, get_password_hash
from core.config import get_settings

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# OAuth2認証の依存関係
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token",
    scopes={
        "user": "一般ユーザー権限",
        "admin": "管理者権限",
        "vc": "ベンチャーキャピタリスト権限"
    }
)

# =====================
# データベース層の依存関係
# =====================

def get_db_client():
    """
    Firestoreクライアントの取得

    Returns:
        firestore.Client: Firestoreクライアント
    """
    return get_firestore_client()

# =====================
# サービス層の依存関係
# =====================

async def get_user_by_email(email: str) -> Optional[UserModel]:
    """
    メールアドレスによるユーザー取得

    Args:
        email: ユーザーのメールアドレス

    Returns:
        Optional[UserModel]: 見つかったユーザーまたはNone
    """
    try:
        db = get_db_client()
        user_ref = db.collection("users").where("email", "==", email).limit(1)
        users = user_ref.get()

        for user_doc in users:
            user_data = user_doc.to_dict()
            return UserModel.from_dict(user_data)

        return None
    except Exception as e:
        logger.error(f"ユーザー取得中にエラーが発生しました: {str(e)}")
        return None

async def authenticate_user(email: str, password: str) -> Optional[UserModel]:
    """
    ユーザー認証

    Args:
        email: ユーザーのメールアドレス
        password: ユーザーのパスワード

    Returns:
        Optional[UserModel]: 認証されたユーザーまたはNone
    """
    user = await get_user_by_email(email)
    if not user:
        return None

    if not verify_password(password, user.hashed_password):
        return None

    return user

# =====================
# セキュリティ層の依存関係
# =====================

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    アクセストークンの作成

    Args:
        data: トークンにエンコードするデータ
        expires_delta: トークンの有効期限

    Returns:
        str: 作成されたJWTトークン
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    security_settings = get_security_settings()

    encoded_jwt = jwt.encode(
        to_encode,
        security_settings.secret_key,
        algorithm=security_settings.algorithm
    )

    return encoded_jwt

async def get_current_user(
    security_scopes: SecurityScopes,
    token: Annotated[str, Depends(oauth2_scheme)]
) -> UserModel:
    """
    現在のユーザーを取得

    Args:
        security_scopes: セキュリティスコープ
        token: アクセストークン

    Returns:
        UserModel: 現在のユーザー

    Raises:
        HTTPException: 認証エラーまたは権限エラーの場合
    """
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="認証情報を検証できませんでした",
        headers={"WWW-Authenticate": authenticate_value},
    )

    security_settings = get_security_settings()

    try:
        payload = jwt.decode(
            token,
            security_settings.secret_key,
            algorithms=[security_settings.algorithm]
        )
        email: str = payload.get("sub")

        if email is None:
            raise credentials_exception

        token_scopes = payload.get("scopes", [])

    except (JWTError, ValidationError):
        logger.exception("トークンの検証に失敗しました")
        raise credentials_exception

    user = await get_user_by_email(email)

    if user is None:
        raise credentials_exception

    for scope in security_scopes.scopes:
        if scope not in token_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"権限が不足しています: {scope}が必要です",
                headers={"WWW-Authenticate": authenticate_value},
            )

    return user

# =====================
# API層の依存関係
# =====================

async def get_current_active_user(
    current_user: Annotated[UserModel, Depends(get_current_user)]
) -> UserModel:
    """
    現在のアクティブユーザーを取得

    Args:
        current_user: 現在のユーザー

    Returns:
        UserModel: アクティブなユーザー

    Raises:
        HTTPException: ユーザーが無効な場合
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="無効なユーザーアカウントです"
        )
    return current_user

async def get_current_vc_user(
    current_user: Annotated[UserModel, Depends(get_current_active_user)]
) -> UserModel:
    """
    現在のVCユーザーを取得

    Args:
        current_user: 現在のアクティブユーザー

    Returns:
        UserModel: VCユーザー

    Raises:
        HTTPException: ユーザーがVCでない場合
    """
    if not current_user.is_vc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="この操作にはVC権限が必要です"
        )
    return current_user

# =====================
# 構成の依存関係
# =====================

def get_app_settings():
    """
    アプリケーション設定の取得

    Returns:
        Settings: アプリケーション設定
    """
    return get_settings()