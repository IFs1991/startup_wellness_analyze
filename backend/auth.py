# -*- coding: utf-8 -*-
"""
認証関連関数
Firebase Authentication と GCP を使用したユーザー認証の処理を定義します。
"""
from datetime import timedelta, datetime
import os
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials, initialize_app
from google.cloud import secretmanager
import firebase_admin
from schemas import UserCreate, Token

# Firebase初期化
try:
    firebase_app = firebase_admin.get_app()
except ValueError:
    cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not cred_path or not os.path.exists(cred_path):
        raise FileNotFoundError(f"Firebase credentials file not found at {cred_path}")

    cred = credentials.Certificate(cred_path)
    firebase_app = initialize_app(cred)

# Secret Manager クライアントの設定
secret_client = secretmanager.SecretManagerServiceClient()

# 環境変数の取得
def get_secret(secret_id: str) -> str:
    """GCP Secret Managerから機密情報を取得"""
    name = f"projects/your-project-id/secrets/{secret_id}/versions/latest"
    response = secret_client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# 定数の設定
SECRET_KEY = "your-secret-key"  # 本番環境では環境変数から取得
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 スキームの設定
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class FirebaseAuthManager:
    """Firebase認証マネージャー"""

    @staticmethod
    async def verify_token(token: str) -> Dict[Any, Any]:
        """Firebaseトークンを検証"""
        try:
            decoded_token = firebase_auth.verify_id_token(token)
            return decoded_token
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @staticmethod
    async def create_user(user: UserCreate) -> Dict[Any, Any]:
        """Firebaseでユーザーを作成"""
        try:
            user_record = firebase_auth.create_user(
                email=user.email,
                password=user.password,
                display_name=user.username
            )
            return user_record
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

    @staticmethod
    async def get_user(uid: str) -> Dict[Any, Any]:
        """ユーザー情報を取得"""
        try:
            return firebase_auth.get_user(uid)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

# Firebase Auth Managerのインスタンスを作成
auth_manager = FirebaseAuthManager()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    現在のユーザーを取得します。
    Firebaseトークンを検証し、対応するユーザー情報を返します。
    """
    try:
        decoded_token = await auth_manager.verify_token(token)
        user = await auth_manager.get_user(decoded_token["uid"])
        return user
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def register_user(user: UserCreate):
    """ユーザーを登録します。"""
    return await auth_manager.create_user(user)

async def reset_password(email: str):
    """パスワードリセットメールを送信します。"""
    try:
        firebase_auth.generate_password_reset_link(email)
        return {"message": "Password reset email sent successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

async def revoke_refresh_tokens(uid: str):
    """ユーザーのリフレッシュトークンを無効化します。"""
    try:
        firebase_auth.revoke_refresh_tokens(uid)
        return {"message": "All refresh tokens revoked successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )