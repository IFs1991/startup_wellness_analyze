from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from firebase_admin import auth as firebase_auth
from typing import Dict, Any
import requests
import json

from auth import auth_manager, get_current_user
from schemas import UserCreate, UserUpdate, Token

router = APIRouter(prefix="/api/users", tags=["users"])

FIREBASE_WEB_API_KEY = "your-web-api-key"  # 環境変数から取得する必要があります
FIREBASE_AUTH_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate) -> Dict[str, Any]:
    """新規ユーザーを登録します。"""
    try:
        # 既存ユーザーのチェック
        try:
            existing_user = firebase_auth.get_user_by_email(user.email)
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
        except firebase_auth.UserNotFoundError:
            pass

        # ユーザーを作成
        user_record = await auth_manager.create_user(user)

        # IDトークンを生成
        custom_token = firebase_auth.create_custom_token(user_record.uid)

        # カスタムトークンをIDトークンに交換
        exchange_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key={FIREBASE_WEB_API_KEY}"
        exchange_data = {
            "token": custom_token.decode(),
            "returnSecureToken": True
        }
        response = requests.post(exchange_url, json=exchange_data)
        id_token = response.json().get("idToken")

        return {
            "message": "User created successfully",
            "user_id": user_record.uid,
            "access_token": id_token,
            "token_type": "bearer"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login")
async def login_user(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    """ユーザーログイン処理を行います。"""
    try:
        # Firebase Auth REST APIを使用してログイン
        login_data = {
            "email": form_data.username,
            "password": form_data.password,
            "returnSecureToken": True
        }
        response = requests.post(FIREBASE_AUTH_URL, json=login_data)

        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        data = response.json()
        return {
            "access_token": data["idToken"],
            "token_type": "bearer"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.get("/me")
async def get_current_user_info(current_user = Depends(get_current_user)) -> Dict[str, Any]:
    """現在のユーザー情報を取得します。"""
    return {
        "uid": current_user.uid,
        "email": current_user.email,
        "display_name": current_user.display_name
    }

@router.put("/me")
async def update_user_info(
    user_update: UserUpdate,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """ユーザー情報を更新します。"""
    try:
        update_data = {}
        if user_update.email:
            update_data["email"] = user_update.email
        if user_update.username:
            update_data["display_name"] = user_update.username

        if update_data:
            updated_user = firebase_auth.update_user(
                current_user.uid,
                **update_data
            )
            return {
                "message": "User updated successfully",
                "user": {
                    "uid": updated_user.uid,
                    "email": updated_user.email,
                    "display_name": updated_user.display_name
                }
            }
        return {"message": "No updates provided"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/change-password")
async def change_password(
    old_password: str,
    new_password: str,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """ユーザーのパスワードを変更します。"""
    try:
        # Firebase Auth REST APIを使用してパスワードを変更
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:update?key={FIREBASE_WEB_API_KEY}"
        data = {
            "idToken": old_password,  # 現在のセッションのIDトークン
            "password": new_password,
            "returnSecureToken": True
        }
        response = requests.post(url, json=data)

        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to change password"
            )

        return {"message": "Password changed successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )