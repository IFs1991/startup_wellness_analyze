# -*- coding: utf-8 -*-
"""
認証 API ルーター
Firebase AuthenticationとCloud Firestoreを使用した認証機能を提供します。
"""
# 1. APIRouterのインポート
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from typing import Optional, Dict
from pydantic import BaseModel, EmailStr
from firebase_admin import auth
from datetime import datetime
from service.firestore.client import FirestoreService

# 2. routerオブジェクトの定義
router = APIRouter(
    prefix="/auth",
    tags=["auth"],
    responses={404: {"description": "Not found"}}
)

# 3. 必要なサービスの初期化
firestore_service = FirestoreService()

# データモデルの定義
class Token(BaseModel):
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    display_name: Optional[str] = None

class UserResponse(BaseModel):
    uid: str
    email: str
    display_name: Optional[str]
    created_at: datetime
    last_login: Optional[datetime]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Firebaseトークンを検証してユーザー情報を取得"""
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/register", response_model=Dict)
async def register_user(user: UserCreate):
    """新規ユーザー登録"""
    try:
        # Firebaseユーザーを作成
        user_record = auth.create_user(
            email=user.email,
            password=user.password,
            display_name=user.display_name or user.email.split('@')[0]
        )

        # ユーザーデータを準備
        user_data = {
            'uid': user_record.uid,
            'email': user.email,
            'display_name': user_record.display_name,
            'is_active': True,
            'profile_completed': False
        }

        # FirestoreServiceを使用してユーザーデータを保存
        await firestore_service.save_results(
            results=[user_data],
            collection_name='users'
        )

        return {
            "message": "User created successfully",
            "uid": user_record.uid,
            "email": user.email
        }
    except auth.EmailAlreadyExistsError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """ユーザー認証とIDトークン発行"""
    try:
        # Firebase Auth で認証
        user = auth.get_user_by_email(form_data.username)
        custom_token = auth.create_custom_token(user.uid)

        # ユーザーの最終ログイン時刻を更新
        update_data = {
            'last_login': datetime.now(),
            'is_active': True
        }

        conditions = [
            {'field': 'uid', 'operator': '==', 'value': user.uid}
        ]

        # FirestoreServiceを使用してユーザーデータを更新
        await firestore_service.save_results(
            results=[update_data],
            collection_name='users'
        )

        return {
            "access_token": custom_token.decode('utf-8'),
            "token_type": "bearer"
        }
    except auth.UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """ユーザーログアウト処理"""
    try:
        logout_data = {
            'last_logout': datetime.now(),
            'is_active': False
        }

        conditions = [
            {'field': 'uid', 'operator': '==', 'value': current_user['uid']}
        ]

        # FirestoreServiceを使用してログアウト情報を更新
        await firestore_service.save_results(
            results=[logout_data],
            collection_name='users'
        )

        # Firebase セッションを無効化
        auth.revoke_refresh_tokens(current_user['uid'])

        return {"message": "Successfully logged out"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """現在のユーザー情報を取得"""
    try:
        conditions = [
            {'field': 'uid', 'operator': '==', 'value': current_user['uid']}
        ]

        # FirestoreServiceを使用してユーザー情報を取得
        user_data = await firestore_service.fetch_documents(
            collection_name='users',
            conditions=conditions,
            limit=1
        )

        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )

        return UserResponse(**user_data[0])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.put("/me", response_model=UserResponse)
async def update_user_profile(
    profile_update: dict,
    current_user: dict = Depends(get_current_user)
):
    """ユーザープロファイルの更新"""
    try:
        # 更新可能なフィールドをフィルタリング
        allowed_fields = {'display_name', 'profile_completed'}
        update_data = {
            k: v for k, v in profile_update.items()
            if k in allowed_fields
        }

        if update_data:
            update_data['updated_at'] = datetime.now()
            update_data['uid'] = current_user['uid']

            # FirestoreServiceを使用してプロファイルを更新
            await firestore_service.save_results(
                results=[update_data],
                collection_name='users'
            )

        # 更新後のユーザー情報を取得
        conditions = [
            {'field': 'uid', 'operator': '==', 'value': current_user['uid']}
        ]
        updated_user = await firestore_service.fetch_documents(
            collection_name='users',
            conditions=conditions,
            limit=1
        )

        return UserResponse(**updated_user[0])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )