# -*- coding: utf-8 -*-
"""
依存関係定義
FastAPI アプリケーションの依存関係を定義します。
"""
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from firebase_admin import firestore
from contextlib import asynccontextmanager
from service.firestore.client import FirestoreService
from database.models import User
from auth import get_current_user

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# FirestoreServiceのインスタンスを作成
firestore_service = FirestoreService()

@asynccontextmanager
async def get_db():
    """
    Firestoreデータベースクライアントを取得します。
    """
    try:
        yield firestore_service
    finally:
        await firestore_service.close()

def get_current_active_user(
    current_user: User = Depends(get_current_user),
):
    """アクティブなユーザーを取得します。"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def get_current_vc_user(
    current_user: User = Depends(get_current_user)
):
    """VCユーザーを取得します"""
    if not current_user.is_vc:
        raise HTTPException(status_code=400, detail="Not a VC user")
    return current_user