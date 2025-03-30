# -*- coding: utf-8 -*-
"""
依存関係定義
FastAPI アプリケーションの依存関係を定義します。
"""
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from firebase_admin import firestore
from contextlib import asynccontextmanager
import logging

# 循環インポートを避けるため、遅延インポートを使用
# from service.firestore.client import FirestoreService
# from database.models import User
# from auth import get_current_user

# ロギングの設定
logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# FirestoreServiceのインスタンスを作成 - 遅延初期化
firestore_service = None

def get_firestore_service():
    """
    FirestoreServiceのインスタンスを遅延初期化して取得します。
    """
    global firestore_service
    if firestore_service is None:
        # 遅延インポート
        from service.firestore.client import FirestoreService
        firestore_service = FirestoreService()
        logger.info("FirestoreServiceを初期化しました")
    return firestore_service

@asynccontextmanager
async def get_db():
    """
    Firestoreデータベースクライアントを取得します。
    """
    try:
        service = get_firestore_service()
        yield service
    finally:
        service = get_firestore_service()
        await service.close()

# 遅延インポート用の関数
def get_user_model():
    """Userモデルクラスを遅延インポートして返します"""
    from backend.database.models import User
    return User

def get_current_user_func():
    """get_current_user関数を遅延インポートして返します"""
    from backend.auth import get_current_user
    return get_current_user