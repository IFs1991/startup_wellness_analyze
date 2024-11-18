# -- coding: utf-8 --
"""
CRUD 操作
Firebaseに対するCRUD操作 (Create, Read, Update, Delete) を定義します。
"""
from typing import Optional, List
from firebase_admin import firestore
from . import models

db = firestore.client()

# ユーザー関連のCRUD操作
def get_user(user_id: str) -> Optional[dict]:
    """ユーザーIDをもとにユーザー情報を取得"""
    user_doc = db.collection('users').document(user_id).get()
    return user_doc.to_dict() if user_doc.exists else None

def get_user_by_username(username: str) -> Optional[dict]:
    """ユーザー名をもとにユーザー情報を取得"""
    users_ref = db.collection('users')
    query = users_ref.where('username', '==', username).limit(1)
    docs = query.stream()
    for doc in docs:
        return doc.to_dict()
    return None

def get_users(skip: int = 0, limit: int = 100) -> List[dict]:
    """全てのユーザー情報を取得"""
    users_ref = db.collection('users')
    docs = users_ref.limit(limit).offset(skip).stream()
    return [doc.to_dict() for doc in docs]

def create_user(user_data: dict) -> dict:
    """新規ユーザーを作成"""
    # パスワードのハッシュ化は別途認証サービスで処理
    user_ref = db.collection('users').document()
    user_data['id'] = user_ref.id
    user_ref.set(user_data)
    return user_data

# スタートアップ企業関連のCRUD操作
def get_startup(startup_id: str) -> Optional[dict]:
    """スタートアップIDをもとにスタートアップ情報を取得"""
    startup_doc = db.collection('startups').document(startup_id).get()
    return startup_doc.to_dict() if startup_doc.exists else None

def get_startups(skip: int = 0, limit: int = 100) -> List[dict]:
    """全てのスタートアップ情報を取得"""
    startups_ref = db.collection('startups')
    docs = startups_ref.limit(limit).offset(skip).stream()
    return [doc.to_dict() for doc in docs]

def create_startup(startup_data: dict) -> dict:
    """新規スタートアップ企業を作成"""
    startup_ref = db.collection('startups').document()
    startup_data['id'] = startup_ref.id
    startup_ref.set(startup_data)
    return startup_data

# VASデータ関連のCRUD操作
def get_vas_data(vas_id: str) -> Optional[dict]:
    """VASデータIDをもとにVASデータを取得"""
    vas_doc = db.collection('vas_data').document(vas_id).get()
    return vas_doc.to_dict() if vas_doc.exists else None

def get_vas_datas(startup_id: str, skip: int = 0, limit: int = 100) -> List[dict]:
    """スタートアップに紐づくVASデータを取得"""
    vas_ref = db.collection('vas_data')
    query = vas_ref.where('startup_id', '==', startup_id).limit(limit).offset(skip)
    return [doc.to_dict() for doc in query.stream()]

def create_vas_data(vas_data: dict) -> dict:
    """新規VASデータを作成"""
    vas_ref = db.collection('vas_data').document()
    vas_data['id'] = vas_ref.id
    vas_ref.set(vas_data)
    return vas_data

# 財務データ関連のCRUD操作
def get_financial_data(financial_id: str) -> Optional[dict]:
    """財務データIDをもとに財務データを取得"""
    financial_doc = db.collection('financial_data').document(financial_id).get()
    return financial_doc.to_dict() if financial_doc.exists else None

def get_financial_datas(startup_id: str, skip: int = 0, limit: int = 100) -> List[dict]:
    """スタートアップに紐づく財務データを取得"""
    financial_ref = db.collection('financial_data')
    query = financial_ref.where('startup_id', '==', startup_id).limit(limit).offset(skip)
    return [doc.to_dict() for doc in query.stream()]

def create_financial_data(financial_data: dict) -> dict:
    """新規財務データを作成"""
    financial_ref = db.collection('financial_data').document()
    financial_data['id'] = financial_ref.id
    financial_ref.set(financial_data)
    return financial_data

# メモ関連のCRUD操作
def get_note(note_id: str) -> Optional[dict]:
    """メモIDをもとにメモを取得"""
    note_doc = db.collection('notes').document(note_id).get()
    return note_doc.to_dict() if note_doc.exists else None

def get_notes(startup_id: str, skip: int = 0, limit: int = 100) -> List[dict]:
    """スタートアップに紐づくメモを取得"""
    notes_ref = db.collection('notes')
    query = notes_ref.where('startup_id', '==', startup_id).limit(limit).offset(skip)
    return [doc.to_dict() for doc in query.stream()]

def create_note(note_data: dict) -> dict:
    """新規メモを作成"""
    note_ref = db.collection('notes').document()
    note_data['id'] = note_ref.id
    note_ref.set(note_data)
    return note_data