"""
CRUD 操作 (PostgreSQL)
SQLAlchemyを使用したCRUD操作 (Create, Read, Update, Delete) を定義します。
"""
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from . import models_sql

# ユーザー関連のCRUD操作
def get_user(db: Session, user_id: str) -> Optional[models_sql.User]:
    """ユーザーIDをもとにユーザー情報を取得"""
    return db.query(models_sql.User).filter(models_sql.User.id == user_id).first()

def get_user_by_username(db: Session, username: str) -> Optional[models_sql.User]:
    """ユーザー名をもとにユーザー情報を取得"""
    return db.query(models_sql.User).filter(models_sql.User.username == username).first()

def get_user_by_email(db: Session, email: str) -> Optional[models_sql.User]:
    """メールアドレスをもとにユーザー情報を取得"""
    return db.query(models_sql.User).filter(models_sql.User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[models_sql.User]:
    """全てのユーザー情報を取得"""
    return db.query(models_sql.User).offset(skip).limit(limit).all()

def create_user(db: Session, user_data: Dict[str, Any]) -> models_sql.User:
    """新規ユーザーを作成"""
    db_user = models_sql.User(**user_data)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(db: Session, user_id: str, user_data: Dict[str, Any]) -> Optional[models_sql.User]:
    """ユーザー情報を更新"""
    db_user = get_user(db, user_id)
    if db_user:
        for key, value in user_data.items():
            setattr(db_user, key, value)
        db.commit()
        db.refresh(db_user)
    return db_user

def delete_user(db: Session, user_id: str) -> bool:
    """ユーザーを削除"""
    db_user = get_user(db, user_id)
    if db_user:
        db.delete(db_user)
        db.commit()
        return True
    return False

# スタートアップ企業関連のCRUD操作
def get_startup(db: Session, startup_id: str) -> Optional[models_sql.Startup]:
    """スタートアップIDをもとにスタートアップ情報を取得"""
    return db.query(models_sql.Startup).filter(models_sql.Startup.id == startup_id).first()

def get_startups(db: Session, skip: int = 0, limit: int = 100) -> List[models_sql.Startup]:
    """全てのスタートアップ情報を取得"""
    return db.query(models_sql.Startup).offset(skip).limit(limit).all()

def get_startups_by_owner(db: Session, owner_id: str) -> List[models_sql.Startup]:
    """オーナーIDをもとにスタートアップ情報を取得"""
    return db.query(models_sql.Startup).filter(models_sql.Startup.owner_id == owner_id).all()

def create_startup(db: Session, startup_data: Dict[str, Any]) -> models_sql.Startup:
    """新規スタートアップ企業を作成"""
    db_startup = models_sql.Startup(**startup_data)
    db.add(db_startup)
    db.commit()
    db.refresh(db_startup)
    return db_startup

def update_startup(db: Session, startup_id: str, startup_data: Dict[str, Any]) -> Optional[models_sql.Startup]:
    """スタートアップ情報を更新"""
    db_startup = get_startup(db, startup_id)
    if db_startup:
        for key, value in startup_data.items():
            setattr(db_startup, key, value)
        db.commit()
        db.refresh(db_startup)
    return db_startup

def delete_startup(db: Session, startup_id: str) -> bool:
    """スタートアップを削除"""
    db_startup = get_startup(db, startup_id)
    if db_startup:
        db.delete(db_startup)
        db.commit()
        return True
    return False

# VASデータ関連のCRUD操作
def get_vas_data(db: Session, vas_id: str) -> Optional[models_sql.VASData]:
    """VASデータIDをもとにVASデータを取得"""
    return db.query(models_sql.VASData).filter(models_sql.VASData.id == vas_id).first()

def get_vas_datas(db: Session, startup_id: str, skip: int = 0, limit: int = 100) -> List[models_sql.VASData]:
    """スタートアップに紐づくVASデータを取得"""
    return db.query(models_sql.VASData)\
        .filter(models_sql.VASData.startup_id == startup_id)\
        .offset(skip).limit(limit).all()

def create_vas_data(db: Session, vas_data: Dict[str, Any]) -> models_sql.VASData:
    """新規VASデータを作成"""
    db_vas = models_sql.VASData(**vas_data)
    db.add(db_vas)
    db.commit()
    db.refresh(db_vas)
    return db_vas

def update_vas_data(db: Session, vas_id: str, vas_data: Dict[str, Any]) -> Optional[models_sql.VASData]:
    """VASデータを更新"""
    db_vas = get_vas_data(db, vas_id)
    if db_vas:
        for key, value in vas_data.items():
            setattr(db_vas, key, value)
        db.commit()
        db.refresh(db_vas)
    return db_vas

def delete_vas_data(db: Session, vas_id: str) -> bool:
    """VASデータを削除"""
    db_vas = get_vas_data(db, vas_id)
    if db_vas:
        db.delete(db_vas)
        db.commit()
        return True
    return False

# 財務データ関連のCRUD操作
def get_financial_data(db: Session, financial_id: str) -> Optional[models_sql.FinancialData]:
    """財務データIDをもとに財務データを取得"""
    return db.query(models_sql.FinancialData).filter(models_sql.FinancialData.id == financial_id).first()

def get_financial_datas(db: Session, startup_id: str, skip: int = 0, limit: int = 100) -> List[models_sql.FinancialData]:
    """スタートアップに紐づく財務データを取得"""
    return db.query(models_sql.FinancialData)\
        .filter(models_sql.FinancialData.startup_id == startup_id)\
        .offset(skip).limit(limit).all()

def create_financial_data(db: Session, financial_data: Dict[str, Any]) -> models_sql.FinancialData:
    """新規財務データを作成"""
    db_financial = models_sql.FinancialData(**financial_data)
    db.add(db_financial)
    db.commit()
    db.refresh(db_financial)
    return db_financial

def update_financial_data(db: Session, financial_id: str, financial_data: Dict[str, Any]) -> Optional[models_sql.FinancialData]:
    """財務データを更新"""
    db_financial = get_financial_data(db, financial_id)
    if db_financial:
        for key, value in financial_data.items():
            setattr(db_financial, key, value)
        db.commit()
        db.refresh(db_financial)
    return db_financial

def delete_financial_data(db: Session, financial_id: str) -> bool:
    """財務データを削除"""
    db_financial = get_financial_data(db, financial_id)
    if db_financial:
        db.delete(db_financial)
        db.commit()
        return True
    return False

# メモ関連のCRUD操作
def get_note(db: Session, note_id: str) -> Optional[models_sql.Note]:
    """メモIDをもとにメモを取得"""
    return db.query(models_sql.Note).filter(models_sql.Note.id == note_id).first()

def get_notes(db: Session, startup_id: str, skip: int = 0, limit: int = 100) -> List[models_sql.Note]:
    """スタートアップに紐づくメモを取得"""
    return db.query(models_sql.Note)\
        .filter(models_sql.Note.startup_id == startup_id)\
        .offset(skip).limit(limit).all()

def create_note(db: Session, note_data: Dict[str, Any]) -> models_sql.Note:
    """新規メモを作成"""
    db_note = models_sql.Note(**note_data)
    db.add(db_note)
    db.commit()
    db.refresh(db_note)
    return db_note

def update_note(db: Session, note_id: str, note_data: Dict[str, Any]) -> Optional[models_sql.Note]:
    """メモを更新"""
    db_note = get_note(db, note_id)
    if db_note:
        for key, value in note_data.items():
            setattr(db_note, key, value)
        db.commit()
        db.refresh(db_note)
    return db_note

def delete_note(db: Session, note_id: str) -> bool:
    """メモを削除"""
    db_note = get_note(db, note_id)
    if db_note:
        db.delete(db_note)
        db.commit()
        return True
    return False