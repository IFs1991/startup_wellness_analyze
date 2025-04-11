from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Float, JSON, Table
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, List, Dict, Any, Type, TypeVar, get_type_hints
import uuid

from .connection import Base

class User(Base):
    """ユーザーモデル"""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_vc = Column(Boolean, default=False)
    hr_system_user_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # リレーションシップ
    startups = relationship("Startup", back_populates="owner")
    notes = relationship("Note", back_populates="user")

class Startup(Base):
    """スタートアップ企業モデル"""
    __tablename__ = "startups"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    industry = Column(String, nullable=True)
    founding_date = Column(DateTime, nullable=True)
    location = Column(String, nullable=True)
    website = Column(String, nullable=True)
    logo_url = Column(String, nullable=True)
    employee_count = Column(Integer, nullable=True)
    funding_stage = Column(String, nullable=True)
    total_funding = Column(Float, nullable=True)
    owner_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # リレーションシップ
    owner = relationship("User", back_populates="startups")
    vas_data = relationship("VASData", back_populates="startup")
    financial_data = relationship("FinancialData", back_populates="startup")
    notes = relationship("Note", back_populates="startup")

class VASData(Base):
    """VAS (Value Assessment Score) データモデル"""
    __tablename__ = "vas_data"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    startup_id = Column(String, ForeignKey("startups.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    product_score = Column(Float)
    team_score = Column(Float)
    business_model_score = Column(Float)
    market_score = Column(Float)
    financial_score = Column(Float)
    total_score = Column(Float)
    comments = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # リレーションシップ
    startup = relationship("Startup", back_populates="vas_data")

class FinancialData(Base):
    """財務データモデル"""
    __tablename__ = "financial_data"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    startup_id = Column(String, ForeignKey("startups.id"))
    year = Column(Integer)
    quarter = Column(Integer)
    revenue = Column(Float, nullable=True)
    expenses = Column(Float, nullable=True)
    profit = Column(Float, nullable=True)
    burn_rate = Column(Float, nullable=True)
    runway = Column(Float, nullable=True)
    cash_balance = Column(Float, nullable=True)
    kpis = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # リレーションシップ
    startup = relationship("Startup", back_populates="financial_data")

class Note(Base):
    """メモモデル"""
    __tablename__ = "notes"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    startup_id = Column(String, ForeignKey("startups.id"))
    user_id = Column(String, ForeignKey("users.id"))
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # リレーションシップ
    startup = relationship("Startup", back_populates="notes")
    user = relationship("User", back_populates="notes")

# エンティティとSQLモデルのマッピングテーブル
_entity_to_orm_mapping = {}

def register_orm_model(entity_class: Type, orm_class: Type):
    """
    エンティティクラスとORMモデルクラスのマッピングを登録

    Args:
        entity_class: エンティティクラス
        orm_class: ORMモデルクラス
    """
    global _entity_to_orm_mapping
    _entity_to_orm_mapping[entity_class.__name__] = orm_class

def get_orm_model_for_entity(entity_class: Type) -> Type:
    """
    エンティティクラスに対応するORMモデルクラスを取得

    Args:
        entity_class: エンティティクラス

    Returns:
        Type: ORMモデルクラス

    Raises:
        ValueError: マッピングが見つからない場合
    """
    global _entity_to_orm_mapping

    # クラス名でマッピングを検索
    class_name = entity_class.__name__
    if class_name in _entity_to_orm_mapping:
        return _entity_to_orm_mapping[class_name]

    # コレクション名/テーブル名で推測を試みる
    if hasattr(entity_class, 'get_collection_name'):
        collection_name = entity_class.get_collection_name()

        # フォールバック: SQLクラスを探すヒューリスティック
        # 例: UserEntityに対してUser、またはUserModelに対してUser
        base_name = class_name
        if base_name.endswith('Entity'):
            base_name = base_name[:-6]  # 'Entity'を削除
        elif base_name.endswith('Model'):
            base_name = base_name[:-5]  # 'Model'を削除

        # 全ての登録済みORMクラスを走査
        for orm_class in Base.__subclasses__():
            # テーブル名の一致をチェック
            if hasattr(orm_class, '__tablename__') and orm_class.__tablename__ == collection_name:
                # マッピングを登録して返す
                register_orm_model(entity_class, orm_class)
                return orm_class

            # クラス名の一致をチェック
            if orm_class.__name__ == base_name:
                register_orm_model(entity_class, orm_class)
                return orm_class

    raise ValueError(f"エンティティクラス {class_name} に対応するORMモデルが見つかりません")