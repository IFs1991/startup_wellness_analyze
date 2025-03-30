from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Float, JSON, Table
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid

from .postgres import Base

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