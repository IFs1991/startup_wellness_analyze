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

# -*- coding: utf-8 -*-
"""
SQLデータモデル定義
Startup Wellness データ分析システムで使用されるSQLAlchemyモデルを定義します。
"""
from typing import Dict, Optional, Any, List
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime,
    Boolean, ForeignKey, Date, JSON, Numeric, BigInteger,
    UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import datetime

Base = declarative_base()


class VASHealthPerformance(Base):
    """VAS健康・パフォーマンスデータモデル"""
    __tablename__ = "vas_health_performance"

    record_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False)
    company_id = Column(String(100), nullable=False)
    record_date = Column(DateTime, nullable=False)
    physical_health = Column(Integer, CheckConstraint("physical_health BETWEEN 0 AND 100"))
    mental_health = Column(Integer, CheckConstraint("mental_health BETWEEN 0 AND 100"))
    work_performance = Column(Integer, CheckConstraint("work_performance BETWEEN 0 AND 100"))
    work_satisfaction = Column(Integer, CheckConstraint("work_satisfaction BETWEEN 0 AND 100"))
    additional_comments = Column(Text)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint('user_id', 'company_id', 'record_date', name='uq_vas_user_company_date'),
    )

    def __repr__(self):
        return f"<VASHealthPerformance(record_id={self.record_id}, user_id={self.user_id}, date={self.record_date})>"


class GoogleFormsConfiguration(Base):
    """Google Forms 設定モデル"""
    __tablename__ = "google_forms_configurations"

    config_id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(100), nullable=False)
    form_type = Column(String(50), nullable=False)
    form_id = Column(String(100), nullable=False)
    sheet_id = Column(String(100))
    field_mappings = Column(JSON, nullable=False, default={})
    active = Column(Boolean, nullable=False, default=True)
    sync_frequency = Column(Integer, nullable=False, default=3600)  # デフォルト1時間（秒）
    last_sync_time = Column(DateTime)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())

    sync_logs = relationship("GoogleFormsSyncLog", back_populates="config")

    __table_args__ = (
        UniqueConstraint('company_id', 'form_type', name='uq_forms_company_form_type'),
    )

    def __repr__(self):
        return f"<GoogleFormsConfiguration(config_id={self.config_id}, form_type={self.form_type})>"


class GoogleFormsSyncLog(Base):
    """Google Forms 同期ログモデル"""
    __tablename__ = "google_forms_sync_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    config_id = Column(Integer, ForeignKey('google_forms_configurations.config_id'), nullable=False)
    sync_start_time = Column(DateTime, nullable=False)
    sync_end_time = Column(DateTime, nullable=False)
    records_processed = Column(Integer, nullable=False, default=0)
    records_created = Column(Integer, nullable=False, default=0)
    records_updated = Column(Integer, nullable=False, default=0)
    status = Column(String(20), nullable=False)
    error_details = Column(Text)
    created_at = Column(DateTime, nullable=False, default=func.now())

    config = relationship("GoogleFormsConfiguration", back_populates="sync_logs")

    def __repr__(self):
        return f"<GoogleFormsSyncLog(log_id={self.log_id}, status={self.status})>"


class MonthlyBusinessPerformance(Base):
    """月次業績データモデル"""
    __tablename__ = "monthly_business_performance"

    report_id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(100), nullable=False)
    report_month = Column(Date, nullable=False)
    revenue = Column(Numeric(15, 2))
    expenses = Column(Numeric(15, 2))
    profit_margin = Column(Numeric(5, 2))
    headcount = Column(Integer)
    new_clients = Column(Integer)
    turnover_rate = Column(Numeric(5, 2))
    notes = Column(Text)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())

    extraction_results = relationship("DocumentExtractionResult", back_populates="report")

    __table_args__ = (
        UniqueConstraint('company_id', 'report_month', name='uq_performance_company_month'),
    )

    def __repr__(self):
        return f"<MonthlyBusinessPerformance(report_id={self.report_id}, company_id={self.company_id}, month={self.report_month})>"


class UploadedDocument(Base):
    """アップロードされたドキュメント情報モデル"""
    __tablename__ = "uploaded_documents"

    document_id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(100), nullable=False)
    file_name = Column(String(255), nullable=False)
    original_file_name = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    upload_path = Column(String(255), nullable=False)
    content_type = Column(String(100))
    processing_status = Column(String(50), nullable=False, default='pending')
    processed_at = Column(DateTime)
    error_details = Column(Text)
    uploaded_by = Column(String(100), nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())

    extraction_results = relationship("DocumentExtractionResult", back_populates="document")

    def __repr__(self):
        return f"<UploadedDocument(document_id={self.document_id}, file_name={self.file_name})>"


class DocumentExtractionResult(Base):
    """ドキュメント抽出結果モデル"""
    __tablename__ = "document_extraction_results"

    result_id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('uploaded_documents.document_id'), nullable=False)
    report_id = Column(Integer, ForeignKey('monthly_business_performance.report_id'))
    extracted_data = Column(JSON, nullable=False, default={})
    confidence_score = Column(Numeric(5, 2))
    review_status = Column(String(50), nullable=False, default='pending')
    reviewed_by = Column(String(100))
    reviewed_at = Column(DateTime)
    review_notes = Column(Text)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())

    document = relationship("UploadedDocument", back_populates="extraction_results")
    report = relationship("MonthlyBusinessPerformance", back_populates="extraction_results")

    def __repr__(self):
        return f"<DocumentExtractionResult result_id={self.result_id}, document_id={self.document_id}>"


class PositionLevel(Base):
    """役職レベルマスターモデル"""
    __tablename__ = "position_levels"

    level_id = Column(Integer, primary_key=True, autoincrement=True)
    level_name = Column(String(100), nullable=False)
    position_title = Column(String(100), nullable=False, unique=True)
    base_weight = Column(Numeric(5, 2), nullable=False)
    theoretical_basis = Column(Text)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<PositionLevel(level_id={self.level_id}, level_name={self.level_name})>"


class Industry(Base):
    """業種マスターモデル"""
    __tablename__ = "industries"

    industry_id = Column(Integer, primary_key=True, autoincrement=True)
    industry_name = Column(String(100), nullable=False)
    industry_code = Column(String(20), nullable=False, unique=True)
    industry_description = Column(Text)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())

    weights = relationship("IndustryWeight", back_populates="industry")

    def __repr__(self):
        return f"<Industry(industry_id={self.industry_id}, industry_name={self.industry_name})>"


class IndustryWeight(Base):
    """業種別重み係数モデル"""
    __tablename__ = "industry_weights"

    weight_id = Column(Integer, primary_key=True, autoincrement=True)
    industry_id = Column(Integer, ForeignKey('industries.industry_id'), nullable=False)
    metric_name = Column(String(100), nullable=False)
    weight_value = Column(Numeric(5, 2), nullable=False)
    weight_description = Column(Text)
    effective_from = Column(Date, nullable=False)
    effective_to = Column(Date)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())

    industry = relationship("Industry", back_populates="weights")

    __table_args__ = (
        UniqueConstraint('industry_id', 'metric_name', 'effective_from', name='uq_weights_industry_metric_date'),
    )

    def __repr__(self):
        return f"<IndustryWeight(weight_id={self.weight_id}, industry_id={self.industry_id}, metric={self.metric_name})>"


class CompanySizeCategory(Base):
    """企業規模分類モデル"""
    __tablename__ = "company_size_categories"

    category_id = Column(Integer, primary_key=True, autoincrement=True)
    category_name = Column(String(100), nullable=False, unique=True)
    min_employees = Column(Integer)
    max_employees = Column(Integer)
    min_revenue = Column(Numeric(15, 2))
    max_revenue = Column(Numeric(15, 2))
    adjustment_factor = Column(Numeric(5, 2), nullable=False, default=1.0)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<CompanySizeCategory(category_id={self.category_id}, name={self.category_name})>"

# 企業情報モデルを追加
class Company(Base):
    """企業モデル（SQLAlchemy ORM）"""
    __tablename__ = "companies"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, index=True)
    industry = Column(String, nullable=True)
    founded_date = Column(DateTime, nullable=True)
    employee_count = Column(Integer, nullable=True)
    location = Column(String, nullable=True)
    website = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Company id={self.id}, name={self.name}>"

# エンティティとSQLモデルのマッピング（ファイルの最後に追加）
from .models.entities import CompanyEntity
register_orm_model(CompanyEntity, Company)