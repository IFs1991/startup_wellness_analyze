# Phase 3 Task 3.1: データベース層の実装
# TDD GREEN段階: SQLAlchemyモデル実装

from sqlalchemy import (
    Column, String, Integer, DateTime, LargeBinary,
    JSON, Boolean, Text, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime, timezone
import uuid
from typing import Dict, Any, List, Optional

Base = declarative_base()


class FLModel(Base):
    """フェデレーテッド学習モデルのORMモデル"""
    __tablename__ = "fl_models"

    # 主キー
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # モデル基本情報
    name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)

    # モデルデータ
    model_bytes = Column(LargeBinary, nullable=False)
    model_size = Column(Integer, nullable=False)  # バイト数

    # メタデータ（JSON形式）
    model_metadata = Column(JSON, nullable=False, default=dict)

    # パフォーマンスメトリクス
    accuracy = Column(String(20), nullable=True)  # 精度値（文字列で格納）
    loss = Column(String(20), nullable=True)      # 損失値（文字列で格納）

    # 差分プライバシーメトリクス
    privacy_epsilon = Column(String(20), nullable=True)
    privacy_delta = Column(String(20), nullable=True)

    # ステータス管理
    status = Column(String(20), nullable=False, default="training")  # training, completed, archived
    is_active = Column(Boolean, default=True, nullable=False)

    # タイムスタンプ
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    # 関連データ
    training_sessions = relationship("TrainingSession", back_populates="model")

    # 制約とインデックス
    __table_args__ = (
        UniqueConstraint('name', 'version', name='uq_model_name_version'),
        Index('idx_model_name_version', 'name', 'version'),
        Index('idx_model_status', 'status'),
        Index('idx_model_created_at', 'created_at'),
    )

    def __repr__(self):
        return f"<FLModel(id={self.id}, name={self.name}, version={self.version})>"


class ClientRegistration(Base):
    """クライアント登録情報のORMモデル"""
    __tablename__ = "client_registrations"

    # 主キー
    client_id = Column(String(255), primary_key=True)

    # 認証情報
    public_key = Column(Text, nullable=False)
    certificate = Column(Text, nullable=False)
    certificate_fingerprint = Column(String(128), nullable=False, index=True)

    # クライアント能力情報
    capabilities = Column(JSON, nullable=False, default=dict)  # GPU, メモリ、計算能力など

    # ネットワーク情報
    last_seen_ip = Column(String(45), nullable=True)  # IPv6対応
    user_agent = Column(String(500), nullable=True)

    # ステータス管理
    status = Column(String(20), nullable=False, default="active")  # active, inactive, suspended, banned
    registration_source = Column(String(50), nullable=False, default="manual")

    # 統計情報
    total_sessions = Column(Integer, default=0, nullable=False)
    total_updates_contributed = Column(Integer, default=0, nullable=False)
    average_response_time = Column(String(20), nullable=True)  # ミリ秒

    # タイムスタンプ
    registered_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    last_active_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    # 関連データは多対多で実装する必要があるため、一旦削除
    # training_sessions = relationship("TrainingSession", back_populates="clients")

    # インデックス
    __table_args__ = (
        Index('idx_client_status', 'status'),
        Index('idx_client_last_active', 'last_active_at'),
        Index('idx_client_capabilities', 'capabilities'),
    )

    def __repr__(self):
        return f"<ClientRegistration(client_id={self.client_id}, status={self.status})>"


class TrainingSession(Base):
    """学習セッションのORMモデル"""
    __tablename__ = "training_sessions"

    # 主キー
    session_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # 関連モデル
    model_id = Column(String(36), ForeignKey("fl_models.id"), nullable=False)

    # セッション情報
    round_number = Column(Integer, nullable=False)
    session_name = Column(String(255), nullable=True)

    # 参加クライアント情報（JSON配列）
    participating_clients = Column(JSON, nullable=False, default=list)
    selected_clients = Column(JSON, nullable=False, default=list)

    # 集約結果
    aggregation_result = Column(JSON, nullable=False, default=dict)

    # パフォーマンスメトリクス
    accuracy_before = Column(String(20), nullable=True)
    accuracy_after = Column(String(20), nullable=True)
    loss_before = Column(String(20), nullable=True)
    loss_after = Column(String(20), nullable=True)

    # 差分プライバシーメトリクス
    privacy_metrics = Column(JSON, nullable=False, default=dict)
    epsilon_consumed = Column(String(20), nullable=True)
    delta_consumed = Column(String(20), nullable=True)

    # セッション状態管理
    status = Column(String(20), nullable=False, default="pending")  # pending, running, completed, failed

    # 統計情報
    total_updates_received = Column(Integer, default=0, nullable=False)
    average_client_response_time = Column(String(20), nullable=True)
    session_duration = Column(Integer, nullable=True)  # 秒

    # エラー情報
    error_message = Column(Text, nullable=True)
    failed_clients = Column(JSON, nullable=True, default=list)

    # タイムスタンプ
    started_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    # 関連データ
    model = relationship("FLModel", back_populates="training_sessions")
    # 多対多関係は今回は省略し、participating_clientsをJSONで管理
    # clients = relationship("ClientRegistration", back_populates="training_sessions")

    # インデックス
    __table_args__ = (
        Index('idx_session_model_round', 'model_id', 'round_number'),
        Index('idx_session_status', 'status'),
        Index('idx_session_started_at', 'started_at'),
        Index('idx_session_round_number', 'round_number'),
    )

    def __repr__(self):
        return f"<TrainingSession(session_id={self.session_id}, model_id={self.model_id}, round={self.round_number})>"


class PrivacyBudgetRecord(Base):
    """プライバシー予算記録のORMモデル"""
    __tablename__ = "privacy_budget_records"

    # 主キー
    record_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # 関連モデル・セッション
    model_id = Column(String(36), ForeignKey("fl_models.id"), nullable=False)
    session_id = Column(String(36), ForeignKey("training_sessions.session_id"), nullable=True)

    # 予算消費情報
    epsilon_consumed = Column(String(20), nullable=False)
    delta_consumed = Column(String(20), nullable=False)
    mechanism_used = Column(String(50), nullable=False)  # gaussian, laplace, etc.

    # 予算状況
    epsilon_remaining = Column(String(20), nullable=False)
    delta_remaining = Column(String(20), nullable=False)

    # 操作詳細
    operation_type = Column(String(50), nullable=False)  # training_round, evaluation, etc.
    operation_details = Column(JSON, nullable=False, default=dict)

    # タイムスタンプ
    consumed_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    # インデックス
    __table_args__ = (
        Index('idx_budget_model_id', 'model_id'),
        Index('idx_budget_consumed_at', 'consumed_at'),
        Index('idx_budget_mechanism', 'mechanism_used'),
    )

    def __repr__(self):
        return f"<PrivacyBudgetRecord(model_id={self.model_id}, epsilon={self.epsilon_consumed})>"


class SystemAuditLog(Base):
    """システム監査ログのORMモデル"""
    __tablename__ = "system_audit_logs"

    # 主キー
    log_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # ログ基本情報
    event_type = Column(String(50), nullable=False, index=True)
    event_category = Column(String(30), nullable=False, index=True)  # security, privacy, operation
    severity = Column(String(20), nullable=False, default="info")  # debug, info, warning, error, critical

    # 関連エンティティ
    entity_type = Column(String(50), nullable=True)  # model, client, session
    entity_id = Column(String(255), nullable=True, index=True)

    # ユーザー・クライアント情報
    client_id = Column(String(255), nullable=True, index=True)
    user_id = Column(String(255), nullable=True)

    # ログ内容
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True, default=dict)

    # ネットワーク情報
    source_ip = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)

    # タイムスタンプ
    timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False, index=True)

    # インデックス
    __table_args__ = (
        Index('idx_audit_timestamp_category', 'timestamp', 'event_category'),
        Index('idx_audit_entity', 'entity_type', 'entity_id'),
        Index('idx_audit_client_timestamp', 'client_id', 'timestamp'),
    )

    def __repr__(self):
        return f"<SystemAuditLog(event_type={self.event_type}, severity={self.severity})>"