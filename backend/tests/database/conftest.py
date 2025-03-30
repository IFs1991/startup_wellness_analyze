import pytest
import os
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session
from datetime import datetime
import uuid

# テスト用のインメモリSQLiteデータベース
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

# テスト用のモデルインポート
from backend.database.models_sql import Base as SQLModels
from backend.database.models_sql import User, Startup, VASData, FinancialData, Note
from backend.database import models

# テスト用のFirestoreモック
@pytest.fixture
def mock_firestore_client():
    """Firestoreクライアントのモックを提供します"""
    mock_client = MagicMock()

    # コレクションリファレンスのモック
    mock_collection = MagicMock()
    mock_client.collection.return_value = mock_collection

    # ドキュメントリファレンスのモック
    mock_doc = MagicMock()
    mock_collection.document.return_value = mock_doc

    # ドキュメントのモック
    mock_doc_snapshot = MagicMock()
    mock_doc.get.return_value = mock_doc_snapshot
    mock_doc_snapshot.exists = True
    mock_doc_snapshot.to_dict.return_value = {"id": "test_id", "name": "Test Name"}

    # クエリのモック
    mock_query = MagicMock()
    mock_collection.where.return_value = mock_query
    mock_query.stream.return_value = [mock_doc_snapshot]

    return mock_client

# テスト用のSQLiteデータベース
@pytest.fixture
def test_db():
    """テスト用のSQLiteインメモリデータベースを提供します"""
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    SQLModels.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # テストデータベースセッションの作成
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        # テスト終了後にテーブルを削除
        SQLModels.metadata.drop_all(bind=engine)

# テスト用DBセッションとFirestoreクライアントを同時にパッチ
@pytest.fixture
def patched_db_session(test_db):
    """テスト用のDBセッションを提供し、get_db関数をパッチします"""
    with patch("backend.database.postgres.get_db") as mock_get_db:
        mock_get_db.return_value.__enter__.return_value = test_db
        mock_get_db.return_value.__exit__.return_value = None
        yield test_db

@pytest.fixture
def patched_firestore(mock_firestore_client):
    """Firestoreクライアントをパッチします"""
    with patch("backend.database.database.get_firestore_client") as mock_get_client:
        mock_get_client.return_value = mock_firestore_client
        yield mock_firestore_client

# テスト用ユーザーデータ
@pytest.fixture
def sample_user_data():
    """テスト用のユーザーデータを提供します"""
    return {
        "id": str(uuid.uuid4()),
        "username": "testuser",
        "email": "test@example.com",
        "hashed_password": "hashed_password",
        "is_active": True,
        "is_vc": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

# テスト用スタートアップデータ
@pytest.fixture
def sample_startup_data(sample_user_data):
    """テスト用のスタートアップデータを提供します"""
    return {
        "id": str(uuid.uuid4()),
        "name": "Test Startup",
        "description": "Test Description",
        "industry": "Tech",
        "founding_date": datetime.utcnow(),
        "location": "Tokyo",
        "website": "https://example.com",
        "employee_count": 10,
        "funding_stage": "Seed",
        "total_funding": 1000000.0,
        "owner_id": sample_user_data["id"],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

# テスト用VASデータ
@pytest.fixture
def sample_vas_data(sample_startup_data):
    """テスト用のVASデータを提供します"""
    return {
        "id": str(uuid.uuid4()),
        "startup_id": sample_startup_data["id"],
        "timestamp": datetime.utcnow(),
        "product_score": 8.5,
        "team_score": 7.8,
        "business_model_score": 8.2,
        "market_score": 7.5,
        "financial_score": 6.9,
        "total_score": 7.8,
        "comments": "Good potential.",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

# テスト用財務データ
@pytest.fixture
def sample_financial_data(sample_startup_data):
    """テスト用の財務データを提供します"""
    return {
        "id": str(uuid.uuid4()),
        "startup_id": sample_startup_data["id"],
        "year": 2023,
        "quarter": 1,
        "revenue": 500000.0,
        "expenses": 400000.0,
        "profit": 100000.0,
        "burn_rate": 80000.0,
        "runway": 12.5,
        "cash_balance": 1000000.0,
        "kpis": {"cac": 500, "ltv": 2500, "mrr": 50000},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

# テスト用のメモデータ
@pytest.fixture
def sample_note_data(sample_startup_data, sample_user_data):
    """テスト用のメモデータを提供します"""
    return {
        "id": str(uuid.uuid4()),
        "startup_id": sample_startup_data["id"],
        "user_id": sample_user_data["id"],
        "content": "This is a test note.",
        "timestamp": datetime.utcnow(),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

# データベースタイプとカテゴリーのパッチ
@pytest.fixture
def patched_database_type():
    """DatabaseTypeをパッチします"""
    with patch("backend.database.database.DatabaseType") as mock_db_type:
        mock_db_type.NOSQL.value = "firestore"
        mock_db_type.SQL.value = "postgresql"
        yield mock_db_type

@pytest.fixture
def patched_data_category():
    """DataCategoryをパッチします"""
    with patch("backend.database.database.DataCategory") as mock_category:
        # 構造化データカテゴリー
        mock_category.STRUCTURED.value = "structured"
        mock_category.TRANSACTIONAL.value = "transactional"
        mock_category.USER_MASTER.value = "users"
        mock_category.COMPANY_MASTER.value = "companies"

        # スケーラブルなデータカテゴリー
        mock_category.REALTIME.value = "realtime"
        mock_category.SCALABLE.value = "scalable"
        mock_category.ANALYTICS_CACHE.value = "analytics_cache"

        yield mock_category