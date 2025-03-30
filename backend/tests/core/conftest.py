import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Generator

from core.auth_manager import User, UserRole

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """テスト用のサンプルDataFrameを提供します"""
    return pd.DataFrame({
        'company_id': ['comp1', 'comp1', 'comp2', 'comp2', 'comp3'],
        'revenue': [1000000, 1200000, 800000, 850000, 1500000],
        'expenses': [800000, 900000, 700000, 720000, 1200000],
        'profit': [200000, 300000, 100000, 130000, 300000],
        'employees': [50, 55, 30, 32, 70],
        'year': [2021, 2022, 2021, 2022, 2022],
        'quarter': [4, 4, 4, 4, 4],
        'growth_rate': [0.05, 0.08, 0.03, 0.04, 0.1],
        'customer_satisfaction': [4.2, 4.3, 3.8, 3.9, 4.5]
    })

@pytest.fixture
def sample_text_data() -> List[Dict[str, Any]]:
    """テスト用のテキストデータを提供します"""
    return [
        {"id": "text1", "content": "顧客満足度が高く、市場での評判も良好。成長率は安定している。", "date": "2022-12-01"},
        {"id": "text2", "content": "社員のモチベーションが高く、チームワークが優れている。", "date": "2022-11-15"},
        {"id": "text3", "content": "収益は安定しているが、経費の増加が懸念される。", "date": "2022-10-20"},
        {"id": "text4", "content": "新規顧客獲得に苦戦しており、マーケティング戦略の見直しが必要。", "date": "2022-09-10"},
        {"id": "text5", "content": "イノベーションが進み、新製品の開発が順調。将来性は高い。", "date": "2022-08-05"}
    ]

@pytest.fixture
def sample_time_series_data() -> pd.DataFrame:
    """テスト用の時系列データを提供します"""
    dates = pd.date_range(start='2021-01-01', end='2022-12-31', freq='M')
    np.random.seed(42)  # 結果の再現性のために

    return pd.DataFrame({
        'date': dates,
        'revenue': np.random.normal(1000000, 100000, len(dates)),
        'expenses': np.random.normal(800000, 50000, len(dates)),
        'new_customers': np.random.poisson(30, len(dates)),
        'churn_rate': np.random.beta(2, 10, len(dates))
    })

@pytest.fixture
def sample_categorical_data() -> pd.DataFrame:
    """テスト用のカテゴリカルデータを提供します"""
    return pd.DataFrame({
        'company_id': ['comp1', 'comp2', 'comp3', 'comp4', 'comp5'] * 4,
        'industry': ['Tech', 'Finance', 'Healthcare', 'Retail', 'Manufacturing'] * 4,
        'company_size': ['Small', 'Medium', 'Large', 'Medium', 'Large'] * 4,
        'region': ['North', 'South', 'East', 'West', 'Central'] * 4,
        'success_status': ['Success', 'Failure', 'Success', 'Success', 'Failure'] * 4
    })

@pytest.fixture
def mock_firebase_app() -> MagicMock:
    """Firebase Appのモックを提供します"""
    mock_app = MagicMock()
    return mock_app

@pytest.fixture
def mock_firestore_client() -> MagicMock:
    """Firestoreクライアントのモックを提供します"""
    mock_client = MagicMock()
    return mock_client

@pytest.fixture
def mock_auth_client() -> MagicMock:
    """Firebase Auth クライアントのモックを提供します"""
    mock_client = MagicMock()
    return mock_client

@pytest.fixture
def mock_user() -> User:
    """テスト用ユーザーを提供します"""
    return User(
        id="test_user_id",
        email="test@example.com",
        display_name="Test User",
        is_active=True,
        role=UserRole.USER,
        company_id="test_company_id"
    )

@pytest.fixture
def mock_ai_service() -> MagicMock:
    """AI サービスのモックを提供します"""
    mock_service = MagicMock()
    mock_service.generate_text.return_value = "生成されたテキスト"
    mock_service.analyze_sentiment.return_value = {"sentiment": "positive", "score": 0.8}
    return mock_service