"""
pytest用の共通フィクスチャと設定

このモジュールは分析モジュールのテストに必要な共通のフィクスチャと設定を提供します。
"""
import pytest
import pandas as pd
import numpy as np
import sys
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Generator
import networkx as nx
from unittest.mock import AsyncMock

# システムに依存するモジュールのモック化
sys.modules['backend.config'] = MagicMock()
sys.modules['backend.config.current_config'] = MagicMock()
sys.modules['backend.database'] = MagicMock()
sys.modules['backend.database.database'] = MagicMock()
sys.modules['backend.service.firestore'] = MagicMock()
sys.modules['backend.service.firestore.client'] = MagicMock()

@pytest.fixture
def sample_company_data() -> pd.DataFrame:
    """企業分析用サンプルデータを提供します"""
    return pd.DataFrame({
        'company_id': ['comp1', 'comp2', 'comp3', 'comp4', 'comp5'],
        'industry': ['Tech', 'Healthcare', 'Finance', 'Retail', 'Manufacturing'],
        'stage': ['Seed', 'Series A', 'Series B', 'Series C', 'Growth'],
        'founding_year': [2019, 2018, 2017, 2016, 2015],
        'total_funding': [500000, 2000000, 5000000, 12000000, 25000000],
        'employee_count': [10, 25, 50, 100, 200],
        'revenue': [100000, 500000, 2000000, 5000000, 10000000],
        'growth_rate': [0.2, 0.5, 0.3, 0.15, 0.1],
        'burn_rate': [50000, 150000, 300000, 400000, 600000],
        'runway_months': [10, 12, 15, 24, 36],
        'customer_count': [50, 200, 1000, 5000, 20000],
        'churn_rate': [0.1, 0.08, 0.06, 0.05, 0.04],
        'wellness_score': [65, 72, 78, 81, 85]
    })

@pytest.fixture
def sample_financial_data() -> pd.DataFrame:
    """財務分析用サンプルデータを提供します"""
    return pd.DataFrame({
        'company_id': ['comp1', 'comp1', 'comp1', 'comp2', 'comp2', 'comp2'],
        'date': pd.date_range(start='2022-01-01', periods=6, freq='Q'),
        'revenue': [100000, 120000, 150000, 500000, 550000, 600000],
        'expenses': [80000, 90000, 100000, 400000, 420000, 450000],
        'profit': [20000, 30000, 50000, 100000, 130000, 150000],
        'cash_on_hand': [250000, 200000, 250000, 1000000, 950000, 1100000],
        'assets': [300000, 320000, 350000, 1200000, 1250000, 1300000],
        'liabilities': [50000, 60000, 70000, 200000, 220000, 230000],
        'equity': [250000, 260000, 280000, 1000000, 1030000, 1070000],
        'debt': [30000, 35000, 40000, 150000, 155000, 160000]
    })

@pytest.fixture
def sample_team_data() -> pd.DataFrame:
    """チーム分析用サンプルデータを提供します"""
    return pd.DataFrame({
        'company_id': ['comp1', 'comp1', 'comp1', 'comp2', 'comp2', 'comp2'],
        'employee_id': ['e1', 'e2', 'e3', 'e4', 'e5', 'e6'],
        'role': ['CEO', 'CTO', 'CFO', 'CEO', 'CTO', 'CFO'],
        'experience_years': [8, 10, 7, 15, 12, 9],
        'education_level': ['Masters', 'PhD', 'MBA', 'MBA', 'Masters', 'MBA'],
        'previous_exits': [0, 1, 0, 2, 1, 0],
        'domain_expertise': [8, 9, 7, 9, 8, 8],
        'leadership_score': [7, 8, 7, 9, 8, 7],
        'technical_score': [6, 9, 5, 7, 9, 5],
        'management_score': [8, 7, 8, 9, 7, 8],
        'network_score': [7, 8, 7, 9, 8, 7]
    })

@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """市場分析用サンプルデータを提供します"""
    return pd.DataFrame({
        'industry': ['Tech', 'Healthcare', 'Finance', 'Retail', 'Manufacturing'],
        'market_size': [1000000000, 800000000, 1200000000, 650000000, 900000000],
        'cagr': [0.15, 0.12, 0.09, 0.07, 0.05],
        'competitors_count': [200, 150, 80, 300, 100],
        'market_concentration': [0.3, 0.4, 0.6, 0.25, 0.5],
        'barrier_to_entry': [7, 8, 9, 5, 7],
        'regulation_level': [6, 9, 9, 4, 7],
        'technology_innovation_rate': [9, 7, 6, 5, 6],
        'customer_acquisition_cost': [500, 800, 400, 300, 600],
        'lifetime_value': [5000, 12000, 8000, 3000, 7000]
    })

@pytest.fixture
def sample_time_series_data() -> pd.DataFrame:
    """時系列分析用サンプルデータを提供します"""
    dates = pd.date_range(start='2021-01-01', end='2022-12-31', freq='M')
    np.random.seed(42)

    return pd.DataFrame({
        'date': dates,
        'company_id': ['comp1'] * len(dates),
        'revenue': np.random.normal(1000000, 100000, len(dates)) * (1 + np.arange(len(dates)) * 0.02),
        'expenses': np.random.normal(800000, 50000, len(dates)) * (1 + np.arange(len(dates)) * 0.015),
        'profit': np.random.normal(200000, 30000, len(dates)) * (1 + np.arange(len(dates)) * 0.025),
        'customers': np.random.poisson(1000, len(dates)) + np.arange(len(dates)) * 20,
        'churn_rate': np.random.beta(2, 10, len(dates)),
        'wellness_score': 70 + np.cumsum(np.random.normal(0.5, 0.3, len(dates)))
    })

@pytest.fixture
def sample_network_data() -> nx.Graph:
    """ネットワーク分析用サンプルグラフを提供します"""
    G = nx.Graph()

    # ノードの追加（会社）
    companies = ['comp1', 'comp2', 'comp3', 'comp4', 'comp5']
    for idx, company in enumerate(companies):
        G.add_node(company,
                   type='company',
                   industry=['Tech', 'Healthcare', 'Finance', 'Retail', 'Manufacturing'][idx],
                   size=[10, 25, 50, 100, 200][idx])

    # ノードの追加（投資家）
    investors = ['inv1', 'inv2', 'inv3', 'inv4']
    for idx, investor in enumerate(investors):
        G.add_node(investor,
                  type='investor',
                  portfolio_size=[10, 20, 5, 15][idx],
                  focus=['Tech', 'Healthcare', 'Multi', 'Finance'][idx])

    # エッジの追加（投資関係）
    investments = [
        ('inv1', 'comp1', {'amount': 500000, 'date': '2021-01-15', 'round': 'Seed'}),
        ('inv1', 'comp2', {'amount': 1000000, 'date': '2021-03-20', 'round': 'Series A'}),
        ('inv2', 'comp2', {'amount': 1000000, 'date': '2021-03-20', 'round': 'Series A'}),
        ('inv2', 'comp3', {'amount': 2000000, 'date': '2021-05-10', 'round': 'Series B'}),
        ('inv3', 'comp3', {'amount': 3000000, 'date': '2021-05-10', 'round': 'Series B'}),
        ('inv3', 'comp4', {'amount': 5000000, 'date': '2021-08-05', 'round': 'Series C'}),
        ('inv4', 'comp4', {'amount': 7000000, 'date': '2021-08-05', 'round': 'Series C'}),
        ('inv4', 'comp5', {'amount': 10000000, 'date': '2021-12-15', 'round': 'Growth'})
    ]

    for src, dst, attrs in investments:
        G.add_edge(src, dst, **attrs)

    return G

@pytest.fixture
def mock_firestore_client():
    """Firestoreクライアントのモックを提供します"""
    client = MagicMock()
    collection_ref = MagicMock()
    client.collection.return_value = collection_ref

    # モックドキュメントの設定
    doc_ref = MagicMock()
    collection_ref.document.return_value = doc_ref

    # 非同期メソッドのモック
    get_mock = MagicMock()
    doc_ref.get = get_mock

    return client

@pytest.fixture
def mock_model():
    """機械学習モデルのモックを提供します"""
    model = MagicMock()
    model.predict = MagicMock(return_value=np.array([1, 0, 1]))
    model.predict_proba = MagicMock(return_value=np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]))
    model.fit = MagicMock(return_value=model)

    return model

@pytest.fixture
def sample_data():
    """テスト用のサンプルデータフレームを提供します"""
    np.random.seed(42)
    data = {
        'revenue': np.random.normal(1000000, 200000, 100),
        'customers': np.random.randint(100, 1000, 100),
        'marketing_spend': np.random.normal(50000, 10000, 100),
        'product_development': np.random.normal(30000, 5000, 100),
        'team_size': np.random.randint(5, 50, 100),
        'funding_amount': np.random.normal(2000000, 500000, 100),
        'growth_rate': np.random.normal(0.2, 0.05, 100),
        'churn_rate': np.random.normal(0.1, 0.02, 100)
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_bq_service():
    """BigQueryServiceのモックを提供します"""
    service = MagicMock()
    service.query = AsyncMock()
    service.save_dataframe = AsyncMock()
    return service

@pytest.fixture
def mock_firestore_service():
    """FirestoreServiceのモックを提供します"""
    service = MagicMock()
    service.add_document = AsyncMock()
    service.add_document.return_value = "test_doc_id"
    service.get_document = AsyncMock()
    service.get_document.return_value = {"id": "test_doc_id", "data": {"test": "data"}}
    service.update_document = AsyncMock()
    service.delete_document = AsyncMock()
    return service

@pytest.fixture
def sample_time_series():
    """時系列データのサンプルを提供します"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = {
        'date': dates,
        'value': np.cumsum(np.random.normal(10, 5, 100)),
        'trend': np.linspace(0, 100, 100)
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_graph():
    """サンプルのネットワークグラフを提供します"""
    G = nx.DiGraph()

    # ノードの追加
    nodes = ['A', 'B', 'C', 'D', 'E']
    for node in nodes:
        G.add_node(node, weight=np.random.random())

    # エッジの追加
    edges = [('A', 'B'), ('B', 'C'), ('A', 'C'), ('C', 'D'), ('D', 'E'), ('B', 'E')]
    for u, v in edges:
        G.add_edge(u, v, weight=np.random.random())

    return G