import pytest
from unittest.mock import MagicMock, patch
from backend.analysis.association_analyzer import AssociationAnalyzer, analyze_associations
from backend.analysis.base import AnalysisConfig
import pandas as pd
import numpy as np

@pytest.fixture(autouse=True)
def mock_firebase_admin():
    """Firebase Adminの初期化をモック化するフィクスチャ"""
    with patch('firebase_admin.initialize_app') as mock:
        yield mock

@pytest.fixture(autouse=True)
def mock_credentials():
    """Firebase認証情報をモック化するフィクスチャ"""
    with patch('firebase_admin.credentials.Certificate') as mock:
        yield mock

@pytest.fixture(autouse=True)
def mock_firestore_async_client():
    """Firestore AsyncClientをモック化するフィクスチャ"""
    with patch('google.cloud.firestore.AsyncClient') as mock:
        mock_instance = mock.return_value
        mock_instance.collection.return_value.document.return_value.set = MagicMock()
        mock_instance.collection.return_value.document.return_value.get = MagicMock()
        mock_instance.collection.return_value.document.return_value.update = MagicMock()
        mock_instance.collection.return_value.document.return_value.delete = MagicMock()
        yield mock

@pytest.fixture
def sample_data():
    """テスト用のサンプルデータを生成するフィクスチャ"""
    return [
        {'item1': True, 'item2': False},
        {'item1': False, 'item2': True}
    ]

@pytest.fixture
def mock_storage_client():
    """Cloud Storageクライアントのモックを生成するフィクスチャ"""
    with patch('google.cloud.storage.Client') as mock:
        yield mock.return_value

@pytest.fixture
def mock_firestore_client(mock_storage_client, mock_firestore_async_client):
    """Firestoreクライアントのモックを生成するフィクスチャ"""
    mock_client = MagicMock()
    mock_client.query_documents.return_value = [
        {'item1': True, 'item2': False},
        {'item1': False, 'item2': True}
    ]
    mock_client.create_document.return_value = {"id": "test_id"}
    mock_client.storage_client = mock_storage_client
    mock_client.client = mock_firestore_async_client.return_value
    return mock_client

@pytest.fixture
def analyzer(mock_firestore_client) -> AssociationAnalyzer:
    """AssociationAnalyzerインスタンスを生成するフィクスチャ"""
    return AssociationAnalyzer(
        min_support=0.3,
        min_confidence=0.5,
        min_lift=1.0,
        firestore_client=mock_firestore_client
    )

@pytest.fixture
def analysis_config() -> AnalysisConfig:
    """分析設定を生成するフィクスチャ"""
    return AnalysisConfig(
        collection_name="test_collection",
        target_fields=["item1", "item2"]
    )

class TestAssociationAnalyzer:
    async def test_validate_data(self, analyzer, sample_data):
        """データ検証のテスト"""
        data = pd.DataFrame(sample_data)
        result = analyzer.validate_data(data)
        assert result is True

    async def test_prepare_data(self, analyzer, sample_data):
        """データ準備のテスト"""
        data = pd.DataFrame(sample_data)
        prepared_data = analyzer.prepare_data(data)
        assert isinstance(prepared_data, pd.DataFrame)
        assert not prepared_data.empty

    async def test_analyze(self, analyzer, analysis_config):
        """分析実行のテスト"""
        result = await analyzer.analyze(
            config=analysis_config,
            target_columns=["item1", "item2"]
        )

        assert isinstance(result, dict)
        assert "frequent_itemsets" in result
        assert "association_rules" in result

    async def test_analyze_with_invalid_data(self, analyzer, analysis_config, mock_firestore_client):
        """無効なデータでの分析実行のテスト"""
        mock_firestore_client.query_documents.return_value = []

        with pytest.raises(Exception):
            await analyzer.analyze(
                config=analysis_config,
                target_columns=["item1", "item2"]
            )

    async def test_analyze_and_store(self, analyzer, analysis_config):
        """分析実行と結果保存のテスト"""
        result = await analyzer.analyze_and_store(
            config=analysis_config,
            target_columns=["item1", "item2"]
        )

        assert isinstance(result, dict)
        assert "analysis_id" in result

    async def test_analyze_associations_helper(self, mock_firestore_client):
        """ヘルパー関数のテスト"""
        result = await analyze_associations(
            collection="test_collection",
            target_columns=["item1", "item2"],
            min_support=0.3,
            min_confidence=0.5,
            min_lift=1.0,
            firestore_client=mock_firestore_client
        )

        assert isinstance(result, dict)
        assert "analysis_id" in result