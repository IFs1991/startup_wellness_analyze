import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from backend.analysis.association_analyzer import AssociationAnalyzer, AnalysisConfig, AnalysisError

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """テスト用のサンプルデータを生成するフィクスチャ"""
    data = {
        'item1': [True, True, False, True, False],
        'item2': [True, False, True, True, True],
        'item3': [False, True, True, True, False]
    }
    return pd.DataFrame(data)

@pytest.fixture
def analyzer() -> AssociationAnalyzer:
    """AssociationAnalyzerインスタンスを生成するフィクスチャ"""
    return AssociationAnalyzer(
        min_support=0.3,
        min_confidence=0.5,
        min_lift=1.0
    )

@pytest.fixture
def analysis_config() -> AnalysisConfig:
    """テスト用の分析設定を生成するフィクスチャ"""
    return AnalysisConfig(
        collection_name="test_collection",
        target_fields=["item1", "item2", "item3"]
    )

class TestAssociationAnalyzer:
    """AssociationAnalyzerのテストクラス"""

    async def test_validate_data(self, analyzer: AssociationAnalyzer, sample_data: pd.DataFrame):
        """データバリデーションのテスト"""
        assert analyzer._validate_data(sample_data) is True

        # 空のデータフレームでテスト
        with pytest.raises(AnalysisError):
            analyzer._validate_data(pd.DataFrame())

        # 非ブール型のデータでテスト
        invalid_data = pd.DataFrame({
            'item1': [1, 2, 3],
            'item2': [4, 5, 6]
        })
        with pytest.raises(AnalysisError):
            analyzer._validate_data(invalid_data)

    async def test_prepare_data(self, analyzer: AssociationAnalyzer):
        """データ前処理のテスト"""
        test_data = pd.DataFrame({
            'item1': [1, 0, 1],
            'item2': [0, 1, 1]
        })
        prepared_data = analyzer._prepare_data(test_data)
        assert prepared_data.dtypes.all() == bool

    async def test_analyze(self, analyzer: AssociationAnalyzer, sample_data: pd.DataFrame):
        """分析実行のテスト"""
        result = await analyzer.analyze(sample_data)

        assert isinstance(result, dict)
        assert 'frequent_itemsets' in result
        assert 'rules' in result
        assert 'summary' in result

        # サマリー情報の検証
        summary = result['summary']
        assert isinstance(summary['total_itemsets'], int)
        assert isinstance(summary['total_rules'], int)
        assert summary['parameters']['min_support'] == analyzer.min_support
        assert summary['parameters']['min_confidence'] == analyzer.min_confidence
        assert summary['parameters']['min_lift'] == analyzer.min_lift

    async def test_analyze_with_invalid_data(self, analyzer: AssociationAnalyzer):
        """無効なデータでの分析テスト"""
        with pytest.raises(AnalysisError):
            await analyzer.analyze(pd.DataFrame())

    async def test_analyze_and_store(
        self,
        analyzer: AssociationAnalyzer,
        analysis_config: AnalysisConfig,
        mocker
    ):
        """分析結果の保存テスト"""
        # Firestoreクライアントのモック
        mock_firestore = mocker.patch('backend.src.database.firestore.client.get_firestore_client')
        mock_firestore.return_value.query_documents.return_value = [
            {'item1': True, 'item2': False, 'item3': True},
            {'item1': False, 'item2': True, 'item3': False}
        ]

        result = await analyzer.analyze_and_store(analysis_config)

        assert isinstance(result, dict)
        assert mock_firestore.return_value.create_document.called

    async def test_analyze_associations_helper(self, mocker):
        """ヘルパー関数のテスト"""
        from backend.analysis.association_analyzer import analyze_associations

        # Firestoreクライアントのモック
        mock_firestore = mocker.patch('backend.src.database.firestore.client.get_firestore_client')
        mock_firestore.return_value.query_documents.return_value = [
            {'item1': True, 'item2': False},
            {'item1': False, 'item2': True}
        ]

        result = await analyze_associations(
            collection="test_collection",
            target_columns=["item1", "item2"],
            min_support=0.3,
            min_confidence=0.5,
            min_lift=1.0
        )

        assert isinstance(result, dict)
        assert mock_firestore.return_value.create_document.called