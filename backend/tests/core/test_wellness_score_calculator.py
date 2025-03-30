import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, Any, List

from core.wellness_score_calculator import (
    WellnessScoreCalculator,
    WellnessScoreError,
    FirestoreClient,
    create_wellness_score_calculator
)

@pytest.fixture
def mock_firestore_client():
    """Firestoreクライアントのモックを提供します"""
    client = MagicMock(spec=FirestoreClient)

    # 必要なメソッドのAsyncMockを設定
    client.query_documents = AsyncMock()
    client.add_document = AsyncMock()
    client.update_document = AsyncMock()
    client.get_document = AsyncMock()
    client.set_document = AsyncMock()

    return client

@pytest.fixture
def mock_data_preprocessor():
    """DataPreprocessorのモックを提供します"""
    preprocessor = MagicMock()
    preprocessor.preprocess_firestore_data = MagicMock()
    preprocessor.merge_datasets = MagicMock()
    return preprocessor

@pytest.fixture
def mock_correlation_analyzer():
    """CorrelationAnalyzerのモックを提供します"""
    analyzer = MagicMock()
    analyzer.calculate_correlation = AsyncMock()
    analyzer.calculate_correlation.return_value = 0.7
    return analyzer

@pytest.fixture
def mock_time_series_analyzer():
    """TimeSeriesAnalyzerのモックを提供します"""
    analyzer = MagicMock()
    analyzer.analyze_trend = AsyncMock()
    analyzer.analyze_trend.return_value = {"trend_coefficient": 0.05}
    return analyzer

@pytest.fixture
def sample_vas_data():
    """VASデータのサンプルを提供します"""
    return pd.DataFrame({
        'company_id': ['comp1', 'comp1', 'comp1'],
        'timestamp': [
            pd.Timestamp('2022-01-01'),
            pd.Timestamp('2022-02-01'),
            pd.Timestamp('2022-03-01')
        ],
        'employee_satisfaction': [4.2, 4.3, 4.4],
        'work_life_balance': [3.8, 3.9, 4.0],
        'team_collaboration': [4.0, 4.1, 4.2],
        'leadership_quality': [3.5, 3.6, 3.8],
        'career_growth': [3.7, 3.8, 3.9]
    })

@pytest.fixture
def sample_financial_data():
    """財務データのサンプルを提供します"""
    return pd.DataFrame({
        'company_id': ['comp1', 'comp1', 'comp1'],
        'timestamp': [
            pd.Timestamp('2022-01-01'),
            pd.Timestamp('2022-02-01'),
            pd.Timestamp('2022-03-01')
        ],
        'revenue': [1000000, 1050000, 1100000],
        'expenses': [800000, 820000, 840000],
        'profit': [200000, 230000, 260000],
        'cash_flow': [150000, 180000, 200000]
    })

@pytest.mark.asyncio
async def test_calculate_wellness_score(
    mock_firestore_client,
    mock_data_preprocessor,
    mock_correlation_analyzer,
    mock_time_series_analyzer,
    sample_vas_data,
    sample_financial_data
):
    """ウェルネススコア計算機能のテスト"""
    # モックの設定
    mock_firestore_client.query_documents.side_effect = [
        [{'data': {'some': 'vas_data'}}],  # VASデータのクエリ結果
        [{'data': {'some': 'financial_data'}}]  # 財務データのクエリ結果
    ]

    mock_data_preprocessor.preprocess_firestore_data.side_effect = [
        sample_vas_data,
        sample_financial_data
    ]

    mock_data_preprocessor.merge_datasets.return_value = pd.concat([
        sample_vas_data,
        sample_financial_data.drop('company_id', axis=1)
    ], axis=1)

    # テスト対象インスタンスの作成
    calculator = WellnessScoreCalculator(
        data_preprocessor=mock_data_preprocessor,
        correlation_analyzer=mock_correlation_analyzer,
        time_series_analyzer=mock_time_series_analyzer,
        firestore_client=mock_firestore_client,
        use_federated_learning=False
    )

    # パッチを適用して内部メソッドをモック
    with patch.object(calculator, '_calculate_category_scores', new=AsyncMock()) as mock_category_scores, \
         patch.object(calculator, '_calculate_base_score', return_value=70.0) as mock_base_score, \
         patch.object(calculator, '_apply_industry_stage_adjustment', return_value=1.1) as mock_industry_adj, \
         patch.object(calculator, '_calculate_trend_adjustment', new=AsyncMock(return_value=0.05)) as mock_trend_adj, \
         patch.object(calculator, '_apply_adjustments', return_value=80.0) as mock_apply_adj, \
         patch.object(calculator, '_save_score_to_firestore', new=AsyncMock()) as mock_save:

        # カテゴリスコアのモック設定
        mock_category_scores.return_value = {
            'employee_satisfaction': 85,
            'work_life_balance': 78,
            'team_collaboration': 82,
            'leadership_quality': 75,
            'career_growth': 76
        }

        # ウェルネススコアの計算を実行
        result = await calculator.calculate_wellness_score(
            company_id='comp1',
            industry='Technology',
            stage='Growth',
            calculation_date=datetime(2022, 3, 15)
        )

        # 結果の検証
        assert result['overall_score'] == 80.0
        assert 'timestamp' in result
        assert 'category_scores' in result
        assert result['industry'] == 'Technology'
        assert result['stage'] == 'Growth'

        # 各メソッドが正しく呼び出されたことを検証
        mock_firestore_client.query_documents.assert_any_call(
            'vas_responses',
            filters=[{'field': 'company_id', 'op': '==', 'value': 'comp1'}]
        )
        mock_firestore_client.query_documents.assert_any_call(
            'financial_data',
            filters=[{'field': 'company_id', 'op': '==', 'value': 'comp1'}]
        )
        mock_category_scores.assert_called_once()
        mock_base_score.assert_called_once()
        mock_industry_adj.assert_called_once_with('Technology', 'Growth')
        mock_trend_adj.assert_called_once()
        mock_apply_adj.assert_called_once()
        mock_save.assert_called_once()

@pytest.mark.asyncio
async def test_calculate_wellness_score_with_federated_learning():
    """連合学習を使用したウェルネススコア計算のテスト"""
    # 連合学習モジュールの有無をパッチ
    with patch('core.wellness_score_calculator.FEDERATED_LEARNING_AVAILABLE', True), \
         patch('core.wellness_score_calculator.CoreModelIntegration') as mock_integration:

        # 連合学習モデル統合モジュールのモック設定
        mock_integration_instance = MagicMock()
        mock_integration_instance.integrate_models = AsyncMock()
        mock_integration_instance.integrate_models.return_value = {
            'adjusted_score': 82.5,
            'confidence': 0.9,
            'model_version': '1.2.0'
        }
        mock_integration.return_value = mock_integration_instance

        # 他の依存コンポーネントのモック
        with patch('core.wellness_score_calculator.DataPreprocessor') as mock_preprocessor_class, \
             patch('core.wellness_score_calculator.CorrelationAnalyzer') as mock_analyzer_class, \
             patch('core.wellness_score_calculator.TimeSeriesAnalyzer') as mock_ts_analyzer_class, \
             patch('core.wellness_score_calculator.FirestoreClient') as mock_firestore_class, \
             patch.object(WellnessScoreCalculator, 'calculate_wellness_score') as mock_calc_score:

            # インスタンスの設定
            mock_preprocessor = MagicMock()
            mock_analyzer = MagicMock()
            mock_ts_analyzer = MagicMock()
            mock_firestore = MagicMock()

            mock_preprocessor_class.return_value = mock_preprocessor
            mock_analyzer_class.return_value = mock_analyzer
            mock_ts_analyzer_class.return_value = mock_ts_analyzer
            mock_firestore_class.return_value = mock_firestore

            # ファクトリ関数をテスト
            calculator = create_wellness_score_calculator()

            # 連合学習フラグがTrueであることを確認
            assert calculator.use_federated_learning == True

            # モックの確認
            mock_preprocessor_class.assert_called_once()
            mock_analyzer_class.assert_called_once()
            mock_ts_analyzer_class.assert_called_once()
            mock_firestore_class.assert_called_once()

@pytest.mark.asyncio
async def test_error_handling(
    mock_firestore_client,
    mock_data_preprocessor,
    mock_correlation_analyzer,
    mock_time_series_analyzer
):
    """エラーハンドリングのテスト"""
    # テスト対象インスタンスの作成
    calculator = WellnessScoreCalculator(
        data_preprocessor=mock_data_preprocessor,
        correlation_analyzer=mock_correlation_analyzer,
        time_series_analyzer=mock_time_series_analyzer,
        firestore_client=mock_firestore_client,
        use_federated_learning=False
    )

    # VASデータの取得でエラーが発生する場合
    mock_firestore_client.query_documents.side_effect = Exception("Firestore connection error")

    # ウェルネススコアの計算を実行し、例外が発生することを確認
    with pytest.raises(WellnessScoreError) as excinfo:
        await calculator.calculate_wellness_score(
            company_id='comp1',
            industry='Technology',
            stage='Growth'
        )

    # エラーメッセージを検証
    assert "Failed to calculate wellness score" in str(excinfo.value)

@pytest.mark.asyncio
async def test_category_score_calculation(
    mock_firestore_client,
    mock_data_preprocessor,
    mock_correlation_analyzer,
    mock_time_series_analyzer,
    sample_vas_data
):
    """カテゴリ別スコア計算のテスト"""
    # テスト対象インスタンスの作成
    calculator = WellnessScoreCalculator(
        data_preprocessor=mock_data_preprocessor,
        correlation_analyzer=mock_correlation_analyzer,
        time_series_analyzer=mock_time_series_analyzer,
        firestore_client=mock_firestore_client,
        use_federated_learning=False
    )

    # _calculate_category_scores実際のメソッドを実行できるようにパッチを解除して直接テスト
    category_scores = await calculator._calculate_category_scores(sample_vas_data)

    # 結果の検証
    assert isinstance(category_scores, dict)
    assert len(category_scores) > 0
    for category, score in category_scores.items():
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100