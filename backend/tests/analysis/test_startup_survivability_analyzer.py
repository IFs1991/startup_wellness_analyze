import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta

from backend.analysis.StartupSurvivabilityAnalyzer import StartupSurvivabilityAnalyzer

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
    return service

@pytest.fixture
def sample_startup_data():
    """スタートアップ生存分析用のサンプルデータを提供します"""
    np.random.seed(42)

    # 200社分のスタートアップデータを生成
    n_samples = 200

    # 設立日を生成（過去5年間でランダム）
    start_date = datetime.now() - timedelta(days=5*365)
    founding_dates = [start_date + timedelta(days=np.random.randint(0, 5*365)) for _ in range(n_samples)]

    # 生存状態を生成（70%は生存、30%は失敗）
    survival_status = np.random.choice([1, 0], size=n_samples, p=[0.7, 0.3])

    # 失敗企業の場合、失敗日を設定
    failure_dates = [None] * n_samples
    for i in range(n_samples):
        if survival_status[i] == 0:  # 失敗した企業
            max_days = (datetime.now() - founding_dates[i]).days
            if max_days > 0:
                days_until_failure = np.random.randint(1, max_days)
                failure_dates[i] = founding_dates[i] + timedelta(days=days_until_failure)

    # 資金調達ラウンドを生成
    funding_rounds = np.random.choice(['Seed', 'Series A', 'Series B', 'Series C', 'None'], size=n_samples, p=[0.3, 0.25, 0.15, 0.1, 0.2])

    # 資金調達額を生成
    funding_amounts = np.zeros(n_samples)
    for i, round_type in enumerate(funding_rounds):
        if round_type == 'Seed':
            funding_amounts[i] = np.random.normal(500000, 200000)
        elif round_type == 'Series A':
            funding_amounts[i] = np.random.normal(3000000, 1000000)
        elif round_type == 'Series B':
            funding_amounts[i] = np.random.normal(10000000, 3000000)
        elif round_type == 'Series C':
            funding_amounts[i] = np.random.normal(30000000, 10000000)
        else:  # No funding
            funding_amounts[i] = 0

    # その他の特徴量を生成
    data = {
        'id': [f'startup_{i}' for i in range(n_samples)],
        'founding_date': founding_dates,
        'failure_date': failure_dates,
        'status': ['active' if s == 1 else 'failed' for s in survival_status],
        'industry': np.random.choice(['Tech', 'Health', 'Finance', 'Retail', 'Education'], size=n_samples),
        'location': np.random.choice(['USA', 'Europe', 'Asia', 'Other'], size=n_samples, p=[0.5, 0.3, 0.15, 0.05]),
        'team_size': np.random.randint(1, 50, size=n_samples),
        'founder_experience': np.random.randint(0, 20, size=n_samples),
        'has_technical_founder': np.random.choice([True, False], size=n_samples),
        'has_business_founder': np.random.choice([True, False], size=n_samples),
        'funding_round': funding_rounds,
        'funding_amount': funding_amounts,
        'burn_rate': np.random.normal(100000, 50000, size=n_samples),
        'revenue': np.random.normal(200000, 100000, size=n_samples) * np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]),  # 40%は収益0
        'customer_count': np.random.randint(0, 1000, size=n_samples),
        'churn_rate': np.random.beta(2, 10, size=n_samples),
        'market_size': np.random.normal(1000000000, 500000000, size=n_samples),
        'competitor_count': np.random.randint(1, 20, size=n_samples)
    }

    # DataFrameを作成し、生存期間を計算
    df = pd.DataFrame(data)
    df['age_days'] = [(datetime.now() - date).days for date in df['founding_date']]

    # 資金が尽きるまでの日数を計算（単純化のため、burn_rateとfunding_amountから計算）
    df['runway_days'] = np.where(df['burn_rate'] > 0, df['funding_amount'] / df['burn_rate'] * 30, 365)

    return df

@pytest.mark.asyncio
async def test_predict_survival_probability(mock_bq_service, sample_startup_data):
    """生存確率予測機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_startup_data

    # 可視化関数をモック
    with patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('io.BytesIO') as mock_bytesio, \
         patch('base64.b64encode') as mock_b64encode, \
         patch('sklearn.ensemble.RandomForestClassifier') as mock_rf, \
         patch('sklearn.model_selection.train_test_split') as mock_split:

        # モックの設定
        mock_rf_instance = MagicMock()
        mock_rf_instance.fit.return_value = mock_rf_instance
        mock_rf_instance.predict_proba.return_value = np.array([[0.3, 0.7]] * len(sample_startup_data))
        mock_rf_instance.feature_importances_ = np.random.random(10)
        mock_rf.return_value = mock_rf_instance

        mock_split.return_value = (None, None, None, None)

        # 画像データのモック
        mock_bytesio_instance = MagicMock()
        mock_bytesio.return_value = mock_bytesio_instance
        mock_bytesio_instance.getvalue.return_value = b'image_data'
        mock_b64encode.return_value = b'base64_encoded_image'

        # アナライザーのインスタンスを作成
        analyzer = StartupSurvivabilityAnalyzer(bq_service=mock_bq_service)

        # 生存確率予測を実行
        result = await analyzer.predict_survival_probability(
            query="SELECT * FROM dataset.table",
            time_horizon=365,  # 1年後の生存確率
            features=['team_size', 'founder_experience', 'funding_amount', 'burn_rate', 'revenue'],
            save_results=False
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'survival_probabilities' in result
        assert 'feature_importance' in result
        assert 'model_metrics' in result
        assert 'risk_factors' in result
        assert 'visualization' in result

        # 予測結果の検証
        assert isinstance(result['survival_probabilities'], pd.DataFrame)
        assert 'id' in result['survival_probabilities'].columns
        assert 'survival_probability' in result['survival_probabilities'].columns

        # 特徴量重要度の検証
        assert isinstance(result['feature_importance'], dict)

        # BigQueryのqueryメソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()

        # 保存メソッドが呼び出されていないことを確認
        mock_bq_service.save_dataframe.assert_not_called()

@pytest.mark.asyncio
async def test_analyze_survival_curve(mock_bq_service, sample_startup_data):
    """生存曲線分析機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_startup_data

    # 可視化関数をモック
    with patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('io.BytesIO') as mock_bytesio, \
         patch('base64.b64encode') as mock_b64encode, \
         patch('lifelines.KaplanMeierFitter') as mock_km:

        # KaplanMeierFitterのモック
        mock_km_instance = MagicMock()
        mock_km_instance.fit.return_value = mock_km_instance
        mock_km_instance.survival_function_ = pd.DataFrame(
            np.linspace(1.0, 0.2, 100),
            index=np.linspace(0, 2000, 100),
            columns=['KM_estimate']
        )
        mock_km_instance.confidence_interval_ = pd.DataFrame(
            np.array([np.linspace(1.0, 0.3, 100), np.linspace(1.0, 0.1, 100)]).T,
            index=np.linspace(0, 2000, 100),
            columns=['lower', 'upper']
        )
        mock_km.return_value = mock_km_instance

        # 画像データのモック
        mock_bytesio_instance = MagicMock()
        mock_bytesio.return_value = mock_bytesio_instance
        mock_bytesio_instance.getvalue.return_value = b'image_data'
        mock_b64encode.return_value = b'base64_encoded_image'

        # アナライザーのインスタンスを作成
        analyzer = StartupSurvivabilityAnalyzer(bq_service=mock_bq_service)

        # 生存曲線分析を実行
        result = await analyzer.analyze_survival_curve(
            query="SELECT * FROM dataset.table",
            time_column='age_days',
            event_column='status',
            event_observed_value='failed',
            stratify_column='industry',
            save_results=True,
            dataset_id="output_dataset",
            table_id="output_table"
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'overall_survival_curve' in result
        assert 'median_survival_time' in result
        assert 'survival_rates' in result
        assert 'stratified_curves' in result
        assert 'log_rank_test' in result
        assert 'visualization' in result

        # 生存曲線データの検証
        assert isinstance(result['overall_survival_curve'], dict)
        assert 'times' in result['overall_survival_curve']
        assert 'survival_probabilities' in result['overall_survival_curve']
        assert 'confidence_intervals' in result['overall_survival_curve']

        # 生存率データの検証
        assert isinstance(result['survival_rates'], dict)
        assert '1_year' in result['survival_rates']
        assert '2_year' in result['survival_rates']
        assert '5_year' in result['survival_rates']

        # BigQueryのクエリと保存メソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()
        mock_bq_service.save_dataframe.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_risk_factors(mock_bq_service, sample_startup_data):
    """リスク要因分析機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_startup_data

    # 統計モデルをモック
    with patch('lifelines.CoxPHFitter') as mock_cox, \
         patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('io.BytesIO') as mock_bytesio, \
         patch('base64.b64encode') as mock_b64encode:

        # CoxPHFitterのモック
        mock_cox_instance = MagicMock()
        mock_cox_instance.fit.return_value = mock_cox_instance
        mock_cox_instance.summary = pd.DataFrame({
            'coef': [0.5, -0.8, 0.3, -0.4, 0.2],
            'exp(coef)': [1.65, 0.45, 1.35, 0.67, 1.22],
            'se(coef)': [0.1, 0.2, 0.15, 0.12, 0.1],
            'p': [0.01, 0.001, 0.05, 0.01, 0.1]
        }, index=['team_size', 'founder_experience', 'funding_amount', 'burn_rate', 'revenue'])
        mock_cox_instance.log_likelihood_ = -100
        mock_cox_instance.concordance_index_ = 0.75
        mock_cox.return_value = mock_cox_instance

        # 画像データのモック
        mock_bytesio_instance = MagicMock()
        mock_bytesio.return_value = mock_bytesio_instance
        mock_bytesio_instance.getvalue.return_value = b'image_data'
        mock_b64encode.return_value = b'base64_encoded_image'

        # アナライザーのインスタンスを作成
        analyzer = StartupSurvivabilityAnalyzer(bq_service=mock_bq_service)

        # リスク要因分析を実行
        result = await analyzer.analyze_risk_factors(
            query="SELECT * FROM dataset.table",
            features=['team_size', 'founder_experience', 'funding_amount', 'burn_rate', 'revenue'],
            time_column='age_days',
            event_column='status',
            event_observed_value='failed',
            save_results=False
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'risk_factors' in result
        assert 'protective_factors' in result
        assert 'statistical_significance' in result
        assert 'model_metrics' in result
        assert 'hazard_ratios' in result
        assert 'visualization' in result

        # リスク要因データの検証
        assert isinstance(result['risk_factors'], list)
        assert isinstance(result['protective_factors'], list)

        # 統計的有意性の検証
        assert isinstance(result['statistical_significance'], dict)

        # モデル指標の検証
        assert isinstance(result['model_metrics'], dict)
        assert 'concordance_index' in result['model_metrics']
        assert 'log_likelihood' in result['model_metrics']

        # BigQueryのqueryメソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_competitive_landscape(mock_bq_service, sample_startup_data):
    """競合環境分析機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_startup_data

    # 可視化関数をモック
    with patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('io.BytesIO') as mock_bytesio, \
         patch('base64.b64encode') as mock_b64encode:

        # 画像データのモック
        mock_bytesio_instance = MagicMock()
        mock_bytesio.return_value = mock_bytesio_instance
        mock_bytesio_instance.getvalue.return_value = b'image_data'
        mock_b64encode.return_value = b'base64_encoded_image'

        # アナライザーのインスタンスを作成
        analyzer = StartupSurvivabilityAnalyzer(bq_service=mock_bq_service)

        # 競合環境分析を実行
        result = await analyzer.analyze_competitive_landscape(
            query="SELECT * FROM dataset.table",
            industry_column='industry',
            location_column='location',
            funding_column='funding_amount',
            save_results=True,
            dataset_id="output_dataset",
            table_id="output_table"
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'industry_saturation' in result
        assert 'funding_concentration' in result
        assert 'regional_competition' in result
        assert 'survival_by_competition' in result
        assert 'competitive_pressure_index' in result
        assert 'visualization' in result

        # 業界飽和度の検証
        assert isinstance(result['industry_saturation'], dict)

        # 資金集中度の検証
        assert isinstance(result['funding_concentration'], dict)

        # 地域別競合の検証
        assert isinstance(result['regional_competition'], dict)

        # BigQueryのクエリと保存メソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()
        mock_bq_service.save_dataframe.assert_called_once()

@pytest.mark.asyncio
async def test_estimate_runway_and_cashflow_risk(mock_bq_service, sample_startup_data):
    """ランウェイとキャッシュフローリスク分析機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_startup_data

    # 可視化関数をモック
    with patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('io.BytesIO') as mock_bytesio, \
         patch('base64.b64encode') as mock_b64encode:

        # 画像データのモック
        mock_bytesio_instance = MagicMock()
        mock_bytesio.return_value = mock_bytesio_instance
        mock_bytesio_instance.getvalue.return_value = b'image_data'
        mock_b64encode.return_value = b'base64_encoded_image'

        # アナライザーのインスタンスを作成
        analyzer = StartupSurvivabilityAnalyzer(bq_service=mock_bq_service)

        # ランウェイ分析を実行
        result = await analyzer.estimate_runway_and_cashflow_risk(
            query="SELECT * FROM dataset.table",
            funding_column='funding_amount',
            burn_rate_column='burn_rate',
            revenue_column='revenue',
            save_results=False
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'runway_analysis' in result
        assert 'cash_flow_risk' in result
        assert 'runway_distribution' in result
        assert 'burn_rate_analysis' in result
        assert 'survival_probability_by_runway' in result
        assert 'visualization' in result

        # ランウェイ分析の検証
        assert isinstance(result['runway_analysis'], dict)
        assert 'median_runway' in result['runway_analysis']
        assert 'runway_quartiles' in result['runway_analysis']

        # キャッシュフローリスクの検証
        assert isinstance(result['cash_flow_risk'], dict)
        assert 'high_risk_count' in result['cash_flow_risk']
        assert 'medium_risk_count' in result['cash_flow_risk']
        assert 'low_risk_count' in result['cash_flow_risk']

        # BigQueryのqueryメソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()

def test_validate_survival_data():
    """生存分析データの検証機能をテストします"""
    # アナライザーのインスタンスを作成
    analyzer = StartupSurvivabilityAnalyzer(bq_service=MagicMock())

    # 有効なデータ
    valid_data = pd.DataFrame({
        'id': ['startup_1', 'startup_2', 'startup_3'],
        'founding_date': [datetime(2019, 1, 1), datetime(2018, 6, 1), datetime(2020, 3, 1)],
        'status': ['active', 'failed', 'active'],
        'age_days': [1000, 800, 500],
        'team_size': [10, 5, 15]
    })

    # 無効なデータ（必須列が不足）
    invalid_missing_column = pd.DataFrame({
        'id': ['startup_1', 'startup_2', 'startup_3'],
        # 'founding_date'が不足
        'status': ['active', 'failed', 'active'],
        'age_days': [1000, 800, 500],
        'team_size': [10, 5, 15]
    })

    # 無効なデータ（ステータス値が不適切）
    invalid_status = pd.DataFrame({
        'id': ['startup_1', 'startup_2', 'startup_3'],
        'founding_date': [datetime(2019, 1, 1), datetime(2018, 6, 1), datetime(2020, 3, 1)],
        'status': ['active', 'unknown', 'active'],  # 'unknown'は不適切
        'age_days': [1000, 800, 500],
        'team_size': [10, 5, 15]
    })

    # 有効なデータの検証
    is_valid, error_msg = analyzer._validate_survival_data(
        valid_data,
        required_columns=['id', 'founding_date', 'status', 'age_days'],
        status_column='status',
        valid_status_values=['active', 'failed', 'acquired']
    )
    assert is_valid is True
    assert error_msg is None

    # 必須列が不足するデータの検証
    is_valid, error_msg = analyzer._validate_survival_data(
        invalid_missing_column,
        required_columns=['id', 'founding_date', 'status', 'age_days'],
        status_column='status',
        valid_status_values=['active', 'failed', 'acquired']
    )
    assert is_valid is False
    assert error_msg is not None
    assert 'required column' in error_msg.lower()

    # ステータス値が不適切なデータの検証
    is_valid, error_msg = analyzer._validate_survival_data(
        invalid_status,
        required_columns=['id', 'founding_date', 'status', 'age_days'],
        status_column='status',
        valid_status_values=['active', 'failed', 'acquired']
    )
    assert is_valid is False
    assert error_msg is not None
    assert 'invalid status' in error_msg.lower()

def test_prepare_survival_data():
    """生存分析データの前処理機能をテストします"""
    # アナライザーのインスタンスを作成
    analyzer = StartupSurvivabilityAnalyzer(bq_service=MagicMock())

    # テスト用のデータ
    data = pd.DataFrame({
        'id': ['startup_1', 'startup_2', 'startup_3', 'startup_4'],
        'founding_date': [
            datetime(2019, 1, 1),
            datetime(2018, 6, 1),
            datetime(2020, 3, 1),
            datetime(2017, 5, 1)
        ],
        'status': ['active', 'failed', 'active', 'acquired'],
        'failure_date': [None, datetime(2020, 6, 1), None, None],
        'acquisition_date': [None, None, None, datetime(2021, 8, 1)],
        'team_size': [10, 5, 15, 20],
        'funding_amount': [1000000, 500000, 2000000, 5000000]
    })

    # 前処理を実行
    processed_data = analyzer._prepare_survival_data(
        data,
        time_column=None,  # 自動計算
        event_column='status',
        event_observed_value='failed'
    )

    # 結果の検証
    assert 'duration' in processed_data.columns  # 生存期間が追加されていること
    assert 'event_observed' in processed_data.columns  # イベント観測フラグが追加されていること

    # 生存期間の計算が正しいか確認
    # startupは失敗するまで、または現在までの日数が期間となる
    today = datetime.now()

    # 失敗したスタートアップ (startup_2)
    failure_duration = (datetime(2020, 6, 1) - datetime(2018, 6, 1)).days
    assert processed_data.loc[1, 'duration'] == failure_duration
    assert processed_data.loc[1, 'event_observed'] == 1  # 失敗イベントが観測された

    # アクティブなスタートアップ (startup_1, startup_3)
    assert processed_data.loc[0, 'event_observed'] == 0  # イベントは観測されていない
    assert processed_data.loc[2, 'event_observed'] == 0  # イベントは観測されていない

    # 買収されたスタートアップ (startup_4) - 失敗ではないのでevent_observed=0
    assert processed_data.loc[3, 'event_observed'] == 0