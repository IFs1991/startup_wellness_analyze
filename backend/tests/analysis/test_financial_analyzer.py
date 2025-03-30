import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List, Any, Tuple

from backend.analysis.FinancialAnalyzer import FinancialAnalyzer

@pytest.fixture
def mock_bq_service():
    """BigQueryServiceのモックを提供します"""
    service = MagicMock()
    service.query = AsyncMock()
    service.query.return_value = pd.DataFrame()
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
def sample_financial_data():
    """財務分析用のサンプルデータを提供します"""
    np.random.seed(42)

    # 3年分の月次データを作成
    periods = 36
    dates = pd.date_range(start='2020-01-01', periods=periods, freq='M')

    # 基本的な財務データを生成
    data = {
        'date': dates,
        'revenue': np.random.normal(100000, 15000, periods) * (1 + np.arange(periods) * 0.01),  # 成長傾向
        'cogs': np.random.normal(50000, 8000, periods) * (1 + np.arange(periods) * 0.008),  # 成長傾向（revenueより小さい率）
        'operating_expenses': np.random.normal(30000, 5000, periods) * (1 + np.arange(periods) * 0.007),
        'marketing_expenses': np.random.normal(10000, 2000, periods) * (1 + np.arange(periods) * 0.009),
        'r_and_d_expenses': np.random.normal(15000, 3000, periods) * (1 + np.arange(periods) * 0.01),
        'cash': np.random.normal(200000, 30000, periods) * (1 + np.arange(periods) * 0.005),
        'accounts_receivable': np.random.normal(25000, 5000, periods) * (1 + np.arange(periods) * 0.008),
        'inventory': np.random.normal(30000, 6000, periods) * (1 + np.arange(periods) * 0.006),
        'fixed_assets': np.random.normal(150000, 10000, periods) * (1 + np.arange(periods) * 0.003),
        'accounts_payable': np.random.normal(20000, 4000, periods) * (1 + np.arange(periods) * 0.007),
        'short_term_debt': np.random.normal(50000, 10000, periods) * (1 - np.arange(periods) * 0.003),  # 減少傾向
        'long_term_debt': np.random.normal(100000, 20000, periods) * (1 - np.arange(periods) * 0.001),  # ゆっくり減少
        'equity': np.random.normal(250000, 30000, periods) * (1 + np.arange(periods) * 0.01),
        'capital_expenditure': np.random.normal(8000, 1500, periods),
        'depreciation': np.random.normal(5000, 800, periods),
        'tax_rate': np.full(periods, 0.25) + np.random.normal(0, 0.01, periods),
    }

    df = pd.DataFrame(data)

    # 計算フィールドを追加
    df['gross_profit'] = df['revenue'] - df['cogs']
    df['operating_profit'] = df['gross_profit'] - df['operating_expenses'] - df['marketing_expenses'] - df['r_and_d_expenses']
    df['ebit'] = df['operating_profit']
    df['taxes'] = df['ebit'] * df['tax_rate']
    df['net_income'] = df['ebit'] - df['taxes']

    # キャッシュフロー関連の計算
    df['cash_flow_operations'] = df['net_income'] + df['depreciation'] + np.random.normal(0, 2000, periods)
    df['free_cash_flow'] = df['cash_flow_operations'] - df['capital_expenditure']

    return df

@pytest.mark.asyncio
async def test_analyze_profitability(mock_bq_service, sample_financial_data):
    """収益性分析機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_financial_data

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

        # FinancialAnalyzerのインスタンスを作成
        analyzer = FinancialAnalyzer(bq_service=mock_bq_service)

        # 収益性分析を実行
        result = await analyzer.analyze_profitability(
            query="SELECT * FROM dataset.table WHERE date BETWEEN '2020-01-01' AND '2022-12-31'",
            period="monthly",
            save_results=False
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'gross_margin' in result
        assert 'operating_margin' in result
        assert 'net_profit_margin' in result
        assert 'roi' in result
        assert 'trend_analysis' in result
        assert 'visualization' in result

        # 利益率指標の検証
        assert isinstance(result['gross_margin'], dict)
        assert 'current' in result['gross_margin']
        assert 'historical' in result['gross_margin']
        assert 'industry_benchmark' in result['gross_margin']

        # BigQueryのqueryメソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()

        # 保存メソッドが呼び出されていないことを確認
        mock_bq_service.save_dataframe.assert_not_called()

@pytest.mark.asyncio
async def test_analyze_cash_flow(mock_bq_service, sample_financial_data):
    """キャッシュフロー分析機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_financial_data

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

        # FinancialAnalyzerのインスタンスを作成
        analyzer = FinancialAnalyzer(bq_service=mock_bq_service)

        # キャッシュフロー分析を実行
        result = await analyzer.analyze_cash_flow(
            query="SELECT * FROM dataset.table WHERE date BETWEEN '2020-01-01' AND '2022-12-31'",
            period="quarterly",
            forecast_periods=4,
            save_results=True,
            dataset_id="output_dataset",
            table_id="output_table"
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'operating_cash_flow' in result
        assert 'investing_cash_flow' in result
        assert 'financing_cash_flow' in result
        assert 'free_cash_flow' in result
        assert 'cash_conversion_cycle' in result
        assert 'burn_rate' in result
        assert 'runway' in result
        assert 'forecast' in result
        assert 'visualization' in result

        # BigQueryのクエリと保存メソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()
        mock_bq_service.save_dataframe.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_financial_health(mock_bq_service, sample_financial_data):
    """財務健全性分析機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_financial_data

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

        # FinancialAnalyzerのインスタンスを作成
        analyzer = FinancialAnalyzer(bq_service=mock_bq_service)

        # 財務健全性分析を実行
        result = await analyzer.analyze_financial_health(
            query="SELECT * FROM dataset.table WHERE date BETWEEN '2020-01-01' AND '2022-12-31'",
            period="yearly",
            save_results=False
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'liquidity_ratios' in result
        assert 'solvency_ratios' in result
        assert 'efficiency_ratios' in result
        assert 'z_score' in result
        assert 'risk_assessment' in result
        assert 'visualization' in result

        # 財務比率の検証
        assert isinstance(result['liquidity_ratios'], dict)
        assert 'current_ratio' in result['liquidity_ratios']
        assert 'quick_ratio' in result['liquidity_ratios']

        assert isinstance(result['solvency_ratios'], dict)
        assert 'debt_to_equity' in result['solvency_ratios']
        assert 'interest_coverage' in result['solvency_ratios']

        # BigQueryのqueryメソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()

@pytest.mark.asyncio
async def test_calculate_roi(mock_bq_service, sample_financial_data):
    """投資収益率計算機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_financial_data

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

        # FinancialAnalyzerのインスタンスを作成
        analyzer = FinancialAnalyzer(bq_service=mock_bq_service)

        # ROI計算を実行
        result = await analyzer.calculate_roi(
            query="SELECT * FROM dataset.table WHERE date BETWEEN '2020-01-01' AND '2022-12-31'",
            investment_column="capital_expenditure",
            return_column="net_income",
            time_period=12,
            discount_rate=0.08,
            save_results=False
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'roi' in result
        assert 'payback_period' in result
        assert 'npv' in result
        assert 'irr' in result
        assert 'roi_breakdown' in result
        assert 'sensitivity_analysis' in result
        assert 'visualization' in result

        # ROI関連指標の検証
        assert isinstance(result['roi'], float)
        assert isinstance(result['npv'], float)
        assert isinstance(result['payback_period'], float)

        # BigQueryのqueryメソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()

@pytest.mark.asyncio
async def test_forecast_financials(mock_bq_service, sample_financial_data):
    """財務予測機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_financial_data

    # 可視化関数をモック
    with patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('io.BytesIO') as mock_bytesio, \
         patch('base64.b64encode') as mock_b64encode, \
         patch('statsmodels.tsa.arima.model.ARIMA') as mock_arima:

        # ARIMAモックの設定
        mock_arima_instance = MagicMock()
        mock_arima_instance.fit.return_value = MagicMock()
        mock_arima_instance.fit.return_value.forecast.return_value = pd.Series(np.random.normal(120000, 20000, 12))
        mock_arima.return_value = mock_arima_instance

        # 画像データのモック
        mock_bytesio_instance = MagicMock()
        mock_bytesio.return_value = mock_bytesio_instance
        mock_bytesio_instance.getvalue.return_value = b'image_data'
        mock_b64encode.return_value = b'base64_encoded_image'

        # FinancialAnalyzerのインスタンスを作成
        analyzer = FinancialAnalyzer(bq_service=mock_bq_service)

        # 財務予測を実行
        result = await analyzer.forecast_financials(
            query="SELECT * FROM dataset.table WHERE date BETWEEN '2020-01-01' AND '2022-12-31'",
            target_columns=["revenue", "net_income", "cash"],
            forecast_periods=12,
            method="arima",
            save_results=True,
            dataset_id="output_dataset",
            table_id="forecast_output"
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'forecasts' in result
        assert 'model_quality' in result
        assert 'confidence_intervals' in result
        assert 'seasonal_patterns' in result
        assert 'visualization' in result

        # 予測結果の検証
        assert isinstance(result['forecasts'], dict)
        assert 'revenue' in result['forecasts']
        assert 'net_income' in result['forecasts']
        assert 'cash' in result['forecasts']

        # BigQueryのクエリと保存メソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()
        mock_bq_service.save_dataframe.assert_called_once()

def test_validate_financial_data():
    """財務データ検証機能をテストします"""
    # FinancialAnalyzerのインスタンスを作成
    analyzer = FinancialAnalyzer(bq_service=MagicMock())

    # 有効なデータ
    valid_data = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=12, freq='M'),
        'revenue': np.random.normal(100000, 15000, 12),
        'cogs': np.random.normal(50000, 8000, 12),
        'operating_expenses': np.random.normal(30000, 5000, 12)
    })

    # 無効なデータ（重要な列が不足）
    invalid_data_missing = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=12, freq='M'),
        'revenue': np.random.normal(100000, 15000, 12),
        # 'cogs'が不足
        'operating_expenses': np.random.normal(30000, 5000, 12)
    })

    # 無効なデータ（負の値）
    invalid_data_negative = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=12, freq='M'),
        'revenue': np.random.normal(100000, 15000, 12),
        'cogs': np.random.normal(-50000, 8000, 12),  # 負の値
        'operating_expenses': np.random.normal(30000, 5000, 12)
    })

    # 有効なデータの検証
    is_valid, error_msg = analyzer._validate_financial_data(valid_data, required_columns=['date', 'revenue', 'cogs'])
    assert is_valid is True
    assert error_msg is None

    # 不足データの検証
    is_valid, error_msg = analyzer._validate_financial_data(invalid_data_missing, required_columns=['date', 'revenue', 'cogs'])
    assert is_valid is False
    assert error_msg is not None
    assert 'required column' in error_msg.lower()

    # 負の値データの検証
    is_valid, error_msg = analyzer._validate_financial_data(invalid_data_negative, required_columns=['date', 'revenue', 'cogs'], non_negative_columns=['cogs'])
    assert is_valid is False
    assert error_msg is not None
    assert 'negative value' in error_msg.lower()

def test_calculate_financial_ratios():
    """財務比率計算機能をテストします"""
    # FinancialAnalyzerのインスタンスを作成
    analyzer = FinancialAnalyzer(bq_service=MagicMock())

    # テスト用の財務データ
    financial_data = pd.DataFrame({
        'revenue': [100000, 110000, 120000],
        'cogs': [50000, 54000, 58000],
        'operating_expenses': [30000, 32000, 34000],
        'net_income': [15000, 18000, 21000],
        'total_assets': [200000, 220000, 240000],
        'current_assets': [80000, 90000, 100000],
        'inventory': [30000, 32000, 34000],
        'current_liabilities': [40000, 42000, 44000],
        'total_liabilities': [100000, 105000, 110000],
        'equity': [100000, 115000, 130000]
    })

    # 財務比率の計算
    ratios = analyzer._calculate_financial_ratios(financial_data)

    # 結果の検証
    assert isinstance(ratios, dict)

    # 収益性比率
    assert 'profitability_ratios' in ratios
    assert 'gross_margin' in ratios['profitability_ratios']
    assert 'net_profit_margin' in ratios['profitability_ratios']
    assert 'return_on_assets' in ratios['profitability_ratios']
    assert 'return_on_equity' in ratios['profitability_ratios']

    # 流動性比率
    assert 'liquidity_ratios' in ratios
    assert 'current_ratio' in ratios['liquidity_ratios']
    assert 'quick_ratio' in ratios['liquidity_ratios']

    # 数値チェック
    # 粗利益率 = (売上 - 売上原価) / 売上
    expected_gross_margin = (financial_data['revenue'].mean() - financial_data['cogs'].mean()) / financial_data['revenue'].mean()
    assert np.isclose(ratios['profitability_ratios']['gross_margin'], expected_gross_margin)

    # 流動比率 = 流動資産 / 流動負債
    expected_current_ratio = financial_data['current_assets'].mean() / financial_data['current_liabilities'].mean()
    assert np.isclose(ratios['liquidity_ratios']['current_ratio'], expected_current_ratio)