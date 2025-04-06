import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # 不要な場合コメントアウト/削除
# import io # 不要な場合コメントアウト/削除
# import base64 # 不要な場合コメントアウト/削除
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta

from backend.analysis.FinancialAnalyzer import FinancialAnalyzer

# bq_service のモックは不要なので削除
# @pytest.fixture
# def mock_bq_service():
#     """BigQueryServiceのモックを提供します"""
#     service = MagicMock()
#     service.query = AsyncMock()
#     service.query.return_value = pd.DataFrame()
#     service.save_dataframe = AsyncMock()
#     return service

@pytest.fixture
def mock_firestore_service():
    """FirestoreServiceのモックを提供します"""
    service = MagicMock()
    service.add_document = AsyncMock()
    service.add_document.return_value = "test_doc_id"
    service.get_document = AsyncMock()
    service.get_document.return_value = {"id": "test_doc_id", "data": {"test": "data"}}
    service.update_document = AsyncMock()
    # BaseAnalyzerで使用される可能性のあるメソッドもモックしておく
    service.query_documents = AsyncMock(return_value=[])
    service.set_document = AsyncMock(return_value="mock-doc-id")
    return service

@pytest.fixture
def sample_financial_data():
    """財務分析用のサンプルデータを提供します"""
    np.random.seed(42)

    # 3年分の月次データを作成
    periods = 36
    dates = pd.date_range(start='2020-01-01', periods=periods, freq='M')

    # 基本的な財務データを生成（カラム名をFinancialAnalyzerの実装に合わせる）
    data = {
        'date': dates,
        'revenue': np.random.normal(100000, 15000, periods) * (1 + np.arange(periods) * 0.01),
        'cogs': np.random.normal(50000, 8000, periods) * (1 + np.arange(periods) * 0.008),
        'operating_expenses': np.random.normal(30000, 5000, periods) * (1 + np.arange(periods) * 0.007),
        'marketing_expenses': np.random.normal(10000, 2000, periods) * (1 + np.arange(periods) * 0.009),
        'r_and_d_expenses': np.random.normal(15000, 3000, periods) * (1 + np.arange(periods) * 0.01),
        'cash_balance': np.random.normal(200000, 30000, periods) * (1 + np.arange(periods) * 0.005), # カラム名変更: cash -> cash_balance
        'accounts_receivable': np.random.normal(25000, 5000, periods) * (1 + np.arange(periods) * 0.008),
        'inventory': np.random.normal(30000, 6000, periods) * (1 + np.arange(periods) * 0.006),
        'fixed_assets': np.random.normal(150000, 10000, periods) * (1 + np.arange(periods) * 0.003),
        'accounts_payable': np.random.normal(20000, 4000, periods) * (1 + np.arange(periods) * 0.007),
        'short_term_debt': np.random.normal(50000, 10000, periods) * (1 - np.arange(periods) * 0.003),
        'long_term_debt': np.random.normal(100000, 20000, periods) * (1 - np.arange(periods) * 0.001),
        'equity': np.random.normal(250000, 30000, periods) * (1 + np.arange(periods) * 0.01),
        'capital_expenditure': np.random.normal(8000, 1500, periods),
        'depreciation': np.random.normal(5000, 800, periods),
        'tax_rate': np.full(periods, 0.25) + np.random.normal(0, 0.01, periods),
    }

    df = pd.DataFrame(data)
    df = df.set_index('date') # 日付をインデックスに設定

    # 計算フィールドを追加 (必要に応じて)
    # df['gross_profit'] = df['revenue'] - df['cogs']
    # df['operating_profit'] = df['gross_profit'] - df['operating_expenses'] - df['marketing_expenses'] - df['r_and_d_expenses']
    # df['ebit'] = df['operating_profit']
    # df['taxes'] = df['ebit'] * df['tax_rate']
    # df['net_income'] = df['ebit'] - df['taxes']
    # df['cash_flow_operations'] = df['net_income'] + df['depreciation'] + np.random.normal(0, 2000, periods)
    # df['free_cash_flow'] = df['cash_flow_operations'] - df['capital_expenditure']

    return df

# ------------------------------------
# 古いテスト関数を削除（またはコメントアウト）
# ------------------------------------
# @pytest.mark.asyncio
# async def test_analyze_profitability(mock_bq_service, sample_financial_data):
#     ...
# @pytest.mark.asyncio
# async def test_analyze_cash_flow(mock_bq_service, sample_financial_data):
#     ...
# @pytest.mark.asyncio
# async def test_analyze_financial_health(mock_bq_service, sample_financial_data):
#     ...
# @pytest.mark.asyncio
# async def test_calculate_roi(mock_bq_service, sample_financial_data):
#     ...
# @pytest.mark.asyncio
# async def test_forecast_financials(mock_bq_service, sample_financial_data):
#     ...
# def test_validate_financial_data():
#     ...
# def test_calculate_financial_ratios():
#     ...

# ------------------------------------
# 新しいテスト関数
# ------------------------------------

def test_calculate_burn_rate(mock_firestore_service, sample_financial_data):
    """calculate_burn_rateメソッドをテストします"""
    # FinancialAnalyzerのインスタンスを作成 (Firestoreモックを使用)
    analyzer = FinancialAnalyzer(firestore_client=mock_firestore_service)

    # テストデータ準備
    test_data = sample_financial_data.copy()
    # 必要であればテストデータをさらに加工

    # メソッド実行 (デフォルト: 月次、現金残高から計算)
    result_monthly_cash = analyzer.calculate_burn_rate(test_data)

    # アサーション (結果の型、主要なキーの存在、値の妥当性など)
    assert isinstance(result_monthly_cash, dict)
    assert 'burn_rate' in result_monthly_cash
    assert 'runway_months' in result_monthly_cash
    assert 'latest_cash' in result_monthly_cash
    assert result_monthly_cash['period'] == 'monthly'
    assert result_monthly_cash['burn_rate'] > 0 # サンプルデータではバーンしているはず
    assert result_monthly_cash['runway_months'] > 0
    assert result_monthly_cash['latest_cash'] == test_data['cash_balance'].iloc[-1]

    # メソッド実行 (四半期、費用項目から計算)
    expense_cols = ['cogs', 'operating_expenses', 'marketing_expenses', 'r_and_d_expenses']
    test_data['total_expenses'] = test_data[expense_cols].sum(axis=1) # 費用合計カラムを追加しておく
    result_quarterly_expense = analyzer.calculate_burn_rate(
        test_data,
        period='quarterly',
        cash_column='cash_balance', # 費用から計算する場合でも最新キャッシュのために必要
        expense_columns=expense_cols # 費用項目を指定
    )

    assert isinstance(result_quarterly_expense, dict)
    assert result_quarterly_expense['period'] == 'quarterly'
    assert result_quarterly_expense['burn_rate'] > 0
    assert result_quarterly_expense['runway_months'] > 0

    # エラーケースのテスト (例: 不正なperiod)
    with pytest.raises(ValueError):
        analyzer.calculate_burn_rate(test_data, period='yearly')

    # エラーケースのテスト (例: インデックスがDatetimeIndexでない)
    with pytest.raises(ValueError):
        analyzer.calculate_burn_rate(test_data.reset_index())

# TODO: 他のメソッドのテスト関数を追加
def test_compare_burn_rate_to_benchmarks(mock_firestore_service):
    """compare_burn_rate_to_benchmarksメソッドをテストします"""
    analyzer = FinancialAnalyzer(firestore_client=mock_firestore_service)

    # サンプルデータ
    company_burn_rate = 50000
    company_runway = 12  # months
    industry_benchmarks = {
        'SaaS': {'burn_rate': 60000, 'runway': 10},
        'FinTech': {'burn_rate': 80000, 'runway': 8},
        'HealthTech': {'burn_rate': 40000, 'runway': 15},
    }
    target_industry = 'SaaS'

    # メソッド実行 (業界が存在する場合)
    result_saas = analyzer.compare_burn_rate_to_benchmarks(
        company_burn_rate, company_runway, industry_benchmarks, target_industry
    )

    # アサーション (SaaS業界)
    assert isinstance(result_saas, dict)
    assert result_saas['industry'] == target_industry
    assert result_saas['benchmark_burn_rate'] == industry_benchmarks[target_industry]['burn_rate']
    assert result_saas['benchmark_runway'] == industry_benchmarks[target_industry]['runway']
    assert 'burn_rate_ratio' in result_saas
    assert 'runway_ratio' in result_saas
    assert 'burn_rate_performance' in result_saas
    assert 'runway_performance' in result_saas
    assert 'overall_score' in result_saas
    assert 0 <= result_saas['overall_score'] <= 100

    # メソッド実行 (業界が存在しない場合 - 平均値を使用)
    unknown_industry = 'Gaming'
    result_unknown = analyzer.compare_burn_rate_to_benchmarks(
        company_burn_rate, company_runway, industry_benchmarks, unknown_industry
    )

    # アサーション (平均値)
    avg_burn = np.mean([b['burn_rate'] for b in industry_benchmarks.values()])
    avg_runway = np.mean([b['runway'] for b in industry_benchmarks.values()])
    assert isinstance(result_unknown, dict)
    assert result_unknown['industry'] == unknown_industry
    assert result_unknown['benchmark_burn_rate'] == avg_burn
    assert result_unknown['benchmark_runway'] == avg_runway
    assert 'overall_score' in result_unknown
    assert 0 <= result_unknown['overall_score'] <= 100

    # エッジケース (ベンチマークが0の場合)
    zero_benchmark = {'ZeroBurn': {'burn_rate': 0, 'runway': 0}}
    result_zero = analyzer.compare_burn_rate_to_benchmarks(
        company_burn_rate, company_runway, zero_benchmark, 'ZeroBurn'
    )
    assert result_zero['burn_rate_ratio'] == float('inf')
    assert result_zero['runway_ratio'] == float('inf')
    # スコアが妥当な範囲にあることを確認 (tanhにより発散しないはず)
    assert 0 <= result_zero['overall_score'] <= 100

def test_analyze_unit_economics(mock_firestore_service):
    """analyze_unit_economicsメソッドをテストします"""
    analyzer = FinancialAnalyzer(firestore_client=mock_firestore_service)

    # サンプルデータの作成
    customer_ids = [f'cust_{i}' for i in range(100)]
    acquisition_dates = pd.to_datetime([datetime(2022, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(100)])
    # 一部の顧客はチャーン
    churn_dates = acquisition_dates + pd.to_timedelta(np.random.randint(30, 720), unit='D')
    churn_dates[np.random.rand(100) > 0.7] = pd.NaT # 約30%がチャーンしていない

    customer_data = pd.DataFrame({
        'customer_id': customer_ids,
        'acquisition_date': acquisition_dates,
        'churn_date': churn_dates
    })

    cost_data = pd.DataFrame({
        'customer_id': customer_ids,
        'acquisition_cost': np.random.normal(100, 20, 100)
    })

    revenue_data_list = []
    for idx, row in customer_data.iterrows():
        start_date = row['acquisition_date']
        end_date = row['churn_date'] if pd.notna(row['churn_date']) else datetime.now()
        num_months = max(1, int((end_date - start_date).days / 30))
        dates = pd.date_range(start=start_date, periods=num_months, freq='M')
        revenues = np.random.normal(50, 10, num_months)
        revenue_data_list.append(pd.DataFrame({
            'customer_id': row['customer_id'],
            'date': dates,
            'revenue': revenues
        }))
    revenue_data = pd.concat(revenue_data_list, ignore_index=True)

    # メソッド実行 (デフォルトカラム名)
    result = analyzer.analyze_unit_economics(
        revenue_data,
        customer_data,
        cost_data
    )

    # アサーション (デフォルトカラム名)
    assert isinstance(result, dict)
    assert 'cac' in result
    assert 'avg_tenure' in result
    assert 'arpu' in result
    assert 'ltv_simple' in result
    assert 'ltv_dcf' in result
    assert 'ltv_cac_ratio_simple' in result
    assert 'ltv_cac_ratio_dcf' in result
    assert result['cac'] > 0
    assert result['avg_tenure'] > 0
    assert result['arpu'] > 0
    assert result['ltv_simple'] > 0
    assert result['ltv_dcf'] > 0
    assert result['ltv_cac_ratio_simple'] > 0
    assert result['ltv_cac_ratio_dcf'] > 0

    # メソッド実行 (カスタムカラム名)
    customer_data_custom = customer_data.rename(columns={'customer_id': 'client_id', 'acquisition_date': 'start_dt', 'churn_date': 'end_dt'})
    cost_data_custom = cost_data.rename(columns={'customer_id': 'client_id', 'acquisition_cost': 'marketing_spend'})
    revenue_data_custom = revenue_data.rename(columns={'customer_id': 'client_id', 'revenue': 'sales'})

    result_custom = analyzer.analyze_unit_economics(
        revenue_data_custom,
        customer_data_custom,
        cost_data_custom,
        customer_id_column='client_id',
        revenue_column='sales',
        acquisition_cost_column='marketing_spend',
        acquisition_date_column='start_dt',
        churn_date_column='end_dt'
    )

    # アサーション (カスタムカラム名 - 結果がほぼ同じであることを確認)
    assert isinstance(result_custom, dict)
    assert np.isclose(result_custom['cac'], result['cac'])
    assert np.isclose(result_custom['avg_tenure'], result['avg_tenure'])
    assert np.isclose(result_custom['arpu'], result['arpu'])
    assert np.isclose(result_custom['ltv_simple'], result['ltv_simple'])
    assert np.isclose(result_custom['ltv_dcf'], result['ltv_dcf'])

    # エラーケース (必須カラムがない)
    with pytest.raises(KeyError):
        analyzer.analyze_unit_economics(revenue_data, customer_data, cost_data.drop(columns=['acquisition_cost']))
    with pytest.raises(KeyError):
        analyzer.analyze_unit_economics(revenue_data.drop(columns=['revenue']), customer_data, cost_data)
    with pytest.raises(KeyError):
        analyzer.analyze_unit_economics(revenue_data, customer_data.drop(columns=['acquisition_date']), cost_data)

def test_calculate_growth_metrics(mock_firestore_service, sample_financial_data):
    """calculate_growth_metricsメソッドをテストします"""
    analyzer = FinancialAnalyzer(firestore_client=mock_firestore_service)

    test_data = sample_financial_data.copy()

    # Rule of 40 テスト用に利益率カラムを追加 (仮の値)
    test_data['profit_margin'] = np.random.normal(0.15, 0.05, len(test_data)) * 100 # パーセント表示

    # メソッド実行 (デフォルト: revenue)
    result_revenue = analyzer.calculate_growth_metrics(test_data)

    # アサーション (revenue)
    assert isinstance(result_revenue, dict)
    assert 'avg_mom_growth' in result_revenue
    assert 'avg_qoq_growth' in result_revenue
    assert 'avg_yoy_growth' in result_revenue
    assert 'latest_mom_growth' in result_revenue
    assert 'latest_qoq_growth' in result_revenue
    assert 'latest_yoy_growth' in result_revenue
    assert 't2d3_score' in result_revenue
    assert 'rule_of_40_score' in result_revenue
    assert result_revenue['rule_of_40_score'] > 0 # 利益率を追加したので計算されるはず

    # メソッド実行 (別の指標: operating_expenses)
    result_opex = analyzer.calculate_growth_metrics(test_data, metric_column='operating_expenses')
    assert isinstance(result_opex, dict)
    assert 'avg_mom_growth' in result_opex # キーの存在を確認
    # Rule of 40 は metric_column が revenue でなくても計算される (利益率は存在するから)
    assert 'rule_of_40_score' in result_opex

    # メソッド実行 (ベンチマークデータ付き)
    benchmark_dates = pd.date_range(start='2020-01-01', periods=36, freq='M')
    benchmark_data = pd.DataFrame({
        'date': benchmark_dates,
        'mom_growth': np.random.normal(0.02, 0.01, 36),
        'qoq_growth': np.random.normal(0.06, 0.02, 36),
        'yoy_growth': np.random.normal(0.25, 0.05, 36)
    }).set_index('date')

    result_benchmark = analyzer.calculate_growth_metrics(test_data, benchmark_data=benchmark_data)
    assert isinstance(result_benchmark, dict)
    assert 'mom_vs_benchmark' in result_benchmark
    assert 'qoq_vs_benchmark' in result_benchmark
    assert 'yoy_vs_benchmark' in result_benchmark

    # エラーケース (存在しないカラム)
    with pytest.raises(KeyError):
        analyzer.calculate_growth_metrics(test_data, metric_column='non_existent')

    # エラーケース (不正なインデックス)
    with pytest.raises(ValueError, match="Could not convert index to DatetimeIndex"):
        analyzer.calculate_growth_metrics(test_data.reset_index())

def test_analyze_funding_efficiency(mock_firestore_service):
    """analyze_funding_efficiencyメソッドをテストします"""
    analyzer = FinancialAnalyzer(firestore_client=mock_firestore_service)

    # サンプルデータ作成
    funding_dates = pd.to_datetime(['2021-01-15', '2022-03-20', '2023-06-01'])
    funding_rounds = ['Seed', 'Series A', 'Series B']
    funding_amounts = [1_000_000, 5_000_000, 15_000_000]
    funding_data = pd.DataFrame({
        'date': funding_dates,
        'round': funding_rounds,
        'amount': funding_amounts,
        'investors': ['VC A', 'VC B, VC C', 'VC D, Growth Fund E'] # 使われていないが形式のため
    })

    valuation_dates = pd.to_datetime(['2021-01-10', '2022-03-15', '2023-05-25'])
    valuations = [5_000_000, 25_000_000, 100_000_000]
    revenue_multiples = [10, 15, 20] # 使われていないが形式のため
    valuation_data = pd.DataFrame({
        'date': valuation_dates,
        'valuation': valuations,
        'revenue_multiple': revenue_multiples
    })

    # 競合データ (オプション)
    comp_a_funding = pd.DataFrame({
        'date': pd.to_datetime(['2021-06-01', '2022-09-01']),
        'round': ['Seed', 'Series A'],
        'amount': [1_500_000, 7_000_000],
        'valuation': [6_000_000, 30_000_000] # 競合データにもvaluationが必要
    })
    comp_b_funding = pd.DataFrame({
        'date': pd.to_datetime(['2021-11-01', '2023-02-01']),
        'round': ['Seed', 'Series A'],
        'amount': [800_000, 4_000_000],
        'valuation': [4_000_000, 20_000_000]
    })
    competitor_funding = {'Competitor A': comp_a_funding, 'Competitor B': comp_b_funding}

    # メソッド実行 (競合データあり)
    result_with_comp = analyzer.analyze_funding_efficiency(
        funding_data, valuation_data, competitor_funding=competitor_funding
    )

    # アサーション (競合データあり)
    assert isinstance(result_with_comp, dict)
    assert 'total_raised' in result_with_comp
    assert 'latest_valuation' in result_with_comp
    assert 'latest_funding_efficiency' in result_with_comp
    assert 'avg_dilution_per_round' in result_with_comp
    assert 'valuation_growth_multiple' in result_with_comp
    assert 'funding_rounds' in result_with_comp
    assert len(result_with_comp['funding_rounds']) == len(funding_data)
    assert 'competitor_comparison' in result_with_comp
    assert 'Competitor A' in result_with_comp['competitor_comparison']
    assert 'summary' in result_with_comp['competitor_comparison']
    assert 'efficiency_percentile' in result_with_comp['competitor_comparison']['summary']
    assert 'dilution_percentile' in result_with_comp['competitor_comparison']['summary']

    # メソッド実行 (競合データなし)
    result_no_comp = analyzer.analyze_funding_efficiency(funding_data, valuation_data)

    # アサーション (競合データなし)
    assert isinstance(result_no_comp, dict)
    assert 'total_raised' in result_no_comp
    assert 'latest_valuation' in result_no_comp
    assert 'competitor_comparison' in result_no_comp
    assert not result_no_comp['competitor_comparison'] # 競合比較は空のはず

    # エラーケース (必須カラムがない)
    with pytest.raises(KeyError):
        analyzer.analyze_funding_efficiency(funding_data.drop(columns=['amount']), valuation_data)
    with pytest.raises(KeyError):
        analyzer.analyze_funding_efficiency(funding_data, valuation_data.drop(columns=['valuation']))

    # エッジケース (資金調達が1ラウンドのみ)
    single_round_funding = funding_data.iloc[:1]
    single_round_valuation = valuation_data.iloc[:1]
    result_single = analyzer.analyze_funding_efficiency(single_round_funding, single_round_valuation)
    assert result_single['valuation_growth_multiple'] == 1.0 # 変化なし