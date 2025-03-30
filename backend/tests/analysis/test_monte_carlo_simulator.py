import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List, Any, Tuple

from backend.analysis.MonteCarloSimulator import MonteCarloSimulator

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
def sample_portfolio_data():
    """ポートフォリオシミュレーション用のサンプルデータを提供します"""
    np.random.seed(42)

    # 20社のスタートアップポートフォリオを生成
    n_startups = 20

    data = {
        'id': [f'startup_{i}' for i in range(n_startups)],
        'investment_amount': np.random.normal(1000000, 500000, n_startups),
        'equity_stake': np.random.uniform(0.05, 0.25, n_startups),
        'success_probability': np.random.beta(2, 5, n_startups),
        'expected_valuation': np.random.lognormal(16, 1, n_startups),  # 対数正規分布で企業価値を生成
        'time_to_exit': np.random.randint(3, 10, n_startups),
        'industry': np.random.choice(['Tech', 'Health', 'Finance', 'Retail', 'Education'], size=n_startups)
    }

    # DataFrameを作成
    df = pd.DataFrame(data)

    # 計算フィールドを追加
    df['expected_return'] = df['equity_stake'] * df['expected_valuation'] - df['investment_amount']
    df['roi'] = df['expected_return'] / df['investment_amount']

    return df

@pytest.fixture
def sample_cashflow_data():
    """キャッシュフローシミュレーション用のサンプルデータを提供します"""
    np.random.seed(42)

    # 36ヶ月のデータを生成
    periods = 36

    # 初期資金
    initial_fund = 10000000

    # スタートアップへの投資パターン（最初の数ヶ月で大きい投資、その後は小さい）
    investments = np.concatenate([
        np.random.normal(1000000, 200000, 5),  # 最初の5ヶ月は大きい投資
        np.random.normal(500000, 100000, 10),  # 次の10ヶ月は中規模の投資
        np.random.normal(200000, 50000, 21)    # 残りは小規模の投資
    ])

    # 運用コスト（毎月増加する傾向）
    operating_costs = np.random.normal(50000, 5000, periods) * (1 + np.arange(periods) * 0.01)

    # リターン（数社からの収入、ほとんど0だがたまに大きい）
    returns = np.zeros(periods)
    # ランダムな月にリターン発生
    return_months = np.random.choice(range(periods), size=5, replace=False)
    returns[return_months] = np.random.normal(2000000, 1000000, 5)

    # キャッシュフロー計算
    cash_flow = returns - investments - operating_costs

    # 残高計算
    balance = np.zeros(periods)
    balance[0] = initial_fund + cash_flow[0]
    for i in range(1, periods):
        balance[i] = balance[i-1] + cash_flow[i]

    # DataFrameを作成
    data = {
        'month': range(1, periods + 1),
        'investment': investments,
        'operating_cost': operating_costs,
        'return': returns,
        'cash_flow': cash_flow,
        'balance': balance
    }

    return pd.DataFrame(data)

@pytest.mark.asyncio
async def test_simulate_portfolio_returns(mock_bq_service, sample_portfolio_data):
    """ポートフォリオリターンシミュレーション機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_portfolio_data

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

        # MonteCarloSimulatorのインスタンスを作成
        simulator = MonteCarloSimulator(bq_service=mock_bq_service)

        # ポートフォリオリターンシミュレーションを実行
        result = await simulator.simulate_portfolio_returns(
            query="SELECT * FROM dataset.table",
            n_simulations=1000,
            investment_amount_column='investment_amount',
            equity_stake_column='equity_stake',
            success_probability_column='success_probability',
            valuation_column='expected_valuation',
            time_horizon=5,
            discount_rate=0.1,
            save_results=False
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'expected_portfolio_return' in result
        assert 'return_distribution' in result
        assert 'probability_of_loss' in result
        assert 'probability_of_5x_return' in result
        assert 'var_95' in result  # 95% Value at Risk
        assert 'expected_exit_values' in result
        assert 'visualization' in result

        # シミュレーション結果の検証
        assert isinstance(result['expected_portfolio_return'], float)
        assert isinstance(result['probability_of_loss'], float)
        assert isinstance(result['probability_of_5x_return'], float)
        assert isinstance(result['var_95'], float)

        # リターン分布の検証
        assert isinstance(result['return_distribution'], dict)
        assert 'percentiles' in result['return_distribution']
        assert 'values' in result['return_distribution']

        # BigQueryのqueryメソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()

        # 保存メソッドが呼び出されていないことを確認
        mock_bq_service.save_dataframe.assert_not_called()

@pytest.mark.asyncio
async def test_simulate_fund_cashflow(mock_bq_service, sample_cashflow_data):
    """ファンドキャッシュフローシミュレーション機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_cashflow_data

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

        # MonteCarloSimulatorのインスタンスを作成
        simulator = MonteCarloSimulator(bq_service=mock_bq_service)

        # ファンドキャッシュフローシミュレーションを実行
        result = await simulator.simulate_fund_cashflow(
            query="SELECT * FROM dataset.table",
            n_simulations=1000,
            initial_fund=10000000,
            investment_pattern_column='investment',
            operating_cost_column='operating_cost',
            return_pattern_column='return',
            time_periods=60,  # 60ヶ月先までシミュレーション
            save_results=True,
            dataset_id="output_dataset",
            table_id="output_table"
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'expected_ending_balance' in result
        assert 'probability_of_running_out' in result
        assert 'minimum_balance_distribution' in result
        assert 'cashflow_projections' in result
        assert 'time_to_depletion' in result
        assert 'visualization' in result

        # キャッシュフロー予測の検証
        assert isinstance(result['expected_ending_balance'], float)
        assert isinstance(result['probability_of_running_out'], float)

        # 最小残高分布の検証
        assert isinstance(result['minimum_balance_distribution'], dict)

        # 枯渇までの時間分布の検証
        assert isinstance(result['time_to_depletion'], dict)

        # BigQueryのクエリと保存メソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()
        mock_bq_service.save_dataframe.assert_called_once()

@pytest.mark.asyncio
async def test_sensitivity_analysis(mock_bq_service, sample_portfolio_data):
    """感度分析機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_portfolio_data

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

        # MonteCarloSimulatorのインスタンスを作成
        simulator = MonteCarloSimulator(bq_service=mock_bq_service)

        # 感度分析を実行
        result = await simulator.sensitivity_analysis(
            query="SELECT * FROM dataset.table",
            n_simulations=100,
            target_variable='expected_return',
            sensitivity_variables={
                'success_probability': {'min': 0.1, 'max': 0.5, 'steps': 5},
                'equity_stake': {'min': 0.05, 'max': 0.3, 'steps': 5}
            },
            save_results=False
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'sensitivity_impact' in result
        assert 'one_way_sensitivity' in result
        assert 'two_way_sensitivity' in result
        assert 'tornado_diagram' in result
        assert 'visualization' in result

        # 感度影響度の検証
        assert isinstance(result['sensitivity_impact'], dict)
        assert 'success_probability' in result['sensitivity_impact']
        assert 'equity_stake' in result['sensitivity_impact']

        # 一方向感度分析の検証
        assert isinstance(result['one_way_sensitivity'], dict)
        assert 'success_probability' in result['one_way_sensitivity']
        assert 'equity_stake' in result['one_way_sensitivity']

        # 二方向感度分析の検証
        assert isinstance(result['two_way_sensitivity'], dict)
        assert 'variables' in result['two_way_sensitivity']
        assert 'values' in result['two_way_sensitivity']

        # BigQueryのqueryメソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()

@pytest.mark.asyncio
async def test_scenario_analysis(mock_bq_service, sample_portfolio_data):
    """シナリオ分析機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_portfolio_data

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

        # MonteCarloSimulatorのインスタンスを作成
        simulator = MonteCarloSimulator(bq_service=mock_bq_service)

        # シナリオ分析を実行
        result = await simulator.scenario_analysis(
            query="SELECT * FROM dataset.table",
            n_simulations=500,
            target_variable='expected_return',
            scenarios={
                'pessimistic': {
                    'success_probability': 0.5,  # 50%減少
                    'expected_valuation': 0.7    # 30%減少
                },
                'base_case': {
                    'success_probability': 1.0,  # 変更なし
                    'expected_valuation': 1.0    # 変更なし
                },
                'optimistic': {
                    'success_probability': 1.2,  # 20%増加
                    'expected_valuation': 1.5    # 50%増加
                }
            },
            save_results=True,
            dataset_id="output_dataset",
            table_id="scenario_output"
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'scenario_results' in result
        assert 'scenario_comparisons' in result
        assert 'probability_distributions' in result
        assert 'scenario_risk_metrics' in result
        assert 'visualization' in result

        # シナリオ結果の検証
        assert isinstance(result['scenario_results'], dict)
        assert 'pessimistic' in result['scenario_results']
        assert 'base_case' in result['scenario_results']
        assert 'optimistic' in result['scenario_results']

        # 各シナリオのリスク指標の検証
        assert isinstance(result['scenario_risk_metrics'], dict)
        assert 'pessimistic' in result['scenario_risk_metrics']
        assert 'base_case' in result['scenario_risk_metrics']
        assert 'optimistic' in result['scenario_risk_metrics']

        # BigQueryのクエリと保存メソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()
        mock_bq_service.save_dataframe.assert_called_once()

@pytest.mark.asyncio
async def test_optimize_portfolio_allocation(mock_bq_service, sample_portfolio_data):
    """ポートフォリオ配分最適化機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_portfolio_data

    # 最適化関連のモジュールをモック
    with patch('scipy.optimize.minimize') as mock_minimize, \
         patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('io.BytesIO') as mock_bytesio, \
         patch('base64.b64encode') as mock_b64encode:

        # 最適化結果のモック
        mock_minimize.return_value = MagicMock(
            success=True,
            x=np.random.dirichlet(np.ones(len(sample_portfolio_data))),  # ランダムな配分
            fun=-0.5  # 負の値（最大化問題のため）
        )

        # 画像データのモック
        mock_bytesio_instance = MagicMock()
        mock_bytesio.return_value = mock_bytesio_instance
        mock_bytesio_instance.getvalue.return_value = b'image_data'
        mock_b64encode.return_value = b'base64_encoded_image'

        # MonteCarloSimulatorのインスタンスを作成
        simulator = MonteCarloSimulator(bq_service=mock_bq_service)

        # ポートフォリオ配分最適化を実行
        result = await simulator.optimize_portfolio_allocation(
            query="SELECT * FROM dataset.table",
            n_simulations=500,
            total_investment=10000000,
            risk_aversion=0.5,
            min_allocation=0.01,  # 最低1%
            max_allocation=0.3,   # 最大30%
            save_results=False
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'optimized_allocations' in result
        assert 'expected_return' in result
        assert 'expected_risk' in result
        assert 'sharpe_ratio' in result
        assert 'efficiency_frontier' in result
        assert 'visualization' in result

        # 最適配分の検証
        assert isinstance(result['optimized_allocations'], pd.DataFrame)
        assert 'id' in result['optimized_allocations'].columns
        assert 'allocation' in result['optimized_allocations'].columns
        assert 'investment_amount' in result['optimized_allocations'].columns

        # パフォーマンス指標の検証
        assert isinstance(result['expected_return'], float)
        assert isinstance(result['expected_risk'], float)
        assert isinstance(result['sharpe_ratio'], float)

        # 効率的フロンティアの検証
        assert isinstance(result['efficiency_frontier'], dict)
        assert 'returns' in result['efficiency_frontier']
        assert 'risks' in result['efficiency_frontier']

        # BigQueryのqueryメソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()

def test_generate_random_returns():
    """ランダムリターン生成機能をテストします"""
    # MonteCarloSimulatorのインスタンスを作成
    simulator = MonteCarloSimulator(bq_service=MagicMock())

    # テストパラメータ
    n_simulations = 1000
    n_companies = 20
    success_probabilities = np.random.beta(2, 5, n_companies)
    valuation_means = np.random.lognormal(16, 1, n_companies)
    valuation_stds = valuation_means * 0.5  # 変動係数50%

    # ランダムリターンを生成
    returns = simulator._generate_random_returns(
        n_simulations=n_simulations,
        success_probabilities=success_probabilities,
        valuation_means=valuation_means,
        valuation_stds=valuation_stds
    )

    # 結果の検証
    assert isinstance(returns, np.ndarray)
    assert returns.shape == (n_simulations, n_companies)

    # いくつかの統計的性質を確認
    # 成功確率に近い割合の会社が0以上のリターンを持つはず
    non_zero_returns = np.mean(returns > 0, axis=0)
    assert np.allclose(non_zero_returns, success_probabilities, atol=0.1)

    # リターンの平均値が期待値に近いことを確認
    mean_returns = np.mean(returns, axis=0)
    expected_means = success_probabilities * valuation_means
    # 大きなデータセットでは近似的に一致するはず
    assert np.allclose(mean_returns, expected_means, rtol=0.5)

def test_calculate_portfolio_metrics():
    """ポートフォリオ指標計算機能をテストします"""
    # MonteCarloSimulatorのインスタンスを作成
    simulator = MonteCarloSimulator(bq_service=MagicMock())

    # テスト用のシミュレーションリターンデータ
    n_simulations = 1000
    n_companies = 20
    np.random.seed(42)
    simulation_returns = np.random.normal(0.2, 0.5, (n_simulations, n_companies))

    # ランダムな配分を生成（合計1）
    allocations = np.random.dirichlet(np.ones(n_companies))

    # ポートフォリオ指標を計算
    metrics = simulator._calculate_portfolio_metrics(
        simulation_returns=simulation_returns,
        allocations=allocations
    )

    # 結果の検証
    assert isinstance(metrics, dict)
    assert 'expected_return' in metrics
    assert 'risk' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'var_95' in metrics
    assert 'cvar_95' in metrics
    assert 'min_return' in metrics
    assert 'max_return' in metrics
    assert 'return_distribution' in metrics

    # 期待リターンの検証（重み付き平均）
    expected_return = np.mean(np.dot(simulation_returns, allocations))
    assert np.isclose(metrics['expected_return'], expected_return)

    # リスク（標準偏差）の検証
    risk = np.std(np.dot(simulation_returns, allocations))
    assert np.isclose(metrics['risk'], risk)

    # シャープレシオの検証
    sharpe = expected_return / risk if risk > 0 else 0
    assert np.isclose(metrics['sharpe_ratio'], sharpe)

    # 分布が正しく計算されているか検証
    assert isinstance(metrics['return_distribution'], dict)
    assert 'percentiles' in metrics['return_distribution']
    assert 'values' in metrics['return_distribution']