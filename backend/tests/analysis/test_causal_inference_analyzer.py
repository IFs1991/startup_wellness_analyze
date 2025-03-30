import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from backend.analysis.CausalInferenceAnalyzer import (
    CausalInferenceAnalyzer,
    CausalImpactResult,
    HeterogeneousTreatmentEffectResult
)

@pytest.fixture
def mock_ci_result():
    """CausalImpactResultのモックを提供します"""
    return CausalImpactResult(
        point_effect=0.25,
        confidence_interval=(0.15, 0.35),
        p_value=0.02,
        cumulative_effect=1.5,
        posterior_samples=np.random.normal(0.25, 0.05, 1000),
        counterfactual_series=pd.Series(np.random.normal(100, 10, 50)),
        effect_series=pd.Series(np.random.normal(25, 5, 50))
    )

@pytest.fixture
def mock_cate_result():
    """HeterogeneousTreatmentEffectResultのモックを提供します"""
    return HeterogeneousTreatmentEffectResult(
        model_type="causal_forest",
        average_effect=0.3,
        conditional_effects=np.random.normal(0.3, 0.1, 100),
        feature_importance={'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.2},
        confidence_intervals=np.array([(0.2, 0.4) for _ in range(100)]),
        p_values=np.random.uniform(0, 0.05, 100)
    )

@pytest.fixture
def sample_time_series():
    """テスト用の時系列データを生成します"""
    dates = pd.date_range('2022-01-01', periods=100, freq='D')
    # 介入前のデータ
    pre_data = np.random.normal(100, 5, 50)
    # 介入後のデータ（効果を追加）
    post_data = np.random.normal(125, 5, 50)
    target = np.concatenate([pre_data, post_data])

    # 対照系列（介入の影響を受けない）
    control1 = np.random.normal(80, 3, 100)
    control2 = np.random.normal(120, 4, 100)

    return pd.DataFrame({
        'date': dates,
        'target': target,
        'control1': control1,
        'control2': control2
    })

@pytest.fixture
def sample_treatment_data():
    """処理効果分析用のサンプルデータを生成します"""
    np.random.seed(42)
    n = 200

    # 特徴量
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X3 = np.random.uniform(0, 1, n)

    # 処理割り当て（X1, X2に依存）
    propensity = 1 / (1 + np.exp(-(0.5*X1 + 0.5*X2)))
    T = np.random.binomial(1, propensity)

    # 真の処理効果（X1に依存）
    true_effect = 2*X1

    # 結果変数
    y_0 = 1 + 2*X1 + 1*X2 + np.random.normal(0, 1, n)  # 対照結果
    y_1 = y_0 + true_effect  # 処理結果
    Y = y_0 * (1 - T) + y_1 * T  # 観測される結果

    return pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'treatment': T,
        'outcome': Y
    })

@pytest.mark.asyncio
async def test_analyze_difference_in_differences(sample_treatment_data):
    """差分の差分法分析のテストを行います"""
    # モックのStatsmodelsの結果オブジェクト
    with patch('statsmodels.formula.api.ols') as mock_ols:
        # モックの結果オブジェクト
        mock_result = MagicMock()
        mock_result.params = {'treatment:time': 0.25}
        mock_result.conf_int.return_value = pd.DataFrame({
            0: {'treatment:time': 0.15},
            1: {'treatment:time': 0.35}
        })
        mock_result.pvalues = {'treatment:time': 0.02}

        # OLSモデルとフィットメソッドのモック
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_result
        mock_ols.return_value = mock_model

        # テスト対象インスタンス
        analyzer = CausalInferenceAnalyzer()

        # メソッド実行
        result = analyzer.analyze_difference_in_differences(
            data=sample_treatment_data,
            treatment_col='treatment',
            time_col='time_period',  # 注: データに追加するためにパッチを適用
            outcome_col='outcome',
            covariates=['X1', 'X2']
        )

        # 検証
        assert isinstance(result, CausalImpactResult)
        assert result.point_effect == 0.25
        assert result.confidence_interval == (0.15, 0.35)
        assert result.p_value == 0.02

@pytest.mark.asyncio
async def test_analyze_causal_impact(sample_time_series):
    """因果的影響分析のテストを行います"""
    with patch('causalimpact.CausalImpact') as mock_ci:
        # モックのCausalImpactインスタンス
        ci_instance = MagicMock()

        # summaryのモック
        ci_instance.summary.return_value = "Summary data"

        # 結果のモック
        ci_instance.summary_data = {
            'average': {'absolute': {'actual': 100, 'predicted': 80, 'prediction_interval': [75, 85]}},
            'average': {'relative': {'effect': 0.25, 'lower': 0.15, 'upper': 0.35}},
            'cumulative': {'absolute': {'effect': 500}},
            'p_value': 0.02
        }

        # シリーズのモック
        ci_instance.series = pd.DataFrame({
            'response': sample_time_series['target'],
            'point_pred': sample_time_series['target'] * 0.8,
            'point_effect': sample_time_series['target'] * 0.2,
            'post_period_response': sample_time_series['target'].iloc[50:]
        })

        # インスタンス作成のモック
        mock_ci.return_value = ci_instance

        # テスト対象インスタンス
        analyzer = CausalInferenceAnalyzer()

        # メソッド実行
        result = analyzer.analyze_causal_impact(
            time_series=sample_time_series,
            intervention_time='2022-02-20',
            target_col='target',
            control_cols=['control1', 'control2']
        )

        # 検証
        assert isinstance(result, CausalImpactResult)
        assert result.point_effect == 0.25
        assert result.confidence_interval == (0.15, 0.35)
        assert result.p_value == 0.02
        assert result.cumulative_effect == 500

@pytest.mark.asyncio
async def test_analyze_causal_impact_bayesian(sample_time_series):
    """ベイジアン因果的影響分析のテストを行います"""
    # PyMCとArvizのモック
    with patch('pymc.Model') as mock_model_cls, \
         patch('pymc.sample') as mock_sample, \
         patch('arviz.summary') as mock_summary:

        # 事後分布サンプルのモック
        mock_trace = MagicMock()
        mock_sample.return_value = mock_trace

        # モデルコンテキストのモック
        mock_model = MagicMock()
        mock_model_cls.return_value.__enter__.return_value = mock_model

        # 要約統計量のモック
        mock_summary.return_value = pd.DataFrame({
            'mean': {'effect': 0.25},
            'hdi_3%': {'effect': 0.15},
            'hdi_97%': {'effect': 0.35}
        })

        # テスト対象インスタンス
        analyzer = CausalInferenceAnalyzer()

        # 計算メソッドをモック
        with patch.object(analyzer, '_calculate_posterior_predictions') as mock_calc:
            mock_calc.return_value = (
                pd.Series(np.random.normal(80, 5, 50), index=sample_time_series.index[50:]),
                pd.Series(np.random.normal(20, 2, 50), index=sample_time_series.index[50:])
            )

            # メソッド実行
            result = await analyzer.analyze_causal_impact_bayesian(
                time_series=sample_time_series,
                intervention_time='2022-02-20',
                target_col='target',
                control_cols=['control1', 'control2']
            )

            # 検証
            assert isinstance(result, CausalImpactResult)
            assert result.point_effect == 0.25
            assert result.confidence_interval == (0.15, 0.35)
            assert result.posterior_samples is not None

@pytest.mark.asyncio
async def test_estimate_revenue_impact():
    """収益影響推定のテストを行います"""
    # テスト用データの作成
    revenue_data = pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=100, freq='D'),
        'company_id': ['comp1'] * 100,
        'revenue': np.concatenate([np.random.normal(100000, 10000, 50),
                                  np.random.normal(120000, 10000, 50)])
    })

    intervention_data = pd.DataFrame({
        'company_id': ['comp1', 'comp2', 'comp3'],
        'intervention_date': ['2022-02-20', '2022-03-15', '2022-04-10'],
        'program_type': ['wellness', 'training', 'leadership']
    })

    # CausalInferenceAnalyzerのインスタンス
    analyzer = CausalInferenceAnalyzer()

    # analyze_causal_impactメソッドをモック
    with patch.object(analyzer, 'analyze_causal_impact') as mock_analyze:
        # 結果のモック
        mock_analyze.return_value = CausalImpactResult(
            point_effect=0.2,
            confidence_interval=(0.1, 0.3),
            p_value=0.01,
            cumulative_effect=1000000,
            counterfactual_series=pd.Series(),
            effect_series=pd.Series()
        )

        # メソッド実行
        result = await analyzer.estimate_revenue_impact(
            revenue_data=revenue_data,
            intervention_data=intervention_data,
            company_id='comp1',
            intervention_date='2022-02-20',
            control_features=None,
            method='causal_impact'
        )

        # 検証
        assert isinstance(result, dict)
        assert 'relative_effect' in result
        assert 'absolute_effect' in result
        assert 'confidence_interval' in result
        assert 'p_value' in result
        assert 'interpretation' in result
        assert result['relative_effect'] == 0.2

@pytest.mark.asyncio
async def test_estimate_heterogeneous_treatment_effects(sample_treatment_data):
    """異質処理効果推定のテストを行います"""
    # EconMLのモデルをモック
    with patch('econml.dml.CausalForestDML') as mock_forest_cls, \
         patch('econml.inference.BootstrapInference') as mock_inference_cls:

        # モデルインスタンスのモック
        mock_model = MagicMock()
        mock_forest_cls.return_value = mock_model

        # 推論インスタンスのモック
        mock_inference = MagicMock()
        mock_inference_cls.return_value = mock_inference

        # effect関連メソッドのモック
        mock_model.effect.return_value = np.array([0.2, 0.3, 0.4, 0.2, 0.3])
        mock_model.effect_interval.return_value = (
            np.array([0.1, 0.2, 0.3, 0.1, 0.2]),
            np.array([0.3, 0.4, 0.5, 0.3, 0.4])
        )
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])

        # テスト対象インスタンス
        analyzer = CausalInferenceAnalyzer()

        # メソッド実行
        result = analyzer.estimate_heterogeneous_treatment_effects(
            data=sample_treatment_data,
            treatment_col='treatment',
            outcome_col='outcome',
            features=['X1', 'X2', 'X3'],
            method='causal_forest'
        )

        # 検証
        assert isinstance(result, HeterogeneousTreatmentEffectResult)
        assert result.model_type == 'causal_forest'
        assert isinstance(result.average_effect, float)
        assert isinstance(result.conditional_effects, np.ndarray)
        assert isinstance(result.feature_importance, dict)
        assert list(result.feature_importance.keys()) == ['X1', 'X2', 'X3']

@pytest.mark.asyncio
async def test_visualize_causal_effect(mock_ci_result):
    """因果効果の可視化テストを行います"""
    # matplotlib.pyplotのモック
    with patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.savefig') as mock_savefig:

        # モックフィギュアとアックス
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_figure.return_value = mock_fig

        # テスト対象インスタンス
        analyzer = CausalInferenceAnalyzer()

        # メソッド実行
        result = analyzer.visualize_causal_effect(
            result=mock_ci_result,
            title="テスト可視化",
            save_path="test_figure.png"
        )

        # 検証
        assert result == mock_fig
        mock_fig.add_subplot.assert_called()
        mock_ax.plot.assert_called()
        mock_savefig.assert_called_once_with("test_figure.png")

@pytest.mark.asyncio
async def test_calculate_roi_components():
    """ROI構成要素計算のテストを行います"""
    # テスト用データの作成
    revenue_data = pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=100, freq='D'),
        'company_id': ['comp1'] * 100,
        'revenue': np.random.normal(100000, 10000, 100)
    })

    valuation_data = pd.DataFrame({
        'company_id': ['comp1', 'comp2', 'comp3'],
        'valuation_date': ['2022-01-01', '2022-01-01', '2022-01-01'],
        'pre_valuation': [10000000, 20000000, 30000000],
        'post_valuation': [12000000, 22000000, 33000000]
    })

    program_cost_data = pd.DataFrame({
        'company_id': ['comp1', 'comp2', 'comp3'],
        'program_type': ['wellness', 'training', 'leadership'],
        'cost': [100000, 150000, 200000]
    })

    investment_data = pd.DataFrame({
        'company_id': ['comp1', 'comp2', 'comp3'],
        'investment_date': ['2022-01-15', '2022-01-15', '2022-01-15'],
        'investment_amount': [2000000, 3000000, 4000000]
    })

    # CausalInferenceAnalyzerのインスタンス
    analyzer = CausalInferenceAnalyzer()

    # estimate_revenue_impactメソッドをモック
    with patch.object(analyzer, 'estimate_revenue_impact') as mock_revenue_impact:
        # 結果のモック
        mock_revenue_impact.return_value = {
            'relative_effect': 0.2,
            'absolute_effect': 1000000,
            'confidence_interval': (500000, 1500000),
            'p_value': 0.01
        }

        # メソッド実行
        result = await analyzer.calculate_roi_components(
            company_id='comp1',
            start_date='2022-01-01',
            end_date='2022-04-10',
            revenue_data=revenue_data,
            valuation_data=valuation_data,
            program_cost_data=program_cost_data,
            investment_data=investment_data
        )

        # 検証
        assert isinstance(result, dict)
        assert 'revenue_impact' in result
        assert 'valuation_impact' in result
        assert 'program_cost' in result
        assert 'roi' in result
        assert 'confidence_interval' in result