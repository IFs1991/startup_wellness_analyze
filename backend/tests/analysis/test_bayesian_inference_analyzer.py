import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List, Any, Tuple

from backend.analysis.BayesianInferenceAnalyzer import BayesianInferenceAnalyzer

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
    return service

@pytest.fixture
def sample_data():
    """ベイズ分析用のサンプルデータを提供します"""
    np.random.seed(42)

    # 基本的なデータフレームを生成
    data = {
        'revenue': np.random.normal(1000000, 200000, 100),
        'marketing_spend': np.random.normal(50000, 10000, 100),
        'product_dev_cost': np.random.normal(30000, 5000, 100),
        'customer_count': np.random.randint(100, 1000, 100),
        'churn_rate': np.random.beta(2, 10, 100),
        'team_size': np.random.randint(5, 50, 100),
        'market_size': np.random.normal(10000000, 2000000, 100),
        'competitor_count': np.random.randint(1, 10, 100)
    }

    return pd.DataFrame(data)

@pytest.fixture
def mock_pymc_trace():
    """PyMCのトレース結果をモックします"""
    mock_trace = MagicMock()
    mock_trace.posterior = {
        'alpha': np.random.normal(0, 1, (4, 1000)),
        'beta': np.random.normal(0, 1, (4, 1000, 3)),
        'sigma': np.random.gamma(1, 1, (4, 1000))
    }
    return mock_trace

@pytest.mark.asyncio
async def test_fit_linear_model(mock_bq_service, sample_data):
    """線形モデルのフィット機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_data

    # PyMCとArvizのモック
    with patch('pymc.Model') as mock_model, \
         patch('pymc.sample') as mock_sample, \
         patch('arviz.summary') as mock_summary, \
         patch('arviz.plot_trace') as mock_plot, \
         patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('io.BytesIO') as mock_bytesio, \
         patch('base64.b64encode') as mock_b64encode:

        # モックの設定
        mock_context = MagicMock()
        mock_model.return_value.__enter__.return_value = mock_context

        mock_trace = MagicMock()
        mock_sample.return_value = mock_trace

        # サマリーデータのモック
        summary_data = pd.DataFrame({
            'mean': [0.5, 2.0, 0.1],
            'sd': [0.1, 0.3, 0.05],
            'hdi_3%': [0.3, 1.5, 0.01],
            'hdi_97%': [0.7, 2.5, 0.2],
            'r_hat': [1.01, 1.0, 1.02]
        }, index=['alpha', 'beta[0]', 'sigma'])
        mock_summary.return_value = summary_data

        # 画像データのモック
        mock_bytesio_instance = MagicMock()
        mock_bytesio.return_value = mock_bytesio_instance
        mock_bytesio_instance.getvalue.return_value = b'image_data'
        mock_b64encode.return_value = b'base64_encoded_image'

        # アナライザーのインスタンスを作成
        analyzer = BayesianInferenceAnalyzer(bq_service=mock_bq_service)

        # 線形モデルをフィット
        result = await analyzer.fit_linear_model(
            query="SELECT * FROM dataset.table",
            target_variable="revenue",
            predictor_variables=["marketing_spend", "product_dev_cost", "team_size"],
            model_type="normal",
            save_results=False
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'model_summary' in result
        assert 'coefficients' in result
        assert 'diagnostics' in result
        assert 'trace_plot' in result
        assert 'posterior_predictive_plot' in result

        # BigQueryのクエリメソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()

        # PyMCのモデルとサンプリング関数が呼び出されたことを確認
        mock_model.assert_called_once()
        mock_sample.assert_called_once()
        mock_summary.assert_called_once()

@pytest.mark.asyncio
async def test_fit_hierarchical_model(mock_bq_service, sample_data):
    """階層モデルのフィット機能をテストします"""
    # サンプルデータにグループ列を追加
    sample_data['group'] = np.random.choice(['A', 'B', 'C'], size=len(sample_data))

    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_data

    # PyMCとArvizのモック
    with patch('pymc.Model') as mock_model, \
         patch('pymc.sample') as mock_sample, \
         patch('arviz.summary') as mock_summary, \
         patch('arviz.plot_trace') as mock_plot, \
         patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('io.BytesIO') as mock_bytesio, \
         patch('base64.b64encode') as mock_b64encode:

        # モックの設定
        mock_context = MagicMock()
        mock_model.return_value.__enter__.return_value = mock_context

        mock_trace = MagicMock()
        mock_sample.return_value = mock_trace

        # サマリーデータのモック
        summary_data = pd.DataFrame({
            'mean': [0.5, 2.0, 0.1, 0.2, 0.3],
            'sd': [0.1, 0.3, 0.05, 0.1, 0.1],
            'hdi_3%': [0.3, 1.5, 0.01, 0.05, 0.1],
            'hdi_97%': [0.7, 2.5, 0.2, 0.35, 0.5],
            'r_hat': [1.01, 1.0, 1.02, 1.0, 1.01]
        }, index=['mu_alpha', 'beta[0]', 'sigma', 'alpha[0]', 'alpha[1]'])
        mock_summary.return_value = summary_data

        # 画像データのモック
        mock_bytesio_instance = MagicMock()
        mock_bytesio.return_value = mock_bytesio_instance
        mock_bytesio_instance.getvalue.return_value = b'image_data'
        mock_b64encode.return_value = b'base64_encoded_image'

        # アナライザーのインスタンスを作成
        analyzer = BayesianInferenceAnalyzer(bq_service=mock_bq_service)

        # 階層モデルをフィット
        result = await analyzer.fit_hierarchical_model(
            query="SELECT * FROM dataset.table",
            target_variable="revenue",
            predictor_variables=["marketing_spend", "product_dev_cost"],
            group_variable="group",
            save_results=True,
            dataset_id="output_dataset",
            table_id="output_table"
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'model_summary' in result
        assert 'group_effects' in result
        assert 'diagnostics' in result
        assert 'trace_plot' in result
        assert 'posterior_predictive_plot' in result

        # BigQueryのクエリメソッドと保存メソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()
        mock_bq_service.save_dataframe.assert_called_once()

@pytest.mark.asyncio
async def test_calculate_bayes_factor(mock_bq_service, sample_data):
    """ベイズファクター計算機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_data

    # PyMCとArvizのモック
    with patch('pymc.Model') as mock_model, \
         patch('pymc.sample') as mock_sample, \
         patch('arviz.waic') as mock_waic, \
         patch('arviz.loo') as mock_loo, \
         patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('io.BytesIO') as mock_bytesio, \
         patch('base64.b64encode') as mock_b64encode:

        # モックの設定
        mock_context = MagicMock()
        mock_model.return_value.__enter__.return_value = mock_context

        mock_trace = MagicMock()
        mock_sample.return_value = mock_trace

        # WAICとLOOのモック結果
        mock_waic.return_value = MagicMock(waic=-100, se=5)
        mock_loo.return_value = MagicMock(loo=-105, se=6)

        # 画像データのモック
        mock_bytesio_instance = MagicMock()
        mock_bytesio.return_value = mock_bytesio_instance
        mock_bytesio_instance.getvalue.return_value = b'image_data'
        mock_b64encode.return_value = b'base64_encoded_image'

        # アナライザーのインスタンスを作成
        analyzer = BayesianInferenceAnalyzer(bq_service=mock_bq_service)

        # ベイズファクターを計算
        result = await analyzer.calculate_bayes_factor(
            query="SELECT * FROM dataset.table",
            target_variable="revenue",
            model1_predictors=["marketing_spend", "product_dev_cost"],
            model2_predictors=["marketing_spend"],
            comparison_method="waic"
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'bayes_factor' in result
        assert 'model1_score' in result
        assert 'model2_score' in result
        assert 'comparison_method' in result
        assert 'interpretation' in result

        # BigQueryのクエリメソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()

        # 適切なメソッドが呼び出されたかを確認
        mock_waic.assert_called()
        assert mock_loo.call_count == 0  # WAICが指定されているのでLOOは呼ばれていない

@pytest.mark.asyncio
async def test_update_beliefs(mock_bq_service, sample_data, mock_firestore_service):
    """事前信念の更新機能をテストします"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_data

    # 事前確率のモック
    prior_beliefs = {
        "parameter_distributions": {
            "conversion_rate": {"distribution": "beta", "alpha": 10, "beta": 90},
            "revenue_per_customer": {"distribution": "normal", "mu": 100, "sigma": 20}
        }
    }

    # Firestoreの取得結果をモック
    mock_firestore_service.get_document.return_value = {"id": "prior_doc", "data": prior_beliefs}

    # PyMCとArvizのモック
    with patch('pymc.Model') as mock_model, \
         patch('pymc.sample') as mock_sample, \
         patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('io.BytesIO') as mock_bytesio, \
         patch('base64.b64encode') as mock_b64encode:

        # モックの設定
        mock_context = MagicMock()
        mock_model.return_value.__enter__.return_value = mock_context

        mock_trace = MagicMock()
        mock_sample.return_value = mock_trace
        mock_trace.posterior = {
            'conversion_rate': np.random.beta(15, 85, (4, 1000)),
            'revenue_per_customer': np.random.normal(105, 15, (4, 1000))
        }

        # 画像データのモック
        mock_bytesio_instance = MagicMock()
        mock_bytesio.return_value = mock_bytesio_instance
        mock_bytesio_instance.getvalue.return_value = b'image_data'
        mock_b64encode.return_value = b'base64_encoded_image'

        # アナライザーのインスタンスを作成
        analyzer = BayesianInferenceAnalyzer(
            bq_service=mock_bq_service,
            firestore_service=mock_firestore_service
        )

        # 事前信念を更新
        result = await analyzer.update_beliefs(
            query="SELECT * FROM dataset.table",
            prior_beliefs_id="prior_doc",
            parameter_column_mapping={
                "conversion_rate": "churn_rate",
                "revenue_per_customer": "revenue"
            },
            save_updated_beliefs=True
        )

        # 結果の検証
        assert isinstance(result, dict)
        assert 'prior_beliefs' in result
        assert 'posterior_beliefs' in result
        assert 'updated_beliefs_id' in result
        assert 'parameter_summaries' in result
        assert 'visualization' in result

        # BigQueryとFirestoreのメソッドが呼び出されたことを確認
        mock_bq_service.query.assert_called_once()
        mock_firestore_service.get_document.assert_called_once()
        mock_firestore_service.add_document.assert_called_once()

def test_validate_prior_beliefs():
    """事前信念の検証機能をテストします"""
    # アナライザーのインスタンスを作成
    analyzer = BayesianInferenceAnalyzer(bq_service=MagicMock())

    # 有効な事前信念
    valid_beliefs = {
        "parameter_distributions": {
            "conversion_rate": {"distribution": "beta", "alpha": 10, "beta": 90},
            "revenue_per_customer": {"distribution": "normal", "mu": 100, "sigma": 20}
        }
    }

    # 無効な事前信念（パラメータが不足）
    invalid_beliefs_missing_params = {
        "parameter_distributions": {
            "conversion_rate": {"distribution": "beta", "alpha": 10},
            "revenue_per_customer": {"distribution": "normal", "mu": 100, "sigma": 20}
        }
    }

    # 無効な事前信念（未サポートの分布）
    invalid_beliefs_unsupported_dist = {
        "parameter_distributions": {
            "conversion_rate": {"distribution": "unsupported", "param1": 10, "param2": 90},
            "revenue_per_customer": {"distribution": "normal", "mu": 100, "sigma": 20}
        }
    }

    # 有効な信念を検証
    is_valid, error_msg = analyzer._validate_prior_beliefs(valid_beliefs)
    assert is_valid is True
    assert error_msg is None

    # パラメータが不足している信念を検証
    is_valid, error_msg = analyzer._validate_prior_beliefs(invalid_beliefs_missing_params)
    assert is_valid is False
    assert error_msg is not None
    assert "missing" in error_msg.lower() or "required" in error_msg.lower()

    # 未サポートの分布を含む信念を検証
    is_valid, error_msg = analyzer._validate_prior_beliefs(invalid_beliefs_unsupported_dist)
    assert is_valid is False
    assert error_msg is not None
    assert "unsupported" in error_msg.lower()

def test_create_visualization():
    """可視化機能をテストします"""
    # アナライザーのインスタンスを作成
    analyzer = BayesianInferenceAnalyzer(bq_service=MagicMock())

    # 可視化関数のモック
    with patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('io.BytesIO') as mock_bytesio, \
         patch('base64.b64encode') as mock_b64encode:

        # 画像データのモック
        mock_bytesio_instance = MagicMock()
        mock_bytesio.return_value = mock_bytesio_instance
        mock_bytesio_instance.getvalue.return_value = b'image_data'
        mock_b64encode.return_value = b'base64_encoded_image'

        # 可視化関数を呼び出し
        result = analyzer._create_visualization({
            'x': np.random.normal(0, 1, 100),
            'y': np.random.normal(0, 1, 100)
        }, "テスト可視化", "x", "y")

        # 結果の検証
        assert isinstance(result, str)
        assert result == 'data:image/png;base64,base64_encoded_image'

        # プロット関数が呼び出されたことを確認
        mock_figure.assert_called_once()
        mock_savefig.assert_called_once()
        mock_bytesio.assert_called_once()
        mock_b64encode.assert_called_once()