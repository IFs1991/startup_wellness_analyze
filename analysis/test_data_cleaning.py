"""
ベイズ推論を用いたデータクリーニングのテスト
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any
from analysis.data_cleaning import BayesianDataCleaner, BayesianCleaningConfig

class SimpleTestService:
    """テスト用のシンプルなデータ生成サービス"""

    def __init__(self, data_type: str = 'outlier'):
        self.data_type = data_type

    async def fetch_data(self, query: str) -> pd.DataFrame:
        """テストデータを生成"""
        if self.data_type == 'outlier':
            # 異常値を含むデータ（より極端な異常値を使用）
            data = np.array([2.1, 2.3, 15.0, 2.2, 2.0, -12.0, 2.4, 2.1])
        else:
            # 欠損値を含むデータ
            data = np.array([2.1, 2.3, np.nan, 2.2, 2.0, np.nan, 2.4, 2.1])

        return pd.DataFrame({'value': data})

@pytest.mark.asyncio
async def test_outlier_detection():
    """異常値検出のテスト"""
    # テストデータの準備
    service = SimpleTestService(data_type='outlier')
    cleaner = BayesianDataCleaner(service, random_seed=42)

    config = BayesianCleaningConfig(
        query="SELECT * FROM test_table",
        target_variable="value",
        cleaning_type="outlier_detection",
        confidence_level=0.99,  # より高い信頼水準を使用
        n_samples=1000,  # テスト用に少なめのサンプル数
        chains=2  # テスト用に少なめのチェーン数
    )

    # 異常値検出の実行
    results = await cleaner.clean_data(config)

    # 基本的な結果の検証
    assert results['status'] == 'success'
    assert 'results' in results
    assert 'outlier_indices' in results['results']
    assert 'diagnostics' in results['results']

    # 異常値の検証
    outlier_indices = results['results']['outlier_indices']
    assert len(outlier_indices) >= 2  # 少なくとも2つの異常値（15.0と-12.0）を検出
    assert 2 in outlier_indices  # インデックス2（値15.0）は異常値
    assert 5 in outlier_indices  # インデックス5（値-12.0）は異常値

    # スコアの検証
    mad_scores = results['results']['mad_scores']
    assert len(mad_scores) == 8  # データ点の数と一致
    assert all(isinstance(s, float) for s in mad_scores)  # すべてfloat型
    assert max(mad_scores) > results['results']['mad_threshold']  # 最大MADスコアは閾値を超える

    # 診断情報の検証
    diagnostics = results['results']['diagnostics']
    assert 'r_hat' in diagnostics
    assert 'ess' in diagnostics
    assert 'divergences' in diagnostics
    assert diagnostics['r_hat'] < 1.1  # R-hat < 1.1は収束の目安
    assert diagnostics['ess'] > 400  # 有効サンプル数の最小値

@pytest.mark.asyncio
async def test_missing_value_imputation():
    """欠損値補完のテスト"""
    # テストデータの準備
    service = SimpleTestService(data_type='missing')
    cleaner = BayesianDataCleaner(service, random_seed=42)

    config = BayesianCleaningConfig(
        query="SELECT * FROM test_table",
        target_variable="value",
        cleaning_type="missing_value_imputation",
        n_samples=1000,  # テスト用に少なめのサンプル数
        chains=2  # テスト用に少なめのチェーン数
    )

    # 欠損値補完の実行
    results = await cleaner.clean_data(config)

    # 基本的な結果の検証
    assert results['status'] == 'success'
    assert 'results' in results
    assert 'imputed_values' in results['results']
    assert 'diagnostics' in results['results']

    # 補完値の検証
    imputed_values = results['results']['imputed_values']
    assert len(imputed_values) == 2  # 2つの欠損値が補完されている
    assert all(1.5 < v < 3.0 for v in imputed_values)  # 補完値は正常値の範囲内

    # 診断情報の検証
    diagnostics = results['results']['diagnostics']
    assert 'r_hat' in diagnostics
    assert 'ess' in diagnostics
    assert 'divergences' in diagnostics
    assert diagnostics['r_hat'] < 1.1  # R-hat < 1.1は収束の目安
    assert diagnostics['ess'] > 400  # 有効サンプル数の最小値

@pytest.mark.asyncio
async def test_invalid_cleaning_type():
    """無効なクリーニングタイプのテスト"""
    service = SimpleTestService()
    cleaner = BayesianDataCleaner(service)

    config = BayesianCleaningConfig(
        query="SELECT * FROM test_table",
        target_variable="value",
        cleaning_type="invalid_type"
    )

    with pytest.raises(ValueError, match="未対応のクリーニングタイプです"):
        await cleaner.clean_data(config)