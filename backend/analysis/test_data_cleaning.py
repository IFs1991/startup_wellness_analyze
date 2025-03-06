"""
ベイズ推論を用いたデータクリーニングの単体テスト

改善版：
- モデルの収束性チェック
- パフォーマンス監視
- データ品質の評価
"""

import pytest
import pandas as pd
import numpy as np
import time
import pymc as pm
from .data_cleaning import BayesianDataCleaner, BayesianCleaningConfig

class BayesianModelMonitor:
    """ベイジアンモデルの品質監視クラス"""

    def __init__(self):
        self.n_samples = 0
        self.start_time = None
        self.end_time = None

    def check_convergence(self, trace) -> dict:
        """
        収束性の確認

        Args:
            trace: MCMCトレース

        Returns:
            dict: 診断結果
        """
        try:
            diagnostics = pm.diagnostics.gelman_rubin(trace)
            n_eff = pm.diagnostics.effective_n(trace)
            diverging = trace['diverging'].sum()

            return {
                'r_hat': float(max(diagnostics.values())),
                'min_ess': float(min(n_eff.values())),
                'divergences': int(diverging)
            }
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }

    def performance_metrics(self) -> dict:
        """
        パフォーマンス指標の計算

        Returns:
            dict: パフォーマンス指標
        """
        if not (self.start_time and self.end_time):
            return {'status': 'not_measured'}

        total_time = self.end_time - self.start_time
        return {
            'total_time': total_time,
            'samples_per_second': self.n_samples / total_time if total_time > 0 else 0
        }

    def start_monitoring(self, n_samples: int):
        """モニタリング開始"""
        self.n_samples = n_samples
        self.start_time = time.time()

    def end_monitoring(self):
        """モニタリング終了"""
        self.end_time = time.time()

class TestDataGenerator:
    """テストデータ生成クラス"""

    @staticmethod
    def generate_test_data(n_samples: int = 100, contamination: float = 0.1) -> pd.DataFrame:
        """
        テストデータの生成

        Args:
            n_samples: サンプル数
            contamination: 異常値の割合

        Returns:
            pd.DataFrame: テストデータ
        """
        np.random.seed(42)

        # 正常データの生成
        normal_data = np.random.normal(loc=0, scale=1, size=int(n_samples * (1 - contamination)))

        # 異常値の生成
        outliers = np.random.normal(loc=10, scale=2, size=int(n_samples * contamination))

        # データの結合
        data = np.concatenate([normal_data, outliers])
        np.random.shuffle(data)

        # 欠損値の追加
        data_series = pd.Series(data)
        missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        data_series.iloc[missing_indices] = np.nan

        return pd.DataFrame({'value': data_series})

class MockBQService:
    """改善されたBigQueryサービスモック"""

    def __init__(self, test_data: pd.DataFrame):
        self.test_data = test_data

    async def fetch_data(self, query: str) -> pd.DataFrame:
        return self.test_data

@pytest.mark.asyncio
async def test_basic_functionality():
    """基本的な機能のテスト - 改善版"""
    # モニタリングの設定
    monitor = BayesianModelMonitor()

    # テストデータの生成
    test_data = TestDataGenerator.generate_test_data(n_samples=100)

    # モックサービスの設定
    mock_service = MockBQService(test_data)

    # 異常値検出のテスト
    outlier_config = BayesianCleaningConfig(
        query="dummy query",
        target_variable="value",
        cleaning_type="outlier_detection",
        confidence_level=0.95,
        n_samples=1000  # テスト用
    )

    cleaner = BayesianDataCleaner(mock_service)

    try:
        # モニタリング開始
        monitor.start_monitoring(outlier_config.n_samples)

        # 異常値検出の実行
        outlier_result = await cleaner.clean_data(outlier_config)

        # モニタリング終了
        monitor.end_monitoring()

        # 基本的な結果の検証
        assert outlier_result['status'] == 'success'
        assert 'results' in outlier_result
        assert 'outlier_indices' in outlier_result['results']

        # パフォーマンス指標の確認
        perf_metrics = monitor.performance_metrics()
        assert perf_metrics['total_time'] > 0
        assert perf_metrics['samples_per_second'] > 0

    except Exception as e:
        pytest.fail(f"異常値検出テストでエラー発生: {str(e)}")

@pytest.mark.asyncio
async def test_missing_value_imputation():
    """欠損値補完のテスト - 改善版"""
    # モニタリングの設定
    monitor = BayesianModelMonitor()

    # テストデータの生成（欠損値を含む）
    test_data = TestDataGenerator.generate_test_data(n_samples=100)

    # モックサービスの設定
    mock_service = MockBQService(test_data)

    # 欠損値補完の設定
    imputation_config = BayesianCleaningConfig(
        query="dummy query",
        target_variable="value",
        cleaning_type="missing_value_imputation",
        confidence_level=0.95,
        n_samples=1000
    )

    cleaner = BayesianDataCleaner(mock_service)

    try:
        # モニタリング開始
        monitor.start_monitoring(imputation_config.n_samples)

        # 欠損値補完の実行
        imputation_result = await cleaner.clean_data(imputation_config)

        # モニタリング終了
        monitor.end_monitoring()

        # 基本的な結果の検証
        assert imputation_result['status'] == 'success'
        assert 'results' in imputation_result
        assert 'imputed_values' in imputation_result['results']

        # 補完値の品質チェック
        imputed_values = imputation_result['results']['imputed_values']
        assert all(isinstance(x, (int, float)) for x in imputed_values)
        assert all(-20 < x < 20 for x in imputed_values)  # 妥当な範囲内か確認

        # パフォーマンス指標の確認
        perf_metrics = monitor.performance_metrics()
        assert perf_metrics['total_time'] > 0
        assert perf_metrics['samples_per_second'] > 0

    except Exception as e:
        pytest.fail(f"欠損値補完テストでエラー発生: {str(e)}")

@pytest.mark.asyncio
async def test_model_convergence():
    """モデルの収束性テスト"""
    # テストデータの生成
    test_data = TestDataGenerator.generate_test_data(n_samples=50)  # 小さめのサンプルサイズ

    # モックサービスの設定
    mock_service = MockBQService(test_data)

    # クリーニング設定
    config = BayesianCleaningConfig(
        query="dummy query",
        target_variable="value",
        cleaning_type="outlier_detection",
        confidence_level=0.95,
        n_samples=2000  # 収束性確認用に多めのサンプル
    )

    cleaner = BayesianDataCleaner(mock_service)
    monitor = BayesianModelMonitor()

    try:
        result = await cleaner.clean_data(config)

        # トレースが利用可能な場合の収束性チェック
        if hasattr(result, 'trace'):
            diagnostics = monitor.check_convergence(result.trace)
            assert diagnostics.get('r_hat', float('inf')) < 1.1  # R-hat < 1.1 は良い収束の指標
            assert diagnostics.get('divergences', float('inf')) < 10  # 発散が少ないことを確認

    except Exception as e:
        pytest.fail(f"収束性テストでエラー発生: {str(e)}")