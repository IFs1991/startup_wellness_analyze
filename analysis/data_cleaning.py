"""
ベイズ推論を用いたデータクリーニングモジュール

異常値検出と欠損値補完のためのベイズモデルを提供します。
"""

from typing import Dict, List, Optional, Tuple, Any, Protocol
import numpy as np
import pymc as pm
import pandas as pd
import xarray as xr
from scipy import stats
from dataclasses import dataclass

class BigQueryService(Protocol):
    """BigQueryサービスのインターフェース"""
    async def fetch_data(self, query: str) -> pd.DataFrame:
        """データを取得"""
        ...

@dataclass
class BayesianCleaningConfig:
    """データクリーニングの設定"""
    query: str
    target_variable: str
    cleaning_type: str  # 'outlier_detection' or 'missing_value_imputation'
    confidence_level: float = 0.95
    n_samples: int = 2000
    n_tune: Optional[int] = None
    chains: int = 4
    target_accept: float = 0.95

class BayesianDataCleaner:
    """
    ベイズ推論を用いたデータクリーニングを実行するクラス
    """

    def __init__(self, bq_service: BigQueryService, random_seed: Optional[int] = None):
        self.bq_service = bq_service
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def _get_max_value(self, values) -> float:
        """xarrayまたは通常の値から最大値を取得"""
        if isinstance(values, xr.Dataset):
            max_values = []
            for var in values.data_vars:
                if isinstance(values[var].values, np.ndarray):
                    max_values.append(float(np.max(values[var].values)))
                else:
                    max_values.append(float(values[var].values))
            return max(max_values)
        elif isinstance(values, xr.DataArray):
            if isinstance(values.values, np.ndarray):
                return float(np.max(values.values))
            return float(values.values)
        return float(values)

    def _get_min_value(self, values) -> float:
        """xarrayまたは通常の値から最小値を取得"""
        if isinstance(values, xr.Dataset):
            min_values = []
            for var in values.data_vars:
                if isinstance(values[var].values, np.ndarray):
                    min_values.append(float(np.min(values[var].values)))
                else:
                    min_values.append(float(values[var].values))
            return min(min_values)
        elif isinstance(values, xr.DataArray):
            if isinstance(values.values, np.ndarray):
                return float(np.min(values.values))
            return float(values.values)
        return float(values)

    async def _run_mcmc(self, model: pm.Model, config: BayesianCleaningConfig) -> Dict[str, Any]:
        """
        MCMCサンプリングを実行し、診断情報を含む結果を返す

        Args:
            model: PyMCモデル
            config: サンプリング設定

        Returns:
            Dict[str, Any]: サンプリング結果と診断情報
        """
        with model:
            # サンプリング設定
            n_tune = config.n_tune if config.n_tune is not None else config.n_samples

            # サンプリング実行
            trace = pm.sample(
                draws=config.n_samples,
                tune=n_tune,
                chains=config.chains,
                target_accept=config.target_accept,
                return_inferencedata=True,
                random_seed=self.random_seed
            )

            # 診断情報の収集
            r_hat_values = pm.rhat(trace)
            ess_values = pm.ess(trace)

            diagnostics = {
                'r_hat': self._get_max_value(r_hat_values),
                'ess': self._get_min_value(ess_values),
                'divergences': int(trace.sample_stats.diverging.sum()),
                'acceptance_rate': float(trace.sample_stats.diverging.mean()),
                'n_tune': n_tune,
                'n_samples': config.n_samples,
                'chains': config.chains
            }

            return {
                'trace': trace,
                'diagnostics': diagnostics
            }

    async def detect_outliers(self, data: pd.DataFrame, config: BayesianCleaningConfig) -> Dict[str, Any]:
        """
        ロバストなベイズモデルによる異常値検出

        Args:
            data: 対象データ
            config: 設定

        Returns:
            Dict[str, Any]: 異常値検出結果
        """
        values = data[config.target_variable].values

        # データの基本統計量を計算
        median = np.median(values)
        mad = np.median(np.abs(values - median))  # MAD（中央絶対偏差）

        with pm.Model() as model:
            # スチューデントのt分布を使用してロバストな推定
            nu = pm.Exponential('nu', 1/10)  # より小さい自由度を許容
            sigma = pm.HalfCauchy('sigma', beta=mad)  # MADを使用
            mu = pm.Normal('mu', mu=median, sigma=mad)

            # 尤度関数
            likelihood = pm.StudentT('likelihood',
                                   nu=nu,
                                   mu=mu,
                                   sigma=sigma,
                                   observed=values)

        # MCMCサンプリング実行
        mcmc_results = await self._run_mcmc(model, config)
        trace = mcmc_results['trace']

        # 予測分布からの確率計算
        with model:
            ppc = pm.sample_posterior_predictive(trace, random_seed=self.random_seed)

        # 異常値の検出
        predicted = ppc.posterior_predictive.likelihood.mean(dim=("chain", "draw")).values
        std = ppc.posterior_predictive.likelihood.std(dim=("chain", "draw")).values

        # 修正済みz-score（MADベース）を計算
        mad_scores = np.abs(values - median) / mad
        z_scores = np.abs((values - predicted) / std)

        # 両方のスコアを組み合わせて異常値を検出
        mad_threshold = 3.5  # MADベースの閾値
        z_threshold = stats.t.ppf(1 - (1 - config.confidence_level)/2, df=3)  # t分布の閾値

        outliers = (mad_scores > mad_threshold) | (z_scores > z_threshold)

        # 異常値の検出結果を返す
        outlier_indices = np.where(outliers)[0]

        # 異常値の重要度でソート（MADスコアとz-scoreの最大値を使用）
        importance_scores = np.maximum(mad_scores, z_scores)
        sorted_indices = outlier_indices[np.argsort(-importance_scores[outlier_indices])]

        return {
            'outlier_indices': sorted_indices.tolist(),
            'z_scores': z_scores.tolist(),
            'mad_scores': mad_scores.tolist(),
            'threshold': float(z_threshold),
            'mad_threshold': float(mad_threshold),
            'diagnostics': mcmc_results['diagnostics']
        }

    async def impute_missing_values(self, data: pd.DataFrame, config: BayesianCleaningConfig) -> Dict[str, Any]:
        """
        ベイズ推論による欠損値の補完

        Args:
            data: 対象データ
            config: 設定

        Returns:
            Dict[str, Any]: 補完結果
        """
        values = data[config.target_variable]
        n_missing = int(values.isna().sum())
        observed_values = values.dropna().values

        with pm.Model() as model:
            # 欠損メカニズムのモデル化
            mu = pm.Normal('mu', mu=np.mean(observed_values), sigma=np.std(observed_values))
            sigma = pm.HalfCauchy('sigma', beta=np.std(observed_values))

            # 欠損値の生成モデル
            missing_values = pm.Normal('missing_values',
                                     mu=mu,
                                     sigma=sigma,
                                     shape=(n_missing,))

            # 観測データの尤度
            observed = pm.Normal('observed',
                               mu=mu,
                               sigma=sigma,
                               observed=observed_values)

        # MCMCサンプリング実行
        mcmc_results = await self._run_mcmc(model, config)
        trace = mcmc_results['trace']

        # 欠損値の補完
        missing_samples = trace.posterior.missing_values.values
        imputed_values = np.mean(missing_samples, axis=(0, 1))
        imputed_std = np.std(missing_samples, axis=(0, 1))

        return {
            'imputed_values': imputed_values.tolist(),
            'imputed_std': imputed_std.tolist(),
            'imputation_locations': np.where(values.isna())[0].tolist(),
            'diagnostics': mcmc_results['diagnostics']
        }

    async def clean_data(self, config: BayesianCleaningConfig) -> Dict[str, Any]:
        """
        データクリーニングを実行

        Args:
            config: クリーニング設定

        Returns:
            Dict[str, Any]: クリーニング結果
        """
        try:
            data = await self.bq_service.fetch_data(config.query)

            if config.cleaning_type == 'outlier_detection':
                results = await self.detect_outliers(data, config)
            elif config.cleaning_type == 'missing_value_imputation':
                results = await self.impute_missing_values(data, config)
            else:
                raise ValueError(f"未対応のクリーニングタイプです: {config.cleaning_type}")

            return {
                'status': 'success',
                'results': results,
                'metadata': {
                    'target_variable': config.target_variable,
                    'cleaning_type': config.cleaning_type,
                    'confidence_level': config.confidence_level,
                    'n_samples': config.n_samples,
                    'timestamp': pd.Timestamp.now().isoformat()
                }
            }

        except ValueError as e:
            # ValueErrorはそのまま再送出
            raise
        except Exception as e:
            raise RuntimeError(f"データクリーニング中にエラーが発生しました: {str(e)}")