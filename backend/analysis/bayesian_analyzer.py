"""
ベイズ分析モジュール

スタートアップの健全性評価のためのベイズ分析を実行します。
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pymc as pm
import pandas as pd
from dataclasses import dataclass
from backend.src.database.bigquery.client import BigQueryService
from config.bayesian_settings import calculate_adjusted_parameters

@dataclass
class BayesianAnalysisConfig:
    """ベイズ分析の設定を保持するデータクラス"""
    query: str
    target_variable: str
    industry: str
    company_stage: str
    prior_parameters: Optional[Dict[str, float]] = None
    n_samples: Optional[int] = None
    n_tune: Optional[int] = None
    chains: Optional[int] = None

class BayesianAnalyzer:
    """
    ベイズ分析を実行するクラス
    """

    def __init__(self, bq_service: BigQueryService):
        self.bq_service = bq_service

    def _get_adjusted_parameters(self, data: pd.DataFrame, config: BayesianAnalysisConfig) -> Dict[str, Any]:
        """
        調整済みパラメータを取得

        Args:
            data: 分析対象データ
            config: 分析設定

        Returns:
            Dict[str, Any]: 調整済みパラメータ
        """
        data_size = len(data)
        params = calculate_adjusted_parameters(
            industry=config.industry,
            stage=config.company_stage,
            data_size=data_size
        )

        # ユーザー指定のパラメータで上書き
        if config.prior_parameters:
            params['prior_parameters'].update(config.prior_parameters)
        if config.n_samples:
            params['mcmc_settings']['n_samples'] = config.n_samples
        if config.n_tune:
            params['mcmc_settings']['n_tune'] = config.n_tune
        if config.chains:
            params['mcmc_settings']['chains'] = config.chains

        return params

    async def analyze_growth_rate(self, data: pd.DataFrame, config: BayesianAnalysisConfig) -> Dict[str, Any]:
        """
        成長率のベイズ推定を実行

        Args:
            data: 分析対象データ
            config: 分析設定

        Returns:
            Dict[str, Any]: 分析結果
        """
        params = self._get_adjusted_parameters(data, config)
        prior_params = params['prior_parameters']
        mcmc_settings = params['mcmc_settings']

        with pm.Model() as model:
            # 成長率の事前分布（正規分布を仮定）
            mu = pm.Normal('mu',
                         mu=prior_params['mu'],
                         sigma=prior_params['sigma'])

            # 標準偏差の事前分布（半コーシー分布を仮定）
            sigma = pm.HalfCauchy('sigma',
                                beta=prior_params.get('beta', 1))

            # 尤度関数
            likelihood = pm.Normal('likelihood',
                                 mu=mu,
                                 sigma=sigma,
                                 observed=data[config.target_variable])

            # MCMCサンプリング
            trace = pm.sample(
                draws=mcmc_settings['n_samples'],
                tune=mcmc_settings['n_tune'],
                chains=mcmc_settings['chains'],
                return_inferencedata=False
            )

        # 結果の集計
        results = {
            'posterior_mean': float(np.mean(trace['mu'])),
            'posterior_std': float(np.std(trace['mu'])),
            'hdi_3%': float(pm.hdi(trace['mu'], hdi_prob=0.94)[0]),
            'hdi_97%': float(pm.hdi(trace['mu'], hdi_prob=0.94)[1]),
            'effective_sample_size': float(pm.ess(trace['mu'])),
            'r_hat': float(pm.rhat(trace['mu']))
        }

        return results

    async def analyze_success_probability(self, data: pd.DataFrame, config: BayesianAnalysisConfig) -> Dict[str, Any]:
        """
        成功確率のベイズ推定を実行

        Args:
            data: 分析対象データ
            config: 分析設定

        Returns:
            Dict[str, Any]: 分析結果
        """
        params = self._get_adjusted_parameters(data, config)
        prior_params = params['prior_parameters']
        mcmc_settings = params['mcmc_settings']

        with pm.Model() as model:
            # 成功確率の事前分布（ベータ分布を仮定）
            theta = pm.Beta('theta',
                          alpha=prior_params['alpha'],
                          beta=prior_params['beta'])

            # 尤度関数（ベルヌーイ分布を仮定）
            likelihood = pm.Bernoulli('likelihood',
                                    p=theta,
                                    observed=data[config.target_variable])

            # MCMCサンプリング
            trace = pm.sample(
                draws=mcmc_settings['n_samples'],
                tune=mcmc_settings['n_tune'],
                chains=mcmc_settings['chains'],
                return_inferencedata=False
            )

        # 結果の集計
        results = {
            'posterior_mean': float(np.mean(trace['theta'])),
            'posterior_std': float(np.std(trace['theta'])),
            'hdi_3%': float(pm.hdi(trace['theta'], hdi_prob=0.94)[0]),
            'hdi_97%': float(pm.hdi(trace['theta'], hdi_prob=0.94)[1]),
            'effective_sample_size': float(pm.ess(trace['theta'])),
            'r_hat': float(pm.rhat(trace['theta']))
        }

        return results

    async def run_analysis(self, config: BayesianAnalysisConfig) -> Dict[str, Any]:
        """
        ベイズ分析を実行

        Args:
            config: 分析設定

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            # データ取得
            data = await self.bq_service.fetch_data(config.query)

            # データの種類に応じて適切な分析を実行
            if pd.api.types.is_numeric_dtype(data[config.target_variable]):
                results = await self.analyze_growth_rate(data, config)
            else:
                results = await self.analyze_success_probability(data, config)

            return {
                'status': 'success',
                'results': results,
                'metadata': {
                    'target_variable': config.target_variable,
                    'industry': config.industry,
                    'company_stage': config.company_stage,
                    'data_size': len(data),
                    'analysis_timestamp': pd.Timestamp.now().isoformat()
                }
            }

        except Exception as e:
            raise RuntimeError(f"ベイズ分析の実行中にエラーが発生しました: {str(e)}")