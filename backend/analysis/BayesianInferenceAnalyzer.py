from typing import Dict, Any, List, Optional, Union, Tuple, ContextManager
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import logging
from .base import BaseAnalyzer, AnalysisError
# PyMCとarvizをインポート
import pymc as pm
import arviz as az
import warnings
import gc
import tempfile
import os
import contextlib
import weakref

class BayesianInferenceAnalyzer(BaseAnalyzer):
    """
    ベイズ推論を用いてROI予測を行うアナライザー

    指定されたデータに基づいて事前分布を更新し、事後分布と95%信用区間を計算します。
    週次で新しいデータを取得し、事後分布を更新します。
    """

    def __init__(self, firestore_client=None, storage_mode: str = 'memory'):
        """
        コンストラクタ

        Args:
            firestore_client: Firestoreクライアントのインスタンス
            storage_mode (str): ストレージモード ('memory', 'disk', 'hybrid')
        """
        super().__init__(analysis_type="bayesian_inference", firestore_client=firestore_client)
        self.logger = logging.getLogger(__name__)
        self.storage_mode = storage_mode
        self._temp_files = []
        self._plot_resources = weakref.WeakValueDictionary()

    def __del__(self):
        """デストラクタ - リソースの解放"""
        self.release_resources()

    def release_resources(self) -> None:
        """すべてのリソースを解放"""
        super().release_resources()
        self._clean_plot_resources()
        self._clean_temp_files()
        gc.collect()

    def _clean_plot_resources(self) -> None:
        """プロットリソースをクリーンアップ"""
        for fig_id in list(self._plot_resources.keys()):
            plt.close(self._plot_resources[fig_id])
        self._plot_resources.clear()

    def _clean_temp_files(self) -> None:
        """一時ファイルをクリーンアップ"""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                self.logger.warning(f"一時ファイルの削除中にエラーが発生しました: {str(e)}")
        self._temp_files = []

    @contextlib.contextmanager
    def _plot_context(self, figsize=(10, 6)) -> ContextManager:
        """
        プロット作成用のコンテキストマネージャ

        Args:
            figsize (tuple): フィギュアサイズ

        Yields:
            tuple: (fig, ax) matplotlib図とaxesオブジェクト
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig_id = id(fig)
        self._plot_resources[fig_id] = fig
        try:
            yield fig, ax
        finally:
            if fig_id in self._plot_resources:
                plt.close(fig)
                del self._plot_resources[fig_id]

    async def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        ベイズ推論分析を実行する

        Args:
            data (pd.DataFrame): 分析対象のデータ
                必要なカラム:
                - revenue_change: 売上変化
                - valuation_change: バリュエーション変化
                - program_cost: プログラムコスト
                - investment_cost: 投資コスト
            **kwargs: 追加パラメータ
                - prior_mean: 事前分布の平均 (デフォルト: 0.05)
                - prior_std: 事前分布の標準偏差 (デフォルト: 0.02)
                - portfolio_id: ポートフォリオID
                - use_historical: 過去データを使用するかどうか (デフォルト: True)

        Returns:
            Dict[str, Any]: 分析結果
                - prior_distribution: 事前分布のパラメータ
                - posterior_distribution: 事後分布のパラメータ
                - likelihood_points: 尤度関数の点群
                - roi_prediction: ROI予測値
                - credible_interval: 95%信用区間
                - plot_base64: プロット画像（Base64エンコード）
                - analyzed_at: 分析実行日時
                - metadata: メタデータ
        """
        try:
            # パラメータの取得
            prior_mean = kwargs.get('prior_mean', 0.05)  # デフォルト値: 5% ROI
            prior_std = kwargs.get('prior_std', 0.02)  # デフォルト標準偏差
            portfolio_id = kwargs.get('portfolio_id')
            use_historical = kwargs.get('use_historical', True)

            # データのバリデーション
            self._validate_data(data)

            # 過去データの取得（必要な場合）
            historical_data = None
            if use_historical and portfolio_id:
                try:
                    historical_data = await self._fetch_historical_data(portfolio_id)
                    self.logger.info(f"過去データを取得しました: {len(historical_data)} 件のレコード")
                except Exception as e:
                    self.logger.warning(f"過去データの取得に失敗しました: {str(e)}")

            # ROIの計算
            roi_values = self._calculate_roi(data)
            self.register_temp_data('roi_values', roi_values)

            # 事前分布の設定
            prior_distribution = {
                'distribution': 'normal',
                'mean': prior_mean,
                'std': prior_std
            }

            # 尤度関数の計算
            likelihood_mean = np.mean(roi_values)
            likelihood_std = np.std(roi_values) if len(roi_values) > 1 else prior_std

            # 事後分布の計算 (ベイズの定理)
            posterior_distribution = self._calculate_posterior(
                prior_mean, prior_std, likelihood_mean, likelihood_std, len(roi_values)
            )

            # 95%信用区間の計算
            credible_interval = self._calculate_credible_interval(
                posterior_distribution['mean'],
                posterior_distribution['std']
            )

            # 可視化
            plot_base64 = self._generate_plot(
                prior_distribution,
                posterior_distribution,
                roi_values
            )

            # 結果を返す
            results = {
                'prior_distribution': prior_distribution,
                'posterior_distribution': posterior_distribution,
                'likelihood_points': roi_values.tolist(),
                'roi_prediction': posterior_distribution['mean'],
                'credible_interval': credible_interval,
                'plot_base64': plot_base64,
                'analyzed_at': datetime.now().isoformat(),
                'metadata': {
                    'portfolio_id': portfolio_id,
                    'data_count': len(data),
                    'historical_data_used': historical_data is not None,
                    'historical_data_count': len(historical_data) if historical_data is not None else 0
                }
            }

            # 使用済みの一時データを解放
            self.release_resource('roi_values')

            return results

        except Exception as e:
            self.logger.error(f"分析実行中にエラーが発生しました: {str(e)}")
            self.release_resources()  # エラー発生時も確実にリソースを解放
            raise AnalysisError(f"ベイズ推論分析に失敗しました: {str(e)}")

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        入力データのバリデーションを行う

        Args:
            data (pd.DataFrame): 検証対象のデータ

        Raises:
            AnalysisError: データが不正な場合
        """
        if data.empty:
            raise AnalysisError("データが空です")

        required_columns = ['revenue_change', 'valuation_change', 'program_cost', 'investment_cost']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise AnalysisError(f"必要なカラムがありません: {', '.join(missing_columns)}")

    async def _fetch_historical_data(self, portfolio_id: str) -> pd.DataFrame:
        """
        過去のデータをFirestoreから取得する

        Args:
            portfolio_id (str): ポートフォリオID

        Returns:
            pd.DataFrame: 過去のデータ
        """
        # 過去3年間のデータを取得
        filters = [
            {'field': 'portfolio_id', 'operator': '==', 'value': portfolio_id},
            {'field': 'analyzed_at', 'operator': '>=', 'value': datetime.now().replace(year=datetime.now().year - 3).isoformat()}
        ]

        historical_data = await self.fetch_data(
            collection='bayesian_results',
            filters=filters,
            order_by=('analyzed_at', 'desc')
        )

        return historical_data

    def _calculate_roi(self, data: pd.DataFrame) -> np.ndarray:
        """
        ROIを計算する

        ROI = ((収益変化 + バリュエーション変化) - プログラムコスト) / 投資コスト * 100

        Args:
            data (pd.DataFrame): 分析対象のデータ

        Returns:
            np.ndarray: ROI値の配列
        """
        # メモリ効率のためにイテレータベースの処理を使用
        roi_values = []

        for _, row in data.iterrows():
            revenue_change = row['revenue_change']
            valuation_change = row['valuation_change']
            program_cost = row['program_cost']
            investment_cost = row['investment_cost']

            # 投資コストが0または無効な値の場合はスキップ
            if investment_cost <= 0:
                continue

            roi = ((revenue_change + valuation_change) - program_cost) / investment_cost
            roi_values.append(roi)

        # 効率的なndarrayに変換
        return np.array(roi_values, dtype=np.float32)

    def _calculate_posterior(
        self,
        prior_mean: float,
        prior_std: float,
        likelihood_mean: float,
        likelihood_std: float,
        n: int
    ) -> Dict[str, Any]:
        """
        事後分布を計算する

        正規分布の場合の共役事前分布を使用します。

        Args:
            prior_mean (float): 事前分布の平均
            prior_std (float): 事前分布の標準偏差
            likelihood_mean (float): 尤度関数の平均
            likelihood_std (float): 尤度関数の標準偏差
            n (int): サンプル数

        Returns:
            Dict[str, Any]: 事後分布のパラメータ
        """
        # 事前分布の精度（precision）
        prior_precision = 1 / (prior_std ** 2)

        # 尤度の精度
        likelihood_precision = n / (likelihood_std ** 2) if likelihood_std > 0 else 0

        # 事後分布の精度
        posterior_precision = prior_precision + likelihood_precision

        # 事後分布の分散
        posterior_var = 1 / posterior_precision if posterior_precision > 0 else prior_std ** 2

        # 事後分布の平均
        posterior_mean = (prior_mean * prior_precision + likelihood_mean * likelihood_precision) / posterior_precision if posterior_precision > 0 else prior_mean

        # 事後分布の標準偏差
        posterior_std = np.sqrt(posterior_var)

        return {
            'distribution': 'normal',
            'mean': float(posterior_mean),
            'std': float(posterior_std)
        }

    def _calculate_credible_interval(self, mean: float, std: float, confidence: float = 0.95) -> List[float]:
        """
        信用区間を計算する

        Args:
            mean (float): 分布の平均
            std (float): 分布の標準偏差
            confidence (float): 信頼度 (デフォルト: 0.95)

        Returns:
            List[float]: 信用区間 [下限, 上限]
        """
        z = stats.norm.ppf((1 + confidence) / 2)
        lower = mean - z * std
        upper = mean + z * std

        return [float(lower), float(upper)]

    def _generate_plot(
        self,
        prior_distribution: Dict[str, Any],
        posterior_distribution: Dict[str, Any],
        roi_values: np.ndarray
    ) -> str:
        """
        分布のプロットを生成する

        Args:
            prior_distribution (Dict[str, Any]): 事前分布のパラメータ
            posterior_distribution (Dict[str, Any]): 事後分布のパラメータ
            roi_values (np.ndarray): ROI値の配列

        Returns:
            str: Base64エンコードされたプロット画像
        """
        with self._plot_context(figsize=(10, 6)) as (fig, ax):
            # x軸の範囲を設定
            x_min = min(
                prior_distribution['mean'] - 3 * prior_distribution['std'],
                posterior_distribution['mean'] - 3 * posterior_distribution['std'],
                np.min(roi_values) if len(roi_values) > 0 else 0
            )

            x_max = max(
                prior_distribution['mean'] + 3 * prior_distribution['std'],
                posterior_distribution['mean'] + 3 * posterior_distribution['std'],
                np.max(roi_values) if len(roi_values) > 0 else 0
            )

            x = np.linspace(x_min, x_max, 1000)

            # 事前分布のプロット
            prior_pdf = stats.norm.pdf(x, prior_distribution['mean'], prior_distribution['std'])
            ax.plot(x, prior_pdf, 'b-', label='事前分布', alpha=0.7)

            # 事後分布のプロット
            posterior_pdf = stats.norm.pdf(x, posterior_distribution['mean'], posterior_distribution['std'])
            ax.plot(x, posterior_pdf, 'r-', label='事後分布', alpha=0.7)

            # 観測データのヒストグラム
            if len(roi_values) > 0:
                ax.hist(roi_values, bins=10, density=True, alpha=0.3, label='観測データ')

            # 95%信用区間の表示
            credible_interval = self._calculate_credible_interval(
                posterior_distribution['mean'],
                posterior_distribution['std']
            )

            ax.axvline(credible_interval[0], color='r', linestyle='--', alpha=0.5)
            ax.axvline(credible_interval[1], color='r', linestyle='--', alpha=0.5)
            ax.axvspan(credible_interval[0], credible_interval[1], alpha=0.2, color='r')

            # ラベルと凡例
            ax.set_xlabel('ROI')
            ax.set_ylabel('確率密度')
            ax.set_title('ROIのベイズ推論結果 (95%信用区間付き)')
            ax.legend()

            # グリッドを追加
            ax.grid(True, alpha=0.3)

            # スタイル設定
            plt.tight_layout()

            # 画像をBase64エンコード
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')

            return img_str

    async def bayesian_update(self, portfolio_id: str) -> Dict[str, Any]:
        """
        ベイズ更新を実行する

        新しいデータが追加されたときに事後分布を更新します。

        Args:
            portfolio_id (str): ポートフォリオID

        Returns:
            Dict[str, Any]: 更新された事後分布
        """
        try:
            # 過去の事後分布を取得
            latest_result = await self._get_latest_result(portfolio_id)

            if not latest_result:
                self.logger.warning(f"ポートフォリオ {portfolio_id} の過去の分析結果が見つかりません。初期分析を実行してください。")
                return None

            # 新しいデータの取得
            new_data = await self._fetch_new_data(portfolio_id, latest_result.get('analyzed_at'))

            if new_data.empty:
                self.logger.info(f"ポートフォリオ {portfolio_id} の新しいデータがありません。更新はスキップします。")
                return latest_result

            # 前回の事後分布を今回の事前分布として使用
            prior_distribution = latest_result.get('posterior_distribution', {
                'distribution': 'normal',
                'mean': 0.05,
                'std': 0.02
            })

            # 分析を実行
            results = await self.analyze(
                new_data,
                prior_mean=prior_distribution['mean'],
                prior_std=prior_distribution['std'],
                portfolio_id=portfolio_id,
                use_historical=False  # 既に過去データを事前分布に組み込んでいるため
            )

            # 結果を保存
            if results:
                document_id = await self.save_results(results, collection='bayesian_results')
                results['document_id'] = document_id

            return results

        except Exception as e:
            self.logger.error(f"ベイズ更新中にエラーが発生しました: {str(e)}")
            self.release_resources()  # エラー発生時も確実にリソースを解放
            raise AnalysisError(f"ベイズ更新に失敗しました: {str(e)}")

    async def _get_latest_result(self, portfolio_id: str) -> Dict[str, Any]:
        """
        最新の分析結果を取得する

        Args:
            portfolio_id (str): ポートフォリオID

        Returns:
            Dict[str, Any]: 最新の分析結果
        """
        filters = [
            {'field': 'metadata.portfolio_id', 'operator': '==', 'value': portfolio_id}
        ]

        results = await self.firestore_client.query_documents(
            collection='bayesian_results',
            filters=filters,
            order_by=('analyzed_at', 'desc'),
            limit=1
        )

        return results[0] if results else None

    async def _fetch_new_data(self, portfolio_id: str, last_analyzed_at: str) -> pd.DataFrame:
        """
        最後の分析以降の新しいデータを取得する

        Args:
            portfolio_id (str): ポートフォリオID
            last_analyzed_at (str): 最後の分析日時

        Returns:
            pd.DataFrame: 新しいデータ
        """
        filters = [
            {'field': 'portfolio_id', 'operator': '==', 'value': portfolio_id},
            {'field': 'created_at', 'operator': '>', 'value': last_analyzed_at}
        ]

        new_data = await self.fetch_data(
            collection='financial_metrics',
            filters=filters,
            order_by=('created_at', 'asc')
        )

        return new_data

    @contextlib.contextmanager
    def _managed_trace(self, trace_path=None):
        """
        PyMCトレースの管理用コンテキストマネージャ

        Args:
            trace_path: トレースの保存先パス（Noneの場合は一時ファイル）

        Yields:
            str: トレースファイルのパス
        """
        # ストレージモードがディスクまたはハイブリッドの場合はファイルに保存
        if self.storage_mode in ['disk', 'hybrid'] or trace_path is not None:
            if trace_path is None:
                # 一時ファイルを作成
                fd, trace_path = tempfile.mkstemp(suffix='.nc')
                os.close(fd)
                self._temp_files.append(trace_path)

            try:
                yield trace_path
            finally:
                # ディスクモードでは保持、それ以外は削除
                if self.storage_mode != 'disk' and trace_path in self._temp_files:
                    try:
                        if os.path.exists(trace_path):
                            os.remove(trace_path)
                        self._temp_files.remove(trace_path)
                    except Exception as e:
                        self.logger.warning(f"トレースファイルの削除中にエラーが発生しました: {str(e)}")
        else:
            # メモリモードではNoneを返す
            yield None

    async def analyze_with_pymc(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        PyMCを使用した高度なベイズ推論分析を実行する

        Args:
            data (pd.DataFrame): 分析対象のデータ
                必要なカラム:
                - revenue_change: 売上変化
                - valuation_change: バリュエーション変化
                - program_cost: プログラムコスト
                - investment_cost: 投資コスト
            **kwargs: 追加のパラメータ
                - hierarchical (bool): 階層モデルを使用するかどうか
                - num_samples (int): MCMCサンプリングの回数
                - chains (int): MCMCチェーンの数
                - storage_mode (str): 一時的にストレージモードを変更する

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            self._validate_data(data)

            # 一時的にストレージモードを変更可能
            temp_storage_mode = kwargs.get('storage_mode', self.storage_mode)
            original_storage_mode = self.storage_mode
            self.storage_mode = temp_storage_mode

            # パラメータの取得
            hierarchical = kwargs.get('hierarchical', False)
            num_samples = kwargs.get('num_samples', 2000)
            tune = kwargs.get('tune', 1000)
            chains = kwargs.get('chains', 4)

            # ROIの計算
            roi_values = self._calculate_roi(data)
            self.register_temp_data('pymc_roi_values', roi_values)

            # PyMCモデルの構築
            if hierarchical:
                results = self._run_hierarchical_model(data, roi_values, num_samples, tune, chains)
            else:
                results = self._run_simple_model(roi_values, num_samples, tune, chains)

            # 結果の保存
            if self.firestore_client and kwargs.get('portfolio_id'):
                await self._save_pymc_results(results, kwargs.get('portfolio_id'))

            # リソース解放
            self.release_resource('pymc_roi_values')
            self.storage_mode = original_storage_mode

            return results

        except Exception as e:
            self.logger.error(f"PyMC分析中にエラーが発生しました: {str(e)}")
            self.storage_mode = original_storage_mode if 'original_storage_mode' in locals() else self.storage_mode
            self.release_resources()  # エラー発生時も確実にリソースを解放
            raise AnalysisError(f"PyMC分析に失敗しました: {str(e)}")

    def _run_simple_model(self, roi_values: np.ndarray, num_samples: int, tune: int, chains: int) -> Dict[str, Any]:
        """
        シンプルなベイズモデルを実行する

        Args:
            roi_values (np.ndarray): ROI値の配列
            num_samples (int): MCMCサンプリングの回数
            tune (int): チューニングステップ数
            chains (int): MCMCチェーンの数

        Returns:
            Dict[str, Any]: モデル実行結果
        """
        with self._managed_trace() as trace_path:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pm.Model() as model:
                    # 事前分布の定義
                    mu = pm.Normal('mu', mu=0.0, sigma=10.0)
                    sigma = pm.HalfNormal('sigma', sigma=1.0)

                    # 尤度の定義
                    roi = pm.Normal('roi', mu=mu, sigma=sigma, observed=roi_values)

                    # サンプリングの実行
                    if trace_path:
                        # ディスクに保存
                        trace = pm.sample(num_samples, tune=tune, chains=chains, return_inferencedata=True,
                                          idata_kwargs={"density_dist_obs": False})
                        trace.to_netcdf(trace_path)
                        trace = az.from_netcdf(trace_path)
                    else:
                        # メモリに保存
                        trace = pm.sample(num_samples, tune=tune, chains=chains, return_inferencedata=True)

                    # 要約統計量の取得
                    summary = az.summary(trace)
                    hdi = az.hdi(trace)

            # 結果の整形
            return {
                "mean_roi": float(summary.loc['mu', 'mean']),
                "std_roi": float(summary.loc['mu', 'sd']),
                "hdi_lower": float(hdi.loc['mu', 'hdi_3%']),
                "hdi_upper": float(hdi.loc['mu', 'hdi_97%']),
                "model_type": "simple",
                "effective_sample_size": float(summary.loc['mu', 'ess_bulk']),
                "r_hat": float(summary.loc['mu', 'r_hat']),
                "plot": self._generate_pymc_plot(trace),
                "timestamp": datetime.now().isoformat(),
                "trace_path": trace_path if self.storage_mode == 'disk' else None
            }

    def _run_hierarchical_model(self, data: pd.DataFrame, roi_values: np.ndarray,
                              num_samples: int, tune: int, chains: int) -> Dict[str, Any]:
        """
        階層ベイズモデルを実行する（クラスタ/グループがあるデータ向け）

        Args:
            data (pd.DataFrame): 元データ
            roi_values (np.ndarray): ROI値の配列
            num_samples (int): MCMCサンプリングの回数
            tune (int): チューニングステップ数
            chains (int): MCMCチェーンの数

        Returns:
            Dict[str, Any]: モデル実行結果
        """
        # データからグループ情報を取得（仮にcategoryカラムがあると仮定）
        if 'category' not in data.columns:
            # カテゴリカラムがなければシンプルモデルにフォールバック
            self.logger.warning("階層モデル用のカテゴリカラムがありません。シンプルモデルにフォールバックします。")
            return self._run_simple_model(roi_values, num_samples, tune, chains)

        categories = data['category'].unique()
        category_idx = [list(categories).index(c) for c in data['category']]

        with self._managed_trace() as trace_path:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pm.Model() as hierarchical_model:
                    # グローバルパラメータ
                    mu_global = pm.Normal('mu_global', mu=0.0, sigma=10.0)
                    sigma_global = pm.HalfNormal('sigma_global', sigma=1.0)

                    # グループレベルのパラメータ
                    mu_group = pm.Normal('mu_group', mu=mu_global, sigma=sigma_global, shape=len(categories))
                    sigma = pm.HalfNormal('sigma', sigma=1.0)

                    # 尤度の定義（グループごと）
                    roi = pm.Normal('roi', mu=mu_group[category_idx], sigma=sigma, observed=roi_values)

                    # サンプリングの実行
                    if trace_path:
                        # ディスクに保存
                        trace = pm.sample(num_samples, tune=tune, chains=chains, return_inferencedata=True,
                                          idata_kwargs={"density_dist_obs": False})
                        trace.to_netcdf(trace_path)
                        trace = az.from_netcdf(trace_path)
                    else:
                        # メモリに保存
                        trace = pm.sample(num_samples, tune=tune, chains=chains, return_inferencedata=True)

                    # 要約統計量の取得
                    summary = az.summary(trace)
                    hdi = az.hdi(trace)

            # グループごとの結果を集計
            group_results = {}
            for i, category in enumerate(categories):
                group_results[str(category)] = {
                    "mean_roi": float(summary.iloc[i + 1, 0]),  # +1 はグローバルパラメータの後
                    "hdi_lower": float(hdi.iloc[i + 1, 0]),
                    "hdi_upper": float(hdi.iloc[i + 1, 1]),
                }

            return {
                "global_mean_roi": float(summary.loc['mu_global', 'mean']),
                "global_std_roi": float(summary.loc['mu_global', 'sd']),
                "global_hdi_lower": float(hdi.loc['mu_global', 'hdi_3%']),
                "global_hdi_upper": float(hdi.loc['mu_global', 'hdi_97%']),
                "group_results": group_results,
                "model_type": "hierarchical",
                "effective_sample_size": float(summary.loc['mu_global', 'ess_bulk']),
                "r_hat": float(summary.loc['mu_global', 'r_hat']),
                "plot": self._generate_pymc_plot(trace),
                "timestamp": datetime.now().isoformat(),
                "trace_path": trace_path if self.storage_mode == 'disk' else None
            }

    def _generate_pymc_plot(self, trace) -> str:
        """
        PyMCモデルの可視化結果をBase64エンコードされた画像として返す

        Args:
            trace: PyMCのトレースオブジェクト

        Returns:
            str: Base64エンコードされたプロット画像
        """
        with self._plot_context(figsize=(12, 8)) as (fig, _):
            # トレースプロット
            ax = az.plot_trace(trace)
            plt.tight_layout()

            # 画像をBase64エンコード
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')

            return img_str

    async def _save_pymc_results(self, results: Dict[str, Any], portfolio_id: str) -> None:
        """
        PyMCモデルの結果をFirestoreに保存する

        Args:
            results (Dict[str, Any]): 分析結果
            portfolio_id (str): ポートフォリオID
        """
        if not self.firestore_client:
            self.logger.warning("Firestoreクライアントが設定されていないため、結果を保存できません。")
            return

        try:
            # 分析結果のドキュメントを作成
            document_data = {
                "portfolio_id": portfolio_id,
                "analysis_type": "bayesian_inference_pymc",
                "result": results,
                "created_at": datetime.now().isoformat(),
            }

            # Firestoreに保存
            await self.firestore_client.set_document(
                collection="analysis_results",
                document_id=None,  # 自動生成
                data=document_data
            )

            self.logger.info(f"PyMC分析結果をFirestoreに保存しました: {portfolio_id}")
        except Exception as e:
            self.logger.error(f"PyMC分析結果の保存中にエラーが発生しました: {str(e)}")
            raise AnalysisError(f"結果の保存に失敗しました: {str(e)}")

    def estimate_memory_usage(self, data_size: int, model_type: str = 'simple', chains: int = 4) -> Dict[str, Any]:
        """
        予想メモリ使用量を計算し、適切なストレージモードを推奨する

        Args:
            data_size: データサイズ
            model_type: モデルタイプ ('simple' または 'hierarchical')
            chains: MCMCチェーン数

        Returns:
            Dict[str, Any]: メモリ使用量予測
        """
        # 簡易的なメモリ使用量予測
        base_memory = 50  # MB
        per_sample_memory = 0.001  # MB
        hierarchical_factor = 2.5 if model_type == 'hierarchical' else 1.0

        # 標準的なサンプリングサイズ
        samples = 2000
        tune = 1000

        total_iterations = (samples + tune) * chains
        estimated_memory = base_memory + (per_sample_memory * total_iterations * data_size * hierarchical_factor)

        # 推奨ストレージモード
        recommended_mode = 'memory'
        if estimated_memory > 1000:  # 1GB以上
            recommended_mode = 'disk'
        elif estimated_memory > 500:  # 500MB以上
            recommended_mode = 'hybrid'

        return {
            'estimated_memory_mb': estimated_memory,
            'recommended_storage_mode': recommended_mode,
            'model_type': model_type,
            'data_size': data_size,
            'chains': chains,
            'samples': samples,
            'tune': tune
        }