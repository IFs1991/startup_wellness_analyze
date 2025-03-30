from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import logging
from .base import BaseAnalyzer, AnalysisError
from tqdm import tqdm
import matplotlib.dates as mdates
import scipy.stats as stats
from .utils import PlotUtility, StatisticsUtility

class MonteCarloSimulator(BaseAnalyzer):
    """
    汎用的なモンテカルロシミュレーションエンジン

    様々な分析モジュールから利用可能な基盤クラスとして実装
    """

    def __init__(self, firestore_client=None):
        """
        初期化

        Args:
            firestore_client: Firestoreクライアントのインスタンス（オプション）
        """
        super().__init__(analysis_type="monte_carlo", firestore_client=firestore_client)

    async def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        データセットに対するモンテカルロシミュレーション分析を実行

        Args:
            data: 分析対象のデータフレーム
            **kwargs: 追加パラメータ
                - scenario: シナリオ名 ("標準", "楽観的", "悲観的" または "カスタム")
                - custom_parameters: カスタムパラメータ (scenarioが"カスタム"の場合)
                - num_simulations: シミュレーション回数
                - forecast_periods: 予測期間
                - confidence_level: 信頼水準 (デフォルト: 0.95)

        Returns:
            分析結果を含む辞書
        """
        self._validate_data(data)

        # パラメータの取得
        scenario = kwargs.get('scenario', '標準')
        custom_parameters = kwargs.get('custom_parameters', {})
        num_simulations = kwargs.get('num_simulations', 1000)
        forecast_periods = kwargs.get('forecast_periods', 24)
        confidence_level = kwargs.get('confidence_level', 0.95)

        # シナリオパラメータの取得
        scenario_params = self._get_scenario_parameters(scenario, custom_parameters, data)

        # シミュレーション実行
        simulation_results = self._run_simulation(
            data,
            num_simulations,
            forecast_periods,
            scenario_params
        )

        # ROI分布の計算
        roi_distribution = self._calculate_roi_distribution(simulation_results)

        # 信頼区間の計算 - 共通ユーティリティを使用
        confidence_intervals = StatisticsUtility.calculate_confidence_intervals(
            roi_distribution['roi_simulations'],
            confidence_level
        )

        # グラフの生成
        plots = self._generate_plots(simulation_results, confidence_intervals, scenario)

        # 結果の集約
        results = {
            'scenario': scenario,
            'parameters': scenario_params,
            'simulation_results': simulation_results,
            'roi_distribution': roi_distribution,
            'confidence_intervals': confidence_intervals,
            'plots': plots
        }

        return results

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        入力データの検証

        Args:
            data: 検証するデータフレーム

        Raises:
            AnalysisError: データが無効な場合
        """
        if data is None:
            raise ValueError("データがNoneです")

        if not isinstance(data, pd.DataFrame):
            raise ValueError("データはpandas DataFrameである必要があります")

        if data.empty:
            raise ValueError("データが空です")

    def _get_base_scenario_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        基本的なシナリオパラメータを取得

        Args:
            data: 過去データ

        Returns:
            基本パラメータを含む辞書
        """
        # この部分は具体的な実装により異なるため、サブクラスでオーバーライドすることを想定
        return {}

    def _get_scenario_parameters(
        self,
        scenario: str,
        custom_parameters: Dict[str, Any],
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        シナリオに基づくパラメータを取得

        Args:
            scenario: シナリオ名
            custom_parameters: カスタムパラメータ
            data: 過去データ

        Returns:
            シナリオパラメータを含む辞書
        """
        # 基本パラメータを取得
        base_params = self._get_base_scenario_parameters(data)

        # シナリオに応じてパラメータを調整
        if scenario == 'カスタム':
            # カスタムパラメータで上書き
            for key, value in custom_parameters.items():
                if key in base_params:
                    base_params[key] = value
        elif scenario == '標準':
            # 標準シナリオはベースのままでOK
            pass
        elif scenario == '楽観的':
            # 楽観的シナリオのパラメータ調整
            # 具体的な調整はサブクラスで実装
            pass
        elif scenario == '悲観的':
            # 悲観的シナリオのパラメータ調整
            # 具体的な調整はサブクラスで実装
            pass
        else:
            raise ValueError(f"不明なシナリオ: {scenario}")

        return base_params

    def run_monte_carlo_simulation(
        self,
        initial_values: Dict[str, float],
        num_simulations: int,
        num_periods: int,
        simulation_func
    ) -> Dict[str, Any]:
        """
        汎用的なモンテカルロシミュレーションを実行

        Args:
            initial_values: 初期値
            num_simulations: シミュレーション回数
            num_periods: シミュレーション期間
            simulation_func: 各期間の値を計算する関数
                signature: func(previous_values, period_index) -> new_values

        Returns:
            シミュレーション結果
        """
        # 結果を格納する配列
        results = {
            'paths': [],
            'statistics': {}
        }

        # シミュレーション実行
        for i in tqdm(range(num_simulations), desc="シミュレーション実行中"):
            # 1つのパスをシミュレーション
            path = self._simulate_single_path(initial_values, num_periods, simulation_func)
            results['paths'].append(path)

        # 統計情報の計算
        results['statistics'] = self._calculate_simulation_statistics(results['paths'])

        return results

    def _simulate_single_path(
        self,
        initial_values: Dict[str, float],
        num_periods: int,
        simulation_func
    ) -> Dict[str, List[float]]:
        """
        単一シミュレーションパスを生成

        Args:
            initial_values: 初期値
            num_periods: シミュレーション期間
            simulation_func: 各期間の値を計算する関数

        Returns:
            シミュレーションパス
        """
        # 結果を格納する辞書
        path = {key: [value] for key, value in initial_values.items()}

        # 各期間のシミュレーション
        for period in range(1, num_periods + 1):
            # 前期の値を取得
            previous_values = {key: values[-1] for key, values in path.items()}

            # 新しい値を計算
            new_values = simulation_func(previous_values, period)

            # 結果を追加
            for key, value in new_values.items():
                if key in path:
                    path[key].append(value)
                else:
                    # 新しいキーが返された場合は初期化
                    path[key] = [None] * (period - 1) + [value]

        return path

    def _calculate_simulation_statistics(self, paths: List[Dict[str, List[float]]]) -> Dict[str, Any]:
        """
        シミュレーション結果の統計情報を計算

        Args:
            paths: シミュレーションパスのリスト

        Returns:
            統計情報
        """
        if not paths:
            return {}

        # 各変数と期間ごとの統計情報を計算
        statistics = {}

        # すべてのパスに含まれる変数を特定
        all_keys = set()
        for path in paths:
            all_keys.update(path.keys())

        # 各変数について統計を計算
        for key in all_keys:
            # その変数のすべての期間のすべてのパスの値を収集
            periods_data = []

            # 最大期間数を特定
            max_periods = max(len(path.get(key, [])) for path in paths)

            for period in range(max_periods):
                period_values = [
                    path.get(key, [])[period]
                    for path in paths
                    if key in path and period < len(path[key]) and path[key][period] is not None
                ]
                periods_data.append(period_values)

            # 統計情報を計算
            period_stats = []
            for period_values in periods_data:
                if period_values:
                    stats = {
                        'mean': np.mean(period_values),
                        'median': np.median(period_values),
                        'std': np.std(period_values),
                        'min': np.min(period_values),
                        'max': np.max(period_values),
                        'q25': np.percentile(period_values, 25),
                        'q75': np.percentile(period_values, 75)
                    }
                else:
                    stats = {
                        'mean': None,
                        'median': None,
                        'std': None,
                        'min': None,
                        'max': None,
                        'q25': None,
                        'q75': None
                    }
                period_stats.append(stats)

            statistics[key] = period_stats

        return statistics

    def _calculate_roi_distribution(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        ROI分布の統計量を計算する

        Args:
            simulation_results (Dict[str, Any]): シミュレーション結果

        Returns:
            Dict[str, Any]: ROI分布の統計量
        """
        roi_simulations = simulation_results['roi_simulations']

        # 分位数の計算
        quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        quantile_values = np.quantile(roi_simulations, quantiles)

        # ヒストグラムデータの計算
        hist, bin_edges = np.histogram(roi_simulations, bins=50, density=True)

        # ROI分布の統計量
        distribution = {
            'quantiles': {str(int(q*100)) + '%': float(val) for q, val in zip(quantiles, quantile_values)},
            'histogram': {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            },
            'positive_roi_probability': float((roi_simulations > 0).mean()),
            'skewness': float(StatisticsUtility.calculate_skewness(roi_simulations)),
            'kurtosis': float(StatisticsUtility.calculate_kurtosis(roi_simulations))
        }

        return distribution

    def _generate_plots(
        self,
        simulation_results: Dict[str, Any],
        confidence_intervals: Dict[str, Any],
        scenario: str
    ) -> Dict[str, str]:
        """
        シミュレーション結果のプロットを生成する

        Args:
            simulation_results (Dict[str, Any]): シミュレーション結果
            confidence_intervals (Dict[str, Any]): 信頼区間
            scenario (str): シナリオ名

        Returns:
            Dict[str, str]: Base64エンコードされたプロット画像
        """
        plots = {}

        # ROI分布のヒストグラムプロット
        roi_hist_base64 = self._generate_roi_histogram_plot(
            simulation_results['roi_simulations'],
            confidence_intervals,
            scenario
        )
        plots['roi_histogram'] = roi_hist_base64

        # 売上予測のプロット
        revenue_forecast_base64 = self._generate_forecast_plot(
            simulation_results['revenue_forecasts'],
            '売上変化予測',
            scenario
        )
        plots['revenue_forecast'] = revenue_forecast_base64

        # バリュエーション予測のプロット
        valuation_forecast_base64 = self._generate_forecast_plot(
            simulation_results['valuation_forecasts'],
            'バリュエーション変化予測',
            scenario
        )
        plots['valuation_forecast'] = valuation_forecast_base64

        return plots

    def _generate_roi_histogram_plot(
        self,
        roi_simulations: np.ndarray,
        confidence_intervals: Dict[str, Any],
        scenario: str
    ) -> str:
        """
        ROI分布のヒストグラムプロットを生成する

        Args:
            roi_simulations (np.ndarray): ROIシミュレーション結果
            confidence_intervals (Dict[str, Any]): 信頼区間
            scenario (str): シナリオ名

        Returns:
            str: Base64エンコードされた画像
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # ヒストグラムの描画
        sns.histplot(roi_simulations, kde=True, ax=ax, color='skyblue')

        # 平均値と中央値の縦線
        ax.axvline(roi_simulations.mean(), color='red', linestyle='--', alpha=0.8, label=f'平均: {roi_simulations.mean():.4f}')
        ax.axvline(np.median(roi_simulations), color='green', linestyle='-.', alpha=0.8, label=f'中央値: {np.median(roi_simulations):.4f}')

        # 信頼区間の縦線
        if 'percentile' in confidence_intervals:
            ci_lower, ci_upper = confidence_intervals['percentile']
            ax.axvline(ci_lower, color='purple', linestyle=':', alpha=0.8,
                      label=f'95%信頼区間: [{ci_lower:.4f}, {ci_upper:.4f}]')
            ax.axvline(ci_upper, color='purple', linestyle=':', alpha=0.8)

        # ゼロの縦線
        ax.axvline(0, color='black', linestyle='-', alpha=0.5, label='ROI=0')

        # プロット領域の着色
        if 'percentile' in confidence_intervals:
            ci_lower, ci_upper = confidence_intervals['percentile']
            ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='purple')

        # ROIが正となる確率を計算して表示
        positive_roi_prob = (roi_simulations > 0).mean()
        ax.text(0.02, 0.95, f'正のROI確率: {positive_roi_prob:.2%}', transform=ax.transAxes, fontsize=10, va='top')

        # タイトルとラベル
        scenario_titles = {
            'base': '基本シナリオ',
            'optimistic': '楽観的シナリオ',
            'pessimistic': '悲観的シナリオ',
            'custom': 'カスタムシナリオ'
        }
        scenario_title = scenario_titles.get(scenario, scenario)

        ax.set_title(f'ROI分布 - {scenario_title}', fontsize=14)
        ax.set_xlabel('ROI (投資収益率)', fontsize=12)
        ax.set_ylabel('確率密度', fontsize=12)
        ax.legend(loc='upper right')

        # グリッドの表示
        ax.grid(True, alpha=0.3)

        # レイアウト調整
        plt.tight_layout()

        # 画像をBase64エンコード
        return PlotUtility.save_plot_to_base64(fig)

    def _generate_forecast_plot(
        self,
        forecast_data: Dict[str, List[float]],
        title: str,
        scenario: str
    ) -> str:
        """
        予測データのプロットを生成する

        Args:
            forecast_data (Dict[str, List[float]]): 予測データ
            title (str): プロットタイトル
            scenario (str): シナリオ名

        Returns:
            str: Base64エンコードされた画像
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # x軸の期間
        periods = range(1, len(forecast_data['mean']) + 1)

        # 平均値のプロット
        ax.plot(periods, forecast_data['mean'], 'b-', linewidth=2, label='平均')

        # 中央値のプロット
        ax.plot(periods, forecast_data['median'], 'g--', linewidth=1.5, label='中央値')

        # 信頼区間のプロット
        ax.fill_between(periods, forecast_data['p05'], forecast_data['p95'], color='b', alpha=0.2, label='90%信頼区間')

        # タイトルとラベル
        scenario_titles = {
            'base': '基本シナリオ',
            'optimistic': '楽観的シナリオ',
            'pessimistic': '悲観的シナリオ',
            'custom': 'カスタムシナリオ'
        }
        scenario_title = scenario_titles.get(scenario, scenario)

        ax.set_title(f'{title} - {scenario_title}', fontsize=14)
        ax.set_xlabel('期間（月）', fontsize=12)
        ax.set_ylabel('値', fontsize=12)
        ax.legend(loc='upper left')

        # グリッドの表示
        ax.grid(True, alpha=0.3)

        # x軸目盛りの調整
        ax.set_xticks(periods)

        # レイアウト調整
        plt.tight_layout()

        # 画像をBase64エンコード
        return PlotUtility.save_plot_to_base64(fig)