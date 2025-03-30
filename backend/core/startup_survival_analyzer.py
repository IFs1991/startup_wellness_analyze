# -*- coding: utf-8 -*-
"""
スタートアップ生存分析モジュール
スタートアップの財務パラメータに基づいて、生存確率とランウェイを分析します。
Firestoreと統合したバージョン。
"""
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from tqdm import tqdm
import scipy.stats as stats
from service.firestore.client import FirestoreService, StorageError
from .utils import PlotUtility, StatisticsUtility, AnalysisError

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class StartupSurvivalAnalysisError(AnalysisError):
    """スタートアップ生存分析に関するエラー"""
    pass

class FirestoreStartupSurvivalAnalyzer:
    """
    スタートアップの生存性を分析し、結果をFirestoreに保存するクラス

    キャッシュフロー、バーンレート、資金枯渇確率などを分析し、
    スタートアップの財務的生存可能性を評価します。
    """

    def __init__(self) -> None:
        """
        Firestoreサービスとの接続を初期化します。

        Raises:
            StorageError: Firestore接続の初期化に失敗した場合
        """
        try:
            self.firestore_service = FirestoreService()
            self.collection_name = 'startup_survival_analysis'
            logger.info("FirestoreStartupSurvivalAnalyzer initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize FirestoreStartupSurvivalAnalyzer: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

        # 財務シミュレーションのデフォルトパラメータ設定
        self.default_params = {
            'initial_cash': 100000000,  # 初期資金（円）
            'monthly_burn_rate': 10000000,  # 月間バーンレート基本値（円）
            'burn_rate_volatility': 0.2,  # バーンレートの変動性（標準偏差）
            'monthly_revenue': 2000000,  # 月間収益基本値（円）
            'revenue_growth_rate': 0.05,  # 月間収益成長率
            'revenue_volatility': 0.3,  # 収益の変動性（標準偏差）
            'months_to_simulate': 24,  # シミュレーション期間（月）
            'simulation_runs': 1000,  # シミュレーション回数
        }

    async def analyze_and_save(
        self,
        data: Optional[pd.DataFrame] = None,
        params: Optional[Dict[str, Any]] = None,
        scenario: str = '標準',
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        generate_plots: bool = True
    ) -> Tuple[Dict[str, Any], str]:
        """
        スタートアップの生存性分析を実行し、結果をFirestoreに保存します。

        Args:
            data: 過去データ（オプション）、指定がない場合はパラメータのみで分析
            params: 分析パラメータ、指定がない場合はデフォルト値を使用
            scenario: シナリオ名 ('標準', '楽観的', '悲観的', 'カスタム')
            user_id: 分析を実行したユーザーのID
            metadata: 追加のメタデータ
            generate_plots: 可視化を生成するかどうか

        Returns:
            Tuple[Dict[str, Any], str]: (分析結果, FirestoreのドキュメントID)

        Raises:
            StartupSurvivalAnalysisError: 分析処理中にエラーが発生した場合
            StorageError: Firestoreへの保存時にエラーが発生した場合
            ValueError: 入力パラメータが不正な場合
        """
        try:
            # パラメータの設定
            simulation_params = self.default_params.copy()
            if params:
                simulation_params.update(params)

            # シナリオに応じてパラメータを調整
            adjusted_params = self._adjust_params_for_scenario(simulation_params, scenario)

            logger.info(f"Starting survival analysis with scenario: {scenario}")

            # シミュレーション実行
            simulation_results = self._run_monte_carlo_simulation(adjusted_params)

            # 生存率分析
            survival_metrics = self._calculate_survival_metrics(simulation_results, adjusted_params)

            # 可視化の生成（オプション）
            plots = {}
            if generate_plots:
                plots = self._generate_survival_visualizations(
                    simulation_results,
                    survival_metrics,
                    adjusted_params,
                    scenario
                )

            # 分析結果の準備
            analysis_result = {
                'scenario': scenario,
                'parameters': adjusted_params,
                'survival_metrics': survival_metrics,
                'simulation_summary': {
                    'num_simulations': adjusted_params['simulation_runs'],
                    'months_simulated': adjusted_params['months_to_simulate'],
                    'initial_cash': adjusted_params['initial_cash'],
                    'monthly_burn_rate': adjusted_params['monthly_burn_rate'],
                    'monthly_revenue': adjusted_params['monthly_revenue'],
                    'terminal_survival_rate': survival_metrics['survival_probability']['end_of_simulation']
                },
                'created_at': datetime.now().isoformat(),
                'user_id': user_id,
                'metadata': metadata or {}
            }

            # 可視化データを追加
            if plots:
                analysis_result['plots'] = plots

            # Firestoreに保存
            doc_ids = await self.firestore_service.save_results(
                results=[analysis_result],
                collection_name=self.collection_name
            )

            if not doc_ids:
                raise StorageError("結果をFirestoreに保存できませんでした")

            doc_id = doc_ids[0]
            logger.info(f"Successfully saved survival analysis results with ID: {doc_id}")

            return analysis_result, doc_id

        except Exception as e:
            error_msg = f"生存性分析中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (StartupSurvivalAnalysisError, StorageError, ValueError)):
                raise
            raise StartupSurvivalAnalysisError(error_msg) from e

    def _adjust_params_for_scenario(
        self,
        params: Dict[str, Any],
        scenario: str
    ) -> Dict[str, Any]:
        """
        シナリオに基づいてパラメータを調整する

        Args:
            params: 基本パラメータ
            scenario: シナリオ名

        Returns:
            調整後のパラメータ
        """
        adjusted_params = params.copy()

        if scenario == '標準':
            # 標準シナリオは基本パラメータをそのまま使用
            pass

        elif scenario == '楽観的':
            # 収益成長率を上げ、バーンレートの変動性を下げる
            adjusted_params['revenue_growth_rate'] = params['revenue_growth_rate'] * 1.5
            adjusted_params['monthly_revenue'] = params['monthly_revenue'] * 1.2
            adjusted_params['burn_rate_volatility'] = params['burn_rate_volatility'] * 0.8
            adjusted_params['monthly_burn_rate'] = params['monthly_burn_rate'] * 0.9

        elif scenario == '悲観的':
            # 収益成長率を下げ、バーンレートの変動性を上げる
            adjusted_params['revenue_growth_rate'] = params['revenue_growth_rate'] * 0.7
            adjusted_params['monthly_revenue'] = params['monthly_revenue'] * 0.8
            adjusted_params['burn_rate_volatility'] = params['burn_rate_volatility'] * 1.3
            adjusted_params['monthly_burn_rate'] = params['monthly_burn_rate'] * 1.2

        elif scenario == 'カスタム':
            # カスタムシナリオは渡されたパラメータをそのまま使用
            pass

        else:
            logger.warning(f"不明なシナリオ: {scenario}、標準シナリオとして扱います")

        return adjusted_params

    def _run_monte_carlo_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        モンテカルロシミュレーションを実行

        Args:
            params: シミュレーションパラメータ

        Returns:
            シミュレーション結果
        """
        # パラメータを取得
        initial_cash = params['initial_cash']
        monthly_burn_rate = params['monthly_burn_rate']
        burn_rate_volatility = params['burn_rate_volatility']
        monthly_revenue = params['monthly_revenue']
        revenue_growth_rate = params['revenue_growth_rate']
        revenue_volatility = params['revenue_volatility']
        months_to_simulate = params['months_to_simulate']
        simulation_runs = params['simulation_runs']

        # 結果格納用の配列
        all_paths = []
        all_cash_out_months = []
        all_terminal_cash = []
        all_terminal_revenue = []

        # シミュレーション実行
        for _ in tqdm(range(simulation_runs), desc="シミュレーション実行中"):
            # 各シミュレーションの結果を格納
            cash_values = [initial_cash]
            revenue_values = [monthly_revenue]
            survived = True
            cash_out_month = None

            # 各月のシミュレーション
            for month in range(1, months_to_simulate + 1):
                if not survived:
                    cash_values.append(0)
                    revenue_values.append(revenue_values[-1])
                    continue

                # 収益の計算（成長率と変動性を考慮）
                growth_factor = (1 + revenue_growth_rate) ** month
                revenue_variation = np.random.normal(1, revenue_volatility)
                revenue = monthly_revenue * growth_factor * revenue_variation
                revenue_values.append(revenue)

                # バーンレートの計算（変動性を考慮）
                burn_rate_variation = np.random.normal(1, burn_rate_volatility)
                burn_rate = monthly_burn_rate * burn_rate_variation

                # 現金残高の更新
                prev_cash = cash_values[-1]
                new_cash = prev_cash + revenue - burn_rate

                # キャッシュアウトの判定
                if new_cash <= 0:
                    new_cash = 0
                    survived = False
                    cash_out_month = month

                cash_values.append(new_cash)

            # 結果を保存
            path = {
                'cash': cash_values,
                'revenue': revenue_values,
                'survived': survived
            }
            all_paths.append(path)
            all_terminal_cash.append(cash_values[-1])
            all_terminal_revenue.append(revenue_values[-1])

            # キャッシュアウト月を記録
            if cash_out_month:
                all_cash_out_months.append(cash_out_month)
            else:
                all_cash_out_months.append(months_to_simulate)

        # 統計情報の計算
        simulation_results = {
            'paths': all_paths,
            'terminal_cash': all_terminal_cash,
            'terminal_revenue': all_terminal_revenue,
            'cash_out_months': all_cash_out_months
        }

        return simulation_results

    def _calculate_survival_metrics(
        self,
        simulation_results: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生存率と関連指標を計算

        Args:
            simulation_results: シミュレーション結果
            params: シミュレーションパラメータ

        Returns:
            生存分析の指標
        """
        paths = simulation_results['paths']
        cash_out_months = simulation_results['cash_out_months']
        months_to_simulate = params['months_to_simulate']
        simulation_runs = params['simulation_runs']

        # 月ごとの生存確率を計算
        survival_probabilities = []
        for month in range(months_to_simulate + 1):
            # 月ごとの生存数をカウント
            survived_count = sum(
                1 for path in paths
                if month < len(path['cash']) and path['cash'][month] > 0
            )
            survival_prob = survived_count / len(paths)
            survival_probabilities.append(float(survival_prob))

        # ランウェイ統計
        runway_stats = {
            'mean_months': float(np.mean(cash_out_months)),
            'median_months': float(np.median(cash_out_months)),
            'std_months': float(np.std(cash_out_months)),
            'min_months': float(np.min(cash_out_months)),
            'max_months': float(np.max(cash_out_months))
        }

        # ランウェイの信頼区間
        runway_ci = StatisticsUtility.calculate_confidence_intervals(
            np.array(cash_out_months),
            confidence_level=0.95
        )

        # 終了時の現金残高統計
        terminal_cash = simulation_results['terminal_cash']
        terminal_cash_stats = {
            'mean': float(np.mean(terminal_cash)),
            'median': float(np.median(terminal_cash)),
            'std': float(np.std(terminal_cash)),
            'min': float(np.min(terminal_cash)),
            'max': float(np.max(terminal_cash)),
            'positive_ratio': float(np.mean([c > 0 for c in terminal_cash]))
        }

        # 各月での資金枯渇確率
        depletion_probabilities = []
        for month in range(1, months_to_simulate + 1):
            depleted_count = sum(1 for m in cash_out_months if m <= month)
            depletion_prob = depleted_count / simulation_runs
            depletion_probabilities.append(float(depletion_prob))

        # 結果をまとめる
        survival_metrics = {
            'survival_probability': {
                'by_month': survival_probabilities,
                'end_of_simulation': survival_probabilities[-1]
            },
            'runway_statistics': runway_stats,
            'runway_confidence_interval': {
                'lower': runway_ci['percentile'][0],
                'upper': runway_ci['percentile'][1]
            },
            'depletion_probability': {
                'by_month': depletion_probabilities
            },
            'terminal_cash_statistics': terminal_cash_stats
        }

        return survival_metrics

    def _generate_survival_visualizations(
        self,
        simulation_results: Dict[str, Any],
        survival_metrics: Dict[str, Any],
        params: Dict[str, Any],
        scenario: str
    ) -> Dict[str, str]:
        """
        生存分析結果の可視化を生成

        Args:
            simulation_results: シミュレーション結果
            survival_metrics: 生存分析の指標
            params: シミュレーションパラメータ
            scenario: シナリオ名

        Returns:
            Dict[str, str]: Base64エンコードされた可視化画像
        """
        plots = {}

        try:
            # シナリオ名の整形
            scenario_titles = {
                '標準': '標準シナリオ',
                '楽観的': '楽観的シナリオ',
                '悲観的': '悲観的シナリオ',
                'カスタム': 'カスタムシナリオ'
            }
            scenario_title = scenario_titles.get(scenario, scenario)

            # 1. 生存率曲線
            months = list(range(params['months_to_simulate'] + 1))
            survival_rates = survival_metrics['survival_probability']['by_month']

            fig = PlotUtility.generate_basic_plot(
                x_data=months,
                y_data=survival_rates,
                title=f'スタートアップ生存率推移 - {scenario_title}',
                x_label='月数',
                y_label='生存確率',
                color='blue'
            )
            plots['survival_curve'] = PlotUtility.save_plot_to_base64(fig)

            # 2. 資金枯渇確率曲線
            months = list(range(1, params['months_to_simulate'] + 1))
            depletion_rates = survival_metrics['depletion_probability']['by_month']

            fig = PlotUtility.generate_basic_plot(
                x_data=months,
                y_data=depletion_rates,
                title=f'資金枯渇確率推移 - {scenario_title}',
                x_label='月数',
                y_label='資金枯渇確率',
                color='red'
            )
            plots['depletion_curve'] = PlotUtility.save_plot_to_base64(fig)

            # 3. ランウェイ分布のヒストグラム
            runway_data = np.array(simulation_results['cash_out_months'])
            ci = survival_metrics['runway_confidence_interval']
            confidence_data = {
                'percentile': [ci['lower'], ci['upper']],
                'mean': survival_metrics['runway_statistics']['mean_months']
            }

            fig = PlotUtility.generate_histogram_plot(
                data=runway_data,
                title=f'ランウェイ分布 - {scenario_title}',
                x_label='ランウェイ（月数）',
                bins=min(30, len(set(runway_data))),
                confidence_intervals=confidence_data
            )
            plots['runway_histogram'] = PlotUtility.save_plot_to_base64(fig)

            # 4. 現金推移パスサンプル（最大100パス）
            fig, ax = plt.subplots(figsize=(12, 8))

            sample_size = min(100, len(simulation_results['paths']))
            sampled_paths = np.random.choice(simulation_results['paths'], size=sample_size, replace=False)

            months = list(range(params['months_to_simulate'] + 1))
            for i, path in enumerate(sampled_paths):
                # 生存/非生存で色分け
                color = 'blue' if path['survived'] else 'grey'
                alpha = 0.1 + 0.4 * path['survived']  # 生存パスの方が濃く表示
                ax.plot(months, path['cash'], color=color, alpha=alpha, linewidth=1)

            # 最後に平均パスを太線で表示
            mean_cash = []
            for month in months:
                mean_value = np.mean([path['cash'][month] for path in simulation_results['paths']])
                mean_cash.append(mean_value)

            ax.plot(months, mean_cash, color='red', linewidth=2, label='平均パス')

            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_title(f'キャッシュフロー推移 - {scenario_title}')
            ax.set_xlabel('月数')
            ax.set_ylabel('現金残高（円）')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)

            # Y軸の表示を整形（百万円単位など）
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x/1000000:.0f}M"))

            plt.tight_layout()
            plots['cash_flow_paths'] = PlotUtility.save_plot_to_base64(fig)

            return plots

        except Exception as e:
            logger.warning(f"生存分析可視化の生成中にエラーが発生しました: {str(e)}")
            return plots

    async def get_analysis_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        過去の分析結果を取得

        Args:
            user_id: 特定ユーザーの結果のみを取得する場合に指定
            limit: 取得する結果の最大数

        Returns:
            List[Dict]: 分析結果の履歴

        Raises:
            StorageError: Firestoreからのデータ取得に失敗した場合
        """
        try:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError("limitは正の整数である必要があります")

            conditions = []
            if user_id:
                conditions.append({
                    'field': 'user_id',
                    'operator': '==',
                    'value': user_id
                })

            results = await self.firestore_service.fetch_documents(
                collection_name=self.collection_name,
                conditions=conditions,
                limit=limit,
                order_by='created_at',
                direction='desc'
            )

            if results is None:
                return []

            logger.info(f"Retrieved {len(results)} analysis results")
            return results

        except Exception as e:
            error_msg = f"分析履歴の取得中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, StorageError):
                raise
            raise StorageError(error_msg) from e

    async def compare_scenarios(
        self,
        scenario_params: Dict[str, Dict[str, Any]],
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], str]:
        """
        複数のシナリオを比較分析

        Args:
            scenario_params: シナリオ名とそのパラメータのマッピング
            user_id: 分析を実行したユーザーのID
            metadata: 追加のメタデータ

        Returns:
            Tuple[Dict[str, Dict[str, Any]], str]: (シナリオ比較結果, FirestoreのドキュメントID)

        Raises:
            StartupSurvivalAnalysisError: 分析処理中にエラーが発生した場合
            StorageError: Firestoreへの保存時にエラーが発生した場合
        """
        try:
            if not scenario_params:
                raise ValueError("少なくとも1つのシナリオパラメータが必要です")

            # 各シナリオの分析結果を格納
            scenario_results = {}
            comparison_plots = {}

            # 各シナリオを分析
            for scenario_name, params in scenario_params.items():
                logger.info(f"Analyzing scenario: {scenario_name}")

                # 分析実行（Firestoreには保存しない）
                adjusted_params = self._adjust_params_for_scenario(
                    self.default_params.copy(),
                    scenario_name
                )
                if params:
                    adjusted_params.update(params)

                simulation_results = self._run_monte_carlo_simulation(adjusted_params)
                survival_metrics = self._calculate_survival_metrics(simulation_results, adjusted_params)

                # 結果を保存
                scenario_results[scenario_name] = {
                    'parameters': adjusted_params,
                    'survival_metrics': survival_metrics
                }

            # シナリオ比較の可視化
            comparison_plots = self._generate_scenario_comparison_plots(scenario_results)

            # 比較結果の準備
            comparison_result = {
                'scenarios': list(scenario_results.keys()),
                'scenario_results': scenario_results,
                'plots': comparison_plots,
                'created_at': datetime.now().isoformat(),
                'user_id': user_id,
                'metadata': metadata or {}
            }

            # Firestoreに保存
            doc_ids = await self.firestore_service.save_results(
                results=[comparison_result],
                collection_name=f"{self.collection_name}_comparisons"
            )

            if not doc_ids:
                raise StorageError("結果をFirestoreに保存できませんでした")

            doc_id = doc_ids[0]
            logger.info(f"Successfully saved scenario comparison with ID: {doc_id}")

            return comparison_result, doc_id

        except Exception as e:
            error_msg = f"シナリオ比較中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (StartupSurvivalAnalysisError, StorageError, ValueError)):
                raise
            raise StartupSurvivalAnalysisError(error_msg) from e

    def _generate_scenario_comparison_plots(
        self,
        scenario_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        シナリオ比較の可視化を生成

        Args:
            scenario_results: 各シナリオの分析結果

        Returns:
            Dict[str, str]: Base64エンコードされた可視化画像
        """
        plots = {}

        try:
            # シナリオのリスト
            scenarios = list(scenario_results.keys())

            # シナリオ名の整形
            scenario_labels = {
                '標準': '標準シナリオ',
                '楽観的': '楽観的シナリオ',
                '悲観的': '悲観的シナリオ',
                'カスタム': 'カスタムシナリオ'
            }

            # 1. 生存率曲線の比較
            fig, ax = plt.subplots(figsize=(12, 8))

            for scenario in scenarios:
                survival_rates = scenario_results[scenario]['survival_metrics']['survival_probability']['by_month']
                months = list(range(len(survival_rates)))
                label = scenario_labels.get(scenario, scenario)
                ax.plot(months, survival_rates, linewidth=2, label=label)

            ax.set_title('シナリオ別 生存率推移の比較')
            ax.set_xlabel('月数')
            ax.set_ylabel('生存確率')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            plt.tight_layout()

            plots['survival_comparison'] = PlotUtility.save_plot_to_base64(fig)

            # 2. ランウェイの箱ひげ図比較
            fig, ax = plt.subplots(figsize=(12, 8))

            runway_data = []
            for scenario in scenarios:
                runway_stats = scenario_results[scenario]['survival_metrics']['runway_statistics']
                runway_data.append({
                    'scenario': scenario_labels.get(scenario, scenario),
                    'mean': runway_stats['mean_months'],
                    'median': runway_stats['median_months'],
                    'min': runway_stats['min_months'],
                    'max': runway_stats['max_months']
                })

            # データフレームに変換
            runway_df = pd.DataFrame(runway_data)

            # 箱ひげ図の描画
            sns.barplot(x='scenario', y='mean', data=runway_df, ax=ax)

            # エラーバーの追加
            for i, row in runway_df.iterrows():
                ax.errorbar(
                    i, row['mean'],
                    yerr=[[row['mean'] - row['min']], [row['max'] - row['mean']]],
                    fmt='o', color='black', capsize=5
                )

            ax.set_title('シナリオ別 平均ランウェイの比較')
            ax.set_xlabel('シナリオ')
            ax.set_ylabel('平均ランウェイ（月数）')
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            plots['runway_comparison'] = PlotUtility.save_plot_to_base64(fig)

            # 3. 終了時の生存率比較（棒グラフ）
            fig, ax = plt.subplots(figsize=(12, 8))

            survival_rates = []
            for scenario in scenarios:
                terminal_rate = scenario_results[scenario]['survival_metrics']['survival_probability']['end_of_simulation']
                survival_rates.append({
                    'scenario': scenario_labels.get(scenario, scenario),
                    'survival_rate': terminal_rate
                })

            # データフレームに変換
            survival_df = pd.DataFrame(survival_rates)

            # 棒グラフの描画
            bars = ax.bar(
                survival_df['scenario'],
                survival_df['survival_rate'],
                color=plt.cm.viridis(np.linspace(0, 1, len(scenarios)))
            )

            # 値を表示
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.1%}',
                    ha='center', va='bottom'
                )

            ax.set_title('シナリオ別 最終生存率の比較')
            ax.set_xlabel('シナリオ')
            ax.set_ylabel('生存確率')
            ax.set_ylim(0, 1.1)
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            plots['terminal_survival_comparison'] = PlotUtility.save_plot_to_base64(fig)

            return plots

        except Exception as e:
            logger.warning(f"シナリオ比較可視化の生成中にエラーが発生しました: {str(e)}")
            return plots

    async def close(self) -> None:
        """
        リソースを解放します。

        Raises:
            StorageError: リソースの解放に失敗した場合
        """
        try:
            await self.firestore_service.close()
            logger.info("FirestoreStartupSurvivalAnalyzer closed successfully")
        except Exception as e:
            error_msg = f"Error closing FirestoreStartupSurvivalAnalyzer: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

def create_startup_survival_analyzer() -> FirestoreStartupSurvivalAnalyzer:
    """
    FirestoreStartupSurvivalAnalyzerのインスタンスを作成します。

    Returns:
        FirestoreStartupSurvivalAnalyzer: 初期化済みのアナライザーインスタンス

    Raises:
        StorageError: アナライザーの初期化に失敗した場合
    """
    return FirestoreStartupSurvivalAnalyzer()

# クラスの別名を定義
StartupSurvivalAnalyzer = FirestoreStartupSurvivalAnalyzer