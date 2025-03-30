import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import seaborn as sns
from tqdm import tqdm
import matplotlib.dates as mdates
from datetime import datetime
import scipy.stats as stats
import os
import json

from .MonteCarloSimulator import MonteCarloSimulator, BaseAnalyzer
from .utils import PlotUtility, StatisticsUtility

class StartupSurvivabilityAnalyzer(MonteCarloSimulator):
    """
    スタートアップの生存性を分析するクラス

    キャッシュフロー、バーンレート、資金枯渇確率などを分析し、
    スタートアップの財務的生存可能性を評価します。
    """

    def __init__(self,
                 initial_cash=100000000,  # 初期資金（円）
                 monthly_burn_rate=10000000,  # 月間バーンレート基本値（円）
                 burn_rate_volatility=0.2,  # バーンレートの変動性（標準偏差）
                 monthly_revenue=2000000,  # 月間収益基本値（円）
                 revenue_growth_rate=0.05,  # 月間収益成長率
                 revenue_volatility=0.3,  # 収益の変動性（標準偏差）
                 months_to_simulate=24,  # シミュレーション期間（月）
                 simulation_runs=1000,  # シミュレーション回数
                 firestore_client=None):  # Firestoreクライアント（オプション）
        """
        初期化

        Args:
            initial_cash: 初期資金（円）
            monthly_burn_rate: 月間バーンレート基本値（円）
            burn_rate_volatility: バーンレートの変動性（標準偏差）
            monthly_revenue: 月間収益基本値（円）
            revenue_growth_rate: 月間収益成長率
            revenue_volatility: 収益の変動性（標準偏差）
            months_to_simulate: シミュレーション期間（月）
            simulation_runs: シミュレーション回数
            firestore_client: Firestoreクライアント（オプション）
        """
        super().__init__(firestore_client)

        # 財務パラメータ
        self.initial_cash = initial_cash
        self.monthly_burn_rate = monthly_burn_rate
        self.burn_rate_volatility = burn_rate_volatility
        self.monthly_revenue = monthly_revenue
        self.revenue_growth_rate = revenue_growth_rate
        self.revenue_volatility = revenue_volatility

        # シミュレーションパラメータ
        self.months_to_simulate = months_to_simulate
        self.simulation_runs = simulation_runs

        # 結果を格納する変数
        self.simulation_results = None
        self.survival_probabilities = None
        self.monthly_runway_means = None
        self.cash_out_months = None

    async def analyze(self, data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        スタートアップの生存性分析を実行

        Args:
            data: 過去データ（オプション）
            **kwargs: 追加パラメータ
                - scenario: シナリオ名
                - custom_parameters: カスタムパラメータ

        Returns:
            分析結果
        """
        # パラメータのオーバーライド
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # シミュレーション実行
        initial_values = {'cash': self.initial_cash}

        # シミュレーションの実行
        simulation_results = self.run_monte_carlo_simulation(
            initial_values,
            self.simulation_runs,
            self.months_to_simulate,
            self._startup_simulation_step
        )

        # 結果の保存
        self.simulation_results = simulation_results

        # 生存確率などの計算
        self._calculate_survival_metrics()

        # 結果のフォーマット
        formatted_results = self._format_results()

        return formatted_results

    def _startup_simulation_step(self, previous_values: Dict[str, float], period: int) -> Dict[str, Any]:
        """
        スタートアップシミュレーションの1ステップを実行

        Args:
            previous_values: 前期の値
            period: 期間番号

        Returns:
            新しい値
        """
        # 前期の残高を取得
        previous_cash = previous_values['cash']

        # 前期で現金がなくなっていたら、0を返す
        if previous_cash <= 0:
            return {'cash': 0, 'is_survived': False}

        # 収益の計算（成長率と変動性を考慮）
        revenue_growth_factor = (1 + self.revenue_growth_rate) ** period
        revenue_variation = np.random.normal(1, self.revenue_volatility)
        revenue = self.monthly_revenue * revenue_growth_factor * revenue_variation

        # バーンレートの計算（変動性を考慮）
        burn_rate_variation = np.random.normal(1, self.burn_rate_volatility)
        burn_rate = self.monthly_burn_rate * burn_rate_variation

        # 現金残高の更新
        new_cash = previous_cash + revenue - burn_rate

        # 0未満にはならないように
        if new_cash < 0:
            new_cash = 0
            is_survived = False
        else:
            is_survived = True

        return {'cash': new_cash, 'is_survived': is_survived}

    def _calculate_survival_metrics(self) -> None:
        """
        生存率と関連指標を計算
        """
        if self.simulation_results is None:
            raise ValueError("シミュレーション結果がありません。analyze()を先に実行してください。")

        paths = self.simulation_results['paths']

        # 月ごとの生存確率を計算
        survival_probabilities = []
        for month in range(self.months_to_simulate + 1):
            # 月ごとの生存数をカウント
            survived_count = sum(
                1 for path in paths
                if month < len(path['cash']) and path['cash'][month] > 0
            )
            survival_prob = survived_count / len(paths)
            survival_probabilities.append(survival_prob)

        self.survival_probabilities = survival_probabilities

        # 現金切れの月を特定
        cash_out_months = []
        for path in paths:
            cash_values = path['cash']
            # 最初の0やNoneを探す
            cash_out_month = None
            for i, cash in enumerate(cash_values):
                if cash is None or cash <= 0:
                    cash_out_month = i
                    break

            # 現金切れがなかった場合はシミュレーション期間の最後
            if cash_out_month is None:
                cash_out_month = self.months_to_simulate

            cash_out_months.append(cash_out_month)

        self.cash_out_months = cash_out_months

    def _format_results(self) -> Dict[str, Any]:
        """
        分析結果をフォーマット

        Returns:
            整形された結果
        """
        if self.simulation_results is None or self.survival_probabilities is None:
            raise ValueError("分析が完了していません")

        # 基本的な統計情報
        cash_out_months_array = np.array(self.cash_out_months)
        cash_out_months_array = cash_out_months_array[~np.isnan(cash_out_months_array)]  # NaN除外

        results = {
            'survival_probability': {
                'by_month': self.survival_probabilities,
                'end_of_simulation': self.survival_probabilities[-1]
            },
            'runway_statistics': {
                'mean_months': float(np.mean(cash_out_months_array)),
                'median_months': float(np.median(cash_out_months_array)),
                'std_months': float(np.std(cash_out_months_array)),
                'min_months': float(np.min(cash_out_months_array)),
                'max_months': float(np.max(cash_out_months_array))
            },
            'confidence_intervals': self._calculate_runway_confidence_intervals(),
            'parameters': {
                'initial_cash': self.initial_cash,
                'monthly_burn_rate': self.monthly_burn_rate,
                'burn_rate_volatility': self.burn_rate_volatility,
                'monthly_revenue': self.monthly_revenue,
                'revenue_growth_rate': self.revenue_growth_rate,
                'revenue_volatility': self.revenue_volatility,
                'months_to_simulate': self.months_to_simulate,
                'simulation_runs': self.simulation_runs
            }
        }

        return results

    def _calculate_runway_confidence_intervals(self, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        ランウェイ（資金が尽きるまでの期間）の信頼区間を計算

        Args:
            confidence_level: 信頼水準

        Returns:
            信頼区間情報
        """
        if self.cash_out_months is None:
            raise ValueError("分析が完了していません")

        # NaNを除外
        valid_months = np.array([m for m in self.cash_out_months if m is not None])

        # 共通ユーティリティの信頼区間計算メソッドを使用
        return StatisticsUtility.calculate_confidence_intervals(valid_months, confidence_level)

    def calculate_depletion_probability(self, month=None):
        """
        指定月までに資金が枯渇する確率を計算

        Args:
            month: 計算対象の月。Noneの場合はシミュレーション期間の最後の月

        Returns:
            資金枯渇確率
        """
        if self.survival_probabilities is None:
            raise ValueError("シミュレーションが実行されていません")

        if month is None:
            month = self.months_to_simulate

        if month < 0 or month > self.months_to_simulate:
            raise ValueError(f"月は0から{self.months_to_simulate}の間で指定してください")

        # 生存確率の逆が枯渇確率
        return 1 - self.survival_probabilities[month]

    def plot_survival_curve(self, save_path=None):
        """
        生存曲線をプロット

        Args:
            save_path: 図を保存するパス（オプション）

        Returns:
            matplotlib図オブジェクト
        """
        if self.survival_probabilities is None:
            raise ValueError("シミュレーションが実行されていません")

        months = list(range(self.months_to_simulate + 1))

        fig = self._generate_basic_plot(
            months,
            self.survival_probabilities,
            "スタートアップ生存確率の推移",
            "月数",
            "生存確率",
            color='blue',
            figsize=(12, 6)
        )

        # 生存確率50%の水平線
        ax = fig.gca()
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)

        # 生存確率50%の月を特定
        survival_array = np.array(self.survival_probabilities)
        median_survival_month = np.argmin(np.abs(survival_array - 0.5))

        # その月に垂直線を表示
        ax.axvline(x=median_survival_month, color='r', linestyle='--', alpha=0.7)

        # テキスト注釈
        ax.text(
            median_survival_month + 0.5,
            0.52,
            f"中央生存期間: {median_survival_month}ヶ月",
            verticalalignment='bottom'
        )

        # 保存
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig

    def plot_cash_distribution(self, month, save_path=None):
        """
        特定の月の現金分布をプロット

        Args:
            month: 対象月
            save_path: 保存先パス（オプション）

        Returns:
            matplotlib図オブジェクト
        """
        if self.simulation_results is None:
            raise ValueError("シミュレーションが実行されていません")

        if month < 0 or month > self.months_to_simulate:
            raise ValueError(f"月は0から{self.months_to_simulate}の間で指定してください")

        # その月の現金残高を収集
        cash_values = []
        for path in self.simulation_results['paths']:
            if month < len(path['cash']) and path['cash'][month] is not None and path['cash'][month] > 0:
                cash_values.append(path['cash'][month])

        if not cash_values:
            raise ValueError(f"{month}ヶ月目の有効な現金残高データがありません")

        # ヒストグラムプロット
        fig = self._generate_histogram_plot(
            np.array(cash_values),
            f"{month}ヶ月目の現金残高分布",
            "現金残高（円）",
            bins=30,
            figsize=(12, 6)
        )

        # 保存
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig

    def plot_cash_flow_paths(self, num_paths=100, save_path=None):
        """
        現金フローのパスをプロット

        Args:
            num_paths: 表示するパスの数
            save_path: 保存先パス（オプション）

        Returns:
            matplotlib図オブジェクト
        """
        if self.simulation_results is None:
            raise ValueError("シミュレーションが実行されていません")

        # 表示するパス数を制限
        paths_to_show = min(num_paths, len(self.simulation_results['paths']))

        fig, ax = plt.subplots(figsize=(12, 6))

        months = list(range(self.months_to_simulate + 1))

        # シミュレーションパスをプロット
        for i in range(paths_to_show):
            path = self.simulation_results['paths'][i]
            cash_values = path['cash']
            # パスが短い場合は最後の要素で埋める
            if len(cash_values) < len(months):
                cash_values = list(cash_values) + [cash_values[-1]] * (len(months) - len(cash_values))
            ax.plot(months, cash_values, alpha=0.3, linewidth=0.5)

        # 平均パスをプロット
        mean_cash = []
        for month in months:
            month_values = [
                path['cash'][month] if month < len(path['cash']) and path['cash'][month] is not None else 0
                for path in self.simulation_results['paths']
            ]
            mean_cash.append(np.mean(month_values))

        ax.plot(months, mean_cash, color='red', linewidth=2, label='平均')

        # 0の水平線
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        ax.set_title("キャッシュフローのシミュレーションパス")
        ax.set_xlabel("月数")
        ax.set_ylabel("現金残高（円）")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        # 保存
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig

    def plot_cash_out_distribution(self, save_path=None):
        """
        資金枯渇月の分布をプロット

        Args:
            save_path: 保存先パス（オプション）

        Returns:
            matplotlib図オブジェクト
        """
        if self.cash_out_months is None:
            raise ValueError("シミュレーションが実行されていません")

        # NaNを除外
        valid_months = [m for m in self.cash_out_months if m is not None]

        fig = self._generate_histogram_plot(
            np.array(valid_months),
            "資金枯渇月の分布",
            "月数",
            bins=min(30, self.months_to_simulate),
            figsize=(12, 6)
        )

        # 保存
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig

    def generate_summary_report(self):
        """
        分析の要約レポートを生成

        Returns:
            要約レポート（文字列）
        """
        if self.simulation_results is None or self.survival_probabilities is None:
            raise ValueError("シミュレーションが実行されていません")

        # 基本統計量
        survival_end = self.survival_probabilities[-1]
        mean_runway = np.mean([m for m in self.cash_out_months if m is not None])
        median_runway = np.median([m for m in self.cash_out_months if m is not None])

        # 信頼区間
        ci = self._calculate_runway_confidence_intervals()
        ci_lower, ci_upper = ci['percentile']

        # レポート生成
        report = [
            "============ スタートアップ資金生存性分析 レポート ============",
            "",
            f"■ パラメータ設定",
            f"  - 初期資金: {self.initial_cash:,.0f}円",
            f"  - 月間バーンレート: {self.monthly_burn_rate:,.0f}円（変動性: {self.burn_rate_volatility:.1%}）",
            f"  - 月間収益: {self.monthly_revenue:,.0f}円（成長率: {self.revenue_growth_rate:.1%}、変動性: {self.revenue_volatility:.1%}）",
            f"  - シミュレーション期間: {self.months_to_simulate}ヶ月",
            f"  - シミュレーション回数: {self.simulation_runs}回",
            "",
            f"■ 生存性分析結果",
            f"  - {self.months_to_simulate}ヶ月後の生存確率: {survival_end:.1%}",
            f"  - 平均資金枯渇期間: {mean_runway:.1f}ヶ月",
            f"  - 中央値資金枯渇期間: {median_runway:.1f}ヶ月",
            f"  - 資金枯渇期間の95%信頼区間: {ci_lower:.1f}ヶ月 ～ {ci_upper:.1f}ヶ月",
            "",
            f"■ 推奨アクション",
            f"  - {'資金調達の検討' if mean_runway < 18 else '現在の戦略の維持'}"
        ]

        # 特定の閾値での生存確率
        report.append("")
        report.append("■ 重要な時点での生存確率")
        milestones = [6, 12, 18, 24]
        for month in milestones:
            if month <= self.months_to_simulate:
                prob = self.survival_probabilities[month]
                report.append(f"  - {month}ヶ月後: {prob:.1%}")

        return "\n".join(report)

    def _get_base_scenario_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        スタートアップ分析の基本シナリオパラメータを取得

        Args:
            data: 過去データ

        Returns:
            基本パラメータ
        """
        # データがない場合は現在のインスタンス変数を使用
        return {
            'initial_cash': self.initial_cash,
            'monthly_burn_rate': self.monthly_burn_rate,
            'burn_rate_volatility': self.burn_rate_volatility,
            'monthly_revenue': self.monthly_revenue,
            'revenue_growth_rate': self.revenue_growth_rate,
            'revenue_volatility': self.revenue_volatility
        }

    def scenario_analysis(self, scenarios, save_path=None):
        """
        複数のシナリオを分析し比較

        Args:
            scenarios: シナリオの辞書（各シナリオはパラメータ設定を含む辞書）
            save_path: 図を保存するパス（オプション）

        Returns:
            各シナリオの結果を含む辞書
        """
        # 結果を格納する辞書
        results = {}

        plt.figure(figsize=(12, 6))

        # 各シナリオについてシミュレーション実行
        for scenario_name, params in scenarios.items():
            print(f"シナリオ '{scenario_name}' を分析中...")

            # 元のパラメータを保存
            original_params = {}
            for param_name, param_value in params.items():
                original_params[param_name] = getattr(self, param_name)
                setattr(self, param_name, param_value)

            # このシナリオでシミュレーション実行
            self.analyze()

            # 結果を保存
            results[scenario_name] = {
                'survival_probabilities': self.survival_probabilities.copy(),
                'mean_runway': np.mean([m for m in self.cash_out_months if m is not None]),
                'parameters': params.copy()
            }

            # パラメータを元に戻す
            for param_name, param_value in original_params.items():
                setattr(self, param_name, param_value)

        # 比較グラフの作成
        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(111)

        # 各シナリオの生存曲線をプロット
        months = list(range(self.months_to_simulate + 1))

        for scenario_name, scenario_results in results.items():
            ax1.plot(months, scenario_results['survival_probabilities'], label=scenario_name)

        ax1.set_title("シナリオ別 生存確率の推移")
        ax1.set_xlabel("月数")
        ax1.set_ylabel("生存確率")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')

        # 保存
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.show()

        return results

    def export_data(self, filename):
        """
        分析結果をJSONファイルとしてエクスポート

        Args:
            filename: エクスポート先ファイル名

        Returns:
            エクスポートが成功したかどうか
        """
        if self.simulation_results is None:
            raise ValueError("シミュレーションが実行されていません")

        # エクスポートするデータの準備
        export_data = {
            'parameters': {
                'initial_cash': self.initial_cash,
                'monthly_burn_rate': self.monthly_burn_rate,
                'burn_rate_volatility': self.burn_rate_volatility,
                'monthly_revenue': self.monthly_revenue,
                'revenue_growth_rate': self.revenue_growth_rate,
                'revenue_volatility': self.revenue_volatility,
                'months_to_simulate': self.months_to_simulate,
                'simulation_runs': self.simulation_runs
            },
            'results': {
                'survival_probabilities': self.survival_probabilities,
                'cash_out_statistics': {
                    'mean': float(np.mean([m for m in self.cash_out_months if m is not None])),
                    'median': float(np.median([m for m in self.cash_out_months if m is not None])),
                    'std': float(np.std([m for m in self.cash_out_months if m is not None])),
                    'confidence_interval': self._calculate_runway_confidence_intervals()['percentile']
                }
            },
            'export_date': datetime.now().isoformat()
        }

        # JSONとしてエクスポート
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            print(f"データを {filename} にエクスポートしました")
            return True
        except Exception as e:
            print(f"エクスポート中にエラーが発生しました: {e}")
            return False

    def sensitivity_analysis(self, parameter_name, values, save_path=None):
        """
        特定のパラメータに対する感度分析を実行

        Args:
            parameter_name: 分析するパラメータ名
            values: テストする値のリスト
            save_path: 図を保存するパス（オプション）

        Returns:
            感度分析結果
        """
        if parameter_name not in [
            'initial_cash', 'monthly_burn_rate', 'burn_rate_volatility',
            'monthly_revenue', 'revenue_growth_rate', 'revenue_volatility'
        ]:
            raise ValueError(f"サポートされていないパラメータです: {parameter_name}")

        # 元の値を保存
        original_value = getattr(self, parameter_name)

        # 結果を格納する辞書
        results = {}

        # 各値についてシミュレーション実行
        for value in values:
            # パラメータを設定
            setattr(self, parameter_name, value)

            # シミュレーション実行
            self.analyze()

            # 結果を保存
            results[value] = {
                'survival_probability_end': self.survival_probabilities[-1],
                'mean_runway': np.mean([m for m in self.cash_out_months if m is not None])
            }

        # パラメータを元に戻す
        setattr(self, parameter_name, original_value)

        # グラフ作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        x_values = values
        survival_probs = [results[v]['survival_probability_end'] for v in values]
        mean_runways = [results[v]['mean_runway'] for v in values]

        # 生存確率グラフ
        ax1.plot(x_values, survival_probs, 'o-', color='blue')
        ax1.set_title(f"{parameter_name}の{self.months_to_simulate}ヶ月後の生存確率への影響")
        ax1.set_xlabel(parameter_name)
        ax1.set_ylabel("生存確率")
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 平均ランウェイグラフ
        ax2.plot(x_values, mean_runways, 'o-', color='green')
        ax2.set_title(f"{parameter_name}の平均ランウェイへの影響")
        ax2.set_xlabel(parameter_name)
        ax2.set_ylabel("平均ランウェイ（月）")
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        # 保存
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.show()

        return results


# テスト用コード
if __name__ == "__main__":
    # スタートアップ生存性分析のインスタンスを作成
    analysis = StartupSurvivabilityAnalyzer(
        initial_cash=100000000,  # 1億円
        monthly_burn_rate=10000000,  # 1000万円/月
        burn_rate_volatility=0.2,
        monthly_revenue=2000000,  # 200万円/月
        revenue_growth_rate=0.05,  # 5%/月の成長
        revenue_volatility=0.3,
        months_to_simulate=24,  # 2年
        simulation_runs=1000
    )

    # シミュレーション実行
    analysis.analyze()

    # 要約レポート生成
    print(analysis.generate_summary_report())

    # 生存曲線のプロット
    analysis.plot_survival_curve()

    # キャッシュフローパスのプロット
    analysis.plot_cash_flow_paths(num_paths=100)

    # 資金枯渇月の分布
    analysis.plot_cash_out_distribution()

    # 感度分析: 収益成長率の影響
    growth_rates = [0.03, 0.05, 0.07, 0.1, 0.15]  # 3%〜15%/月
    analysis.sensitivity_analysis('revenue_growth_rate', growth_rates)

    # シナリオ分析
    scenarios = {
        '標準シナリオ': {
            'monthly_burn_rate': 10000000,
            'revenue_growth_rate': 0.05
        },
        '楽観シナリオ': {
            'monthly_burn_rate': 9000000,
            'revenue_growth_rate': 0.1
        },
        '悲観シナリオ': {
            'monthly_burn_rate': 12000000,
            'revenue_growth_rate': 0.03
        }
    }
    analysis.scenario_analysis(scenarios)