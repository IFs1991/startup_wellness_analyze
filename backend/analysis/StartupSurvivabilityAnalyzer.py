import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import seaborn as sns
from tqdm import tqdm
import matplotlib.dates as mdates
from datetime import datetime
import scipy.stats as stats
import os
import json
import weakref
import gc
import tempfile
import logging
from contextlib import contextmanager
import pickle

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
                 firestore_client=None,  # Firestoreクライアント（オプション）
                 storage_mode="memory"):  # 結果の保存モード: "memory", "disk", "hybrid"
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
            storage_mode: 結果の保存モード ("memory", "disk", "hybrid")
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
        self.storage_mode = storage_mode

        # 結果を格納する変数
        self.simulation_results = None
        self.survival_probabilities = None
        self.monthly_runway_means = None
        self.cash_out_months = None

        # 一時ファイル管理
        self._temp_files = set()

        # ロガーの設定
        self.logger = logging.getLogger(self.__class__.__name__)

    def __del__(self):
        """デストラクタ - リソースを解放"""
        self.release_resources()

    def release_resources(self):
        """明示的にリソースを解放するメソッド"""
        # 一時ファイルの削除
        self._clean_temp_files()

        # 大きなデータ構造への参照を解除
        if hasattr(self, 'simulation_results') and self.simulation_results is not None:
            if self.storage_mode == "memory":
                self.simulation_results['paths'] = None
            self.simulation_results = None

        # 明示的なガベージコレクション呼び出し
        gc.collect()

    def _clean_temp_files(self):
        """一時ファイルをクリーンアップ"""
        for temp_file in self._temp_files.copy():
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    self.logger.debug(f"一時ファイルを削除: {temp_file}")
                self._temp_files.remove(temp_file)
            except Exception as e:
                self.logger.warning(f"一時ファイル削除中にエラー: {str(e)}")

    @contextmanager
    def _managed_simulation_data(self, simulation_data, key=None):
        """シミュレーションデータのコンテキスト管理

        Args:
            simulation_data: 管理するシミュレーションデータ
            key: データの識別キー（オプション）

        Yields:
            管理対象のシミュレーションデータ
        """
        try:
            # ストレージモードに応じたデータの準備
            if self.storage_mode == "disk" and key is not None:
                data = self._load_data_from_disk(key)
                yield data
            else:
                yield simulation_data
        finally:
            # リソースのクリーンアップ
            if self.storage_mode == "disk" and key is not None:
                # ディスクモードでは、必要に応じてデータを保存
                self._store_data_to_disk(simulation_data, key)

    async def analyze(self, data: Optional[pd.DataFrame] = None,
                     progress_callback=None,
                     **kwargs) -> Dict[str, Any]:
        """
        スタートアップの生存性分析を実行

        Args:
            data: 過去データ（オプション）
            progress_callback: 進捗コールバック関数
            **kwargs: 追加パラメータ
                - scenario: シナリオ名
                - custom_parameters: カスタムパラメータ
                - storage_mode: 保存モード（オプション）

        Returns:
            分析結果
        """
        # パラメータのオーバーライド
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        try:
            # シミュレーション実行前に一時データをクリア
            if hasattr(self, 'simulation_results') and self.simulation_results is not None:
                self.simulation_results = None
                gc.collect()

            # シミュレーション実行
            initial_values = {'cash': self.initial_cash}

            # シミュレーションの実行
            simulation_results = self.run_monte_carlo_simulation(
                initial_values,
                self.simulation_runs,
                self.months_to_simulate,
                self._startup_simulation_step,
                progress_callback=progress_callback
            )

            # 結果の保存（ストレージモードに応じた処理）
            if self.storage_mode == "memory":
                self.simulation_results = simulation_results
            elif self.storage_mode == "disk":
                # ディスクに保存し、メモリには最小限の情報だけ保持
                temp_file = self._store_simulation_results_to_disk(simulation_results)
                self.simulation_results = {
                    'storage': 'disk',
                    'file_path': temp_file,
                    'summary': self._extract_simulation_summary(simulation_results)
                }
            elif self.storage_mode == "hybrid":
                # 統計情報はメモリに、パス情報はディスクに保存
                self.simulation_results = simulation_results.copy()
                temp_file = self._store_paths_to_disk(simulation_results['paths'])
                self.simulation_results['paths'] = {
                    'storage': 'disk',
                    'file_path': temp_file,
                    'count': len(simulation_results['paths'])
                }
            else:
                raise ValueError(f"不明なストレージモード: {self.storage_mode}")

            # 生存確率などの計算
            self._calculate_survival_metrics()

            # 不要な中間結果をクリア
            self._clean_intermediate_results()

            # 結果のフォーマット
            formatted_results = self._format_results()

            return formatted_results

        except Exception as e:
            self.logger.error(f"分析実行中にエラーが発生: {str(e)}")
            # リソースをクリーンアップ
            self.release_resources()
            raise

    def _extract_simulation_summary(self, simulation_results):
        """シミュレーション結果から要約情報を抽出

        Args:
            simulation_results: シミュレーション結果

        Returns:
            要約情報
        """
        return {
            'run_count': len(simulation_results['paths']),
            'parameters': {
                'initial_cash': self.initial_cash,
                'months_to_simulate': self.months_to_simulate
            }
        }

    def _store_simulation_results_to_disk(self, simulation_results):
        """シミュレーション結果をディスクに保存

        Args:
            simulation_results: 保存するシミュレーション結果

        Returns:
            保存したファイルのパス
        """
        temp_file = tempfile.mktemp(suffix='.pickle')
        with open(temp_file, 'wb') as f:
            pickle.dump(simulation_results, f, protocol=4)

        self._temp_files.add(temp_file)
        return temp_file

    def _store_paths_to_disk(self, paths):
        """シミュレーションパスデータをディスクに保存

        Args:
            paths: 保存するパスデータ

        Returns:
            保存したファイルのパス
        """
        temp_file = tempfile.mktemp(suffix='.pickle')
        with open(temp_file, 'wb') as f:
            pickle.dump(paths, f, protocol=4)

        self._temp_files.add(temp_file)
        return temp_file

    def _load_data_from_disk(self, key):
        """ディスクからデータを読み込む

        Args:
            key: データの識別キーまたはファイルパス

        Returns:
            読み込んだデータ
        """
        if isinstance(key, dict) and 'file_path' in key:
            file_path = key['file_path']
        else:
            file_path = key

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def _clean_intermediate_results(self):
        """不要な中間結果を削除"""
        # すでに集計済みのデータは個別パスから削除可能
        if self.storage_mode == "memory" and self.simulation_results is not None:
            # 必要な統計値を計算したらパスデータの詳細は削減可能
            if hasattr(self, 'survival_probabilities') and self.survival_probabilities is not None:
                # 各パスから不要な詳細データを削除（マイルストーン情報のみ残す）
                if 'paths' in self.simulation_results:
                    for path in self.simulation_results['paths']:
                        # 最低限必要な情報以外を保持しない
                        if 'detailed_metrics' in path:
                            del path['detailed_metrics']
                        # 他の不要なフィールドの削除も追加可能

    def _get_simulation_paths(self):
        """現在のストレージモードに応じてシミュレーションパスを取得

        Returns:
            シミュレーションパスのリスト
        """
        if self.simulation_results is None:
            raise ValueError("シミュレーション結果がありません")

        if self.storage_mode == "memory" or 'paths' in self.simulation_results and isinstance(self.simulation_results['paths'], list):
            return self.simulation_results['paths']
        elif self.storage_mode == "disk" and 'file_path' in self.simulation_results:
            return self._load_data_from_disk(self.simulation_results)['paths']
        elif self.storage_mode == "hybrid" and 'paths' in self.simulation_results and isinstance(self.simulation_results['paths'], dict):
            return self._load_data_from_disk(self.simulation_results['paths'])
        else:
            raise ValueError("シミュレーションパスデータが見つかりません")

    def _calculate_survival_metrics(self) -> None:
        """
        生存率と関連指標を計算
        """
        if self.simulation_results is None:
            raise ValueError("シミュレーション結果がありません。analyze()を先に実行してください。")

        try:
            # パスデータの取得
            paths = self._get_simulation_paths()
            path_count = len(paths)

            # 月ごとの生存確率を計算
            survival_probabilities = []
            for month in range(self.months_to_simulate + 1):
                # 月ごとの生存数をカウント
                survived_count = sum(
                    1 for path in paths
                    if month < len(path['cash']) and path['cash'][month] > 0
                )
                survival_prob = survived_count / path_count
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

            # メモリ解放
            if self.storage_mode != "memory":
                gc.collect()
        except Exception as e:
            self.logger.error(f"生存メトリクス計算中にエラー: {str(e)}")
            raise

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

        try:
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
                plt.close(fig)  # 明示的にリソース解放
                return None  # 保存した場合は図オブジェクトを返さない

            return fig
        except Exception as e:
            self.logger.error(f"生存曲線プロット中にエラー: {str(e)}")
            raise
        finally:
            # 明示的なGCを促進
            gc.collect()

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

        try:
            # パスデータの取得
            paths = self._get_simulation_paths()

            # その月の現金残高を収集
            cash_values = []
            for path in paths:
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
                plt.close(fig)  # 明示的にリソース解放
                return None

            return fig
        except Exception as e:
            self.logger.error(f"現金分布プロット中にエラー: {str(e)}")
            raise
        finally:
            # メモリ使用量削減のためGC実行
            if self.storage_mode != "memory":
                gc.collect()

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

        try:
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
                plt.close(fig)  # 明示的にリソース解放
                return None

            return fig
        except Exception as e:
            self.logger.error(f"資金枯渇分布プロット中にエラー: {str(e)}")
            raise
        finally:
            # メモリ解放の促進
            gc.collect()

    def _generate_basic_plot(self, x_data, y_data, title, xlabel, ylabel, color='blue', figsize=(10, 6)):
        """基本的なプロットを生成する内部メソッド

        Args:
            x_data: X軸データ
            y_data: Y軸データ
            title: タイトル
            xlabel: X軸ラベル
            ylabel: Y軸ラベル
            color: 線の色
            figsize: 図のサイズ

        Returns:
            matplotlib図オブジェクト
        """
        try:
            with PlotUtility.plot_context():
                fig, ax = plt.subplots(figsize=figsize)
                ax.plot(x_data, y_data, '-', color=color)
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                return fig
        except Exception as e:
            self.logger.error(f"プロット生成中にエラー: {str(e)}")
            raise

    def _generate_histogram_plot(self, data, title, xlabel, bins=30, figsize=(10, 6)):
        """ヒストグラムプロットを生成する内部メソッド

        Args:
            data: プロットするデータ
            title: タイトル
            xlabel: X軸ラベル
            bins: ビンの数
            figsize: 図のサイズ

        Returns:
            matplotlib図オブジェクト
        """
        try:
            with PlotUtility.plot_context():
                fig, ax = plt.subplots(figsize=figsize)
                ax.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("頻度")
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                return fig
        except Exception as e:
            self.logger.error(f"ヒストグラム生成中にエラー: {str(e)}")
            raise

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

    def scenario_analysis(self, scenarios, save_path=None, show_plot=True):
        """
        複数のシナリオを分析し比較

        Args:
            scenarios: シナリオの辞書（各シナリオはパラメータ設定を含む辞書）
            save_path: 図を保存するパス（オプション）
            show_plot: プロットを表示するかどうか

        Returns:
            各シナリオの結果を含む辞書
        """
        # 結果を格納する辞書
        results = {}

        try:
            # 各シナリオについてシミュレーション実行
            for scenario_name, params in scenarios.items():
                self.logger.info(f"シナリオ '{scenario_name}' を分析中...")

                # 元のパラメータを保存
                original_params = {}
                for param_name, param_value in params.items():
                    if hasattr(self, param_name):
                        original_params[param_name] = getattr(self, param_name)
                        setattr(self, param_name, param_value)

                # このシナリオでシミュレーション実行
                self.analyze()

                # 結果を保存（最小限のデータのみ）
                results[scenario_name] = {
                    'survival_probabilities': self.survival_probabilities.copy(),
                    'mean_runway': float(np.mean([m for m in self.cash_out_months if m is not None])),
                    'parameters': params.copy()
                }

                # パラメータを元に戻す
                for param_name, param_value in original_params.items():
                    setattr(self, param_name, param_value)

                # シナリオごとに中間結果をクリア
                self._clean_intermediate_results()
                gc.collect()

            # 比較グラフの作成
            with PlotUtility.plot_context():
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

                # 表示
                if show_plot:
                    plt.show()
                else:
                    plt.close()

            return results
        except Exception as e:
            self.logger.error(f"シナリオ分析中にエラー: {str(e)}")
            raise
        finally:
            # リソース解放の促進
            gc.collect()

    def sensitivity_analysis(self, parameter_name, values, save_path=None, show_plot=True):
        """
        特定のパラメータに対する感度分析を実行

        Args:
            parameter_name: 分析するパラメータ名
            values: テストする値のリスト
            save_path: 図を保存するパス（オプション）
            show_plot: プロットを表示するかどうか

        Returns:
            感度分析結果
        """
        if parameter_name not in [
            'initial_cash', 'monthly_burn_rate', 'burn_rate_volatility',
            'monthly_revenue', 'revenue_growth_rate', 'revenue_volatility'
        ]:
            raise ValueError(f"サポートされていないパラメータです: {parameter_name}")

        try:
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
                    'mean_runway': float(np.mean([m for m in self.cash_out_months if m is not None]))
                }

                # 中間結果をクリア
                self._clean_intermediate_results()
                gc.collect()

            # パラメータを元に戻す
            setattr(self, parameter_name, original_value)

            # グラフ作成
            with PlotUtility.plot_context():
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

                # 表示
                if show_plot:
                    plt.show()
                else:
                    plt.close()

            return results
        except Exception as e:
            self.logger.error(f"感度分析中にエラー: {str(e)}")
            raise
        finally:
            # リソース解放
            gc.collect()

    def estimate_memory_usage(self, simulation_runs=None, months_to_simulate=None, storage_mode=None):
        """
        シミュレーション実行時のメモリ使用量を推定

        Args:
            simulation_runs: シミュレーション回数（Noneの場合は現在の設定を使用）
            months_to_simulate: シミュレーション期間（Noneの場合は現在の設定を使用）
            storage_mode: ストレージモード（Noneの場合は現在の設定を使用）

        Returns:
            推定メモリ使用量（バイト単位）と人間可読な形式
        """
        # 現在の設定を使用
        if simulation_runs is None:
            simulation_runs = self.simulation_runs
        if months_to_simulate is None:
            months_to_simulate = self.months_to_simulate
        if storage_mode is None:
            storage_mode = self.storage_mode

        # 1つのパスあたりのメモリ使用量を推定
        # 各月のcash値（浮動小数点8バイト）+ is_survived（ブール値1バイト）+ その他のオーバーヘッド
        bytes_per_month = 8 + 1 + 3  # 浮動小数点 + ブール値 + オーバーヘッド
        bytes_per_path = bytes_per_month * (months_to_simulate + 1) + 50  # +50 はパスオブジェクト自体のオーバーヘッド

        # 合計メモリ使用量
        if storage_mode == "memory":
            # メモリ内にすべてのパスを保持
            total_bytes = bytes_per_path * simulation_runs
            # 追加の統計データとオーバーヘッド
            total_bytes += (months_to_simulate + 1) * 8 * 10  # 各種統計データ用
            total_bytes += 1000  # その他のオーバーヘッド
        elif storage_mode == "disk":
            # ディスクに保存するので、メモリ使用量は最小限
            total_bytes = (months_to_simulate + 1) * 8 * 10  # 統計データのみ
            total_bytes += 1000  # その他のオーバーヘッド
        else:  # "hybrid"
            # 統計データはメモリに、パスデータはディスクに
            total_bytes = (months_to_simulate + 1) * 8 * 10  # 統計データ
            # 一部のパスデータがキャッシュされる可能性を考慮
            total_bytes += bytes_per_path * min(100, simulation_runs)  # 最大100パスをキャッシュと仮定
            total_bytes += 2000  # その他のオーバーヘッド

        # 人間可読な形式に変換
        readable_size = self._format_bytes(total_bytes)

        return {
            'bytes': total_bytes,
            'human_readable': readable_size,
            'parameters': {
                'simulation_runs': simulation_runs,
                'months_to_simulate': months_to_simulate,
                'storage_mode': storage_mode
            }
        }

    def _format_bytes(self, size_bytes):
        """バイト数を人間可読な形式に変換

        Args:
            size_bytes: バイト数

        Returns:
            人間可読な形式の文字列
        """
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


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
        simulation_runs=1000,
        storage_mode="hybrid"  # ハイブリッドストレージモードを使用
    )

    # シミュレーション実行
    analysis.analyze()

    # メモリ使用状況確認
    print(f"現在のメモリ使用モード: {analysis.storage_mode}")

    # 要約レポート生成
    print(analysis.generate_summary_report())

    # 生存曲線のプロット
    analysis.plot_survival_curve()

    # キャッシュフローパスのプロット（明示的にリソース解放）
    fig = analysis.plot_cash_flow_paths(num_paths=100)
    plt.close(fig)

    # リソース解放
    analysis.release_resources()