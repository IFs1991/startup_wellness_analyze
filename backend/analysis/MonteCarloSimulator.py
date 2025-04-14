from typing import Dict, Any, List, Optional, Union, Tuple, Iterator, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import logging
import gc
from functools import lru_cache
import h5py
from pathlib import Path
import tempfile
import os
import weakref
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

    def __init__(self, firestore_client=None, use_disk_storage: bool = False,
                 max_memory_simulations: int = 1000, temp_file_dir: Optional[str] = None):
        """
        初期化

        Args:
            firestore_client: Firestoreクライアントのインスタンス（オプション）
            use_disk_storage: シミュレーション結果をディスクに保存するかどうか
            max_memory_simulations: メモリに保持する最大シミュレーション数
            temp_file_dir: 一時ファイルの保存ディレクトリ
        """
        super().__init__(analysis_type="monte_carlo", firestore_client=firestore_client)

        # ディスクへの保存オプション設定
        self.use_disk_storage = use_disk_storage
        self.max_memory_simulations = max_memory_simulations

        # 一時ファイルの保存先
        self._temp_dir = temp_file_dir
        self._temp_file = None
        self._temp_file_path = None

        # シミュレーション結果の保存先 - 循環参照を避けるためにweakrefを使用
        self._simulation_storage = None
        # ストレージクリーンアップ用のフィナライザー
        self._storage_finalizer = weakref.finalize(self, self._cleanup_storage)

    def __del__(self):
        """デストラクタ：一時ファイルの削除やリソース解放を行う"""
        self._cleanup_storage()
        super().__del__()

    def _initialize_storage(self):
        """シミュレーション結果の保存先を初期化する"""
        try:
            if self.use_disk_storage:
                self._initialize_disk_storage()
            else:
                # メモリストレージの初期化
                self._simulation_storage = {'memory': {'paths': []}, 'paths': []}
        except Exception as e:
            error_msg = f"ストレージの初期化に失敗しました: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    def _initialize_disk_storage(self):
        """ディスクストレージを初期化する"""
        if self._temp_file is None:
            try:
                # 一時ファイルの作成
                if self._temp_dir:
                    os.makedirs(self._temp_dir, exist_ok=True)
                    fd, self._temp_file_path = tempfile.mkstemp(suffix='.h5', dir=self._temp_dir)
                else:
                    fd, self._temp_file_path = tempfile.mkstemp(suffix='.h5')
                os.close(fd)

                # HDF5ファイルを開く
                self._temp_file = h5py.File(self._temp_file_path, 'w')
                self._simulation_storage = {'disk': self._temp_file, 'paths': []}
                self.logger.info(f"シミュレーション結果をディスクに保存します: {self._temp_file_path}")
            except Exception as e:
                self.logger.error(f"ディスクストレージの初期化に失敗しました: {str(e)}")
                if self._temp_file_path and os.path.exists(self._temp_file_path):
                    try:
                        os.unlink(self._temp_file_path)
                    except:
                        pass
                raise

    def _cleanup_storage(self):
        """ストレージのクリーンアップを行う"""
        if self._temp_file is not None:
            try:
                self._temp_file.close()
                if os.path.exists(self._temp_file_path):
                    os.unlink(self._temp_file_path)
                self.logger.info(f"一時ファイルを削除しました: {self._temp_file_path}")
            except Exception as e:
                self.logger.error(f"一時ファイルの削除に失敗しました: {str(e)}")

        # メモリ解放
        self._simulation_storage = None
        gc.collect()

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
                - use_disk_storage: シミュレーション結果をディスクに保存するかどうか（上書き）
                - compute_partial_results: 部分的な結果を計算するかどうか
                - partial_results_interval: 部分結果を計算する間隔

        Returns:
            分析結果を含む辞書
        """
        try:
            self._validate_data(data)

            # パラメータの取得
            scenario = kwargs.get('scenario', '標準')
            custom_parameters = kwargs.get('custom_parameters', {})
            num_simulations = kwargs.get('num_simulations', 1000)
            forecast_periods = kwargs.get('forecast_periods', 24)
            confidence_level = kwargs.get('confidence_level', 0.95)

            # ストレージオプション
            if 'use_disk_storage' in kwargs:
                self.use_disk_storage = kwargs['use_disk_storage']

            # 部分結果の計算オプション
            compute_partial_results = kwargs.get('compute_partial_results', False)
            partial_results_interval = kwargs.get('partial_results_interval', num_simulations // 10)

            # シナリオパラメータの取得
            scenario_params = self._get_scenario_parameters(scenario, custom_parameters, data)

            # ストレージの初期化
            self._initialize_storage()

            # シミュレーション実行
            simulation_results = self._run_simulation(
                data,
                num_simulations,
                forecast_periods,
                scenario_params,
                compute_partial_results,
                partial_results_interval
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
                'simulation_summary': {
                    'num_simulations': num_simulations,
                    'forecast_periods': forecast_periods,
                    'statistics': simulation_results['statistics']
                },
                'roi_distribution': roi_distribution,
                'confidence_intervals': confidence_intervals,
                'plots': plots
            }

            # 大きなシミュレーション結果データは必要に応じて別途アクセスできるようにする
            self.register_temp_data('full_simulation_results', simulation_results)

            # ストレージのクリーンアップ
            if not kwargs.get('keep_simulation_data', False):
                self._cleanup_storage()

            return results
        except Exception as e:
            self.logger.error(f"モンテカルロシミュレーション分析中にエラーが発生しました: {str(e)}")
            # クリーンアップを試みる
            try:
                self._cleanup_storage()
            except:
                pass
            raise AnalysisError(f"モンテカルロシミュレーション分析に失敗しました: {str(e)}") from e

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
        simulation_func: Callable,
        batch_size: int = 100,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """
        汎用的なモンテカルロシミュレーションを実行

        Args:
            initial_values: 初期値
            num_simulations: シミュレーション回数
            num_periods: シミュレーション期間
            simulation_func: 各期間の値を計算する関数
                signature: func(previous_values, period_index) -> new_values
            batch_size: バッチサイズ（メモリ効率を上げるため）
            progress_callback: 進捗報告コールバック関数 (progress_percentage, status_message) -> None
            cancel_check: キャンセル確認用コールバック関数。Trueを返すと処理を中断

        Returns:
            シミュレーション結果
        """
        try:
            # ストレージが初期化されていなければ初期化
            if self._simulation_storage is None:
                self._initialize_storage()

            # 結果を格納する配列
            results = {
                'paths': [],
                'statistics': {},
                'canceled': False
            }

            # 最適なバッチサイズの決定（メモリ使用量と処理速度のバランスを取る）
            adjusted_batch_size = self._optimize_batch_size(batch_size, num_periods)

            # バッチ処理
            total_batches = (num_simulations + adjusted_batch_size - 1) // adjusted_batch_size
            progress_step = max(1, total_batches // 100)  # 進捗更新の頻度制御

            self.logger.info(f"シミュレーション開始: {num_simulations}回 x {num_periods}期間, バッチサイズ: {adjusted_batch_size}")

            start_time = datetime.now()

            for batch_idx in range(total_batches):
                # キャンセルチェック
                if cancel_check and cancel_check():
                    self.logger.info("シミュレーションがキャンセルされました")
                    results['canceled'] = True
                    break

                start_idx = batch_idx * adjusted_batch_size
                end_idx = min(start_idx + adjusted_batch_size, num_simulations)
                batch_size_actual = end_idx - start_idx

                # 進捗報告
                if progress_callback and (batch_idx % progress_step == 0 or batch_idx == total_batches - 1):
                    progress_percentage = (batch_idx + 1) / total_batches * 100
                    elapsed = (datetime.now() - start_time).total_seconds()
                    estimated_total = elapsed / (batch_idx + 1) * total_batches if batch_idx > 0 else 0
                    remaining = max(0, estimated_total - elapsed)

                    status_message = (
                        f"処理中: {batch_idx + 1}/{total_batches}バッチ "
                        f"({end_idx}/{num_simulations}シミュレーション) "
                        f"残り約{int(remaining)}秒"
                    )
                    progress_callback(progress_percentage, status_message)

                # バッチ内のシミュレーション実行（並列処理の可能性も考慮）
                batch_paths = self._execute_simulation_batch(
                    initial_values,
                    num_periods,
                    simulation_func,
                    batch_size_actual
                )

                # ストレージに保存
                self._store_simulation_batch(batch_paths, start_idx)

                # メモリ解放
                del batch_paths
                gc.collect()

            # キャンセルされなかった場合のみ統計情報を計算
            if not results.get('canceled', False):
                # インデックス情報を保持
                results['total_simulations'] = num_simulations
                results['storage_info'] = {
                    'use_disk': self.use_disk_storage,
                    'path_count': len(self._simulation_storage['paths']),
                    'temp_file': self._temp_file_path if self.use_disk_storage else None
                }

                # 統計情報はイテレータを使って計算
                results['statistics'] = self._calculate_statistics_with_iterator(num_simulations, num_periods)

                # 所要時間の記録
                total_time = (datetime.now() - start_time).total_seconds()
                results['performance'] = {
                    'total_seconds': total_time,
                    'simulations_per_second': num_simulations / total_time if total_time > 0 else 0
                }
                self.logger.info(f"シミュレーション完了: 所要時間 {total_time:.2f}秒 "
                               f"({results['performance']['simulations_per_second']:.2f} シミュレーション/秒)")

            return results
        except Exception as e:
            error_msg = f"モンテカルロシミュレーション実行中にエラーが発生しました: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    def _optimize_batch_size(self, requested_batch_size: int, num_periods: int) -> int:
        """
        最適なバッチサイズを計算する

        Args:
            requested_batch_size: 要求されたバッチサイズ
            num_periods: シミュレーション期間

        Returns:
            最適化されたバッチサイズ
        """
        # シミュレーション期間が長い場合はバッチサイズを小さくする
        if num_periods > 100:
            return min(requested_batch_size, 50)
        elif num_periods > 50:
            return min(requested_batch_size, 100)

        # デフォルトのまま
        return requested_batch_size

    def _execute_simulation_batch(
        self,
        initial_values: Dict[str, float],
        num_periods: int,
        simulation_func: Callable,
        batch_size: int
    ) -> List[Dict[str, List[float]]]:
        """
        シミュレーションのバッチを実行する

        Args:
            initial_values: 初期値
            num_periods: シミュレーション期間
            simulation_func: シミュレーション関数
            batch_size: 実行するシミュレーション数

        Returns:
            シミュレーション結果のリスト
        """
        batch_paths = []

        # 並列処理を活用する余地あり（現状はシーケンシャル実行）
        for i in range(batch_size):
            # 1つのパスをシミュレーション
            path = self._simulate_single_path(initial_values, num_periods, simulation_func)
            batch_paths.append(path)

        return batch_paths

    def _store_simulation_batch(self, batch_paths: List[Dict[str, List[float]]], start_idx: int) -> None:
        """シミュレーションバッチをストレージに保存する"""
        try:
            if self.use_disk_storage:
                self._store_batch_to_disk(batch_paths, start_idx)
            else:
                self._store_batch_to_memory(batch_paths, start_idx)
        except Exception as e:
            error_msg = f"シミュレーションバッチの保存に失敗しました: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    def _store_batch_to_disk(self, batch_paths: List[Dict[str, List[float]]], start_idx: int) -> None:
        """バッチをディスクに保存する"""
        # ディスクストレージに保存
        batch_group = self._temp_file.create_group(f"batch_{start_idx}")

        for i, path in enumerate(batch_paths):
            path_group = batch_group.create_group(f"path_{start_idx + i}")
            for key, values in path.items():
                path_group.create_dataset(key, data=np.array(values, dtype=np.float32))

            # インデックス情報を更新
            self._simulation_storage['paths'].append((start_idx + i, f"batch_{start_idx}/path_{start_idx + i}"))

    def _store_batch_to_memory(self, batch_paths: List[Dict[str, List[float]]], start_idx: int) -> None:
        """バッチをメモリに保存する"""
        # メモリストレージに保存
        for i, path in enumerate(batch_paths):
            # 最大メモリシミュレーション数を超えた場合、古いものから削除
            if len(self._simulation_storage['memory']['paths']) >= self.max_memory_simulations:
                idx_to_remove = start_idx + i - self.max_memory_simulations
                if idx_to_remove >= 0 and idx_to_remove in self._simulation_storage['memory']:
                    del self._simulation_storage['memory'][idx_to_remove]

            # パスを保存
            self._simulation_storage['memory'][start_idx + i] = path
            self._simulation_storage['memory']['paths'].append(start_idx + i)
            self._simulation_storage['paths'].append((start_idx + i, 'memory'))

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

    def get_simulation_iterator(self, start_idx: int = 0, end_idx: Optional[int] = None) -> Iterator[Dict[str, List[float]]]:
        """
        シミュレーション結果のイテレータを取得する

        Args:
            start_idx: 開始インデックス
            end_idx: 終了インデックス（指定しない場合は全て）

        Yields:
            シミュレーションパス
        """
        if self._simulation_storage is None:
            raise ValueError("シミュレーションが実行されていません")

        paths = self._simulation_storage['paths']
        end_idx = end_idx or len(paths)

        for i in range(start_idx, min(end_idx, len(paths))):
            try:
                path_idx, path_loc = paths[i]

                if self.use_disk_storage:
                    yield self._get_path_from_disk(path_loc)
                else:
                    yield self._get_path_from_memory(path_idx)
            except Exception as e:
                self.logger.warning(f"パス {i} の取得中にエラーが発生しました: {str(e)}")
                yield {}

    def _get_path_from_disk(self, path_loc: str) -> Dict[str, List[float]]:
        """ディスクからパスを取得する"""
        path_data = {}
        path_group = self._temp_file[path_loc]

        for key in path_group:
            path_data[key] = list(path_group[key][()])

        return path_data

    def _get_path_from_memory(self, path_idx: int) -> Dict[str, List[float]]:
        """メモリからパスを取得する"""
        if path_idx in self._simulation_storage['memory']:
            return self._simulation_storage['memory'][path_idx]
        else:
            self.logger.warning(f"パス {path_idx} はメモリに存在しません")
            return {}

    def _calculate_statistics_with_iterator(self, num_simulations: int, num_periods: int) -> Dict[str, Any]:
        """
        イテレータを使って統計情報を計算する（メモリ効率的）

        Args:
            num_simulations: シミュレーション回数
            num_periods: シミュレーション期間

        Returns:
            統計情報
        """
        try:
            # 変数と統計情報の初期化
            statistics_data = self._initialize_statistics_data(num_periods)
            if not statistics_data:
                return {}

            # データを集計
            self._accumulate_statistics_data(statistics_data)

            # 最終的な統計量を計算
            return self._finalize_statistics_calculations(statistics_data)
        except Exception as e:
            error_msg = f"統計情報の計算中にエラーが発生しました: {str(e)}"
            self.logger.error(error_msg)
            return {}

    def _initialize_statistics_data(self, num_periods: int) -> Dict[str, List[Dict[str, Any]]]:
        """統計データの構造を初期化する"""
        statistics = {}

        # シミュレーションの変数を特定するため最初のパスを取得
        for path in self.get_simulation_iterator(0, 1):
            if path:
                # 変数ごとに統計情報の初期化
                for key in path:
                    statistics[key] = [{'sum': 0, 'sum_sq': 0, 'min': float('inf'), 'max': float('-inf'), 'count': 0}
                                     for _ in range(num_periods)]
                return statistics
        return {}

    def _accumulate_statistics_data(self, statistics_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """イテレータを使用してデータを集計する"""
        # イテレータを使って統計量を計算
        for path in self.get_simulation_iterator():
            for key, values in path.items():
                if key in statistics_data:
                    for period, value in enumerate(values):
                        if value is not None and period < len(statistics_data[key]):
                            stats = statistics_data[key][period]
                            stats['sum'] += value
                            stats['sum_sq'] += value ** 2
                            stats['min'] = min(stats['min'], value)
                            stats['max'] = max(stats['max'], value)
                            stats['count'] += 1

    def _finalize_statistics_calculations(self, statistics_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """集計されたデータから最終的な統計量を計算する"""
        results = {}
        for key, periods_stats in statistics_data.items():
            results[key] = []
            for period_stats in periods_stats:
                count = period_stats['count']
                if count > 0:
                    mean = period_stats['sum'] / count
                    variance = (period_stats['sum_sq'] / count) - (mean ** 2)
                    std = np.sqrt(max(0, variance))  # 数値誤差対策

                    results[key].append({
                        'mean': mean,
                        'median': None,  # メディアンは全データが必要なので省略
                        'std': std,
                        'min': period_stats['min'],
                        'max': period_stats['max'],
                        'q25': None,  # 同様に省略
                        'q75': None   # 同様に省略
                    })
                else:
                    results[key].append(self._get_empty_stats())
        return results

    def _get_empty_stats(self) -> Dict[str, Any]:
        """空の統計情報を返す"""
        return {
            'mean': None,
            'median': None,
            'std': None,
            'min': None,
            'max': None,
            'q25': None,
            'q75': None
        }

    @lru_cache(maxsize=8)
    def _calculate_roi_distribution(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        ROI分布の統計量を計算する（キャッシュ機能付き）

        Args:
            simulation_results (Dict[str, Any]): シミュレーション結果

        Returns:
            Dict[str, Any]: ROI分布の統計量
        """
        try:
            # キャッシュのために、ROIデータの型を調整
            roi_simulations = np.asarray(simulation_results['roi_simulations'], dtype=np.float32)

            # メモリ効率のため、統計計算に必要な値だけを抽出
            roi_count = len(roi_simulations)
            roi_mean = np.mean(roi_simulations)
            roi_positive_prob = np.sum(roi_simulations > 0) / roi_count

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
                'positive_roi_probability': float(roi_positive_prob),
                'skewness': float(StatisticsUtility.calculate_skewness(roi_simulations)),
                'kurtosis': float(StatisticsUtility.calculate_kurtosis(roi_simulations)),
                'mean': float(roi_mean),
                'count': roi_count
            }

            return distribution
        except Exception as e:
            error_msg = f"ROI分布の計算中にエラーが発生しました: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg}

    def _generate_plots(
        self,
        simulation_results: Dict[str, Any],
        confidence_intervals: Dict[str, Any],
        scenario: str
    ) -> Dict[str, str]:
        """
        シミュレーション結果のプロットを生成する

        Args:
            simulation_results: シミュレーション結果
            confidence_intervals: 信頼区間
            scenario: シナリオ名

        Returns:
            Base64エンコードされたプロット画像
        """
        plots = {}

        try:
            # ROI分布のヒストグラムプロット
            plots['roi_histogram'] = self._generate_roi_histogram(
                simulation_results['roi_simulations'],
                confidence_intervals
            )

            # 売上予測のプロット - 必要に応じて生成
            if 'revenue_forecasts' in simulation_results:
                plots['revenue_forecast'] = self._generate_forecast_plot(
                    simulation_results['revenue_forecasts'],
                    '売上変化予測',
                    scenario
                )

            # バリュエーション予測のプロット - 必要に応じて生成
            if 'valuation_forecasts' in simulation_results:
                plots['valuation_forecast'] = self._generate_forecast_plot(
                    simulation_results['valuation_forecasts'],
                    'バリュエーション変化予測',
                    scenario
                )

            return plots
        except Exception as e:
            error_msg = f"プロット生成中にエラーが発生しました: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg}

    def _generate_roi_histogram(self, roi_data: List[float], confidence_intervals: Dict[str, Any]) -> str:
        """ROIヒストグラムを生成する"""
        # PlotUtilityを使って一括でプロット生成
        plot_data = {
            'data': roi_data,
            'title': 'ROI分布',
            'x_label': 'ROI (投資収益率)',
            'confidence_intervals': confidence_intervals
        }
        return PlotUtility.create_and_save_plot('histogram', plot_data)

    def _generate_forecast_plot(
        self,
        forecast_data: Dict[str, List[float]],
        title: str,
        scenario: str
    ) -> str:
        """
        予測データのプロットを生成する

        Args:
            forecast_data: 予測データ
            title: プロットタイトル
            scenario: シナリオ名

        Returns:
            Base64エンコードされた画像
        """
        # PlotUtility のコンテキストマネージャーを活用
        with PlotUtility.plot_context(figsize=(10, 6)) as fig:
            ax = fig.gca()

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

            # レイアウト調整
            plt.tight_layout()

            # 画像をBase64エンコード
            return PlotUtility.save_plot_to_base64(fig)

    def save_simulation_to_file(self, file_path: str) -> bool:
        """
        シミュレーション結果を外部ファイルに保存する

        Args:
            file_path: 保存先ファイルパス

        Returns:
            保存成功の場合True
        """
        if self._simulation_storage is None:
            self.logger.error("保存するシミュレーション結果がありません")
            return False

        try:
            with h5py.File(file_path, 'w') as f:
                # メタデータの保存
                metadata = f.create_group('metadata')
                metadata.attrs['created_at'] = str(datetime.now())
                metadata.attrs['path_count'] = len(self._simulation_storage['paths'])

                # シミュレーションパスの保存
                for i, (path_idx, path_loc) in enumerate(self._simulation_storage['paths']):
                    path_data = None

                    if self.use_disk_storage:
                        # 元のHDF5ファイルからデータを読み込む
                        path_data = self._get_path_from_disk(path_loc)
                    else:
                        # メモリからデータを取得
                        path_data = self._get_path_from_memory(path_idx)

                    if path_data:
                        # 新しいファイルにデータを書き込む
                        path_group = f.create_group(f"path_{i}")
                        for key, values in path_data.items():
                            path_group.create_dataset(key, data=np.array(values, dtype=np.float32))

            self.logger.info(f"シミュレーション結果を{file_path}に保存しました")
            return True
        except Exception as e:
            self.logger.error(f"シミュレーション結果の保存中にエラーが発生しました: {str(e)}")
            return False

    def load_simulation_from_file(self, file_path: str) -> bool:
        """
        外部ファイルからシミュレーション結果を読み込む

        Args:
            file_path: 読み込むファイルパス

        Returns:
            読み込み成功の場合True
        """
        if not os.path.exists(file_path):
            self.logger.error(f"ファイル{file_path}が存在しません")
            return False

        try:
            # 現在のストレージをクリーンアップ
            self._cleanup_storage()

            # ディスクストレージを使用するように設定
            self.use_disk_storage = True
            self._initialize_disk_storage()

            # ファイルからデータをコピー
            with h5py.File(file_path, 'r') as src_file:
                # メタデータの確認
                if 'metadata' in src_file:
                    self.logger.info(f"シミュレーション作成日時: {src_file['metadata'].attrs.get('created_at', '不明')}")

                # パスのコピー
                path_count = 0
                for path_name in src_file:
                    if path_name != 'metadata':
                        # パスグループをコピー
                        path_group = self._temp_file.create_group(path_name)
                        for key in src_file[path_name]:
                            path_group.create_dataset(key, data=src_file[path_name][key][()])

                        # インデックス情報を更新
                        path_idx = int(path_name.split('_')[1])
                        self._simulation_storage['paths'].append((path_idx, path_name))
                        path_count += 1

            self.logger.info(f"{path_count}個のシミュレーションパスを{file_path}から読み込みました")
            return True
        except Exception as e:
            self.logger.error(f"シミュレーション結果の読み込み中にエラーが発生しました: {str(e)}")
            # エラー時のクリーンアップ
            self._cleanup_storage()
            return False

    def export_results_to_csv(self, directory: str, prefix: str = "sim_") -> Dict[str, str]:
        """
        シミュレーション結果をCSVファイルにエクスポートする

        Args:
            directory: 出力ディレクトリ
            prefix: 出力ファイル名の接頭辞

        Returns:
            出力ファイルのパスを含む辞書
        """
        if self._simulation_storage is None:
            raise ValueError("エクスポートするシミュレーション結果がありません")

        try:
            os.makedirs(directory, exist_ok=True)
            result_files = {}

            # 統計情報のエクスポート
            stats_data = self._calculate_statistics_with_iterator(
                len(self._simulation_storage['paths']),
                self._get_simulation_periods()
            )

            for key, stats_list in stats_data.items():
                # 統計情報をDataFrameに変換
                stats_df = pd.DataFrame(stats_list)
                stats_df['period'] = range(1, len(stats_list) + 1)

                # CSVに保存
                file_path = os.path.join(directory, f"{prefix}{key}_stats.csv")
                stats_df.to_csv(file_path, index=False)
                result_files[f"{key}_stats"] = file_path

            # サンプルパスのエクスポート（最大100パス）
            max_paths = min(100, len(self._simulation_storage['paths']))
            for var_name in self._get_simulation_variables():
                # 各変数ごとにデータを集める
                sample_data = []
                periods = []

                for i, path in enumerate(self.get_simulation_iterator(0, max_paths)):
                    if var_name in path:
                        sample_data.append(path[var_name])
                        if not periods and len(path[var_name]) > 0:
                            periods = range(1, len(path[var_name]) + 1)

                # DataFrameに変換
                sample_df = pd.DataFrame(sample_data).T
                sample_df.columns = [f"path_{i}" for i in range(len(sample_data))]
                sample_df['period'] = periods

                # CSVに保存
                file_path = os.path.join(directory, f"{prefix}{var_name}_samples.csv")
                sample_df.to_csv(file_path, index=False)
                result_files[f"{var_name}_samples"] = file_path

            return result_files
        except Exception as e:
            error_msg = f"シミュレーション結果のエクスポート中にエラーが発生しました: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    def _get_simulation_periods(self) -> int:
        """シミュレーション期間を取得する"""
        for path in self.get_simulation_iterator(0, 1):
            for key, values in path.items():
                return len(values)
        return 0

    def _get_simulation_variables(self) -> List[str]:
        """シミュレーションで使用されている変数名のリストを取得する"""
        variables = set()
        for path in self.get_simulation_iterator(0, 1):
            for key in path.keys():
                variables.add(key)
        return list(variables)

    def validate_simulation_parameters(
        self,
        initial_values: Dict[str, float],
        num_simulations: int,
        num_periods: int,
        simulation_func: Callable
    ) -> Tuple[bool, str]:
        """
        シミュレーションパラメータを検証する

        Args:
            initial_values: 初期値
            num_simulations: シミュレーション回数
            num_periods: シミュレーション期間
            simulation_func: シミュレーション関数

        Returns:
            (検証結果, エラーメッセージ)のタプル
        """
        # 初期値の検証
        if not initial_values:
            return False, "初期値が空です"

        for key, value in initial_values.items():
            if not isinstance(value, (int, float)):
                return False, f"初期値 '{key}' が数値ではありません: {value}"

        # シミュレーション回数の検証
        if num_simulations <= 0:
            return False, f"シミュレーション回数は正の整数である必要があります: {num_simulations}"

        if num_simulations > 100000:
            return False, f"シミュレーション回数が大きすぎます({num_simulations}): 100,000以下を推奨します"

        # 期間の検証
        if num_periods <= 0:
            return False, f"シミュレーション期間は正の整数である必要があります: {num_periods}"

        if num_periods > 1000:
            return False, f"シミュレーション期間が長すぎます({num_periods}): 1,000以下を推奨します"

        # シミュレーション関数のテスト
        try:
            test_result = simulation_func(initial_values, 1)
            if not isinstance(test_result, dict):
                return False, "シミュレーション関数は辞書型を返す必要があります"

            for key, value in test_result.items():
                if not isinstance(value, (int, float)):
                    return False, f"シミュレーション関数の戻り値に数値でない値が含まれています: {key}={value}"
        except Exception as e:
            return False, f"シミュレーション関数のテスト実行中にエラーが発生しました: {str(e)}"

        # すべての検証をパス
        return True, ""

    def estimate_memory_usage(
        self,
        initial_values: Dict[str, float],
        num_simulations: int,
        num_periods: int
    ) -> Dict[str, Any]:
        """
        シミュレーション実行時のメモリ使用量を概算する

        Args:
            initial_values: 初期値
            num_simulations: シミュレーション回数
            num_periods: シミュレーション期間

        Returns:
            メモリ使用量の概算情報
        """
        # 1パスあたりの変数数
        num_variables = len(initial_values)

        # 1パスあたりの期間数（各期間の各変数に対して浮動小数点数を保存）
        values_per_path = num_variables * num_periods

        # 1つの浮動小数点数のバイト数（float32）
        bytes_per_value = 4

        # 合計メモリ使用量（概算）
        total_bytes = num_simulations * values_per_path * bytes_per_value

        # オーバーヘッド（辞書構造、インデックス情報など）
        overhead_factor = 1.2  # 20%のオーバーヘッドを仮定
        total_bytes_with_overhead = int(total_bytes * overhead_factor)

        # メモリ使用量をMB/GBに変換
        total_mb = total_bytes_with_overhead / (1024 * 1024)
        total_gb = total_mb / 1024

        # ディスク使用時と未使用時の推奨設定
        recommendation = "メモリ内に保持"
        if total_gb > 1.0:  # 1GB以上の場合
            recommendation = "ディスク保存を推奨"

        return {
            "estimated_bytes": total_bytes_with_overhead,
            "estimated_mb": total_mb,
            "estimated_gb": total_gb,
            "num_variables": num_variables,
            "values_per_path": values_per_path,
            "recommendation": recommendation
        }

    def optimize_memory_usage(self):
        """メモリ使用量を最適化する"""
        # 未使用のシミュレーションストレージをクリーンアップ
        if self._simulation_storage is not None:
            # メモリストレージが存在する場合
            if not self.use_disk_storage and 'memory' in self._simulation_storage:
                # 重複パスの削除
                unique_paths = set()
                duplicates = []

                for i, path_idx in enumerate(self._simulation_storage['memory']['paths']):
                    if path_idx in unique_paths:
                        duplicates.append(i)
                    else:
                        unique_paths.add(path_idx)

                # 重複を削除
                for i in sorted(duplicates, reverse=True):
                    del self._simulation_storage['memory']['paths'][i]

                if duplicates:
                    self.logger.info(f"{len(duplicates)}個の重複シミュレーションパスを削除しました")

                # 未使用のパスデータを削除
                unused_keys = []
                for key in list(self._simulation_storage['memory'].keys()):
                    if key != 'paths' and key not in self._simulation_storage['memory']['paths']:
                        unused_keys.append(key)

                for key in unused_keys:
                    del self._simulation_storage['memory'][key]

                if unused_keys:
                    self.logger.info(f"{len(unused_keys)}個の未使用シミュレーションデータを削除しました")

        # 明示的なガベージコレクションを実行
        gc.collect()

    def release_temp_results(self):
        """一時的な結果データを解放する"""
        result_keys = [key for key in self._temp_data_registry.keys() if 'simulation_results' in key]
        for key in result_keys:
            self.release_resource(key)

        self.logger.info(f"{len(result_keys)}個の一時的な結果リソースを解放しました")
        gc.collect()