import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, Generator
from dataclasses import dataclass
import copy
import json
import logging
import gc
import weakref
from functools import lru_cache
from .base import BaseAnalyzer, AnalysisError
from .utils import PlotUtility, StatisticsUtility

@dataclass
class Parameter:
    """パラメータ情報を保持するクラス"""
    name: str  # パラメータ名
    base_value: float  # 基準値
    min_value: float  # 最小値
    max_value: float  # 最大値
    step_size: Optional[float] = None  # ステップサイズ（オプション）
    distribution: Optional[str] = "uniform"  # 分布タイプ（uniform, normal, triangular, etc.）
    distribution_params: Optional[Dict[str, float]] = None  # 分布パラメータ

class SensitivityAnalyzer(BaseAnalyzer):
    """
    感度分析とトルネードチャート生成を行うクラス

    機能:
    - パラメータの感度分析
    - トルネードチャートの生成
    - 最も影響力の高いパラメータの特定
    - 様々なモデル型への対応
    """

    def __init__(self, db=None, cache_size: int = 32):
        """
        初期化メソッド

        Parameters:
        -----------
        db : データベース接続オブジェクト（オプション）
        cache_size : 結果キャッシュの最大サイズ
        """
        super().__init__(analysis_type="sensitivity", firestore_client=db)
        self.parameters = {}  # パラメータ辞書
        self.model_fn = None  # モデル関数
        self.baseline_output = None  # ベースラインの出力値
        self.logger = logging.getLogger(__name__)

        # キャッシュとメモリ最適化のための変数
        self._cache_size = cache_size
        self._model_cache = weakref.WeakValueDictionary()  # 弱参照を使用して自動メモリ管理
        self._model_cache_keys = []
        self._cache_hits = 0
        self._cache_misses = 0

        # 結果の一時保存用
        self._sensitivity_results = None

        # ファイナライザーの設定
        self._finalizer = weakref.finalize(self, self._cleanup_resources)

    def __del__(self):
        """クリーンアップ処理"""
        self._cleanup_resources()
        super().__del__()

    def _cleanup_resources(self):
        """リソースのクリーンアップ"""
        self._clear_model_cache()
        self._release_sensitivity_results()

    def add_parameter(self,
                      name: str,
                      base_value: float,
                      min_value: float,
                      max_value: float,
                      step_size: Optional[float] = None,
                      distribution: str = "uniform",
                      distribution_params: Optional[Dict[str, float]] = None) -> None:
        """
        感度分析のためのパラメータを追加する

        Parameters:
        -----------
        name : str
            パラメータ名
        base_value : float
            基準値
        min_value : float
            最小値
        max_value : float
            最大値
        step_size : float, optional
            ステップサイズ
        distribution : str, default="uniform"
            パラメータの分布タイプ
        distribution_params : Dict[str, float], optional
            分布のパラメータ
        """
        try:
            self._validate_parameter_input(name, base_value, min_value, max_value)

            self.parameters[name] = Parameter(
                name=name,
                base_value=base_value,
                min_value=min_value,
                max_value=max_value,
                step_size=step_size,
                distribution=distribution,
                distribution_params=distribution_params
            )
            self.logger.info(f"パラメータ '{name}' を追加しました（基準値: {base_value}, 範囲: [{min_value}, {max_value}]）")

            # パラメータが変更されたらベースライン出力を再計算する必要がある
            self.baseline_output = None

            # キャッシュをクリア
            self._clear_model_cache()
        except ValueError as e:
            self.logger.error(f"パラメータ追加エラー: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"予期しないエラー: {str(e)}")
            raise AnalysisError(f"パラメータ追加中にエラーが発生しました: {str(e)}") from e

    def _validate_parameter_input(self, name: str, base_value: float, min_value: float, max_value: float):
        """パラメータ入力値を検証する"""
        if name in self.parameters:
            self.logger.warning(f"パラメータ '{name}' は既に存在するため上書きされます")

        if min_value > max_value:
            raise ValueError(f"最小値({min_value})が最大値({max_value})より大きいため設定できません")

        if base_value < min_value or base_value > max_value:
            raise ValueError(f"基準値({base_value})が範囲[{min_value}, {max_value}]の外にあります")

    def set_model(self, model_fn: Callable) -> None:
        """
        感度分析の対象となるモデル関数を設定する

        Parameters:
        -----------
        model_fn : Callable
            モデル関数（パラメータ辞書を入力として受け取り、出力値を返す関数）
        """
        try:
            if not callable(model_fn):
                raise TypeError("モデル関数は呼び出し可能なオブジェクトである必要があります")

            self.model_fn = model_fn

            # モデル関数が変更されたらキャッシュをクリア
            self._clear_model_cache()

            # ベースライン出力の計算
            if self.parameters:
                self._calculate_baseline_output()

        except TypeError as e:
            self.logger.error(f"モデル関数設定エラー: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"予期しないエラー: {str(e)}")
            raise AnalysisError(f"モデル関数設定中にエラーが発生しました: {str(e)}") from e

    def _calculate_baseline_output(self):
        """ベースライン出力値を計算する"""
        base_params = {name: param.base_value for name, param in self.parameters.items()}
        try:
            self.baseline_output = self._call_model_with_cache(base_params)
            self.logger.info(f"ベースライン出力値: {self.baseline_output}")
        except Exception as e:
            error_msg = f"ベースライン計算エラー: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    @lru_cache(maxsize=128)
    def _compute_model_result(self, param_tuple):
        """
        モデル関数を実行して結果を計算する（LRUキャッシュ付き）

        Parameters:
        -----------
        param_tuple : tuple
            パラメータのタプル表現

        Returns:
        --------
        float
            モデル出力値
        """
        # タプルからパラメータ辞書を再構築
        params = {k: v for k, v in param_tuple}

        # モデル実行
        return self.model_fn(params)

    def _call_model_with_cache(self, params: Dict[str, float]) -> float:
        """
        キャッシュを使用してモデル関数を呼び出す
        同じパラメータでの再計算を避けるため

        Parameters:
        -----------
        params : Dict[str, float]
            モデルパラメータ

        Returns:
        --------
        float
            モデル出力値
        """
        if not self.model_fn:
            raise ValueError("モデル関数が設定されていません")

        try:
            # パラメータをソートしてキャッシュキーを作成
            key = tuple(sorted((k, v) for k, v in params.items()))

            # LRUキャッシュ付き関数を呼び出し
            return self._compute_model_result(key)
        except Exception as e:
            self.logger.error(f"モデル実行エラー: {str(e)}")
            raise AnalysisError(f"モデル関数の実行中にエラーが発生しました: {str(e)}") from e

    def _clear_model_cache(self) -> None:
        """モデルのキャッシュをクリアする"""
        # LRUキャッシュをクリア
        self._compute_model_result.cache_clear()

        # キャッシュ統計情報のリセット
        self._cache_hits = 0
        self._cache_misses = 0

        # 明示的なガベージコレクション
        gc.collect()

    def _release_sensitivity_results(self) -> None:
        """感度分析結果を解放する"""
        if self._sensitivity_results is not None:
            del self._sensitivity_results
            self._sensitivity_results = None
            gc.collect()

    def run_one_way_sensitivity_analysis(self,
                                        num_points: int = 10,
                                        return_raw_data: bool = False,
                                        parameter_subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        1ウェイ感度分析を実行する
        各パラメータを個別に変化させ、出力への影響を測定

        Parameters:
        -----------
        num_points : int, default=10
            各パラメータに対するテストポイント数
        return_raw_data : bool, default=False
            生データを返すかどうか
        parameter_subset : List[str], optional
            分析対象のパラメータサブセット（指定しない場合は全パラメータ）

        Returns:
        --------
        Dict[str, Any]
            感度分析の結果
        """
        try:
            self._validate_analysis_prerequisites()

            # 以前の結果を解放
            self._release_sensitivity_results()

            # 結果を格納するための辞書
            results = self._initialize_results_dict()
            raw_data = {}

            # 分析対象のパラメータを決定
            target_parameters = self._determine_target_parameters(parameter_subset)

            # 各パラメータに対して感度分析を実行
            for param_name in target_parameters:
                self.logger.info(f"パラメータ '{param_name}' の感度分析を実行中...")

                param_results = self._analyze_single_parameter(param_name, num_points)
                results['parameters'][param_name] = param_results

                if return_raw_data:
                    raw_data[param_name] = param_results['raw_data']
                    # 生データは結果から削除（メモリ効率化）
                    del results['parameters'][param_name]['raw_data']

            # ベースライン基準での結果のソート
            results['sorted_impacts'] = self._sort_impacts(results['parameters'])

            # キャッシュ統計の追加
            results['cache_stats'] = {
                'hits': self._cache_hits,
                'misses': self._cache_misses,
                'hit_ratio': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
            }

            # 結果を一時保存
            self._sensitivity_results = results.copy()
            if return_raw_data:
                self._sensitivity_results['raw_data'] = raw_data

            # 生データを含める場合
            if return_raw_data:
                results['raw_data'] = raw_data

            return results

        except ValueError as e:
            self.logger.error(f"感度分析実行エラー: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"予期しないエラー: {str(e)}")
            raise AnalysisError(f"感度分析実行中にエラーが発生しました: {str(e)}") from e

    def _validate_analysis_prerequisites(self):
        """分析前の前提条件を検証する"""
        if not self.model_fn:
            raise ValueError("モデル関数が設定されていません")

        if not self.parameters:
            raise ValueError("パラメータが設定されていません")

    def _initialize_results_dict(self):
        """結果辞書を初期化する"""
        # ベースラインパラメータの取得
        base_params = {name: param.base_value for name, param in self.parameters.items()}

        # ベースライン出力の計算（まだ計算されていない場合）
        if self.baseline_output is None:
            self.baseline_output = self._call_model_with_cache(base_params)

        return {
            'baseline_output': self.baseline_output,
            'baseline_params': base_params,
            'parameters': {},
            'sorted_impacts': []
        }

    def _determine_target_parameters(self, parameter_subset):
        """分析対象のパラメータを決定する"""
        target_parameters = parameter_subset if parameter_subset else self.parameters.keys()
        # 存在しないパラメータをフィルタリング
        return [param for param in target_parameters if param in self.parameters]

    def _analyze_single_parameter(self, param_name, num_points):
        """単一パラメータの感度分析を実行する"""
        param = self.parameters[param_name]

        # テスト値の生成
        test_values = self._generate_test_values(param, num_points)

        # 各テスト値に対するモデル出力を計算
        outputs = []
        base_params = {name: p.base_value for name, p in self.parameters.items()}

        for value in test_values:
            # パラメータの値だけを変更
            test_params = base_params.copy()
            test_params[param_name] = value

            # モデルを実行
            output = self._call_model_with_cache(test_params)
            outputs.append(output)

        # 結果の分析
        min_output = min(outputs)
        max_output = max(outputs)
        min_index = outputs.index(min_output)
        max_index = outputs.index(max_output)

        # ベースラインからの変化率を計算
        if self.baseline_output != 0:
            min_change_percent = (min_output - self.baseline_output) / abs(self.baseline_output) * 100
            max_change_percent = (max_output - self.baseline_output) / abs(self.baseline_output) * 100
        else:
            min_change_percent = 0
            max_change_percent = 0

        # 絶対変化量の計算
        min_change_absolute = min_output - self.baseline_output
        max_change_absolute = max_output - self.baseline_output

        # 変動幅の計算
        range_absolute = max_output - min_output
        range_percent = range_absolute / abs(self.baseline_output) * 100 if self.baseline_output != 0 else 0

        return {
            'test_values': test_values.tolist(),
            'outputs': outputs,
            'min_output': min_output,
            'max_output': max_output,
            'min_value': test_values[min_index],
            'max_value': test_values[max_index],
            'min_change_percent': min_change_percent,
            'max_change_percent': max_change_percent,
            'min_change_absolute': min_change_absolute,
            'max_change_absolute': max_change_absolute,
            'range_absolute': range_absolute,
            'range_percent': range_percent,
            'raw_data': {'values': test_values.tolist(), 'outputs': outputs}
        }

    def _generate_test_values(self, param, num_points):
        """テスト値を生成する"""
        if param.step_size:
            # ステップサイズが指定されている場合
            step_count = int((param.max_value - param.min_value) / param.step_size) + 1
            return np.linspace(param.min_value, param.max_value, min(step_count, num_points))
        else:
            # ポイント数に基づいて均等に分割
            return np.linspace(param.min_value, param.max_value, num_points)

    def _sort_impacts(self, parameter_results):
        """パラメータの影響度でソートする"""
        impact_list = []
        for name, results in parameter_results.items():
            impact_list.append({
                'parameter': name,
                'range_absolute': results['range_absolute'],
                'range_percent': results['range_percent'],
                'min_change_percent': results['min_change_percent'],
                'max_change_percent': results['max_change_percent']
            })

        # 変動幅の絶対値でソート
        return sorted(impact_list, key=lambda x: abs(x['range_absolute']), reverse=True)

    def generate_tornado_chart(self, results=None, top_n=None, sort_by='max_change_percent') -> Dict[str, Any]:
        """
        トルネードチャートを生成する

        Parameters:
        -----------
        results : Dict, optional
            感度分析の結果（指定しない場合は最後の結果を使用）
        top_n : int, optional
            表示するパラメータの数（上位N個）
        sort_by : str, default='max_change_percent'
            ソート基準（'max_change_percent', 'min_change_percent', 'range_absolute', 'range_percent'）

        Returns:
        --------
        Dict[str, Any]
            トルネードチャート情報
        """
        try:
            # 結果の取得
            sensitivity_results = results or self._sensitivity_results

            if not sensitivity_results:
                raise ValueError("感度分析結果がありません。先にrun_one_way_sensitivity_analysisを実行してください")

            # トルネードチャート用にデータを準備
            tornado_data = self._prepare_tornado_data(sensitivity_results, top_n, sort_by)

            # プロットの生成
            plt_image = self._plot_tornado_chart(tornado_data, sort_by)

            return {
                'chart_data': tornado_data,
                'plot': plt_image,
                'sort_by': sort_by,
                'baseline_output': sensitivity_results['baseline_output']
            }

        except ValueError as e:
            self.logger.error(f"トルネードチャート生成エラー: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"予期しないエラー: {str(e)}")
            raise AnalysisError(f"トルネードチャート生成中にエラーが発生しました: {str(e)}") from e

    def _prepare_tornado_data(self, sensitivity_results, top_n, sort_by):
        """トルネードチャート用のデータを準備する"""
        # パラメータとその影響をソートする
        params_sorted = sorted(
            [
                (name, results)
                for name, results in sensitivity_results['parameters'].items()
            ],
            key=lambda x: abs(x[1][sort_by]),
            reverse=True
        )

        # 上位N個に制限
        if top_n is not None:
            params_sorted = params_sorted[:top_n]

        # トルネードチャート用のデータ形式に変換
        tornado_data = []
        for name, results in params_sorted:
            param = self.parameters[name]
            tornado_data.append({
                'parameter': name,
                'base_value': param.base_value,
                'min_value': results['min_value'],
                'max_value': results['max_value'],
                'min_output': results['min_output'],
                'max_output': results['max_output'],
                'min_change_percent': results['min_change_percent'],
                'max_change_percent': results['max_change_percent']
            })

        return tornado_data

    def _plot_tornado_chart(self, tornado_data, sort_by):
        """トルネードチャートをプロットする"""
        # PlotUtilityクラスを使用
        plot_data = {
            'tornado_data': tornado_data,
            'title': 'パラメータ感度分析 - トルネードチャート',
            'sort_by': sort_by,
            'baseline_output': self.baseline_output
        }

        return PlotUtility.create_and_save_plot('tornado', plot_data)