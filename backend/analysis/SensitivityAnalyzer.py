import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
import copy
import json
import logging
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

    def __init__(self, db=None):
        """
        初期化メソッド

        Parameters:
        -----------
        db : データベース接続オブジェクト（オプション）
        """
        super().__init__(analysis_type="sensitivity", firestore_client=db)
        self.parameters = {}  # パラメータ辞書
        self.model_fn = None  # モデル関数
        self.last_results = None  # 最後の分析結果
        self.baseline_output = None  # ベースラインの出力値
        self.logger = logging.getLogger(__name__)

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
        if name in self.parameters:
            self.logger.warning(f"パラメータ '{name}' は既に存在するため上書きされます")

        if min_value > max_value:
            raise ValueError(f"最小値({min_value})が最大値({max_value})より大きいため設定できません")

        if base_value < min_value or base_value > max_value:
            raise ValueError(f"基準値({base_value})が範囲[{min_value}, {max_value}]の外にあります")

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

    def set_model(self, model_fn: Callable) -> None:
        """
        感度分析の対象となるモデル関数を設定する

        Parameters:
        -----------
        model_fn : Callable
            モデル関数（パラメータ辞書を入力として受け取り、出力値を返す関数）
        """
        if not callable(model_fn):
            raise TypeError("モデル関数は呼び出し可能なオブジェクトである必要があります")

        self.model_fn = model_fn

        # ベースライン出力の計算
        if self.parameters:
            base_params = {name: param.base_value for name, param in self.parameters.items()}
            try:
                self.baseline_output = self.model_fn(base_params)
                self.logger.info(f"ベースライン出力値: {self.baseline_output}")
            except Exception as e:
                error_msg = f"ベースライン計算エラー: {str(e)}"
                self.logger.error(error_msg)
                raise AnalysisError(error_msg) from e

    def run_one_way_sensitivity_analysis(self,
                                        num_points: int = 10,
                                        return_raw_data: bool = False) -> Dict[str, Any]:
        """
        1ウェイ感度分析を実行する
        各パラメータを個別に変化させ、出力への影響を測定

        Parameters:
        -----------
        num_points : int, default=10
            各パラメータに対するテストポイント数
        return_raw_data : bool, default=False
            生データを返すかどうか

        Returns:
        --------
        Dict[str, Any]
            感度分析の結果
        """
        if not self.model_fn:
            raise ValueError("モデル関数が設定されていません")

        if not self.parameters:
            raise ValueError("パラメータが設定されていません")

        # ベースラインパラメータの取得
        base_params = {name: param.base_value for name, param in self.parameters.items()}

        # ベースライン出力の計算（まだ計算されていない場合）
        if self.baseline_output is None:
            self.baseline_output = self.model_fn(base_params)

        results = {}
        raw_data = {}

        # 各パラメータに対して感度分析を実行
        for param_name, param in self.parameters.items():
            print(f"パラメータ '{param_name}' の感度分析を実行中...")

            # テスト値の生成
            if param.step_size:
                # ステップサイズが指定されている場合
                step_count = int((param.max_value - param.min_value) / param.step_size) + 1
                test_values = np.linspace(param.min_value, param.max_value, min(step_count, num_points))
            else:
                # ポイント数に基づいて均等に分割
                test_values = np.linspace(param.min_value, param.max_value, num_points)

            outputs = []

            # 各テスト値でモデルを実行
            for test_value in test_values:
                # パラメータのコピーを作成し、テスト値を設定
                test_params = base_params.copy()
                test_params[param_name] = test_value

                # モデル実行
                try:
                    output = self.model_fn(test_params)
                    outputs.append(output)
                except Exception as e:
                    print(f"  エラー（{param_name}={test_value}）: {str(e)}")
                    outputs.append(None)

            # 出力変化の範囲を計算
            valid_outputs = [o for o in outputs if o is not None]
            if valid_outputs:
                output_range = max(valid_outputs) - min(valid_outputs)

                # 基準値からの変化率の最大値
                max_change_percent = 0
                if self.baseline_output != 0:
                    changes = [(o - self.baseline_output) / abs(self.baseline_output) * 100 for o in valid_outputs]
                    max_change_percent = max(abs(min(changes)), abs(max(changes)))

                # 結果を格納
                results[param_name] = {
                    'output_range': output_range,
                    'max_change_percent': max_change_percent,
                    'min_output': min(valid_outputs),
                    'max_output': max(valid_outputs)
                }

                # 生データを格納（オプション）
                if return_raw_data:
                    raw_data[param_name] = {
                        'test_values': test_values.tolist(),
                        'outputs': outputs
                    }

            print(f"  結果: 出力範囲={output_range}, 最大変化率={max_change_percent:.2f}%")

        # 出力変化の大きさでパラメータをソート
        sorted_results = sorted(
            [(param_name, data) for param_name, data in results.items()],
            key=lambda x: x[1]['output_range'],
            reverse=True
        )

        # 結果をまとめる
        analysis_results = {
            'baseline_output': self.baseline_output,
            'sorted_parameters': [name for name, _ in sorted_results],
            'parameter_impacts': {name: data for name, data in sorted_results},
        }

        if return_raw_data:
            analysis_results['raw_data'] = raw_data

        self.last_results = analysis_results
        return analysis_results

    def generate_tornado_chart(self, results=None, top_n=None, sort_by='max_change_percent') -> Dict[str, Any]:
        """
        トルネードチャートを生成する

        Parameters:
        -----------
        results : Dict[str, Any], optional
            感度分析の結果（指定されていない場合は最後の結果を使用）
        top_n : int, optional
            表示するパラメータの数（上位n個）
        sort_by : str, default='max_change_percent'
            ソート基準 ('output_range' または 'max_change_percent')

        Returns:
        --------
        Dict[str, Any]
            チャートデータとBase64エンコードされた画像
        """
        if results is None:
            if self.last_results is None:
                raise ValueError("感度分析の結果がありません。先に感度分析を実行してください。")
            results = self.last_results

        # 結果をソート
        try:
            sorted_results = sorted(
                [(param_name, data) for param_name, data in results.items()],
                key=lambda x: x[1][sort_by],
                reverse=True
            )

            # 上位n個を取得
            if top_n is not None:
                sorted_results = sorted_results[:top_n]
        except KeyError as e:
            error_msg = f"ソートキー '{sort_by}' が結果に存在しません: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e
        except Exception as e:
            error_msg = f"結果のソート中にエラーが発生しました: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e

        # プロットデータの作成
        param_names = [param_name for param_name, _ in sorted_results]
        max_changes = [data[sort_by] for _, data in sorted_results]

        try:
            # プロット作成
            fig, ax = plt.subplots(figsize=(10, 8))

            # 横向きの棒グラフを作成
            bars = ax.barh(param_names, max_changes, color='skyblue')

            # 値の追加
            for i, bar in enumerate(bars):
                ax.text(bar.get_width() + bar.get_width() * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f'{max_changes[i]:.2f}',
                        va='center')

            # タイトルと軸ラベル
            metric_label = '最大変化率 (%)' if sort_by == 'max_change_percent' else '出力範囲'
            ax.set_title('パラメータ感度分析 - トルネードチャート')
            ax.set_xlabel(metric_label)
            ax.set_ylabel('パラメータ')

            # グリッド線の追加
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)

            # レイアウト調整
            plt.tight_layout()

            # 画像をBase64エンコード
            chart_image = PlotUtility.save_plot_to_base64(fig)

            # 結果データを作成
            result = {
                'parameters': param_names,
                'values': max_changes,
                'metric': sort_by,
                'chart_base64': chart_image
            }

            return result

        except Exception as e:
            error_msg = f"トルネードチャート生成中にエラーが発生しました: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    def run_two_way_sensitivity_analysis(self,
                                       param1_name: str,
                                       param2_name: str,
                                       num_points: int = 10) -> Dict[str, Any]:
        """
        2ウェイ感度分析を実行する
        2つのパラメータを同時に変化させ、出力への影響を測定

        Parameters:
        -----------
        param1_name : str
            1つ目のパラメータ名
        param2_name : str
            2つ目のパラメータ名
        num_points : int, default=10
            各パラメータのテストポイント数

        Returns:
        --------
        Dict[str, Any]
            2ウェイ感度分析の結果
        """
        if not self.model_fn:
            raise ValueError("モデル関数が設定されていません")

        if param1_name not in self.parameters or param2_name not in self.parameters:
            raise ValueError(f"パラメータ '{param1_name}' または '{param2_name}' が見つかりません")

        # ベースラインパラメータの取得
        base_params = {name: param.base_value for name, param in self.parameters.items()}

        # パラメータ範囲の取得
        param1 = self.parameters[param1_name]
        param2 = self.parameters[param2_name]

        # テスト値の生成
        param1_values = np.linspace(param1.min_value, param1.max_value, num_points)
        param2_values = np.linspace(param2.min_value, param2.max_value, num_points)

        # 結果格納用の配列
        results = np.zeros((len(param1_values), len(param2_values)))

        # 2ウェイ感度分析の実行
        for i, val1 in enumerate(param1_values):
            for j, val2 in enumerate(param2_values):
                # パラメータのコピーを作成し、テスト値を設定
                test_params = base_params.copy()
                test_params[param1_name] = val1
                test_params[param2_name] = val2

                # モデル実行
                try:
                    output = self.model_fn(test_params)
                    results[i, j] = output
                except Exception as e:
                    print(f"  エラー（{param1_name}={val1}, {param2_name}={val2}）: {str(e)}")
                    results[i, j] = np.nan

        # ヒートマップの作成
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(results,
                        xticklabels=np.round(param2_values, 2),
                        yticklabels=np.round(param1_values, 2),
                        cmap='viridis',
                        annot=False)

        plt.title(f'{param1_name} と {param2_name} の2ウェイ感度分析')
        plt.xlabel(param2_name)
        plt.ylabel(param1_name)
        plt.tight_layout()

        # 結果をまとめる
        analysis_results = {
            'param1': {
                'name': param1_name,
                'values': param1_values.tolist()
            },
            'param2': {
                'name': param2_name,
                'values': param2_values.tolist()
            },
            'results': results.tolist(),
            'baseline_output': self.baseline_output if self.baseline_output is not None else None
        }

        return analysis_results

    def identify_critical_parameters(self,
                                   threshold_percent: float = 5.0) -> Dict[str, Any]:
        """
        重要なパラメータを特定する
        出力に対して一定以上の影響を持つパラメータを特定

        Parameters:
        -----------
        threshold_percent : float, default=5.0
            重要と判断するパーセント変化の閾値

        Returns:
        --------
        Dict[str, Any]
            重要パラメータのリストと情報
        """
        if not self.last_results:
            raise ValueError("まず感度分析を実行してください")

        # パラメータ影響データの取得
        param_impacts = self.last_results['parameter_impacts']

        # 重要パラメータの特定
        critical_params = []

        for param_name, impact in param_impacts.items():
            if impact['max_change_percent'] >= threshold_percent:
                critical_params.append({
                    'name': param_name,
                    'impact_percent': impact['max_change_percent'],
                    'output_range': impact['output_range']
                })

        # 影響度でソート
        critical_params = sorted(critical_params, key=lambda x: x['impact_percent'], reverse=True)

        result = {
            'threshold_percent': threshold_percent,
            'critical_parameters': critical_params,
            'critical_count': len(critical_params),
            'total_parameters': len(param_impacts)
        }

        # 出力
        print(f"重要パラメータ（影響度 >= {threshold_percent}%）:")
        for param in critical_params:
            print(f"  {param['name']}: 影響度 {param['impact_percent']:.2f}%")

        return result

    def export_analysis_results(self, file_path: str) -> None:
        """
        感度分析の結果をJSONファイルにエクスポートする

        Parameters:
        -----------
        file_path : str
            エクスポート先のファイルパス
        """
        if not self.last_results:
            raise ValueError("エクスポートする結果がありません")

        # パラメータ情報を追加
        export_data = copy.deepcopy(self.last_results)
        export_data['parameters'] = {
            name: {
                'base_value': param.base_value,
                'min_value': param.min_value,
                'max_value': param.max_value,
                'distribution': param.distribution
            } for name, param in self.parameters.items()
        }

        # タイムスタンプを追加
        from datetime import datetime
        export_data['timestamp'] = datetime.now().isoformat()

        # JSONファイルに保存
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        print(f"分析結果を '{file_path}' にエクスポートしました")

    def visualize_parameter_impact(self,
                                   param_name: str,
                                   num_points: int = 50,
                                   show_baseline: bool = True) -> Dict[str, Any]:
        """
        特定のパラメータの影響を可視化する

        Parameters:
        -----------
        param_name : str
            可視化するパラメータ名
        num_points : int, default=50
            プロット用のポイント数
        show_baseline : bool, default=True
            ベースライン値を表示するかどうか

        Returns:
        --------
        Dict[str, Any]
            可視化データ
        """
        if param_name not in self.parameters:
            raise ValueError(f"パラメータ '{param_name}' が見つかりません")

        if not self.model_fn:
            raise ValueError("モデル関数が設定されていません")

        # ベースラインパラメータの取得
        base_params = {name: param.base_value for name, param in self.parameters.items()}

        # パラメータ情報の取得
        param = self.parameters[param_name]

        # テスト値の生成
        test_values = np.linspace(param.min_value, param.max_value, num_points)
        outputs = []

        # 各テスト値でモデルを実行
        for test_value in test_values:
            # パラメータのコピーを作成し、テスト値を設定
            test_params = base_params.copy()
            test_params[param_name] = test_value

            # モデル実行
            try:
                output = self.model_fn(test_params)
                outputs.append(output)
            except Exception as e:
                error_msg = f"  エラー（{param_name}={test_value}）: {str(e)}"
                self.logger.warning(error_msg)
                outputs.append(None)

        # 有効な出力のみを抽出
        valid_indices = [i for i, o in enumerate(outputs) if o is not None]
        valid_test_values = [test_values[i] for i in valid_indices]
        valid_outputs = [outputs[i] for i in valid_indices]

        # utils.pyのPlotUtilityを使用してプロットを作成
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(valid_test_values, valid_outputs, 'o-', linewidth=2)

        # ベースライン値の表示
        if show_baseline and self.baseline_output is not None:
            ax.axhline(y=self.baseline_output, color='r', linestyle='--',
                      label=f'ベースライン出力 ({self.baseline_output:.2f})')
            ax.axvline(x=param.base_value, color='g', linestyle='--',
                      label=f'ベースライン入力 ({param.base_value:.2f})')
            ax.legend()

        ax.set_title(f'パラメータ {param_name} の影響')
        ax.set_xlabel(param_name)
        ax.set_ylabel('出力')
        ax.grid(True)
        plt.tight_layout()

        # 結果データを作成
        result = {
            'parameter': param_name,
            'test_values': valid_test_values,
            'outputs': valid_outputs,
            'baseline_input': param.base_value,
            'baseline_output': self.baseline_output,
            'plot_base64': PlotUtility.save_plot_to_base64(fig)
        }

        return result

# 使用例
if __name__ == "__main__":
    # ビジネスモデルの例：シンプルなSaaS収益モデル
    def saas_revenue_model(params):
        """
        シンプルなSaaS収益モデル

        Parameters:
        - users_initial: 初期ユーザー数
        - growth_rate: 月間成長率（%）
        - churn_rate: 月間解約率（%）
        - arpu: 平均ユーザー単価（円）
        - cac: 顧客獲得コスト（円）
        - months: シミュレーション期間（月）

        Returns:
        - 12ヶ月後の累積収益（円）
        """
        users = params['users_initial']
        total_revenue = 0
        total_costs = params['users_initial'] * params['cac']  # 初期ユーザー獲得コスト

        for _ in range(int(params['months'])):
            # 新規ユーザー
            new_users = users * (params['growth_rate'] / 100)
            # 解約ユーザー
            churned_users = users * (params['churn_rate'] / 100)
            # ユーザー数の更新
            users = users + new_users - churned_users
            # 当月の収益
            monthly_revenue = users * params['arpu']
            total_revenue += monthly_revenue
            # 新規ユーザー獲得コスト
            total_costs += new_users * params['cac']

        # 純利益を返す
        return total_revenue - total_costs

    # 感度分析の実行
    analyzer = SensitivityAnalyzer()

    # パラメータの追加
    analyzer.add_parameter('users_initial', 100, 50, 200)
    analyzer.add_parameter('growth_rate', 10, 5, 20)
    analyzer.add_parameter('churn_rate', 5, 1, 10)
    analyzer.add_parameter('arpu', 5000, 3000, 8000)
    analyzer.add_parameter('cac', 20000, 10000, 30000)
    analyzer.add_parameter('months', 12, 6, 24)

    # モデル関数の設定
    analyzer.set_model(saas_revenue_model)

    # 感度分析の実行
    results = analyzer.run_one_way_sensitivity_analysis()

    # トルネードチャートの生成
    tornado_data = analyzer.generate_tornado_chart(results=results)

    # 重要パラメータの特定
    critical_params = analyzer.identify_critical_parameters()

    # 特定のパラメータの影響を詳細に可視化
    param_effect = analyzer.visualize_parameter_impact('growth_rate')