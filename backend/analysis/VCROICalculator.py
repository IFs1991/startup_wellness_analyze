import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import traceback
import gc
import contextlib
import time
from .base import BaseAnalyzer
from .utils import HealthImpactWeightUtility

class VCROICalculator(BaseAnalyzer):
    """
    VC向けROI計算クラス

    Startup Wellnessプログラムの投資対効果を計算するためのクラス
    このクラスは様々な方法でROI（投資利益率）を計算するメソッドを提供します。

    主な機能:
    - 基本ROI計算
    - 時系列データからのROI計算
    - リスク調整済みROI計算
    - 健康影響を考慮したROI計算
    - 感度分析
    - 役職・組織階層別の重み付けROI計算

    基本ROI計算式:
    ROI_{VC} = ((ΔRevenue + ΔValuation) - C_{program}) / C_{investment} × 100
    """

    # 初期化とリソース管理
    # ------------------

    def __init__(self):
        """
        VCROICalculatorの初期化

        ロガーとリソース管理のための初期設定を行います。
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info("VCROICalculatorを初期化しました")
        self._temp_resources = []  # 一時リソース追跡用

    def __del__(self):
        """
        デストラクタ：リソースの自動解放
        """
        self.release_resources()

    def release_resources(self):
        """
        使用したリソースを解放する
        """
        try:
            # 一時リソースのクリーンアップ
            self._temp_resources.clear()

            # 明示的にガベージコレクションを実行
            gc.collect()
            self.logger.debug("リソースを解放しました")
        except Exception as e:
            self.logger.error(f"リソース解放中にエラーが発生しました: {str(e)}")

    @contextlib.contextmanager
    def _managed_dataframe(self, df: pd.DataFrame, copy: bool = False):
        """
        データフレームのリソース管理を行うコンテキストマネージャ

        Parameters
        ----------
        df : pd.DataFrame
            管理するデータフレーム
        copy : bool, default=False
            データをコピーするかどうか

        Yields
        ------
        pd.DataFrame
            管理対象のデータフレーム
        """
        try:
            if copy:
                df_copy = df.copy()
                yield df_copy
            else:
                yield df
        finally:
            # 明示的なクリーンアップ
            if copy:
                del df_copy

            # メモリ使用状況のログ（デバッグレベル）
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("データフレーム管理コンテキスト終了")

    # 基本ROI計算メソッド
    # ------------------

    def calculate_roi(self, delta_revenue: float, delta_valuation: float, program_cost: float, investment_cost: float) -> float:
        """
        基本的なROIを計算します

        Parameters
        ----------
        delta_revenue : float
            収益の変化量
        delta_valuation : float
            企業価値の変化量
        program_cost : float
            Startup Wellnessプログラムのコスト
        investment_cost : float
            投資コスト

        Returns
        -------
        float
            計算されたROI値（％）

        Raises
        ------
        ValueError
            入力値が無効な場合
        """
        # 入力値の検証
        self._validate_basic_roi_inputs(delta_revenue, delta_valuation, program_cost, investment_cost)

        try:
            # 総便益を計算
            total_benefit = delta_revenue + delta_valuation

            # 総コストを計算
            total_cost = program_cost + investment_cost

            # ROI計算
            roi = (total_benefit - total_cost) / total_cost * 100

            self.logger.info(f"基本ROI計算: {roi:.2f}%")
            return roi
        except Exception as e:
            self.logger.error(f"ROI計算中にエラーが発生しました: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise ValueError(f"ROI計算に失敗しました: {str(e)}")

    def _validate_basic_roi_inputs(self, delta_revenue: float, delta_valuation: float, program_cost: float, investment_cost: float) -> None:
        """
        基本的なROI計算の入力値を検証します

        Parameters
        ----------
        delta_revenue : float
            収益の変化量
        delta_valuation : float
            企業価値の変化量
        program_cost : float
            Startup Wellnessプログラムのコスト
        investment_cost : float
            投資コスト

        Raises
        ------
        ValueError
            入力値が無効な場合
        """
        if not all(isinstance(param, (int, float)) for param in [delta_revenue, delta_valuation, program_cost, investment_cost]):
            raise ValueError("全ての入力値は数値である必要があります")

        if program_cost <= 0:
            raise ValueError("プログラムコストは正の値である必要があります")

        if investment_cost <= 0:
            raise ValueError("投資コストは正の値である必要があります")

        total_cost = program_cost + investment_cost
        if total_cost == 0:
            raise ValueError("総コストが0になるため、ROIを計算できません")

    def calculate_roi_from_time_series(self,
                                       revenue_before: pd.Series,
                                       revenue_after: pd.Series,
                                       valuation_before: float,
                                       valuation_after: float,
                                       program_cost: float,
                                       investment_cost: float) -> Dict:
        """
        時系列データからROIを計算

        Parameters
        ----------
        revenue_before : pd.Series
            プログラム導入前の収益時系列データ
        revenue_after : pd.Series
            プログラム導入後の収益時系列データ
        valuation_before : float
            プログラム導入前の企業価値
        valuation_after : float
            プログラム導入後の企業価値
        program_cost : float
            Startup Wellnessプログラムのコスト
        investment_cost : float
            投資コスト

        Returns
        -------
        Dict
            ROI値とその他の計算メトリクスを含む辞書
        """
        try:
            # 収益の変化を計算
            delta_revenue = revenue_after.mean() - revenue_before.mean()

            # 企業価値の変化を計算
            delta_valuation = valuation_after - valuation_before

            # ROI計算
            roi = self.calculate_roi(delta_revenue, delta_valuation, program_cost, investment_cost)

            # 追加メトリクスの計算
            revenue_growth_rate = ((revenue_after.mean() / revenue_before.mean()) - 1) * 100 if revenue_before.mean() > 0 else float('inf')
            valuation_growth_rate = ((valuation_after / valuation_before) - 1) * 100 if valuation_before > 0 else float('inf')

            result = {
                'roi': roi,
                'delta_revenue': delta_revenue,
                'delta_valuation': delta_valuation,
                'revenue_growth_rate': revenue_growth_rate,
                'valuation_growth_rate': valuation_growth_rate,
            }

            self.logger.info(f"時系列ROI計算が完了しました: {result}")
            return result
        except Exception as e:
            self.logger.error(f"時系列データからのROI計算中にエラーが発生しました: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise ValueError(f"時系列ROI計算に失敗しました: {str(e)}")

    # リスク調整ROI計算メソッド
    # ------------------

    def calculate_risk_adjusted_roi(self,
                                   delta_revenue: float,
                                   delta_valuation: float,
                                   program_cost: float,
                                   investment_cost: float,
                                   risk_factor: float = 0.2) -> float:
        """
        リスク調整済みROI計算

        Parameters
        ----------
        delta_revenue : float
            収益の変化量
        delta_valuation : float
            企業価値の変化量
        program_cost : float
            Startup Wellnessプログラムのコスト
        investment_cost : float
            投資コスト
        risk_factor : float, optional
            リスク調整係数（デフォルト: 0.2）

        Returns
        -------
        float
            リスク調整済みROI値
        """
        try:
            # 基本ROI計算
            base_roi = self.calculate_roi(delta_revenue, delta_valuation, program_cost, investment_cost)

            # リスク調整
            risk_adjusted_roi = base_roi * (1 - risk_factor)

            self.logger.info(f"リスク調整済みROI計算: {risk_adjusted_roi:.2f}%")
            return risk_adjusted_roi
        except Exception as e:
            self.logger.error(f"リスク調整済みROI計算中にエラーが発生しました: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise ValueError(f"リスク調整済みROI計算に失敗しました: {str(e)}")

    # 健康影響を考慮したROI計算メソッド
    # ------------------

    def calculate_roi_with_health_impact(self,
                                        delta_revenue: float,
                                        delta_valuation: float,
                                        program_cost: float,
                                        investment_cost: float,
                                        vas_before: pd.DataFrame,
                                        vas_after: pd.DataFrame) -> Dict:
        """
        健康状態の変化を考慮したROI計算

        Parameters
        ----------
        delta_revenue : float
            収益の変化量
        delta_valuation : float
            企業価値の変化量
        program_cost : float
            Startup Wellnessプログラムのコスト
        investment_cost : float
            投資コスト
        vas_before : pd.DataFrame
            プログラム導入前のVASスケールデータ
        vas_after : pd.DataFrame
            プログラム導入後のVASスケールデータ

        Returns
        -------
        Dict
            健康影響を考慮したROI値とその他のメトリクスを含む辞書

        Raises
        ------
        ValueError
            入力データが無効な場合
        """
        try:
            # 入力データの検証
            self._validate_health_impact_inputs(vas_before, vas_after)
            self._validate_basic_roi_inputs(delta_revenue, delta_valuation, program_cost, investment_cost)

            # 基本ROI計算
            base_roi = self.calculate_roi(delta_revenue, delta_valuation, program_cost, investment_cost)

            # 健康指標の改善を計算
            health_impact_factor, vas_improvement = self._calculate_health_impact_factor(vas_before, vas_after)

            # 健康状態を考慮したROI計算
            health_adjusted_roi = self._apply_health_impact_to_roi(base_roi, health_impact_factor)

            # 結果の整形
            result = self._format_health_impact_results(base_roi, health_adjusted_roi, health_impact_factor, vas_improvement)

            self.logger.info(f"健康調整済みROI: {health_adjusted_roi:.2f}% (健康影響係数: {health_impact_factor:.2f})")
            return result

        except Exception as e:
            self.logger.error(f"健康影響を考慮したROI計算中にエラーが発生しました: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise ValueError(f"健康影響を考慮したROI計算に失敗しました: {str(e)}")

    # 健康影響ROI計算の補助メソッド
    # ------------------

    def _validate_health_impact_inputs(self, vas_before: pd.DataFrame, vas_after: pd.DataFrame) -> None:
        """
        健康影響計算の入力データを検証します

        Parameters
        ----------
        vas_before : pd.DataFrame
            プログラム導入前のVASスケールデータ
        vas_after : pd.DataFrame
            プログラム導入後のVASスケールデータ

        Raises
        ------
        ValueError
            入力データが無効な場合
        """
        if vas_before is None or vas_after is None:
            raise ValueError("VASスケールデータがNoneです")

        if not isinstance(vas_before, pd.DataFrame) or not isinstance(vas_after, pd.DataFrame):
            raise ValueError("VASスケールデータはDataFrameである必要があります")

        if vas_before.empty or vas_after.empty:
            raise ValueError("VASスケールデータが空です")

        # 共通の列があるかチェック
        common_columns = set(vas_before.columns).intersection(set(vas_after.columns))
        if not common_columns:
            raise ValueError("VASスケールデータに共通の列がありません")

        # データ型の検証
        for df, name in [(vas_before, "導入前"), (vas_after, "導入後")]:
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(f"{name}のVASスケールデータの列 '{col}' が数値型ではありません")

    def _calculate_health_impact_factor(self, vas_before: pd.DataFrame, vas_after: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """
        健康影響係数と項目ごとの改善率を計算します

        Parameters
        ----------
        vas_before : pd.DataFrame
            プログラム導入前のVASスケールデータ
        vas_after : pd.DataFrame
            プログラム導入後のVASスケールデータ

        Returns
        -------
        Tuple[float, Dict[str, float]]
            (健康影響係数, 項目ごとの改善率の辞書)
        """
        # 結果を格納する変数の初期化
        vas_improvement = {}

        # 各健康指標について改善率を計算
        with self._managed_dataframe(vas_before) as before_df, self._managed_dataframe(vas_after) as after_df:
            common_columns = set(before_df.columns).intersection(set(after_df.columns))

            for column in common_columns:
                before_mean = before_df[column].mean()
                after_mean = after_df[column].mean()

                # 改善率を計算 (VASスケールは値が小さいほど良いと仮定)
                if before_mean > 0:
                    improvement = (before_mean - after_mean) / before_mean * 100
                    vas_improvement[column] = improvement
                else:
                    self.logger.warning(f"列 '{column}' の導入前平均値が0以下です。改善率の計算をスキップします。")
                    vas_improvement[column] = 0.0

        # 健康影響係数を計算
        health_impact_factor = self._compute_health_impact_factor_from_improvements(vas_improvement)

        return health_impact_factor, vas_improvement

    def _compute_health_impact_factor_from_improvements(self, vas_improvement: Dict[str, float]) -> float:
        """
        VAS改善率から健康影響係数を計算します

        Parameters
        ----------
        vas_improvement : Dict[str, float]
            項目ごとの改善率の辞書

        Returns
        -------
        float
            計算された健康影響係数
        """
        if not vas_improvement:
            self.logger.warning("VAS改善率データがありません。健康影響係数は1.0とします。")
            return 1.0

        # 基本係数の設定
        health_impact_factor = 1.0

        # 正の改善がある項目のみを考慮して係数を調整
        positive_improvements = [imp for imp in vas_improvement.values() if imp > 0]

        if positive_improvements:
            # 改善率の平均値を計算
            avg_improvement = sum(positive_improvements) / len(positive_improvements)

            # 改善1%ごとにROIを1%上乗せする仮定
            health_impact_factor += avg_improvement * 0.01

        return health_impact_factor

    def _apply_health_impact_to_roi(self, base_roi: float, health_impact_factor: float) -> float:
        """
        基本ROIに健康影響係数を適用します

        Parameters
        ----------
        base_roi : float
            基本的なROI値
        health_impact_factor : float
            健康影響係数

        Returns
        -------
        float
            健康影響を考慮したROI値
        """
        return base_roi * health_impact_factor

    def _format_health_impact_results(self, base_roi: float, health_adjusted_roi: float,
                                     health_impact_factor: float, vas_improvement: Dict[str, float]) -> Dict:
        """
        健康影響を考慮したROI計算の結果を整形します

        Parameters
        ----------
        base_roi : float
            基本的なROI値
        health_adjusted_roi : float
            健康影響を考慮したROI値
        health_impact_factor : float
            健康影響係数
        vas_improvement : Dict[str, float]
            項目ごとの改善率の辞書

        Returns
        -------
        Dict
            整形された結果辞書
        """
        return {
            'base_roi': base_roi,
            'health_adjusted_roi': health_adjusted_roi,
            'health_impact_factor': health_impact_factor,
            'vas_improvements': vas_improvement,
            'calculation_timestamp': time.time()
        }

    def _calculate_health_improvements(self, vas_before: pd.DataFrame, vas_after: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """
        健康指標の改善を計算します (レガシーメソッド - 廃止予定)

        この関数は後方互換性のために維持されていますが、今後は _calculate_health_impact_factor を使用してください。

        Parameters
        ----------
        vas_before : pd.DataFrame
            プログラム導入前のVASスケールデータ
        vas_after : pd.DataFrame
            プログラム導入後のVASスケールデータ

        Returns
        -------
        Tuple[float, Dict[str, float]]
            (健康影響係数, 項目ごとの改善率の辞書)
        """
        self.logger.warning("_calculate_health_improvements は廃止予定です。代わりに _calculate_health_impact_factor を使用してください。")
        return self._calculate_health_impact_factor(vas_before, vas_after)

    # 感度分析メソッド
    # ------------------

    def sensitivity_analysis(self,
                            delta_revenue_range: Tuple[float, float],
                            delta_valuation_range: Tuple[float, float],
                            program_cost_range: Tuple[float, float],
                            investment_cost_range: Tuple[float, float],
                            steps: int = 10) -> Dict[str, pd.DataFrame]:
        """
        ROIの感度分析を実行します

        各パラメータの範囲におけるROIの変動を計算します

        Parameters
        ----------
        delta_revenue_range : Tuple[float, float]
            収益変化量の範囲（最小値, 最大値）
        delta_valuation_range : Tuple[float, float]
            企業価値変化量の範囲（最小値, 最大値）
        program_cost_range : Tuple[float, float]
            プログラムコストの範囲（最小値, 最大値）
        investment_cost_range : Tuple[float, float]
            投資コストの範囲（最小値, 最大値）
        steps : int, optional
            分析のステップ数, by default 10

        Returns
        -------
        Dict[str, pd.DataFrame]
            各パラメータに対するROIの感度分析結果
        """
        try:
            self.logger.info("感度分析を開始します")

            # 入力パラメータの検証
            self._validate_sensitivity_analysis_inputs(
                delta_revenue_range, delta_valuation_range,
                program_cost_range, investment_cost_range, steps
            )

            # 基準点の定義（各パラメータ範囲の中央値）
            base_params = self._compute_base_parameters(
                delta_revenue_range, delta_valuation_range,
                program_cost_range, investment_cost_range
            )

            # 各パラメータのステップシーケンスを生成
            param_sequences = self._generate_parameter_sequences(
                delta_revenue_range, delta_valuation_range,
                program_cost_range, investment_cost_range, steps
            )

            # 各パラメータに対するROI感度を計算
            sensitivity_results = self._calculate_parameter_sensitivities(
                base_params, param_sequences, steps
            )

            self.logger.info("感度分析が完了しました")
            return sensitivity_results

        except Exception as e:
            self.logger.error(f"感度分析中にエラーが発生しました: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise ValueError(f"感度分析に失敗しました: {str(e)}")

    def _validate_sensitivity_analysis_inputs(self,
                                             delta_revenue_range: Tuple[float, float],
                                             delta_valuation_range: Tuple[float, float],
                                             program_cost_range: Tuple[float, float],
                                             investment_cost_range: Tuple[float, float],
                                             steps: int) -> None:
        """
        感度分析の入力パラメータを検証します

        Parameters
        ----------
        delta_revenue_range : Tuple[float, float]
            収益変化量の範囲
        delta_valuation_range : Tuple[float, float]
            企業価値変化量の範囲
        program_cost_range : Tuple[float, float]
            プログラムコストの範囲
        investment_cost_range : Tuple[float, float]
            投資コストの範囲
        steps : int
            分析のステップ数

        Raises
        ------
        ValueError
            入力パラメータが無効な場合
        """
        # 範囲のチェック
        for param_name, param_range in [
            ("収益変化量", delta_revenue_range),
            ("企業価値変化量", delta_valuation_range),
            ("プログラムコスト", program_cost_range),
            ("投資コスト", investment_cost_range)
        ]:
            if not isinstance(param_range, tuple) or len(param_range) != 2:
                raise ValueError(f"{param_name}の範囲はタプル(最小値, 最大値)で指定してください")

            if param_range[0] >= param_range[1]:
                raise ValueError(f"{param_name}の範囲は最小値 < 最大値である必要があります")

        # ステップ数のチェック
        if not isinstance(steps, int) or steps < 2:
            raise ValueError("ステップ数は2以上の整数である必要があります")

        # 投資コストが0にならないことを確認
        if investment_cost_range[0] <= 0:
            raise ValueError("投資コストは0より大きい値である必要があります")

    def _compute_base_parameters(self,
                               delta_revenue_range: Tuple[float, float],
                               delta_valuation_range: Tuple[float, float],
                               program_cost_range: Tuple[float, float],
                               investment_cost_range: Tuple[float, float]) -> Dict[str, float]:
        """
        感度分析の基準点パラメータを計算します

        Parameters
        ----------
        delta_revenue_range : Tuple[float, float]
            収益変化量の範囲
        delta_valuation_range : Tuple[float, float]
            企業価値変化量の範囲
        program_cost_range : Tuple[float, float]
            プログラムコストの範囲
        investment_cost_range : Tuple[float, float]
            投資コストの範囲

        Returns
        -------
        Dict[str, float]
            基準点パラメータの辞書
        """
        return {
            'delta_revenue': (delta_revenue_range[0] + delta_revenue_range[1]) / 2,
            'delta_valuation': (delta_valuation_range[0] + delta_valuation_range[1]) / 2,
            'program_cost': (program_cost_range[0] + program_cost_range[1]) / 2,
            'investment_cost': (investment_cost_range[0] + investment_cost_range[1]) / 2
        }

    def _generate_parameter_sequences(self,
                                     delta_revenue_range: Tuple[float, float],
                                     delta_valuation_range: Tuple[float, float],
                                     program_cost_range: Tuple[float, float],
                                     investment_cost_range: Tuple[float, float],
                                     steps: int) -> Dict[str, np.ndarray]:
        """
        感度分析の各パラメータシーケンスを生成します

        Parameters
        ----------
        delta_revenue_range : Tuple[float, float]
            収益変化量の範囲
        delta_valuation_range : Tuple[float, float]
            企業価値変化量の範囲
        program_cost_range : Tuple[float, float]
            プログラムコストの範囲
        investment_cost_range : Tuple[float, float]
            投資コストの範囲
        steps : int
            分析のステップ数

        Returns
        -------
        Dict[str, np.ndarray]
            各パラメータのシーケンス配列
        """
        return {
            'delta_revenue': np.linspace(delta_revenue_range[0], delta_revenue_range[1], steps),
            'delta_valuation': np.linspace(delta_valuation_range[0], delta_valuation_range[1], steps),
            'program_cost': np.linspace(program_cost_range[0], program_cost_range[1], steps),
            'investment_cost': np.linspace(investment_cost_range[0], investment_cost_range[1], steps)
        }

    def _calculate_parameter_sensitivities(self,
                                          base_params: Dict[str, float],
                                          param_sequences: Dict[str, np.ndarray],
                                          steps: int) -> Dict[str, pd.DataFrame]:
        """
        各パラメータに対するROI感度を計算します

        Parameters
        ----------
        base_params : Dict[str, float]
            基準点パラメータの辞書
        param_sequences : Dict[str, np.ndarray]
            各パラメータのシーケンス配列
        steps : int
            分析のステップ数

        Returns
        -------
        Dict[str, pd.DataFrame]
            各パラメータに対するROI感度分析結果
        """
        results = {}

        for param_name, param_values in param_sequences.items():
            # 結果を格納するための配列を初期化
            sensitivity_data = []

            for value in param_values:
                # 基準パラメータをコピー
                params = base_params.copy()
                # 現在のパラメータを変更
                params[param_name] = value

                # ROIを計算
                roi = self.calculate_roi(
                    params['delta_revenue'],
                    params['delta_valuation'],
                    params['program_cost'],
                    params['investment_cost']
                )

                # 結果を追加
                sensitivity_data.append({
                    'parameter_value': value,
                    'roi': roi,
                    'relative_change': 100 * (value - base_params[param_name]) / base_params[param_name] if base_params[param_name] != 0 else 0
                })

            # 結果をDataFrameに変換
            results[param_name] = pd.DataFrame(sensitivity_data)

        return results

    # 役職別ROI計算メソッド
    # ------------------

    def calculate_weighted_roi_by_position(self,
                                         positions_data: Dict[str, Dict[str, float]],
                                         program_cost: float,
                                         investment_cost: float,
                                         company_data: Dict[str, str],
                                         db_connection) -> Dict:
        """
        業種・役職別の重み付け係数を考慮したROI計算

        Parameters
        ----------
        positions_data : Dict[str, Dict[str, float]]
            役職ごとの指標データ
            形式: {
                '役職名1': {
                    'delta_revenue': 収益変化量,
                    'delta_valuation': 企業価値変化量
                },
                '役職名2': { ... }
            }
        program_cost : float
            Startup Wellnessプログラムのコスト
        investment_cost : float
            投資コスト
        company_data : Dict[str, str]
            企業データ（業種などの情報を含む）
            必須キー: 'industry'（業種名）
        db_connection
            PostgreSQLデータベース接続オブジェクト

        Returns
        -------
        Dict
            計算結果を含む辞書
            - 'roi': 最終的なROI値（パーセンテージ）
            - 'weighted_by_role': 役職別の重み付けROI値
            - 'role_weights': 使用された役職別の重み係数
            - 'details': 詳細な計算情報
        """
        try:
            self.logger.info("役職別重み付けROI計算を開始")

            # 入力検証
            self._validate_weighted_roi_inputs(positions_data, company_data)

            # 業種情報の取得
            industry_name = company_data['industry']
            self.logger.info(f"対象業種: {industry_name}")

            # 役職ごとの重み係数と個別ROIを計算
            role_weights, role_roi_values, role_impacts = self._calculate_role_based_impacts(
                positions_data, program_cost, investment_cost, industry_name, db_connection
            )

            # 正規化された重みを計算
            normalized_weights = self._normalize_role_weights(role_weights)

            # 重み付け集計値の計算
            weighted_values = self._calculate_weighted_values(
                positions_data, normalized_weights
            )

            # 最終的な重み付けROIを計算
            final_roi = self.calculate_roi(
                weighted_values['delta_revenue'],
                weighted_values['delta_valuation'],
                program_cost,
                investment_cost
            )

            # 結果の整形
            result = self._format_weighted_roi_results(
                final_roi, role_roi_values, role_weights, normalized_weights,
                weighted_values, industry_name, role_impacts
            )

            self.logger.info(f"役職別重み付けROI計算が完了しました。最終ROI値: {final_roi:.2f}%")
            return result

        except Exception as e:
            self.logger.error(f"役職別重み付けROI計算中にエラーが発生しました: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise ValueError(f"役職別重み付けROI計算に失敗しました: {str(e)}")

    def _validate_weighted_roi_inputs(self, positions_data: Dict[str, Dict[str, float]], company_data: Dict[str, str]) -> None:
        """
        重み付けROI計算の入力を検証します

        Parameters
        ----------
        positions_data : Dict[str, Dict[str, float]]
            役職ごとの指標データ
        company_data : Dict[str, str]
            企業データ

        Raises
        ------
        ValueError
            入力データが無効な場合
        """
        if not positions_data:
            raise ValueError("役職データが空です")

        if 'industry' not in company_data:
            raise ValueError("企業データに業種情報（industry）が含まれていません")

        # 各役職データの必須キーをチェック
        for role, data in positions_data.items():
            required_keys = ['delta_revenue', 'delta_valuation']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"役職 '{role}' のデータに必須キー '{key}' がありません")

    def _calculate_role_based_impacts(
            self,
            positions_data: Dict[str, Dict[str, float]],
            program_cost: float,
            investment_cost: float,
            industry_name: str,
            db_connection) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        役職ごとの重み係数とROI、影響度を計算します

        Parameters
        ----------
        positions_data : Dict[str, Dict[str, float]]
            役職ごとの指標データ
        program_cost : float
            プログラムコスト
        investment_cost : float
            投資コスト
        industry_name : str
            業種名
        db_connection
            データベース接続

        Returns
        -------
        Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]
            (役職別重み, 役職別ROI値, 役職別影響度詳細)
        """
        role_weights = {}
        role_roi_values = {}
        role_impacts = {}

        for role, data in positions_data.items():
            # 役職の健康影響度の重み係数を取得
            weight = HealthImpactWeightUtility.get_health_impact_weight(
                db_connection, industry_name, role
            )
            role_weights[role] = weight

            # 役職ごとのROIを計算
            role_roi = self.calculate_roi(
                data['delta_revenue'],
                data['delta_valuation'],
                program_cost * weight,  # 重み付けされたプログラムコスト
                investment_cost * weight  # 重み付けされた投資コスト
            )
            role_roi_values[role] = role_roi

            # 各役職の影響度を計算
            impact = {
                'delta_revenue': data['delta_revenue'],
                'delta_valuation': data['delta_valuation'],
                'weighted_program_cost': program_cost * weight,
                'weighted_investment_cost': investment_cost * weight,
                'roi': role_roi
            }
            role_impacts[role] = impact

        return role_weights, role_roi_values, role_impacts

    def _normalize_role_weights(self, role_weights: Dict[str, float]) -> Dict[str, float]:
        """
        役職の重み係数を正規化します

        Parameters
        ----------
        role_weights : Dict[str, float]
            役職ごとの重み係数

        Returns
        -------
        Dict[str, float]
            正規化された重み係数

        Raises
        ------
        ValueError
            重み係数の合計が0以下の場合
        """
        total_weight = sum(role_weights.values())
        if total_weight <= 0:
            raise ValueError("役職の重み係数の合計が0以下です")

        return {role: weight/total_weight for role, weight in role_weights.items()}

    def _calculate_weighted_values(
            self,
            positions_data: Dict[str, Dict[str, float]],
            normalized_weights: Dict[str, float]) -> Dict[str, float]:
        """
        正規化された重みに基づいて加重平均値を計算します

        Parameters
        ----------
        positions_data : Dict[str, Dict[str, float]]
            役職ごとの指標データ
        normalized_weights : Dict[str, float]
            正規化された重み係数

        Returns
        -------
        Dict[str, float]
            重み付けされた集計値
        """
        total_weighted_delta_revenue = sum(
            positions_data[role]['delta_revenue'] * normalized_weights[role]
            for role in positions_data.keys()
        )

        total_weighted_delta_valuation = sum(
            positions_data[role]['delta_valuation'] * normalized_weights[role]
            for role in positions_data.keys()
        )

        return {
            'delta_revenue': total_weighted_delta_revenue,
            'delta_valuation': total_weighted_delta_valuation
        }

    def _format_weighted_roi_results(
            self,
            final_roi: float,
            role_roi_values: Dict[str, float],
            role_weights: Dict[str, float],
            normalized_weights: Dict[str, float],
            weighted_values: Dict[str, float],
            industry_name: str,
            role_impacts: Dict[str, Dict[str, float]]) -> Dict:
        """
        重み付けROI計算の結果を整形します

        Parameters
        ----------
        final_roi : float
            最終的なROI値
        role_roi_values : Dict[str, float]
            役職別のROI値
        role_weights : Dict[str, float]
            役職別の重み係数
        normalized_weights : Dict[str, float]
            正規化された重み係数
        weighted_values : Dict[str, float]
            重み付けされた集計値
        industry_name : str
            業種名
        role_impacts : Dict[str, Dict[str, float]]
            役職別影響度詳細

        Returns
        -------
        Dict
            整形された結果辞書
        """
        return {
            'roi': final_roi,
            'weighted_by_role': role_roi_values,
            'role_weights': role_weights,
            'normalized_weights': normalized_weights,
            'details': {
                'total_weighted_delta_revenue': weighted_values['delta_revenue'],
                'total_weighted_delta_valuation': weighted_values['delta_valuation'],
                'industry': industry_name,
                'role_impacts': role_impacts,
                'calculation_timestamp': time.time()
            }
        }

    # 組織階層別ROI計算メソッド
    # ------------------

    def calculate_hierarchy_impact_roi(self,
                                    executive_data: Dict[str, float],
                                    management_data: Dict[str, float],
                                    staff_data: Dict[str, float],
                                    program_cost: float,
                                    investment_cost: float,
                                    company_data: Dict[str, str],
                                    db_connection) -> Dict:
        """
        組織階層ごとの影響度を考慮したROI計算の簡易版

        Parameters
        ----------
        executive_data : Dict[str, float]
            経営層の指標データ (delta_revenue, delta_valuation)
        management_data : Dict[str, float]
            管理職層の指標データ (delta_revenue, delta_valuation)
        staff_data : Dict[str, float]
            一般職員層の指標データ (delta_revenue, delta_valuation)
        program_cost : float
            Startup Wellnessプログラムのコスト
        investment_cost : float
            投資コスト
        company_data : Dict[str, str]
            企業データ（業種などの情報を含む）
            必須キー: 'industry'（業種名）
        db_connection
            PostgreSQLデータベース接続オブジェクト

        Returns
        -------
        Dict
            計算結果を含む辞書
        """
        try:
            self.logger.info("組織階層別ROI計算を開始")

            # 階層別データの検証
            self._validate_hierarchy_data(executive_data, management_data, staff_data)

            # 階層別データを役職別データに変換
            positions_data = {
                'C級役員/経営層': executive_data,
                '上級管理職': management_data,
                '一般職員': staff_data
            }

            # 役職別重み付けROI計算を実行
            result = self.calculate_weighted_roi_by_position(
                positions_data,
                program_cost,
                investment_cost,
                company_data,
                db_connection
            )

            self.logger.info(f"組織階層別ROI計算が完了しました。最終ROI値: {result['roi']:.2f}%")
            return result

        except Exception as e:
            self.logger.error(f"組織階層別ROI計算中にエラーが発生しました: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise ValueError(f"組織階層別ROI計算に失敗しました: {str(e)}")

    def _validate_hierarchy_data(self,
                               executive_data: Dict[str, float],
                               management_data: Dict[str, float],
                               staff_data: Dict[str, float]) -> None:
        """
        階層別データを検証します

        Parameters
        ----------
        executive_data : Dict[str, float]
            経営層の指標データ
        management_data : Dict[str, float]
            管理職層の指標データ
        staff_data : Dict[str, float]
            一般職員層の指標データ

        Raises
        ------
        ValueError
            入力データが無効な場合
        """
        required_keys = ['delta_revenue', 'delta_valuation']

        for hierarchy_name, data in [
            ('経営層', executive_data),
            ('管理職層', management_data),
            ('一般職員層', staff_data)
        ]:
            if not isinstance(data, dict):
                raise ValueError(f"{hierarchy_name}のデータが辞書型ではありません")

            for key in required_keys:
                if key not in data:
                    raise ValueError(f"{hierarchy_name}のデータに必須キー '{key}' がありません")

                if not isinstance(data[key], (int, float)):
                    raise ValueError(f"{hierarchy_name}の '{key}' が数値型ではありません")