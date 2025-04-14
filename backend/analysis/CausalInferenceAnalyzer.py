# backend/analysis.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Literal
import logging
import warnings
import pymc as pm
import arviz as az
import cvxpy as cp
from causalimpact import CausalImpact
import dowhy
from dowhy import CausalModel
from dataclasses import dataclass
from datetime import datetime
import gc
import os
import tempfile
import joblib
import weakref
import shutil
from contextlib import contextmanager
import time

# EconMLのインポート
from econml.dml import CausalForestDML, LinearDML
from econml.dr import DRLearner
from econml.inference import BootstrapInference
from econml.metalearners import TLearner, SLearner, XLearner

# 独自のロガー設定
logger = logging.getLogger(__name__)

@dataclass
class CausalImpactResult:
    """因果推論分析の結果を格納するクラス"""
    point_effect: float  # 推定された効果（ポイント推定値）
    confidence_interval: Tuple[float, float]  # 信頼区間
    posterior_samples: Optional[np.ndarray] = None  # ベイズ推論の場合の事後分布サンプル
    p_value: Optional[float] = None  # 仮説検定のp値
    model_summary: Optional[Dict] = None  # モデルの要約統計量
    counterfactual_series: Optional[pd.Series] = None  # 反事実予測系列
    effect_series: Optional[pd.Series] = None  # 効果の時系列
    cumulative_effect: Optional[float] = None  # 累積効果

    def to_dict(self) -> Dict:
        """結果を辞書形式に変換"""
        return {
            "point_effect": self.point_effect,
            "confidence_interval": self.confidence_interval,
            "p_value": self.p_value,
            "cumulative_effect": self.cumulative_effect,
            # その他必要に応じて追加
        }

@dataclass
class HeterogeneousTreatmentEffectResult:
    """異質処理効果（CATE）分析の結果を格納するクラス"""
    model_type: str  # 使用したモデルタイプ（例: 'causal_forest', 'linear_dml'）
    average_effect: float  # 平均処理効果（ATE）
    conditional_effects: np.ndarray  # 条件付き平均処理効果（CATE）
    feature_importance: Optional[Dict[str, float]] = None  # 特徴量重要度
    confidence_intervals: Optional[np.ndarray] = None  # 信頼区間
    p_values: Optional[np.ndarray] = None  # p値
    model_instance: Optional[Any] = None  # モデルインスタンス

    def to_dict(self) -> Dict:
        """結果を辞書形式に変換"""
        result = {
            "model_type": self.model_type,
            "average_effect": self.average_effect,
            "conditional_effects_summary": {
                "mean": float(np.mean(self.conditional_effects)),
                "median": float(np.median(self.conditional_effects)),
                "min": float(np.min(self.conditional_effects)),
                "max": float(np.max(self.conditional_effects)),
                "std": float(np.std(self.conditional_effects))
            }
        }

        if self.feature_importance is not None:
            result["feature_importance"] = self.feature_importance

        return result

class CausalInferenceAnalyzer:
    """
    因果推論分析モジュール

    Startup Wellness プログラムの効果分析と、VC特化ROI計算のための
    時系列因果推論を実装するクラス
    """

    def __init__(self, storage_mode: Literal['memory', 'disk', 'hybrid'] = 'memory',
                 temp_dir: Optional[str] = None, max_memory_size: int = 1000):
        """
        初期化

        Parameters:
        -----------
        storage_mode : str
            結果保存モード ('memory'/'disk'/'hybrid')
            - memory: すべてメモリに保存（小・中規模データに適合）
            - disk: 大きなデータや中間結果をディスクに保存（大規模データに適合）
            - hybrid: 統計情報はメモリに、詳細データはディスクに保存
        temp_dir : str, optional
            一時ファイル保存ディレクトリ（Noneの場合は自動生成）
        max_memory_size : int, optional
            メモリ使用量の目安値（MB単位）
        """
        self.logger = logger
        self.logger.info("因果推論分析モジュールを初期化しました")

        # ストレージ設定
        self.storage_mode = storage_mode
        self.max_memory_size = max_memory_size

        # 一時ディレクトリの設定
        if storage_mode in ['disk', 'hybrid']:
            if temp_dir:
                os.makedirs(temp_dir, exist_ok=True)
                self.temp_dir = temp_dir
            else:
                self.temp_dir = tempfile.mkdtemp(prefix="causal_analysis_")
            self.logger.info(f"一時ディレクトリを作成しました: {self.temp_dir}")
        else:
            self.temp_dir = None

        # 一時ファイル追跡用
        self.temp_files = set()

        # モデルインスタンス追跡用
        self.model_cache = weakref.WeakValueDictionary()

        # 結果キャッシュ
        self._result_cache = {}

    def __del__(self):
        """デストラクタ - リソースの自動解放"""
        self.release_resources()

    def release_resources(self):
        """リソースを明示的に解放"""
        # キャッシュのクリア
        self._result_cache.clear()
        self.model_cache.clear()

        # 一時ファイルの削除
        self._cleanup_temp_files()

        # 一時ディレクトリの削除
        if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"一時ディレクトリを削除しました: {self.temp_dir}")
            except Exception as e:
                self.logger.warning(f"一時ディレクトリの削除に失敗: {str(e)}")

        # メモリの明示的解放
        gc.collect()
        self.logger.info("リソースを解放しました")

    def _cleanup_temp_files(self):
        """一時ファイルのクリーンアップ"""
        for file_path in self.temp_files.copy():
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    self.temp_files.remove(file_path)
                    self.logger.debug(f"一時ファイルを削除しました: {file_path}")
                except Exception as e:
                    self.logger.warning(f"一時ファイル削除に失敗: {str(e)}")

    @contextmanager
    def _managed_data(self, data):
        """データの効率的な管理のためのコンテキストマネージャー"""
        try:
            yield data
        finally:
            # 明示的にオブジェクト参照を削除
            del data
            # メモリ使用量が閾値を超えた場合にガベージコレクションを強制実行
            if self._check_memory_usage():
                gc.collect()

    def _check_memory_usage(self) -> bool:
        """
        メモリ使用量をチェックし、閾値を超えているか判断

        Returns:
        --------
        bool
            メモリ使用量が閾値を超えている場合はTrue
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / (1024 * 1024)

            # 閾値を超えているかチェック
            if memory_usage_mb > self.max_memory_size:
                self.logger.warning(f"メモリ使用量が閾値を超えています: {memory_usage_mb:.1f}MB > {self.max_memory_size}MB")
                return True
            return False
        except ImportError:
            # psutilが利用できない場合は単純にFalseを返す
            return False

    def _save_to_temp_file(self, data, prefix="data", suffix=".pkl"):
        """
        データを一時ファイルに保存

        Parameters:
        -----------
        data : Any
            保存するデータ
        prefix : str
            ファイル名のプレフィックス
        suffix : str
            ファイル名のサフィックス

        Returns:
        --------
        str
            保存したファイルのパス
        """
        if not self.temp_dir:
            raise ValueError("一時ディレクトリが設定されていません")

        file_path = os.path.join(self.temp_dir, f"{prefix}_{int(time.time())}_{id(data)}{suffix}")

        # データサイズに基づいて保存方法を選択
        try:
            # メモリ消費を抑えるために適切な方法で保存
            joblib.dump(data, file_path, compress=3, protocol=4)
            self.temp_files.add(file_path)
            self.logger.debug(f"データを一時ファイルに保存しました: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"データの一時ファイル保存に失敗: {str(e)}")
            raise

    def _load_from_temp_file(self, file_path):
        """
        一時ファイルからデータを読み込み

        Parameters:
        -----------
        file_path : str
            読み込むファイルのパス

        Returns:
        --------
        Any
            読み込んだデータ
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"指定されたファイルが見つかりません: {file_path}")

        try:
            data = joblib.load(file_path)
            return data
        except Exception as e:
            self.logger.error(f"一時ファイルからの読み込みに失敗: {str(e)}")
            raise

    def _save_model(self, model, model_id=None):
        """
        モデルを保存（メモリまたはディスク）

        Parameters:
        -----------
        model : Any
            保存するモデルインスタンス
        model_id : str, optional
            モデルの識別子

        Returns:
        --------
        str or Any
            ディスク保存の場合はファイルパス、メモリ保存の場合はモデル自体
        """
        if model_id is None:
            model_id = str(id(model))

        if self.storage_mode == 'memory':
            # メモリに保存
            self.model_cache[model_id] = model
            return model
        else:
            # ディスクに保存
            file_path = self._save_to_temp_file(model, prefix=f"model_{model_id}")
            return file_path

    def _load_model(self, model_ref, model_id=None):
        """
        保存したモデルを読み込み

        Parameters:
        -----------
        model_ref : str or Any
            モデルの参照（ファイルパスまたはモデル自体）
        model_id : str, optional
            モデルの識別子

        Returns:
        --------
        Any
            読み込んだモデルインスタンス
        """
        if isinstance(model_ref, str) and os.path.exists(model_ref):
            # ディスクから読み込み
            return self._load_from_temp_file(model_ref)
        else:
            # メモリから取得
            return model_ref

    def _manage_result(self, result, result_type, save_to_disk=False):
        """
        分析結果の管理（メモリ効率のため、必要に応じてディスクに保存）

        Parameters:
        -----------
        result : Any
            管理する結果オブジェクト
        result_type : str
            結果の種類を表す識別子
        save_to_disk : bool, optional
            強制的にディスクに保存するかどうか

        Returns:
        --------
        Any or str
            結果オブジェクトまたはディスクパス
        """
        # ハイブリッドモードや大きな結果データは、ディスクに保存
        if save_to_disk or self.storage_mode in ['disk', 'hybrid']:
            if hasattr(result, 'posterior_samples') and result.posterior_samples is not None:
                # ベイジアン分析の後続サンプルは特に大きいので別途保存
                posterior_path = self._save_to_temp_file(
                    result.posterior_samples,
                    prefix=f"posterior_{result_type}"
                )
                # 参照をパスに置き換え
                result.posterior_samples = posterior_path

            # 結果全体を保存
            result_path = self._save_to_temp_file(result, prefix=f"result_{result_type}")
            return result_path
        else:
            # メモリに保存
            return result

    def _get_result(self, result_ref):
        """
        結果参照から結果オブジェクトを取得

        Parameters:
        -----------
        result_ref : Any or str
            結果オブジェクトまたはディスクパス

        Returns:
        --------
        Any
            結果オブジェクト
        """
        if isinstance(result_ref, str) and os.path.exists(result_ref):
            # ディスクから読み込み
            result = self._load_from_temp_file(result_ref)

            # posterior_samplesが参照であれば読み込み
            if hasattr(result, 'posterior_samples') and isinstance(result.posterior_samples, str) and os.path.exists(result.posterior_samples):
                result.posterior_samples = self._load_from_temp_file(result.posterior_samples)

            return result
        else:
            # メモリオブジェクトをそのまま返す
            return result_ref

    def estimate_memory_usage(self, data_rows, data_cols, method='causal_impact_bayesian'):
        """
        メモリ使用量の概算と最適なストレージモードの推奨

        Parameters:
        -----------
        data_rows : int
            データの行数
        data_cols : int
            データの列数
        method : str
            使用する分析手法

        Returns:
        --------
        Dict
            メモリ使用量の概算と推奨設定
        """
        # 単一データポイントの概算サイズ (バイト単位)
        bytes_per_element = 8  # 浮動小数点数

        # 入力データサイズ
        input_size_bytes = data_rows * data_cols * bytes_per_element

        # 中間データと結果の推定サイズ
        if method == 'causal_impact':
            # 状態空間モデルの中間データ
            multiplier = 3
        elif method == 'causal_impact_bayesian':
            # ベイズサンプリングの結果は元データよりもずっと大きい
            multiplier = 20
        elif method == 'synthetic_control':
            multiplier = 2
        elif method == 'estimate_heterogeneous_treatment_effects':
            # 異質効果推定はモデル複雑性でメモリ使用量が増加
            multiplier = 5
        else:
            multiplier = 2

        # 総メモリ使用量の推定 (MB単位)
        estimated_memory_mb = (input_size_bytes * multiplier) / (1024 * 1024)

        # 推奨ストレージモード
        if estimated_memory_mb > self.max_memory_size:
            if estimated_memory_mb > self.max_memory_size * 3:
                recommended_mode = 'disk'
            else:
                recommended_mode = 'hybrid'
        else:
            recommended_mode = 'memory'

        return {
            'estimated_memory_mb': estimated_memory_mb,
            'recommended_storage_mode': recommended_mode,
            'max_allowed_memory_mb': self.max_memory_size,
            'data_size_mb': input_size_bytes / (1024 * 1024)
        }

    @contextmanager
    def _progress_context(self, description="処理中", total=None):
        """処理の進捗を管理するコンテキストマネージャー"""
        start_time = time.time()
        self.logger.info(f"{description}を開始")

        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.logger.info(f"{description}が完了しました（経過時間: {elapsed:.2f}秒）")

    def analyze_difference_in_differences(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        time_col: str,
        outcome_col: str,
        covariates: List[str] = None
    ) -> CausalImpactResult:
        """
        差分の差分法(DiD)による因果効果の推定

        Parameters:
        -----------
        data : pd.DataFrame
            分析対象データ
        treatment_col : str
            処置変数のカラム名（0/1）
        time_col : str
            時間変数のカラム名（処置前=0, 処置後=1）
        outcome_col : str
            アウトカム変数のカラム名
        covariates : List[str], optional
            制御変数のリスト

        Returns:
        --------
        CausalImpactResult
            因果効果の推定結果
        """
        self.logger.info(f"差分の差分法による分析を開始: 対象変数={outcome_col}")

        try:
            # コピーを作成して元データを変更しない
            df = data.copy()

            # 交互作用項を作成
            df['treatment_time'] = df[treatment_col] * df[time_col]

            # 回帰式の構築
            formula = f"{outcome_col} ~ {treatment_col} + {time_col} + treatment_time"

            # 共変量がある場合は追加
            if covariates and len(covariates) > 0:
                cov_formula = " + " + " + ".join(covariates)
                formula += cov_formula

            # モデルの推定
            model = sm.formula.ols(formula=formula, data=df)
            results = model.fit(cov_type='HC1')  # 頑健な標準誤差を使用

            # 処置効果の抽出（交互作用項の係数）
            effect = results.params['treatment_time']
            ci = results.conf_int().loc['treatment_time'].tolist()
            p_value = results.pvalues['treatment_time']

            # 結果の整形
            impact_result = CausalImpactResult(
                point_effect=effect,
                confidence_interval=tuple(ci),
                p_value=p_value,
                model_summary=results.summary().as_dict()
            )

            self.logger.info(f"差分の差分法による分析が完了: 効果={effect}, p値={p_value}")
            return impact_result

        except Exception as e:
            self.logger.error(f"差分の差分法による分析中にエラーが発生: {str(e)}")
            raise

    def analyze_synthetic_control(
        self,
        data: pd.DataFrame,
        target_unit: str,
        control_units: List[str],
        time_col: str,
        outcome_col: str,
        pre_period: List[str],
        post_period: List[str]
    ) -> CausalImpactResult:
        """
        合成コントロール法による因果効果の推定

        Parameters:
        -----------
        data : pd.DataFrame
            パネルデータ形式のデータフレーム
        target_unit : str
            処置を受けたユニット（企業）の識別子
        control_units : List[str]
            コントロールプールのユニット（企業）識別子リスト
        time_col : str
            時間変数のカラム名
        outcome_col : str
            アウトカム変数のカラム名
        pre_period : List[str]
            処置前期間 [開始日, 終了日]
        post_period : List[str]
            処置後期間 [開始日, 終了日]

        Returns:
        --------
        CausalImpactResult
            因果効果の推定結果
        """
        self.logger.info(f"合成コントロール法による分析を開始: 対象ユニット={target_unit}")

        try:
            # データの準備
            df = data.copy()
            pre_mask = (df[time_col] >= pre_period[0]) & (df[time_col] <= pre_period[1])
            post_mask = (df[time_col] >= post_period[0]) & (df[time_col] <= post_period[1])

            # 分析対象ユニットと対照ユニットの選択
            target_pre = df[pre_mask & (df.index == target_unit)][outcome_col].values
            target_post = df[post_mask & (df.index == target_unit)][outcome_col].values

            # コントロールユニットのデータ
            control_pre = np.vstack([
                df[pre_mask & (df.index == unit)][outcome_col].values
                for unit in control_units
            ])

            # 重みの最適化（二次計画法）
            # cvxpyを使用した実装

            # 決定変数: 重み
            weights = cp.Variable(len(control_units))

            # 目的関数: 二乗誤差の最小化
            objective = cp.Minimize(cp.sum_squares(target_pre - control_pre.T @ weights))

            # 制約条件
            constraints = [
                cp.sum(weights) == 1,  # 重みの合計が1
                weights >= 0  # 非負制約
            ]

            # 問題の定義と解決
            problem = cp.Problem(objective, constraints)
            problem.solve()

            if problem.status != cp.OPTIMAL:
                raise ValueError(f"最適化に失敗しました: {problem.status}")

            # 最適な重み
            optimal_weights = weights.value

            # 合成コントロールの予測値
            control_post = np.vstack([
                df[post_mask & (df.index == unit)][outcome_col].values
                for unit in control_units
            ])
            synthetic_post = control_post.T @ optimal_weights

            # 処置効果の計算
            effect = np.mean(target_post - synthetic_post)

            # 結果の保存
            times_post = df[post_mask & (df.index == target_unit)][time_col].values
            counterfactual = pd.Series(synthetic_post, index=times_post)
            actual = pd.Series(target_post, index=times_post)
            effect_series = actual - counterfactual
            cumulative_effect = effect_series.sum()

            # 信頼区間は実際にはプラセボテストに基づいて計算するが
            # ここでは簡易的に標準偏差の2倍を使用
            std_dev = np.std(effect_series)
            ci = (effect - 2 * std_dev, effect + 2 * std_dev)

            result = CausalImpactResult(
                point_effect=effect,
                confidence_interval=ci,
                counterfactual_series=counterfactual,
                effect_series=effect_series,
                cumulative_effect=cumulative_effect
            )

            self.logger.info(f"合成コントロール法による分析が完了: 効果={effect}")
            return result

        except Exception as e:
            self.logger.error(f"合成コントロール法による分析中にエラーが発生: {str(e)}")
            raise

    def analyze_causal_impact(
        self,
        time_series: pd.DataFrame,
        intervention_time: str,
        target_col: str,
        control_cols: List[str] = None,
        model_args: Dict = None
    ) -> CausalImpactResult:
        """
        Google CausalImpactに基づく時系列因果推論

        Parameters:
        -----------
        time_series : pd.DataFrame
            時系列データ（インデックスは日付）
        intervention_time : str
            介入時点（日付文字列）
        target_col : str
            ターゲット変数（効果を測定したい変数）のカラム名
        control_cols : List[str], optional
            コントロール変数（予測モデルに使用する変数）のカラム名リスト
        model_args : Dict, optional
            モデルのパラメータ

        Returns:
        --------
        CausalImpactResult
            因果効果の推定結果
        """
        self.logger.info(f"CausalImpactによる分析を開始: 対象変数={target_col}, 介入時点={intervention_time}")

        try:
            # データの準備
            df = time_series.copy()

            # 日付型への変換
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            intervention_date = pd.to_datetime(intervention_time)

            # 介入前後の分割点
            pre_period = [df.index.min(), intervention_date - pd.Timedelta(days=1)]
            post_period = [intervention_date, df.index.max()]

            # 使用するカラムの選択
            if control_cols and len(control_cols) > 0:
                data = df[[target_col] + control_cols]
            else:
                data = df[[target_col]]

            # CausalImpactライブラリを使用
            ci = CausalImpact(data, pre_period, post_period, model_args=model_args)

            # 結果の抽出
            summary = ci.summary()
            report = ci.summary(output='report')

            # 効果の抽出
            point_effect = ci.summary_data.loc['average', 'abs_effect']
            ci_lower = ci.summary_data.loc['average', 'abs_effect_lower']
            ci_upper = ci.summary_data.loc['average', 'abs_effect_upper']
            p_value = ci.summary_data.loc['average', 'p']

            # 累積効果
            cumulative_effect = ci.summary_data.loc['cumulative', 'abs_effect']

            # 時系列データの抽出
            counterfactual = ci.inferences['series'].prediction
            actual = ci.inferences['series'].response
            effect = ci.inferences['series'].point_effect

            result = CausalImpactResult(
                point_effect=point_effect,
                confidence_interval=(ci_lower, ci_upper),
                p_value=p_value,
                counterfactual_series=counterfactual,
                effect_series=effect,
                cumulative_effect=cumulative_effect,
                model_summary={'report': report, 'summary': summary}
            )

            self.logger.info(f"CausalImpactによる分析が完了: 効果={point_effect}, p値={p_value}")
            return result

        except Exception as e:
            self.logger.error(f"CausalImpactによる分析中にエラーが発生: {str(e)}")
            raise

    def analyze_causal_impact_bayesian(
        self,
        time_series: pd.DataFrame,
        intervention_time: str,
        target_col: str,
        control_cols: List[str] = None,
        model_args: Dict = None,
        batch_size: int = 200,
        progress_callback: callable = None
    ) -> CausalImpactResult:
        """
        PyMCを使用したベイズ時系列因果推論

        大規模データセットに最適化されたバージョン

        Parameters:
        -----------
        time_series : pd.DataFrame
            時系列データ（インデックスは日付）
        intervention_time : str
            介入時点（日付文字列）
        target_col : str
            ターゲット変数（効果を測定したい変数）のカラム名
        control_cols : List[str], optional
            コントロール変数（予測モデルに使用する変数）のカラム名リスト
        model_args : Dict, optional
            モデルのパラメータ
        batch_size : int, optional
            予測バッチのサイズ（メモリ使用量の調整に使用）
        progress_callback : callable, optional
            進捗を報告するコールバック関数

        Returns:
        --------
        CausalImpactResult
            因果効果の推定結果
        """
        with self._progress_context(description=f"ベイズ時系列因果推論分析 - {target_col}", total=100):
            self.logger.info(f"ベイズ時系列因果推論による分析を開始: 対象変数={target_col}, 介入時点={intervention_time}")

            # メモリ使用量の概算
            data_rows = len(time_series)
            data_cols = len(time_series.columns)
            memory_estimate = self.estimate_memory_usage(data_rows, data_cols, 'causal_impact_bayesian')

            if memory_estimate['estimated_memory_mb'] > self.max_memory_size:
                self.logger.warning(
                    f"メモリ使用量が多い可能性があります: 推定{memory_estimate['estimated_memory_mb']:.1f}MB > "
                    f"設定{self.max_memory_size}MB。自動的に効率モードを有効化します。"
                )

            try:
                # 進捗報告 - 10%
                if progress_callback:
                    progress_callback(10, "データの準備中")

                # データの準備 - コピーではなく参照を使用
                with self._managed_data(time_series) as df:
                    # 日付型への変換
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)

                    intervention_date = pd.to_datetime(intervention_time)

                    # 介入前後のデータを分割 - コピーではなく参照を使用
                    pre_data = df.loc[:intervention_time]
                    post_data = df.loc[intervention_time:]

                    # 事前処理で必要なデータのみを抽出
                    if control_cols and len(control_cols) > 0:
                        # 多変量モデル用データ抽出
                        X_pre = pre_data[control_cols].values
                        y_pre = pre_data[target_col].values
                        X_post = post_data[control_cols].values
                    else:
                        # 単変量モデル用データ抽出
                        y_pre = pre_data[target_col].values
                        X_pre = None
                        X_post = None

                    # 進捗報告 - 20%
                    if progress_callback:
                        progress_callback(20, "ベイズモデルの構築中")

                    # モデル構築とサンプリングは一時的に大量のメモリを使用するため、
                    # 別のスコープで実行してすぐに解放できるようにする
                    model_results = self._run_bayesian_model(
                        X_pre, y_pre, control_cols, model_args, progress_callback
                    )

                    # 進捗報告 - 60%
                    if progress_callback:
                        progress_callback(60, "介入後の予測計算中")

                    # 介入後期間の反事実予測 - バッチ処理による最適化
                    post_samples = model_results['trace']

                    # バッチ処理で予測を計算
                    post_pred = self._predict_counterfactual_batches(
                        post_samples, X_post, post_data,
                        control_cols, model_results, batch_size,
                        progress_callback
                    )

                    # 予測の要約
                    predicted_mean = np.mean(post_pred, axis=0)
                    predicted_ci = np.percentile(post_pred, [2.5, 97.5], axis=0).T

                    # 実際の値との差を計算して効果を推定
                    actual = post_data[target_col].values
                    counterfactual = pd.Series(predicted_mean, index=post_data.index)
                    actual_series = pd.Series(actual, index=post_data.index)
                    effect_series = actual_series - counterfactual

                    # 効果の要約統計量
                    point_effect = effect_series.mean()

                    # メモリ効率のためバッチで計算
                    # 事後効果サンプルを元データの配列形式に再構築せず、
                    # 平均値のみを保持
                    posterior_effect_means = np.array([
                        np.mean(actual - post_pred[i])
                        for i in range(len(post_pred))
                    ])

                    # 信頼区間
                    ci = np.percentile(posterior_effect_means, [2.5, 97.5])

                    # 累積効果
                    cumulative_effect = effect_series.sum()

                    # Bayesian p-value（事後確率）
                    # 効果がゼロより大きい（または小さい）確率
                    if point_effect > 0:
                        p_value = (posterior_effect_means <= 0).mean()
                    else:
                        p_value = (posterior_effect_means >= 0).mean()

                    # 進捗報告 - 90%
                    if progress_callback:
                        progress_callback(90, "結果のまとめ中")

                    # メモリ効率のためにposterior_samplesは効果の平均値のみ保存
                    result = CausalImpactResult(
                        point_effect=point_effect,
                        confidence_interval=(ci[0], ci[1]),
                        p_value=p_value,
                        posterior_samples=posterior_effect_means if self.storage_mode == 'memory' else None,
                        counterfactual_series=counterfactual,
                        effect_series=effect_series,
                        cumulative_effect=cumulative_effect,
                        model_summary=model_results.get('summary', {})
                    )

                    # メモリ効率のためにディスクストレージを使用する場合、ここで保存
                    if self.storage_mode != 'memory':
                        result = self._manage_result(result, f"bayes_{target_col}", save_to_disk=True)

                    # 明示的に大きなオブジェクトを解放
                    del post_pred, predicted_mean, predicted_ci, posterior_effect_means
                    gc.collect()

                    # 進捗報告 - 100%
                    if progress_callback:
                        progress_callback(100, "分析完了")

                    self.logger.info(f"ベイズ時系列因果推論による分析が完了: 効果={point_effect}, 事後確率={1-p_value:.3f}")
                    return result

            except Exception as e:
                self.logger.error(f"ベイズ時系列因果推論による分析中にエラーが発生: {str(e)}")
                # 途中で例外が発生した場合でもリソースを解放
                gc.collect()
                raise

    def _run_bayesian_model(self, X_pre, y_pre, control_cols, model_args=None, progress_callback=None):
        """
        ベイズモデルを構築し、MCMCサンプリングを実行

        Parameters:
        -----------
        X_pre : np.ndarray or None
            説明変数の配列（多変量モデルの場合）
        y_pre : np.ndarray
            目的変数の配列
        control_cols : List[str] or None
            コントロール変数のリスト
        model_args : Dict, optional
            モデル構築用の追加パラメータ
        progress_callback : callable, optional
            進捗コールバック関数

        Returns:
        --------
        Dict
            サンプリング結果とモデル情報を含む辞書
        """
        # デフォルトのモデルパラメータ
        if model_args is None:
            model_args = {}

        # サンプリングパラメータの設定
        n_samples = model_args.get('n_samples', 2000)
        n_tune = model_args.get('n_tune', 1000)
        n_chains = model_args.get('n_chains', 2)
        n_cores = model_args.get('n_cores', 1)
        random_seed = model_args.get('random_seed', 42)

        # モデル構築
        with pm.Model() as model:
            # 進捗報告 - 25%
            if progress_callback:
                progress_callback(25, "モデルの事前分布を定義中")

            # 事前分布
            if control_cols and len(control_cols) > 0 and X_pre is not None:
                # 多変量モデル
                coeffs = pm.Normal('coeffs', mu=0, sigma=1, shape=len(control_cols))
                intercept = pm.Normal('intercept', mu=0, sigma=10)
                sigma = pm.HalfCauchy('sigma', beta=1)

                # 線形予測
                mu = intercept + pm.math.dot(X_pre, coeffs)
            else:
                # 単変量モデル - ローカルレベルモデル
                sigma_level = pm.HalfCauchy('sigma_level', beta=1)
                sigma_obs = pm.HalfCauchy('sigma_obs', beta=1)

                # 状態空間表現
                level = pm.GaussianRandomWalk('level', sigma=sigma_level, shape=len(y_pre))
                mu = level
                sigma = sigma_obs

            # 尤度
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y_pre)

            # 進捗報告 - 30%
            if progress_callback:
                progress_callback(30, "サンプリング初期化中")

            # サンプリング - メモリ効率を考慮した設定
            trace = pm.sample(
                n_samples,
                tune=n_tune,
                chains=n_chains,
                cores=n_cores,
                progressbar=False,
                random_seed=random_seed,
                return_inferencedata=False
            )

            # 進捗報告 - 50%
            if progress_callback:
                progress_callback(50, "サンプリング完了")

        # モデル結果をまとめる
        try:
            # 結果の要約
            summary = az.summary(trace)
            summary_dict = summary.to_dict()
        except Exception as e:
            self.logger.warning(f"モデル要約の生成に失敗: {str(e)}")
            summary_dict = {"error": str(e)}

        # モデル変数情報を抽出
        var_names = trace.varnames
        model_info = {
            'has_coeffs': 'coeffs' in var_names,
            'has_intercept': 'intercept' in var_names,
            'has_level': 'level' in var_names
        }

        # トレースをディスクに保存するかどうかを決定
        if self.storage_mode != 'memory':
            trace_path = self._save_to_temp_file(trace, prefix="bayesian_trace")
            trace = trace_path

        return {
            'trace': trace,
            'summary': summary_dict,
            'model_info': model_info
        }

    def _predict_counterfactual_batches(self, trace, X_post, post_data, control_cols, model_results,
                                      batch_size=200, progress_callback=None):
        """
        バッチ処理による反事実予測

        Parameters:
        -----------
        trace : MultiTrace or str
            PyMCサンプリングのトレース、またはトレースファイルへのパス
        X_post : np.ndarray or None
            介入後期間の説明変数
        post_data : pd.DataFrame
            介入後期間のデータ
        control_cols : List[str] or None
            コントロール変数リスト
        model_results : Dict
            モデル構築結果
        batch_size : int
            バッチサイズ
        progress_callback : callable, optional
            進捗コールバック関数

        Returns:
        --------
        np.ndarray
            反事実予測値の配列
        """
        # トレースの読み込み（必要に応じて）
        if isinstance(trace, str) and os.path.exists(trace):
            trace = self._load_from_temp_file(trace)

        model_info = model_results['model_info']
        post_len = len(post_data)

        # モデルタイプに基づいてサンプル数を決定
        if hasattr(trace, 'nchains'):
            n_samples = len(trace) * trace.nchains
        else:
            # ArviZ InferenceData形式
            n_samples = len(trace)

        # トレースから必要なパラメータを抽出
        if model_info['has_coeffs'] and model_info['has_intercept'] and X_post is not None:
            # 多変量モデル - パラメータ抽出
            all_params = []

            # バッチでパラメータを抽出（メモリ効率向上）
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_indices = range(i, batch_end)

                # パラメータの抽出
                batch_params = []
                for idx in batch_indices:
                    sample_idx = idx % len(trace)
                    chain_idx = idx // len(trace) if hasattr(trace, 'nchains') else 0

                    if hasattr(trace, 'get_values'):
                        # PyMC3互換
                        coeffs = trace.get_values('coeffs', chains=chain_idx)[sample_idx]
                        intercept = trace.get_values('intercept', chains=chain_idx)[sample_idx]
                        sigma = trace.get_values('sigma', chains=chain_idx)[sample_idx]
                    else:
                        # PyMC新バージョン
                        coeffs = trace['coeffs'][sample_idx]
                        intercept = trace['intercept'][sample_idx]
                        sigma = trace['sigma'][sample_idx]

                    batch_params.append((coeffs, intercept, sigma))

                all_params.extend(batch_params)

                # 進捗報告
                if progress_callback:
                    progress_percent = 60 + (i / n_samples) * 20
                    progress_callback(int(progress_percent), f"パラメータ抽出中 {i}/{n_samples}")

            # 予測計算用の配列を初期化
            post_pred = np.zeros((n_samples, post_len))

            # バッチで予測を計算
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_size_actual = batch_end - i

                # このバッチのパラメータ
                batch_params = all_params[i:batch_end]

                # 各サンプルの予測値を計算
                for j, (coeffs, intercept, sigma) in enumerate(batch_params):
                    # 予測平均
                    mu = intercept + np.dot(X_post, coeffs)

                    # 予測からのサンプリング
                    post_pred[i+j] = np.random.normal(mu, sigma)

                # 進捗報告
                if progress_callback:
                    progress_percent = 80 + (i / n_samples) * 10
                    progress_callback(int(progress_percent), f"予測計算中 {i}/{n_samples}")

                # 一時的なメモリ解放
                if i % (batch_size * 5) == 0:
                    gc.collect()

        else:
            # 単変量モデル - ローカルレベルモデルの予測
            # トレースから最後のレベル値を抽出
            if hasattr(trace, 'get_values'):
                level_samples = trace.get_values('level', chains=None)
            else:
                level_samples = trace['level']

            # 各サンプルの最後のレベル値
            last_levels = np.array([sample[-1] for sample in level_samples])

            if hasattr(trace, 'get_values'):
                sigma_level_samples = trace.get_values('sigma_level', chains=None)
                sigma_obs_samples = trace.get_values('sigma_obs', chains=None)
            else:
                sigma_level_samples = trace['sigma_level']
                sigma_obs_samples = trace['sigma_obs']

            # 予測計算用の配列を初期化
            post_pred = np.zeros((n_samples, post_len))

            # バッチで予測を計算
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_indices = range(i, batch_end)

                for j, idx in enumerate(batch_indices):
                    # ランダムウォークを継続
                    last_level = last_levels[idx % len(last_levels)]
                    sigma_level = sigma_level_samples[idx % len(sigma_level_samples)]
                    sigma_obs = sigma_obs_samples[idx % len(sigma_obs_samples)]

                    # 新しいレベルをシミュレート
                    new_levels = np.zeros(post_len)
                    new_levels[0] = last_level + np.random.normal(0, sigma_level)

                    for t in range(1, post_len):
                        new_levels[t] = new_levels[t-1] + np.random.normal(0, sigma_level)

                    # 観測値をシミュレート
                    post_pred[i+j] = new_levels + np.random.normal(0, sigma_obs, post_len)

                # 進捗報告
                if progress_callback:
                    progress_percent = 80 + (i / n_samples) * 10
                    progress_callback(int(progress_percent), f"予測計算中 {i}/{n_samples}")

                # 一時的なメモリ解放
                if i % (batch_size * 5) == 0:
                    gc.collect()

        return post_pred

    def estimate_revenue_impact(
        self,
        revenue_data: pd.DataFrame,
        intervention_data: pd.DataFrame,
        company_id: str,
        intervention_date: str,
        control_features: List[str] = None,
        method: str = 'causal_impact',
        progress_callback: callable = None,
        storage_mode: str = None
    ) -> Dict:
        """
        Startup Wellnessプログラムによる売上への影響（ΔRevenue）を推定

        メモリ効率化版

        Parameters:
        -----------
        revenue_data : pd.DataFrame
            売上データ（時系列、複数企業）
        intervention_data : pd.DataFrame
            介入データ（Startup Wellnessプログラムの開始日などの情報）
        company_id : str
            分析対象の企業ID
        intervention_date : str
            介入日（Startup Wellnessプログラム開始日）
        control_features : List[str], optional
            コントロール変数（共変量）のリスト
        method : str, optional
            使用する因果推論手法（'causal_impact', 'causal_impact_bayesian', 'synthetic_control', 'did'のいずれか）
        progress_callback : callable, optional
            進捗報告コールバック関数
        storage_mode : str, optional
            この分析用の一時的なストレージモード（Noneの場合はクラスのデフォルト設定を使用）

        Returns:
        --------
        Dict
            推定結果の辞書
            - delta_revenue: 推定された売上増加額
            - confidence_interval: 信頼区間
            - relative_impact: 相対的な影響度（%）
            - details: 詳細な分析結果
        """
        with self._progress_context(description=f"売上影響度推定 - {company_id} - {method}"):
            self.logger.info(f"売上影響度の推定を開始: 企業ID={company_id}, 介入日={intervention_date}")

            # この分析のための一時的なストレージモード設定
            original_storage_mode = self.storage_mode
            if storage_mode is not None:
                self.storage_mode = storage_mode

            try:
                # 進捗報告 - 10%
                if progress_callback:
                    progress_callback(10, "データの準備中")

                # 対象企業のデータを抽出 - 不要なコピーを避ける
                with self._managed_data(revenue_data) as rd:
                    company_revenue = rd[rd['company_id'] == company_id]

                    # 日付列を確認し、必要に応じて変換
                    date_cols = [col for col in company_revenue.columns if 'date' in col.lower()]
                    if not date_cols:
                        raise ValueError("日付カラムが見つかりません")

                    date_col = date_cols[0]

                    # 日付でソート
                    company_revenue = company_revenue.sort_values(date_col)

                    # 日付をインデックスに設定
                    if date_col != company_revenue.index.name:
                        company_revenue = company_revenue.set_index(date_col)

                    # 介入日以降のデータが十分にあるか確認
                    if len(company_revenue.loc[intervention_date:]) < 3:
                        raise ValueError("介入後のデータポイントが不足しています（最低3ポイント必要）")

                    # 進捗報告 - 20%
                    if progress_callback:
                        progress_callback(20, "分析手法の準備中")

                    result = None

                    if method == 'causal_impact' or method == 'causal_impact_bayesian':
                        # 売上カラムを特定
                        revenue_cols = [col for col in company_revenue.columns if 'revenue' in col.lower()]
                        if not revenue_cols:
                            raise ValueError("売上データのカラムが見つかりません")

                        revenue_col = revenue_cols[0]  # 最初の売上カラムを使用

                        # コントロール変数の準備
                        if not control_features:
                            # 進捗報告 - 30%
                            if progress_callback:
                                progress_callback(30, "コントロール変数の構築中")

                            # コントロール変数が指定されていない場合、同業他社のデータを使用
                            other_companies = revenue_data[revenue_data['company_id'] != company_id]

                            # メモリ効率向上のため、必要な列のみを取得
                            other_companies = other_companies[[date_col, 'company_id', revenue_col]]

                            # 他社の売上データを集約（例: 平均売上）
                            other_companies_pivot = other_companies.pivot_table(
                                index=date_col,
                                columns='company_id',
                                values=revenue_col,
                                aggfunc='mean'
                            )

                            # 大量のコントロール変数がある場合は上位のみ使用
                            if other_companies_pivot.shape[1] > 20:
                                # 相関係数に基づいて上位のコントロール変数を選択
                                corr_with_target = {}
                                target_series = company_revenue[revenue_col]

                                for col in other_companies_pivot.columns:
                                    # 両方のデータがある部分のみを使用して相関を計算
                                    common_idx = target_series.index.intersection(other_companies_pivot.index)
                                    if len(common_idx) > 0:
                                        corr = np.abs(np.corrcoef(
                                            target_series.loc[common_idx],
                                            other_companies_pivot.loc[common_idx, col]
                                        )[0, 1])
                                        if not np.isnan(corr):
                                            corr_with_target[col] = corr

                                # 相関が高い順にソート
                                sorted_controls = sorted(
                                    corr_with_target.items(),
                                    key=lambda x: x[1],
                                    reverse=True
                                )

                                # 上位20社のみを使用
                                top_controls = [c[0] for c in sorted_controls[:20]]
                                other_companies_pivot = other_companies_pivot[top_controls]
                                self.logger.info(f"メモリ効率化のため、上位20社のみをコントロール変数として使用します")

                            # ターゲット企業のデータとマージ
                            merged_data = pd.DataFrame(company_revenue[revenue_col])
                            merged_data = merged_data.join(other_companies_pivot, how='inner')
                            control_cols = other_companies_pivot.columns.tolist()
                        else:
                            # 指定されたコントロール変数を使用
                            control_cols = [col for col in control_features if col in company_revenue.columns]
                            merged_data = company_revenue[[revenue_col] + control_cols]

                        # 進捗報告 - 40%
                        if progress_callback:
                            progress_callback(40, "因果推論分析の実行中")

                        # 因果推論分析の実行
                        if method == 'causal_impact':
                            result = self.analyze_causal_impact(
                                time_series=merged_data,
                                intervention_time=intervention_date,
                                target_col=revenue_col,
                                control_cols=control_cols
                            )
                        else:  # method == 'causal_impact_bayesian'
                            result = self.analyze_causal_impact_bayesian(
                                time_series=merged_data,
                                intervention_time=intervention_date,
                                target_col=revenue_col,
                                control_cols=control_cols,
                                progress_callback=lambda p, m: progress_callback(40 + p * 0.5, m) if progress_callback else None
                            )

                    elif method == 'synthetic_control':
                        # 進捗報告 - 40%
                        if progress_callback:
                            progress_callback(40, "合成コントロール法分析の準備中")

                        # 合成コントロール法による分析
                        # メモリ効率化: 必要な列のみ使用
                        revenue_cols = [col for col in revenue_data.columns if 'revenue' in col.lower()]
                        if not revenue_cols:
                            raise ValueError("売上データのカラムが見つかりません")

                        revenue_col = revenue_cols[0]

                        # パネルデータ構築の効率化
                        panel_data = revenue_data.pivot(
                            index='company_id',
                            columns=date_col,
                            values=revenue_col
                        )

                        # 介入前後の期間を定義
                        all_dates = sorted(panel_data.columns)
                        try:
                            intervention_idx = all_dates.index(intervention_date)
                        except ValueError:
                            # 正確な日付が見つからない場合は最も近い日付を使用
                            intervention_dates = [d for d in all_dates if d >= intervention_date]
                            if not intervention_dates:
                                raise ValueError(f"介入日付 {intervention_date} 以降のデータがありません")
                            intervention_date = intervention_dates[0]
                            intervention_idx = all_dates.index(intervention_date)

                        pre_period = [all_dates[0], all_dates[intervention_idx-1]]
                        post_period = [all_dates[intervention_idx], all_dates[-1]]

                        # 対照企業の選定（介入を受けていない企業）
                        intervention_companies = intervention_data['company_id'].unique().tolist()
                        control_units = [c for c in panel_data.index.unique()
                                       if c != company_id and c not in intervention_companies]

                        # メモリ効率化: 必要なデータのみを含むパネルデータの構築
                        reduced_panel = panel_data.loc[[company_id] + control_units]

                        # パネルデータを長形式に変換
                        long_data = reduced_panel.reset_index().melt(
                            id_vars='company_id',
                            var_name=date_col,
                            value_name=revenue_col
                        )

                        # 進捗報告 - 60%
                        if progress_callback:
                            progress_callback(60, "合成コントロール法分析の実行中")

                        # 合成コントロール法による分析実行
                        result = self.analyze_synthetic_control(
                            data=long_data,
                            target_unit=company_id,
                            control_units=control_units,
                            time_col=date_col,
                            outcome_col=revenue_col,
                            pre_period=pre_period,
                            post_period=post_period
                        )

                    elif method == 'did':
                        # 進捗報告 - 40%
                        if progress_callback:
                            progress_callback(40, "差分の差分法分析の準備中")

                        # 差分の差分法による分析
                        # 処置グループ（対象企業）と対照グループ（その他企業）の設定
                        did_data = revenue_data.copy()
                        did_data['treatment'] = did_data['company_id'] == company_id

                        # 処置前後の期間設定
                        did_data['post'] = pd.to_datetime(did_data[date_col]) >= pd.to_datetime(intervention_date)

                        # 進捗報告 - 60%
                        if progress_callback:
                            progress_callback(60, "差分の差分法分析の実行中")

                        # DiD分析の実行
                        result = self.analyze_difference_in_differences(
                            data=did_data,
                            treatment_col='treatment',
                            time_col='post',
                            outcome_col='revenue',
                            covariates=control_features
                        )

                    # 進捗報告 - 80%
                    if progress_callback:
                        progress_callback(80, "結果の分析中")

                    # 基準期間の平均売上（介入前）を計算
                    pre_avg_revenue = company_revenue.loc[:intervention_date]['revenue'].mean()

                    # 相対的な影響度（%）を計算
                    relative_impact = (result.point_effect / pre_avg_revenue) * 100

                    # 結果をまとめる
                    output = {
                        'delta_revenue': result.point_effect,
                        'confidence_interval': result.confidence_interval,
                        'relative_impact': relative_impact,
                        'cumulative_effect': result.cumulative_effect,
                        'details': {
                            'method': method,
                            'model_summary': result.model_summary,
                            'p_value': result.p_value,
                        }
                    }

                    # 進捗報告 - 100%
                    if progress_callback:
                        progress_callback(100, "分析完了")

                    self.logger.info(f"売上影響度の推定が完了: ΔRevenue={result.point_effect}, 相対影響度={relative_impact}%")
                    return output

            except Exception as e:
                self.logger.error(f"売上影響度の推定中にエラーが発生: {str(e)}")
                # リソース解放
                gc.collect()
                raise
            finally:
                # ストレージモードを元に戻す
                if storage_mode is not None:
                    self.storage_mode = original_storage_mode

    def __enter__(self):
        """コンテキストマネージャプロトコルをサポート"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストブロック終了時にリソースを解放"""
        self.release_resources()
        return False  # 例外が発生した場合は再送出

    def calculate_roi_components(
        self,
        company_id: str,
        start_date: str,
        end_date: str,
        revenue_data: pd.DataFrame,
        valuation_data: pd.DataFrame,
        program_cost_data: pd.DataFrame,
        investment_data: pd.DataFrame,
        control_features: List[str] = None
    ) -> Dict:
        """
        VC特化ROI計算エンジンのコンポーネントを計算

        Parameters:
        -----------
        company_id : str
            分析対象の企業ID
        start_date : str
            プログラム開始日
        end_date : str
            分析終了日
        revenue_data : pd.DataFrame
            売上データ
        valuation_data : pd.DataFrame
            企業価値評価データ
        program_cost_data : pd.DataFrame
            プログラムコストデータ
        investment_data : pd.DataFrame
            投資額データ
        control_features : List[str], optional
            コントロール変数のリスト

        Returns:
        --------
        Dict
            ROI計算のコンポーネント
            - delta_revenue: 売上増加額
            - delta_valuation: 企業価値増加額
            - program_cost: プログラムコスト
            - investment_cost: 投資額
            - roi: ROI値（%）
        """
        self.logger.info(f"ROIコンポーネントの計算を開始: 企業ID={company_id}")

        try:
            # 売上増加額（ΔRevenue）の推定
            revenue_impact = self.estimate_revenue_impact(
                revenue_data=revenue_data,
                intervention_data=pd.DataFrame([{'company_id': company_id, 'date': start_date}]),
                company_id=company_id,
                intervention_date=start_date,
                control_features=control_features,
                method='causal_impact_bayesian'  # ベイズ推論を使用
            )
            delta_revenue = revenue_impact['cumulative_effect']  # 累積効果を使用

            # 企業価値増加額（ΔValuation）の計算
            # EV/EBITDA倍率を適用
            # バリュエーションデータから平均EV/EBITDA倍率を取得
            company_valuation = valuation_data[valuation_data['company_id'] == company]
            latest_valuation = company_valuation.sort_values('date', ascending=False).iloc[0]
            ev_ebitda_ratio = latest_valuation.get('ev_ebitda_ratio', 8.0)  # デフォルト値は8

            # EBITDA増加額を推定（簡略化のため、ΔRevenueの30%と仮定）
            delta_ebitda = delta_revenue * 0.3

            # 企業価値増加額を計算
            delta_valuation = delta_ebitda * ev_ebitda_ratio

            # プログラムコスト（C_program）の計算
            company_program_cost = program_cost_data[program_cost_data['company_id'] == company]
            program_cost = company_program_cost[(company_program_cost['date'] >= start_date) &
                                              (company_program_cost['date'] <= end_date)]['cost'].sum()

            # 投資額（C_investment）の取得
            company_investment = investment_data[investment_data['company_id'] == company]
            investment_cost = company_investment['amount'].sum()

            # ROIの計算
            # ROI_VC = ((ΔRevenue + ΔValuation) - C_program) / C_investment × 100
            roi = ((delta_revenue + delta_valuation) - program_cost) / investment_cost * 100

            # 結果の整形
            result = {
                'delta_revenue': delta_revenue,
                'delta_valuation': delta_valuation,
                'program_cost': program_cost,
                'investment_cost': investment_cost,
                'roi': roi,
                'components': {
                    'delta_ebitda': delta_ebitda,
                    'ev_ebitda_ratio': ev_ebitda_ratio,
                    'revenue_impact_details': revenue_impact
                }
            }

            self.logger.info(f"ROIコンポーネントの計算が完了: ROI={roi}%")
            return result

        except Exception as e:
            self.logger.error(f"ROIコンポーネントの計算中にエラーが発生: {str(e)}")
            raise

    def bayesian_update_process(
        self,
        prior_distribution: Dict,
        likelihood_data: pd.DataFrame,
        target_variable: str = 'roi'
    ) -> Dict:
        """
        ベイズ更新プロセスによるROI分布の更新

        Parameters:
        -----------
        prior_distribution : Dict
            事前分布のパラメータ
            例: {'distribution': 'normal', 'mu': 15.0, 'sigma': 5.0}
        likelihood_data : pd.DataFrame
            新しいデータ
        target_variable : str, optional
            対象変数名（デフォルト: 'roi'）

        Returns:
        --------
        Dict
            事後分布のパラメータ
        """
        self.logger.info("ベイズ更新プロセスを開始")

        try:
            # 事前分布の設定
            if prior_distribution['distribution'] == 'normal':
                prior_mu = prior_distribution['mu']
                prior_sigma = prior_distribution['sigma']

                # 新しいデータからの尤度計算
                data_mean = likelihood_data[target_variable].mean()
                data_std = likelihood_data[target_variable].std()
                n = len(likelihood_data)

                # 事後分布のパラメータを計算（正規-正規の共役事前分布）
                posterior_precision = 1/prior_sigma**2 + n/data_std**2
                posterior_sigma = np.sqrt(1/posterior_precision)
                posterior_mu = (prior_mu/prior_sigma**2 + n*data_mean/data_std**2) / posterior_precision

                # 事後分布のパラメータを返却
                posterior = {
                    'distribution': 'normal',
                    'mu': posterior_mu,
                    'sigma': posterior_sigma,
                    'prior': prior_distribution,
                    'data_summary': {
                        'mean': data_mean,
                        'std': data_std,
                        'n': n
                    }
                }

                self.logger.info(f"ベイズ更新完了: 事前分布({prior_mu:.2f}, {prior_sigma:.2f}) → "
                               f"事後分布({posterior_mu:.2f}, {posterior_sigma:.2f})")
                return posterior

            elif prior_distribution['distribution'] == 'beta':
                # ベータ分布の場合の更新（0-1の範囲の値に適用、例：成功率）
                prior_alpha = prior_distribution['alpha']
                prior_beta = prior_distribution['beta']

                # 新しいデータからの尤度計算（成功回数と試行回数）
                # ROIをベータ分布で表現する場合は、スケーリングが必要
                scaled_data = likelihood_data[target_variable] / 100  # ROIを0-1スケールに変換
                success_count = (scaled_data > 0).sum()  # プラスのROIを「成功」とみなす
                trials = len(scaled_data)

                # 事後分布のパラメータを計算
                posterior_alpha = prior_alpha + success_count
                posterior_beta = prior_beta + (trials - success_count)

                # 事後分布のパラメータを返却
                posterior = {
                    'distribution': 'beta',
                    'alpha': posterior_alpha,
                    'beta': posterior_beta,
                    'prior': prior_distribution,
                    'data_summary': {
                        'success_count': success_count,
                        'trials': trials
                    }
                }

                self.logger.info(f"ベイズ更新完了: 事前分布(α={prior_alpha:.2f}, β={prior_beta:.2f}) → "
                               f"事後分布(α={posterior_alpha:.2f}, β={posterior_beta:.2f})")
                return posterior

            else:
                raise ValueError(f"未対応の分布タイプ: {prior_distribution['distribution']}")

        except Exception as e:
            self.logger.error(f"ベイズ更新プロセス中にエラーが発生: {str(e)}")
            raise

    def compute_portfolio_ecosystem_impact(
        self,
        companies: List[str],
        roi_data: pd.DataFrame,
        network_data: pd.DataFrame = None
    ) -> Dict:
        """
        ポートフォリオネットワーク効果の計算

        Parameters:
        -----------
        companies : List[str]
            分析対象の企業IDリスト
        roi_data : pd.DataFrame
            各企業のROIデータ
        network_data : pd.DataFrame, optional
            企業間の関係データ（存在する場合）

        Returns:
        --------
        Dict
            ネットワーク効果の分析結果
        """
        self.logger.info(f"ポートフォリオネットワーク効果の分析を開始: 企業数={len(companies)}")

        try:
            import networkx as nx

            # ネットワークデータがない場合は、シンプルな完全グラフを作成
            if network_data is None:
                G = nx.complete_graph(len(companies))
                mapping = {i: company for i, company in enumerate(companies)}
                G = nx.relabel_nodes(G, mapping)
            else:
                # ネットワークデータからグラフを構築
                G = nx.Graph()
                for _, row in network_data.iterrows():
                    G.add_edge(row['company1'], row['company2'], weight=row['strength'])

            # 各企業のROI値を取得
            roi_values = {}
            for company in companies:
                company_roi = roi_data[roi_data['company_id'] == company]['roi'].values
                if len(company_roi) > 0:
                    roi_values[company] = company_roi[0]
                else:
                    roi_values[company] = 0

            # ノードにROI値を属性として追加
            nx.set_node_attributes(G, roi_values, 'roi')

            # エコシステム係数の計算
            # シンプルな実装: 企業のROIとその隣接企業のROIの相関
            ecosystem_impact = {}
            for company in companies:
                neighbors = list(G.neighbors(company))
                if not neighbors:
                    ecosystem_impact[company] = 0
                    continue

                # 隣接企業のROI平均との比較
                neighbor_rois = [roi_values[n] for n in neighbors]
                avg_neighbor_roi = np.mean(neighbor_rois)
                company_roi = roi_values[company]

                # 基本的なエコシステム係数: 自社ROIと隣接企業ROIの比率
                if avg_neighbor_roi > 0:
                    impact = min(company_roi / avg_neighbor_roi, 2.0)  # 最大値を2.0に制限
                else:
                    impact = 1.0 if company_roi > 0 else 0.5

                ecosystem_impact[company] = max(0, min(impact, 1.0))  # 0-1の範囲に収める

            # 知識移転指数の計算（同業種間のROI相関）
            industry_data = roi_data[['company_id', 'industry', 'roi']].drop_duplicates()

            # 業種ごとのグループ化
            industries = industry_data['industry'].unique()
            knowledge_transfer = {}

            for industry in industries:
                industry_companies = industry_data[industry_data['industry'] == industry]['company_id'].tolist()
                if len(industry_companies) <= 1:
                    knowledge_transfer[industry] = 0
                    continue

                # 同業種企業間のROI相関を計算
                industry_rois = [roi_values[c] for c in industry_companies if c in roi_values]
                if len(industry_rois) <= 1:
                    knowledge_transfer[industry] = 0
                    continue

                # ROIの標準偏差を計算
                roi_std = np.std(industry_rois)

                # 標準偏差が小さいほど知識移転が進んでいると仮定
                knowledge_transfer[industry] = max(0, 1 - min(roi_std / 20, 1))  # 標準偏差20%で0、0%で1

            # 結果の整形
            result = {
                'ecosystem_impact': ecosystem_impact,
                'knowledge_transfer': knowledge_transfer,
                'network_metrics': {
                    'density': nx.density(G),
                    'avg_clustering': nx.average_clustering(G),
                    'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
                }
            }

            self.logger.info("ポートフォリオネットワーク効果の分析が完了")
            return result

        except Exception as e:
            self.logger.error(f"ポートフォリオネットワーク効果の分析中にエラーが発生: {str(e)}")
            raise

    def visualize_causal_effect(
        self,
        result: CausalImpactResult,
        title: str = "因果効果の可視化",
        save_path: str = None
    ) -> plt.Figure:
        """
        因果効果の可視化

        メモリリークを防ぐための最適化版

        Parameters:
        -----------
        result : CausalImpactResult
            因果推論分析の結果
        title : str, optional
            図のタイトル
        save_path : str, optional
            保存先のパス

        Returns:
        --------
        plt.Figure
            matplotlib図オブジェクト
        """
        # 結果がディスク上にある場合はロード
        if isinstance(result, str) and os.path.exists(result):
            result = self._get_result(result)

        # 必要なデータの確認
        if result.counterfactual_series is None or result.effect_series is None:
            raise ValueError("可視化に必要な時系列データがありません")

        try:
            # コンテキストマネージャーを使用してプロットリソースを管理
            with plt.style.context('default'):
                # 新しい図を作成
                plt.close('all')  # 既存の図をクリア
                fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

                # 実測値と反事実予測の比較
                ax1 = axes[0]
                actual = result.counterfactual_series + result.effect_series
                actual.plot(ax=ax1, label='実測値', color='black', linewidth=2)
                result.counterfactual_series.plot(ax=ax1, label='反事実予測（介入がなかった場合）',
                                   color='blue', linestyle='--', linewidth=2)

                # 介入時点に縦線を引く
                intervention_date = result.counterfactual_series.index[0]
                ax1.axvline(intervention_date, color='red', linestyle='-', alpha=0.5, label='介入時点')

                ax1.set_title('実測値と反事実予測の比較')
                ax1.set_ylabel('値')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # 効果の時系列
                ax2 = axes[1]
                result.effect_series.plot(ax=ax2, label='推定効果', color='green', linewidth=2)
                ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
                ax2.axvline(intervention_date, color='red', linestyle='-', alpha=0.5)

                # 累積効果情報を追加
                cumulative_text = f"累積効果: {result.cumulative_effect:.2f}"
                avg_effect_text = f"平均効果: {result.point_effect:.2f}"
                p_value_text = f"p値: {result.p_value:.3f}" if result.p_value is not None else ""

                info_text = cumulative_text + "\n" + avg_effect_text
                if p_value_text:
                    info_text += "\n" + p_value_text

                ax2.text(0.02, 0.95, info_text, transform=ax2.transAxes,
                         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

                ax2.set_title('推定効果の時系列変化')
                ax2.set_ylabel('効果')
                ax2.set_xlabel('日付')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                fig.suptitle(title, fontsize=16)
                fig.tight_layout()

                if save_path:
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)  # ファイル保存後に図を閉じる

                return fig
        except Exception as e:
            self.logger.error(f"因果効果の可視化中にエラーが発生: {str(e)}")
            plt.close('all')  # エラー発生時も図を閉じる
            raise
        finally:
            # リソースのクリーンアップ
            gc.collect()

    def visualize_bayesian_posterior(
        self,
        posterior_distribution: Dict,
        title: str = "ベイズ事後分布",
        save_path: str = None
    ) -> plt.Figure:
        """
        ベイズ事後分布の可視化

        メモリリークを防ぐための最適化版

        Parameters:
        -----------
        posterior_distribution : Dict
            ベイズ更新後の事後分布パラメータ
        title : str, optional
            図のタイトル
        save_path : str, optional
            保存先のパス

        Returns:
        --------
        plt.Figure
            matplotlib図オブジェクト
        """
        try:
            # 新しい図を作成
            plt.close('all')  # 既存の図をクリア
            fig, ax = plt.subplots(figsize=(10, 6))

            # 分布タイプに応じてプロット
            if posterior_distribution['distribution'] == 'normal':
                mu = posterior_distribution['mu']
                sigma = posterior_distribution['sigma']

                # 95%信用区間
                lower = mu - 1.96 * sigma
                upper = mu + 1.96 * sigma

                # x軸の範囲 - 効率的な実装
                x = np.linspace(lower - 2 * sigma, upper + 2 * sigma, 1000)

                # 確率密度関数 - ベクトル化した実装
                y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

                # プロット
                ax.plot(x, y, 'b-', linewidth=2)

                # 95%信用区間
                ax.axvline(mu, color='r', linestyle='-', label=f'平均: {mu:.2f}')
                ax.axvline(lower, color='g', linestyle='--', label=f'95%信用区間: [{lower:.2f}, {upper:.2f}]')
                ax.axvline(upper, color='g', linestyle='--')

                # 事前分布も表示
                prior_mu = posterior_distribution['prior']['mu']
                prior_sigma = posterior_distribution['prior']['sigma']
                prior_y = 1 / (prior_sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - prior_mu)**2 / (2 * prior_sigma**2))
                ax.plot(x, prior_y, 'k--', alpha=0.5, linewidth=1.5, label='事前分布')

                ax.set_xlabel('ROI (%)')

            elif posterior_distribution['distribution'] == 'beta':
                alpha = posterior_distribution['alpha']
                beta = posterior_distribution['beta']

                # x軸の範囲
                x = np.linspace(0, 1, 1000)

                # ベータ分布の確率密度関数
                from scipy.special import beta as beta_func
                y = x**(alpha-1) * (1-x)**(beta-1) / beta_func(alpha, beta)

                # 平均と95%信用区間
                mean = alpha / (alpha + beta)
                from scipy import stats
                lower, upper = stats.beta.ppf([0.025, 0.975], alpha, beta)

                # プロット
                ax.plot(x, y, 'b-', linewidth=2)

                # 95%信用区間
                ax.axvline(mean, color='r', linestyle='-', label=f'平均: {mean:.2f}')
                ax.axvline(lower, color='g', linestyle='--', label=f'95%信用区間: [{lower:.2f}, {upper:.2f}]')
                ax.axvline(upper, color='g', linestyle='--')

                # 事前分布も表示
                prior_alpha = posterior_distribution['prior']['alpha']
                prior_beta = posterior_distribution['prior']['beta']
                prior_y = x**(prior_alpha-1) * (1-x)**(prior_beta-1) / beta_func(prior_alpha, prior_beta)
                ax.plot(x, prior_y, 'k--', alpha=0.5, linewidth=1.5, label='事前分布')

                ax.set_xlabel('成功確率')

            ax.set_ylabel('確率密度')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)  # ファイル保存後に図を閉じる

            return fig

        except Exception as e:
            self.logger.error(f"ベイズ事後分布の可視化中にエラーが発生: {str(e)}")
            plt.close('all')  # エラー発生時も図を閉じる
            raise
        finally:
            # リソースのクリーンアップ
            gc.collect()

    def visualize_network_effect(
        self,
        network_result: Dict,
        roi_data: pd.DataFrame,
        title: str = "ポートフォリオネットワーク効果",
        save_path: str = None,
        max_nodes: int = 50  # 大規模ネットワークの自動縮小のための閾値
    ) -> plt.Figure:
        """
        ポートフォリオネットワーク効果の可視化

        メモリリークを防ぐための最適化版

        Parameters:
        -----------
        network_result : Dict
            ネットワーク分析の結果
        roi_data : pd.DataFrame
            ROIデータ
        title : str, optional
            図のタイトル
        save_path : str, optional
            保存先のパス
        max_nodes : int, optional
            表示する最大ノード数（大規模ネットワークの自動サンプリング用）

        Returns:
        --------
        plt.Figure
            matplotlib図オブジェクト
        """
        import networkx as nx

        try:
            # コンテキストマネージャーを使用してネットワークリソースを管理
            with self._managed_data(network_result) as result:
                # 新しい図を作成
                plt.close('all')  # 既存の図をクリア
                fig, ax = plt.subplots(figsize=(12, 10))

                # ネットワークグラフの構築
                G = nx.Graph()

                # ノードの追加
                ecosystem_impact = result['ecosystem_impact']

                # 大規模ネットワークの場合は自動的にサンプリングして表示ノード数を制限
                companies = list(ecosystem_impact.keys())
                if len(companies) > max_nodes:
                    # 影響度の高い順にノードを選択
                    sorted_companies = sorted(
                        companies,
                        key=lambda c: ecosystem_impact[c],
                        reverse=True
                    )[:max_nodes]
                    self.logger.info(f"ネットワーク表示を最適化: {len(companies)}ノードから{max_nodes}ノードにサンプリング")
                    companies = sorted_companies

                # ノード属性の設定
                for company in companies:
                    # ROI値の取得
                    company_roi = roi_data[roi_data['company_id'] == company]['roi'].values
                    roi = company_roi[0] if len(company_roi) > 0 else 0

                    # ノード属性の設定
                    G.add_node(company, impact=ecosystem_impact[company], roi=roi)

                # エッジの追加（単純なモデル: すべての企業間に弱いつながりがあると仮定）
                for i, company1 in enumerate(companies):
                    for company2 in companies[i+1:]:
                        # 共通の業界がある場合は強い関係
                        try:
                            industry1 = roi_data[roi_data['company_id'] == company1]['industry'].values[0]
                            industry2 = roi_data[roi_data['company_id'] == company2]['industry'].values[0]

                            if industry1 == industry2:
                                # 同じ業界の場合は強い関係
                                weight = 0.8
                            else:
                                # 異なる業界の場合は弱い関係
                                weight = 0.2

                            G.add_edge(company1, company2, weight=weight)
                        except IndexError:
                            # インデックスエラーが発生した場合は弱い関係を追加
                            G.add_edge(company1, company2, weight=0.1)

                # ノードの位置を計算
                pos = nx.spring_layout(G, seed=42)

                # ノードサイズをエコシステム影響度に基づいて設定
                node_sizes = [ecosystem_impact[node] * 1000 + 200 for node in G.nodes()]

                # ノード色をROI値に基づいて設定
                node_colors = [G.nodes[node]['roi'] for node in G.nodes()]

                # エッジの幅を重みに基づいて設定
                edge_widths = [G.edges[edge]['weight'] * 2 for edge in G.edges()]

                # ネットワークの描画
                nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                                              cmap=plt.cm.viridis, ax=ax)
                edges = nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray', ax=ax)

                # メモリ効率のため、ラベルは表示ノード数が少ない場合のみ表示
                if len(G.nodes()) <= 20:
                    labels = nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', ax=ax)

                # カラーバーの追加
                cbar = plt.colorbar(nodes, ax=ax, label='ROI (%)')

                # 凡例の追加
                sizes = [200, 600, 1000]
                labels = ['低', '中', '高']
                for size, label in zip(sizes, labels):
                    plt.scatter([], [], s=size, label=f'エコシステム影響度: {label}')

                ax.legend(scatterpoints=1, frameon=True, labelspacing=1)

                # タイトルと軸ラベル
                ax.set_title(title)
                ax.axis('off')

                if save_path:
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)  # ファイル保存後に図を閉じる

                return fig

        except Exception as e:
            self.logger.error(f"ポートフォリオネットワーク効果の可視化中にエラーが発生: {str(e)}")
            plt.close('all')  # エラー発生時も図を閉じる
            raise
        finally:
            # リソースのクリーンアップ
            del G
            gc.collect()

    def visualize_heterogeneous_effects(
        self,
        result: HeterogeneousTreatmentEffectResult,
        data: pd.DataFrame,
        features: List[str],
        top_features: int = 3,
        title: str = "異質処理効果（CATE）の分析",
        save_path: str = None,
        max_scatter_points: int = 1000  # スキャッタープロット用の最大データポイント数
    ) -> plt.Figure:
        """
        異質処理効果（CATE）の視覚化

        メモリリークを防ぐための最適化版

        Parameters:
        -----------
        result : HeterogeneousTreatmentEffectResult
            異質処理効果の推定結果
        data : pd.DataFrame
            元の分析データ
        features : List[str]
            使用した特徴量のリスト
        top_features : int, optional
            表示する特徴量の数
        title : str, optional
            図のタイトル
        save_path : str, optional
            図の保存先パス
        max_scatter_points : int, optional
            散布図に表示する最大のデータポイント数（大規模データセット用）

        Returns:
        --------
        plt.Figure
            matplotlib図オブジェクト
        """
        # 結果がディスク上にある場合はロード
        if isinstance(result, str) and os.path.exists(result):
            result = self._get_result(result)

        self.logger.info(f"異質処理効果（CATE）の視覚化を開始")

        try:
            # 新しい図を作成
            plt.close('all')  # 既存の図をクリア
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 2)

            # 1. 個別処理効果のヒストグラム
            ax1 = fig.add_subplot(gs[0, 0])
            # 効率的なヒストグラム計算
            ax1.hist(result.conditional_effects, bins=min(30, len(result.conditional_effects)//20),
                     alpha=0.7, color='blue')
            ax1.axvline(result.average_effect, color='red', linestyle='--',
                        label=f'平均効果: {result.average_effect:.4f}')
            ax1.set_title('個別処理効果の分布')
            ax1.set_xlabel('処理効果の大きさ')
            ax1.set_ylabel('頻度')
            ax1.legend()
            ax1.grid(alpha=0.3)

            # 2. 特徴量重要度（利用可能な場合）
            ax2 = fig.add_subplot(gs[0, 1])
            if result.feature_importance is not None:
                # 重要度でソート
                sorted_importance = dict(sorted(
                    result.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:min(top_features, len(result.feature_importance))])

                feature_names = list(sorted_importance.keys())
                importance_values = list(sorted_importance.values())

                ax2.barh(feature_names, importance_values, color='green', alpha=0.7)
                ax2.set_title('特徴量重要度')
                ax2.set_xlabel('重要度')
                ax2.set_ylabel('特徴量')
                ax2.grid(alpha=0.3)
            else:
                ax2.text(0.5, 0.5, '特徴量重要度は利用できません',
                        ha='center', va='center', fontsize=12)
                ax2.set_title('特徴量重要度 (利用不可)')
                ax2.axis('off')


            # 3. 処理効果の散布図（上位2つの特徴量に対して）
            ax3 = fig.add_subplot(gs[1, :])
            if len(features) >= 2 and result.feature_importance:
                # 上位2つの特徴量を取得
                top_two_features = list(sorted_importance.keys())[:2]

                # 散布図の作成
                scatter = ax3.scatter(
                    data[top_two_features[0]],
                    data[top_two_features[1]],
                    c=result.conditional_effects,
                    cmap='coolwarm',
                    alpha=0.7,
                    s=50
                )

                # カラーバーの追加
                cbar = plt.colorbar(scatter, ax=ax3)
                cbar.set_label('処理効果の大きさ')

                ax3.set_title(f'特徴量空間における処理効果の分布')
                ax3.set_xlabel(top_two_features[0])
                ax3.set_ylabel(top_two_features[1])
                ax3.grid(alpha=0.3)
            else:
                ax3.text(0.5, 0.5, '特徴量空間での可視化に必要な情報がありません',
                        ha='center', va='center', fontsize=12)
                ax3.set_title('特徴量空間における処理効果 (利用不可)')
                ax3.axis('off')

            # サマリー情報
            summary = (
                f"モデル: {result.model_type}\n"
                f"平均効果 (ATE): {result.average_effect:.4f}\n"
                f"効果の範囲: [{np.min(result.conditional_effects):.4f}, "
                f"{np.max(result.conditional_effects):.4f}]\n"
                f"効果の標準偏差: {np.std(result.conditional_effects):.4f}"
            )
            fig.text(0.02, 0.02, summary, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

            # レイアウト調整とタイトル
            fig.suptitle(title, fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])

            # 保存
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')

            self.logger.info(f"異質処理効果（CATE）の視覚化が完了")
            return fig

        except Exception as e:
            self.logger.error(f"異質処理効果（CATE）の視覚化中にエラーが発生: {str(e)}")
            raise