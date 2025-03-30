# backend/analysis.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
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

    def __init__(self):
        """初期化"""
        self.logger = logger
        self.logger.info("因果推論分析モジュールを初期化しました")

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
        model_args: Dict = None
    ) -> CausalImpactResult:
        """
        PyMCを使用したベイズ時系列因果推論

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
        self.logger.info(f"ベイズ時系列因果推論による分析を開始: 対象変数={target_col}, 介入時点={intervention_time}")

        try:
            # データの準備
            df = time_series.copy()

            # 介入前後のデータを分割
            pre_data = df.loc[:intervention_time].copy()
            post_data = df.loc[intervention_time:].copy()

            # 多変量ベイズ構造時系列モデル
            # PyMCを使用したベイズモデル
            with pm.Model() as model:
                # 事前分布
                if control_cols and len(control_cols) > 0:
                    # 多変量モデル
                    coeffs = pm.Normal('coeffs', mu=0, sigma=1, shape=len(control_cols))
                    intercept = pm.Normal('intercept', mu=0, sigma=10)
                    sigma = pm.HalfCauchy('sigma', beta=1)

                    # 回帰モデル（介入前データで学習）
                    X_pre = pre_data[control_cols].values
                    y_pre = pre_data[target_col].values

                    # 線形予測
                    mu = intercept + pm.math.dot(X_pre, coeffs)
                else:
                    # 単変量モデル - ローカルレベルモデル
                    sigma_level = pm.HalfCauchy('sigma_level', beta=1)
                    sigma_obs = pm.HalfCauchy('sigma_obs', beta=1)

                    # 状態空間表現
                    level = pm.GaussianRandomWalk('level', sigma=sigma_level, shape=len(pre_data))
                    mu = level
                    sigma = sigma_obs

                    y_pre = pre_data[target_col].values

                # 尤度
                likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y_pre)

                # サンプリング
                trace = pm.sample(2000, tune=1000, cores=2, progressbar=False, random_seed=42)

            # 介入後期間の反事実予測
            if control_cols and len(control_cols) > 0:
                # 多変量モデル - コントロール変数を使用
                X_post = post_data[control_cols].values

                # 予測分布からサンプリング
                post_pred = np.zeros((len(trace) * 2, len(post_data)))
                for i, sample in enumerate(az.extract(trace, var_names=['coeffs', 'intercept', 'sigma']).to_dict('records')):
                    coeffs_sample = sample['coeffs']
                    intercept_sample = sample['intercept']
                    sigma_sample = sample['sigma']

                    # 予測平均
                    mu = intercept_sample + np.dot(X_post, coeffs_sample)

                    # 予測からのサンプリング
                    post_pred[i] = np.random.normal(mu, sigma_sample)
            else:
                # 単変量モデル - 予測
                with model:
                    # 状態空間モデルの予測
                    forecast = pm.sample_posterior_predictive(
                        trace,
                        var_names=['y'],
                        samples=1000,
                        random_seed=42
                    )

                # 予測値の抽出
                post_pred = forecast['y']

            # 予測の要約
            predicted_mean = post_pred.mean(axis=0)
            predicted_ci = np.percentile(post_pred, [2.5, 97.5], axis=0).T

            # 実際の値との差を計算して効果を推定
            counterfactual = pd.Series(predicted_mean, index=post_data.index)
            actual = post_data[target_col]
            effect_series = actual - counterfactual

            # 効果の要約統計量
            point_effect = effect_series.mean()
            posterior_samples = np.array([actual.values - sample for sample in post_pred])
            ci = np.percentile(posterior_samples.mean(axis=1), [2.5, 97.5])
            cumulative_effect = effect_series.sum()

            # Bayesian p-value（事後確率）
            # 効果がゼロより大きい（または小さい）確率
            if point_effect > 0:
                p_value = (posterior_samples.mean(axis=1) <= 0).mean()
            else:
                p_value = (posterior_samples.mean(axis=1) >= 0).mean()

            result = CausalImpactResult(
                point_effect=point_effect,
                confidence_interval=(ci[0], ci[1]),
                p_value=p_value,
                posterior_samples=posterior_samples,
                counterfactual_series=counterfactual,
                effect_series=effect_series,
                cumulative_effect=cumulative_effect,
                model_summary={'report': report, 'summary': summary}
            )

            self.logger.info(f"ベイズ時系列因果推論による分析が完了: 効果={point_effect}, 事後確率={1-p_value:.3f}")
            return result

        except Exception as e:
            self.logger.error(f"ベイズ時系列因果推論による分析中にエラーが発生: {str(e)}")
            raise

    def estimate_revenue_impact(
        self,
        revenue_data: pd.DataFrame,
        intervention_data: pd.DataFrame,
        company_id: str,
        intervention_date: str,
        control_features: List[str] = None,
        method: str = 'causal_impact'
    ) -> Dict:
        """
        Startup Wellnessプログラムによる売上への影響（ΔRevenue）を推定

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

        Returns:
        --------
        Dict
            推定結果の辞書
            - delta_revenue: 推定された売上増加額
            - confidence_interval: 信頼区間
            - relative_impact: 相対的な影響度（%）
            - details: 詳細な分析結果
        """
        self.logger.info(f"売上影響度の推定を開始: 企業ID={company_id}, 介入日={intervention_date}")

        try:
            # 対象企業のデータを抽出
            company_revenue = revenue_data[revenue_data['company_id'] == company_id].copy()

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

            result = None

            if method == 'causal_impact' or method == 'causal_impact_bayesian':
                # 売上カラムを特定
                revenue_cols = [col for col in company_revenue.columns if 'revenue' in col.lower()]
                if not revenue_cols:
                    raise ValueError("売上データのカラムが見つかりません")

                revenue_col = revenue_cols[0]  # 最初の売上カラムを使用

                # コントロール変数の準備
                if not control_features:
                    # コントロール変数が指定されていない場合、同業他社のデータを使用
                    other_companies = revenue_data[revenue_data['company_id'] != company_id]
                    # 他社の売上データを集約（例: 平均売上）
                    other_companies_pivot = other_companies.pivot_table(
                        index=date_col,
                        columns='company_id',
                        values=revenue_col,
                        aggfunc='mean'
                    )
                    # ターゲット企業のデータとマージ
                    merged_data = pd.DataFrame(company_revenue[revenue_col])
                    merged_data = merged_data.join(other_companies_pivot, how='inner')
                    control_cols = other_companies_pivot.columns.tolist()
                else:
                    # 指定されたコントロール変数を使用
                    control_cols = [col for col in control_features if col in company_revenue.columns]
                    merged_data = company_revenue[[revenue_col] + control_cols]

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
                        control_cols=control_cols
                    )

            elif method == 'synthetic_control':
                # 合成コントロール法による分析
                # 全企業のデータをパネル形式に整形
                panel_data = revenue_data.pivot(index='company_id', columns=date_col, values='revenue')
                panel_data = panel_data.reset_index()

                # 介入前後の期間を定義
                all_dates = sorted(panel_data.columns[1:])  # 日付のみ（company_idを除く）
                intervention_idx = all_dates.index(intervention_date)
                pre_period = [all_dates[0], all_dates[intervention_idx-1]]
                post_period = [all_dates[intervention_idx], all_dates[-1]]

                # 対照企業の選定（介入を受けていない企業）
                intervention_companies = intervention_data['company_id'].unique().tolist()
                control_units = [c for c in panel_data['company_id'].unique()
                               if c != company_id and c not in intervention_companies]

                # パネルデータを長形式に変換
                long_data = panel_data.melt(id_vars='company_id', var_name=date_col, value_name='revenue')

                # 合成コントロール法による分析実行
                result = self.analyze_synthetic_control(
                    data=long_data,
                    target_unit=company_id,
                    control_units=control_units,
                    time_col=date_col,
                    outcome_col='revenue',
                    pre_period=pre_period,
                    post_period=post_period
                )

            elif method == 'did':
                # 差分の差分法による分析
                # 処置グループ（対象企業）と対照グループ（その他企業）の設定
                revenue_data['treatment'] = revenue_data['company_id'] == company_id

                # 処置前後の期間設定
                revenue_data['post'] = pd.to_datetime(revenue_data[date_col]) >= pd.to_datetime(intervention_date)

                # DiD分析の実行
                result = self.analyze_difference_in_differences(
                    data=revenue_data,
                    treatment_col='treatment',
                    time_col='post',
                    outcome_col='revenue',
                    covariates=control_features
                )

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

            self.logger.info(f"売上影響度の推定が完了: ΔRevenue={result.point_effect}, 相対影響度={relative_impact}%")
            return output

        except Exception as e:
            self.logger.error(f"売上影響度の推定中にエラーが発生: {str(e)}")
            raise

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
        if result.counterfactual_series is None or result.effect_series is None:
            raise ValueError("可視化に必要な時系列データがありません")

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

        return fig

    def visualize_bayesian_posterior(
        self,
        posterior_distribution: Dict,
        title: str = "ベイズ事後分布",
        save_path: str = None
    ) -> plt.Figure:
        """
        ベイズ事後分布の可視化

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
        fig, ax = plt.subplots(figsize=(10, 6))

        # 分布タイプに応じてプロット
        if posterior_distribution['distribution'] == 'normal':
            mu = posterior_distribution['mu']
            sigma = posterior_distribution['sigma']

            # 95%信用区間
            lower = mu - 1.96 * sigma
            upper = mu + 1.96 * sigma

            # x軸の範囲
            x = np.linspace(lower - 2 * sigma, upper + 2 * sigma, 1000)

            # 確率密度関数
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

        return fig

    def visualize_network_effect(
        self,
        network_result: Dict,
        roi_data: pd.DataFrame,
        title: str = "ポートフォリオネットワーク効果",
        save_path: str = None
    ) -> plt.Figure:
        """
        ポートフォリオネットワーク効果の可視化

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

        Returns:
        --------
        plt.Figure
            matplotlib図オブジェクト
        """
        import networkx as nx

        # ネットワークグラフの構築
        G = nx.Graph()

        # ノードの追加
        ecosystem_impact = network_result['ecosystem_impact']
        for company, impact in ecosystem_impact.items():
            # ROI値の取得
            company_roi = roi_data[roi_data['company_id'] == company]['roi'].values
            roi = company_roi[0] if len(company_roi) > 0 else 0

            # ノード属性の設定
            G.add_node(company, impact=impact, roi=roi)

        # エッジの追加（単純なモデル: すべての企業間に弱いつながりがあると仮定）
        companies = list(ecosystem_impact.keys())
        for i, company1 in enumerate(companies):
            for company2 in companies[i+1:]:
                # 共通の業界がある場合は強い関係
                industry1 = roi_data[roi_data['company_id'] == company1]['industry'].values[0]
                industry2 = roi_data[roi_data['company_id'] == company2]['industry'].values[0]

                if industry1 == industry2:
                    # 同じ業界の場合は強い関係
                    weight = 0.8
                else:
                    # 異なる業界の場合は弱い関係
                    weight = 0.2

                G.add_edge(company1, company2, weight=weight)

        # 図の作成
        fig, ax = plt.subplots(figsize=(12, 10))

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

        return fig

    def estimate_heterogeneous_treatment_effects(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        features: List[str],
        method: str = 'causal_forest',
        inference_method: str = 'bootstrap',
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_leaf: int = 10,
        bootstrap_samples: int = 1000,
        random_state: int = 42
    ) -> HeterogeneousTreatmentEffectResult:
        """
        EconMLを使用した異質処理効果（CATE）の推定

        Parameters:
        -----------
        data : pd.DataFrame
            分析対象データ
        treatment_col : str
            処置変数のカラム名（0/1のバイナリ変数）
        outcome_col : str
            アウトカム変数のカラム名
        features : List[str]
            特徴量（共変量）のカラム名リスト
        method : str, optional
            使用するモデル（'causal_forest', 'linear_dml', 'dr_learner', 't_learner', 's_learner', 'x_learner'）
        inference_method : str, optional
            推論方法（'bootstrap', 'auto'）
        n_estimators : int, optional
            モデルの推定器数（ツリーベースのモデル用）
        max_depth : int, optional
            ツリーの最大深さ
        min_samples_leaf : int, optional
            リーフノードに必要な最小サンプル数
        bootstrap_samples : int, optional
            ブートストラップサンプル数
        random_state : int, optional
            乱数シード

        Returns:
        --------
        HeterogeneousTreatmentEffectResult
            異質処理効果の推定結果
        """
        self.logger.info(f"異質処理効果（CATE）推定を開始: モデル={method}, 対象変数={outcome_col}")

        try:
            # データの準備
            X = data[features].copy()
            T = data[treatment_col].values
            Y = data[outcome_col].values

            # 特徴量の標準化
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            # 推論方法の設定
            if inference_method == 'bootstrap':
                inference = BootstrapInference(n_bootstrap_samples=bootstrap_samples)
            else:
                inference = 'auto'

            # モデルの選択と初期化
            if method == 'causal_forest':
                model = CausalForestDML(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                    inference=inference
                )
            elif method == 'linear_dml':
                model = LinearDML(
                    model_y=LassoCV(cv=5, random_state=random_state),
                    model_t=LassoCV(cv=5, random_state=random_state),
                    random_state=random_state,
                    inference=inference
                )
            elif method == 'dr_learner':
                model = DRLearner(
                    model_propensity=LassoCV(cv=5, random_state=random_state),
                    model_regression=LassoCV(cv=5, random_state=random_state),
                    model_final=LassoCV(cv=5, random_state=random_state),
                    random_state=random_state,
                    inference=inference
                )
            elif method == 't_learner':
                model = TLearner(
                    models=LassoCV(cv=5, random_state=random_state),
                    random_state=random_state
                )
            elif method == 's_learner':
                model = SLearner(
                    overall_model=LassoCV(cv=5, random_state=random_state),
                    random_state=random_state
                )
            elif method == 'x_learner':
                model = XLearner(
                    models=LassoCV(cv=5, random_state=random_state),
                    propensity_model=LassoCV(cv=5, random_state=random_state),
                    random_state=random_state
                )
            else:
                raise ValueError(f"サポートされていないメソッド: {method}")

            # モデルの学習
            model.fit(Y, T, X=X_scaled)

            # 個別の処理効果を予測
            cate_estimates = model.effect(X_scaled)

            # 平均処理効果（ATE）
            average_effect = float(np.mean(cate_estimates))

            # 特徴量重要度（CausalForestの場合のみ）
            feature_importance = None
            if method == 'causal_forest' and hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(features, model.feature_importances_))

            # 信頼区間と統計的推論（可能な場合）
            confidence_intervals = None
            p_values = None

            if inference_method != 'none' and hasattr(model, 'effect_inference'):
                try:
                    effect_inference = model.effect_inference(X_scaled)
                    confidence_intervals = np.column_stack([
                        effect_inference.point_estimate - 1.96 * effect_inference.stderr,
                        effect_inference.point_estimate + 1.96 * effect_inference.stderr
                    ])
                    p_values = effect_inference.pvalue
                except Exception as e:
                    self.logger.warning(f"効果の統計的推論に失敗しました: {str(e)}")

            # 結果の作成
            result = HeterogeneousTreatmentEffectResult(
                model_type=method,
                average_effect=average_effect,
                conditional_effects=cate_estimates,
                feature_importance=feature_importance,
                confidence_intervals=confidence_intervals,
                p_values=p_values,
                model_instance=model
            )

            self.logger.info(f"異質処理効果（CATE）推定が完了: 平均効果={average_effect}")
            return result

        except Exception as e:
            self.logger.error(f"異質処理効果（CATE）推定中にエラーが発生: {str(e)}")
            raise

    def visualize_heterogeneous_effects(
        self,
        result: HeterogeneousTreatmentEffectResult,
        data: pd.DataFrame,
        features: List[str],
        top_features: int = 3,
        title: str = "異質処理効果（CATE）の分析",
        save_path: str = None
    ) -> plt.Figure:
        """
        異質処理効果（CATE）の視覚化

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

        Returns:
        --------
        plt.Figure
            matplotlib図オブジェクト
        """
        self.logger.info(f"異質処理効果（CATE）の視覚化を開始")

        try:
            # 図の作成
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 2)

            # 1. 個別処理効果のヒストグラム
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.hist(result.conditional_effects, bins=30, alpha=0.7, color='blue')
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
                )[:top_features])

                features = list(sorted_importance.keys())
                importance = list(sorted_importance.values())

                ax2.barh(features, importance, color='green', alpha=0.7)
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