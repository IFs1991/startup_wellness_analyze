import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .base import BaseAnalyzer
from .utils import HealthImpactWeightUtility

class VCROICalculator(BaseAnalyzer):
    """
    VC向けROI計算クラス

    Startup Wellnessプログラムの投資対効果を計算するためのクラス

    ROI_{VC} = ((ΔRevenue + ΔValuation) - C_{program}) / C_{investment} × 100
    """

    def __init__(self):
        """
        VCROICalculatorの初期化
        """
        super().__init__()
        self.logger.info("VCROICalculator initialized")

    def calculate_roi(self,
                      delta_revenue: float,
                      delta_valuation: float,
                      program_cost: float,
                      investment_cost: float) -> float:
        """
        基本的なROI計算を実行

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
            ROI値（パーセンテージ）
        """
        try:
            # 分母が0の場合のエラー処理
            if investment_cost <= 0:
                raise ValueError("Investment cost must be greater than zero")

            roi = ((delta_revenue + delta_valuation) - program_cost) / investment_cost * 100
            self.logger.info(f"ROI calculated: {roi:.2f}%")
            return roi
        except Exception as e:
            self.logger.error(f"Error calculating ROI: {str(e)}")
            raise

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

            self.logger.info(f"Time series ROI calculation completed: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error calculating ROI from time series: {str(e)}")
            raise

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

            self.logger.info(f"Risk-adjusted ROI calculated: {risk_adjusted_roi:.2f}%")
            return risk_adjusted_roi
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted ROI: {str(e)}")
            raise

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
        """
        try:
            # 基本ROI計算
            base_roi = self.calculate_roi(delta_revenue, delta_valuation, program_cost, investment_cost)

            # VASスケールの変化を計算
            vas_improvement = {}
            health_impact_factor = 1.0

            for column in vas_before.columns:
                if column in vas_after.columns:
                    before_mean = vas_before[column].mean()
                    after_mean = vas_after[column].mean()

                    # 改善率を計算 (VASスケールは値が小さいほど良いと仮定)
                    if before_mean > 0:
                        improvement = (before_mean - after_mean) / before_mean * 100
                        vas_improvement[column] = improvement

                        # 健康影響係数の計算（簡易的な例）
                        # 健康状態の改善がROIにプラスの影響を与える仮定
                        if improvement > 0:
                            health_impact_factor += improvement * 0.01  # 改善1%ごとにROIを1%上乗せする仮定

            # 健康状態を考慮したROI計算
            health_adjusted_roi = base_roi * health_impact_factor

            result = {
                'base_roi': base_roi,
                'health_adjusted_roi': health_adjusted_roi,
                'health_impact_factor': health_impact_factor,
                'vas_improvements': vas_improvement
            }

            self.logger.info(f"Health-adjusted ROI calculated: {health_adjusted_roi:.2f}%")
            return result
        except Exception as e:
            self.logger.error(f"Error calculating health-adjusted ROI: {str(e)}")
            raise

    def sensitivity_analysis(self,
                           delta_revenue_range: Tuple[float, float],
                           delta_valuation_range: Tuple[float, float],
                           program_cost: float,
                           investment_cost: float,
                           steps: int = 10) -> pd.DataFrame:
        """
        ROI計算の感度分析

        Parameters
        ----------
        delta_revenue_range : Tuple[float, float]
            収益変化の範囲 (最小値, 最大値)
        delta_valuation_range : Tuple[float, float]
            企業価値変化の範囲 (最小値, 最大値)
        program_cost : float
            Startup Wellnessプログラムのコスト
        investment_cost : float
            投資コスト
        steps : int, optional
            分析ステップ数 (デフォルト: 10)

        Returns
        -------
        pd.DataFrame
            感度分析の結果を含むDataFrame
        """
        try:
            # 範囲内の値を生成
            delta_revenues = np.linspace(delta_revenue_range[0], delta_revenue_range[1], steps)
            delta_valuations = np.linspace(delta_valuation_range[0], delta_valuation_range[1], steps)

            # 結果格納用のリスト
            results = []

            # 全ての組み合わせでROIを計算
            for dr in delta_revenues:
                for dv in delta_valuations:
                    roi = self.calculate_roi(dr, dv, program_cost, investment_cost)
                    results.append({
                        'delta_revenue': dr,
                        'delta_valuation': dv,
                        'roi': roi
                    })

            # DataFrameに変換
            df_results = pd.DataFrame(results)

            self.logger.info(f"Sensitivity analysis completed with {len(results)} scenarios")
            return df_results
        except Exception as e:
            self.logger.error(f"Error in sensitivity analysis: {str(e)}")
            raise

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

            if 'industry' not in company_data:
                raise ValueError("企業データに業種情報（industry）が含まれていません")

            industry_name = company_data['industry']
            self.logger.info(f"対象業種: {industry_name}")

            # 役職ごとの重み係数と個別ROIを計算
            role_roi_values = {}
            role_weights = {}
            role_impacts = {}

            for role, data in positions_data.items():
                # 必須キーのチェック
                required_keys = ['delta_revenue', 'delta_valuation']
                for key in required_keys:
                    if key not in data:
                        raise ValueError(f"役職 '{role}' のデータに必須キー '{key}' がありません")

                # 役職の健康影響度の重み係数を取得
                weight = HealthImpactWeightUtility.get_health_impact_weight(
                    db_connection, industry_name, role
                )
                role_weights[role] = weight

                # 役職ごとのROIを計算
                # 各役職が「全体のコスト」に占める割合を重みとして計算
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

            # 全体の重み付け合計を計算
            total_weight = sum(role_weights.values())
            if total_weight <= 0:
                raise ValueError("役職の重み係数の合計が0以下です")

            # 正規化された役職の重み
            normalized_weights = {role: weight/total_weight for role, weight in role_weights.items()}

            # 重み付け集計値の計算
            total_weighted_delta_revenue = sum(
                positions_data[role]['delta_revenue'] * normalized_weights[role]
                for role in positions_data.keys()
            )

            total_weighted_delta_valuation = sum(
                positions_data[role]['delta_valuation'] * normalized_weights[role]
                for role in positions_data.keys()
            )

            # 最終的な重み付けROIを計算
            final_roi = self.calculate_roi(
                total_weighted_delta_revenue,
                total_weighted_delta_valuation,
                program_cost,
                investment_cost
            )

            result = {
                'roi': final_roi,
                'weighted_by_role': role_roi_values,
                'role_weights': role_weights,
                'normalized_weights': normalized_weights,
                'details': {
                    'total_weighted_delta_revenue': total_weighted_delta_revenue,
                    'total_weighted_delta_valuation': total_weighted_delta_valuation,
                    'industry': industry_name,
                    'role_impacts': role_impacts
                }
            }

            self.logger.info(f"役職別重み付けROI計算が完了しました。最終ROI値: {final_roi:.2f}%")
            return result

        except Exception as e:
            self.logger.error(f"役職別重み付けROI計算中にエラーが発生しました: {str(e)}")
            raise

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
        # 階層別データを役職別データに変換
        positions_data = {
            'C級役員/経営層': executive_data,
            '上級管理職': management_data,
            '一般職員': staff_data
        }

        # 役職別重み付けROI計算を実行
        return self.calculate_weighted_roi_by_position(
            positions_data,
            program_cost,
            investment_cost,
            company_data,
            db_connection
        )