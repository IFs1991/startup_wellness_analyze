import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .base import BaseAnalyzer
from .utils import HealthImpactWeightUtility

class HealthInvestmentEffectIndexCalculator(BaseAnalyzer):
    """
    健康投資効果指数（HIEI）計算クラス

    Startup Wellnessプログラムの健康投資効果を指数化するためのクラス
    """

    def __init__(self):
        """
        HealthInvestmentEffectIndexCalculatorの初期化
        """
        super().__init__()
        self.logger.info("HealthInvestmentEffectIndexCalculator initialized")

    def calculate_hiei(self,
                      vas_improvement: float,
                      productivity_improvement: float,
                      turnover_reduction: float,
                      weights: Dict[str, float] = None) -> float:
        """
        基本的なHIEI計算を実行

        Parameters
        ----------
        vas_improvement : float
            VASスケールの改善度合い（%）
        productivity_improvement : float
            生産性の向上率（%）
        turnover_reduction : float
            離職率の減少（%ポイント）
        weights : Dict[str, float], optional
            各指標の重み付け係数 (デフォルト: None - 均等配分)

        Returns
        -------
        float
            HIEI値（0-100のスケール）
        """
        try:
            # デフォルトの重み
            if weights is None:
                weights = {
                    'vas': 0.4,
                    'productivity': 0.4,
                    'turnover': 0.2
                }

            # 重みの合計が1になることを確認
            total_weight = sum(weights.values())
            if not np.isclose(total_weight, 1.0):
                self.logger.warning(f"Weights do not sum to 1.0 (sum: {total_weight}), normalizing")
                weights = {k: v/total_weight for k, v in weights.items()}

            # 各指標を0-100のスケールに正規化（単純な例）
            normalized_vas = min(max(vas_improvement, 0), 100)
            normalized_productivity = min(max(productivity_improvement, 0), 100)
            normalized_turnover = min(max(turnover_reduction * 10, 0), 100)  # 離職率は通常10%前後なので、10倍

            # 重み付け計算
            hiei = (
                weights['vas'] * normalized_vas +
                weights['productivity'] * normalized_productivity +
                weights['turnover'] * normalized_turnover
            )

            self.logger.info(f"HIEI calculated: {hiei:.2f}")
            return hiei
        except Exception as e:
            self.logger.error(f"Error calculating HIEI: {str(e)}")
            raise

    def calculate_hiei_from_data(self,
                               vas_before: pd.DataFrame,
                               vas_after: pd.DataFrame,
                               financial_data_before: pd.DataFrame,
                               financial_data_after: pd.DataFrame,
                               turnover_before: float,
                               turnover_after: float,
                               weights: Dict[str, float] = None) -> Dict:
        """
        実データからHIEIを計算

        Parameters
        ----------
        vas_before : pd.DataFrame
            プログラム導入前のVASスケールデータ
        vas_after : pd.DataFrame
            プログラム導入後のVASスケールデータ
        financial_data_before : pd.DataFrame
            プログラム導入前の財務データ
        financial_data_after : pd.DataFrame
            プログラム導入後の財務データ
        turnover_before : float
            プログラム導入前の離職率
        turnover_after : float
            プログラム導入後の離職率
        weights : Dict[str, float], optional
            各指標の重み付け係数

        Returns
        -------
        Dict
            HIEI値とその他の計算メトリクスを含む辞書
        """
        try:
            # VASスケールの改善度合いを計算
            vas_improvement = self._calculate_vas_improvement(vas_before, vas_after)

            # 生産性の向上率を計算
            productivity_improvement = self._calculate_productivity_improvement(
                financial_data_before, financial_data_after
            )

            # 離職率の減少を計算
            turnover_reduction = turnover_before - turnover_after

            # HIEI計算
            hiei = self.calculate_hiei(
                vas_improvement, productivity_improvement, turnover_reduction, weights
            )

            result = {
                'hiei': hiei,
                'vas_improvement': vas_improvement,
                'productivity_improvement': productivity_improvement,
                'turnover_reduction': turnover_reduction,
                'vas_details': self._get_vas_improvement_details(vas_before, vas_after)
            }

            self.logger.info(f"HIEI calculation from data completed: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error calculating HIEI from data: {str(e)}")
            raise

    def _calculate_vas_improvement(self, vas_before: pd.DataFrame, vas_after: pd.DataFrame) -> float:
        """
        VASスケールの改善度合いを計算

        Parameters
        ----------
        vas_before : pd.DataFrame
            プログラム導入前のVASスケールデータ
        vas_after : pd.DataFrame
            プログラム導入後のVASスケールデータ

        Returns
        -------
        float
            VASスケールの平均改善率（%）
        """
        improvements = []

        for column in vas_before.columns:
            if column in vas_after.columns:
                before_mean = vas_before[column].mean()
                after_mean = vas_after[column].mean()

                # 改善率を計算 (VASスケールは値が小さいほど良いと仮定)
                if before_mean > 0:
                    improvement = (before_mean - after_mean) / before_mean * 100
                    improvements.append(improvement)

        if improvements:
            return np.mean(improvements)
        else:
            return 0.0

    def _get_vas_improvement_details(self, vas_before: pd.DataFrame, vas_after: pd.DataFrame) -> Dict[str, float]:
        """
        VASスケール項目ごとの改善度合いを計算

        Parameters
        ----------
        vas_before : pd.DataFrame
            プログラム導入前のVASスケールデータ
        vas_after : pd.DataFrame
            プログラム導入後のVASスケールデータ

        Returns
        -------
        Dict[str, float]
            VASスケール項目ごとの改善率を含む辞書
        """
        details = {}

        for column in vas_before.columns:
            if column in vas_after.columns:
                before_mean = vas_before[column].mean()
                after_mean = vas_after[column].mean()

                # 改善率を計算 (VASスケールは値が小さいほど良いと仮定)
                if before_mean > 0:
                    improvement = (before_mean - after_mean) / before_mean * 100
                    details[column] = improvement

        return details

    def _calculate_productivity_improvement(self,
                                          financial_data_before: pd.DataFrame,
                                          financial_data_after: pd.DataFrame) -> float:
        """
        生産性の向上率を計算

        Parameters
        ----------
        financial_data_before : pd.DataFrame
            プログラム導入前の財務データ
        financial_data_after : pd.DataFrame
            プログラム導入後の財務データ

        Returns
        -------
        float
            生産性の向上率（%）
        """
        try:
            # 従業員1人あたりの収益で生産性を計算する想定
            if 'revenue' in financial_data_before.columns and 'employees' in financial_data_before.columns:
                revenue_before = financial_data_before['revenue'].mean()
                employees_before = financial_data_before['employees'].mean()

                revenue_after = financial_data_after['revenue'].mean()
                employees_after = financial_data_after['employees'].mean()

                # 従業員数が0の場合のエラー処理
                if employees_before <= 0 or employees_after <= 0:
                    self.logger.warning("Employee count is zero or negative, using alternative productivity metric")
                    return (revenue_after - revenue_before) / revenue_before * 100 if revenue_before > 0 else 0

                productivity_before = revenue_before / employees_before
                productivity_after = revenue_after / employees_after

                if productivity_before > 0:
                    return (productivity_after - productivity_before) / productivity_before * 100
                else:
                    return 0.0
            else:
                self.logger.warning("Required columns not found in financial data, returning 0")
                return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating productivity improvement: {str(e)}")
            return 0.0

    def calculate_ecosystem_impact(self,
                                 hiei_values: Dict[str, float],
                                 network_adjacency: pd.DataFrame) -> Dict[str, float]:
        """
        企業エコシステム内でのHIEI影響度を計算

        Parameters
        ----------
        hiei_values : Dict[str, float]
            各企業のHIEI値
        network_adjacency : pd.DataFrame
            企業間のネットワーク隣接行列

        Returns
        -------
        Dict[str, float]
            各企業のエコシステム内での影響度
        """
        try:
            # 結果格納用の辞書
            ecosystem_impact = {}

            # 企業ごとの計算
            for company in hiei_values.keys():
                if company in network_adjacency.index and company in network_adjacency.columns:
                    # 当該企業と関連のある企業の重みを取得
                    connected_companies = network_adjacency.loc[company]

                    # 影響度を計算
                    impact = hiei_values[company]  # 自社のHIEI

                    # 接続企業からの影響を加算
                    for other_company, weight in connected_companies.items():
                        if other_company in hiei_values and other_company != company:
                            impact += hiei_values[other_company] * weight * 0.1  # 接続企業のHIEIの10%を重み付けで加算

                    ecosystem_impact[company] = impact
                else:
                    # ネットワークに含まれていない場合は自社のHIEIをそのまま使用
                    ecosystem_impact[company] = hiei_values[company]

            self.logger.info(f"Ecosystem impact calculation completed for {len(ecosystem_impact)} companies")
            return ecosystem_impact
        except Exception as e:
            self.logger.error(f"Error calculating ecosystem impact: {str(e)}")
            raise

    def calculate_industry_benchmarks(self, hiei_values: Dict[str, float],
                                    industry_mapping: Dict[str, str]) -> Dict[str, float]:
        """
        業界ごとのHIEIベンチマークを計算

        Parameters
        ----------
        hiei_values : Dict[str, float]
            各企業のHIEI値
        industry_mapping : Dict[str, str]
            企業IDと業界のマッピング

        Returns
        -------
        Dict[str, float]
            業界ごとの平均HIEI値
        """
        try:
            # 業界ごとにHIEI値をグループ化
            industry_hiei = {}

            for company, industry in industry_mapping.items():
                if company in hiei_values:
                    if industry not in industry_hiei:
                        industry_hiei[industry] = []

                    industry_hiei[industry].append(hiei_values[company])

            # 業界ごとの平均値を計算
            industry_benchmarks = {
                industry: np.mean(values) if values else 0
                for industry, values in industry_hiei.items()
            }

            self.logger.info(f"Industry benchmarks calculated for {len(industry_benchmarks)} industries")
            return industry_benchmarks
        except Exception as e:
            self.logger.error(f"Error calculating industry benchmarks: {str(e)}")
            raise

    def calculate_role_weighted_hiei(self,
                                    vas_improvements: Dict[str, float],
                                    productivity_improvements: Dict[str, float],
                                    turnover_reductions: Dict[str, float],
                                    company_data: Dict[str, str],
                                    db_connection,
                                    custom_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        業種・役職別の重み付けを考慮したHIEI計算を実行

        Parameters
        ----------
        vas_improvements : Dict[str, float]
            役職ごとのVASスケールの改善度合い（%）
            キーは役職名（例：'C級役員/経営層', '上級管理職'）
        productivity_improvements : Dict[str, float]
            役職ごとの生産性の向上率（%）
        turnover_reductions : Dict[str, float]
            役職ごとの離職率の減少（%ポイント）
        company_data : Dict[str, str]
            企業データ（業種などの情報を含む）
            必須キー: 'industry'（業種名）
        db_connection
            PostgreSQLデータベース接続オブジェクト
        custom_weights : Dict[str, float], optional
            VAS, 生産性, 離職率の間の重み付け係数 (デフォルト: None - 規定値を使用)

        Returns
        -------
        Dict[str, float]
            計算結果を含む辞書
            - 'hiei': 最終的なHIEI値（0-100のスケール）
            - 'weighted_by_role': 役職別の重み付けHIEI値
            - 'role_weights': 使用された役職別の重み係数
            - 'overall_weights': 全体の影響度重み
        """
        try:
            self.logger.info("役職別重み付けHIEI計算を開始")

            if 'industry' not in company_data:
                raise ValueError("企業データに業種情報（industry）が含まれていません")

            industry_name = company_data['industry']
            self.logger.info(f"対象業種: {industry_name}")

            # デフォルトの指標間重み
            if custom_weights is None:
                custom_weights = {
                    'vas': 0.4,
                    'productivity': 0.4,
                    'turnover': 0.2
                }

            # 役職ごとの役職別HIEI値を計算
            role_hiei_values = {}
            role_weights = {}

            # 各役職について処理
            for role in vas_improvements.keys():
                # 役職の健康影響度の重み係数を取得
                weight = HealthImpactWeightUtility.get_health_impact_weight(
                    db_connection, industry_name, role
                )
                role_weights[role] = weight

                # 役職ごとのHIEI値を計算
                role_hiei = self.calculate_hiei(
                    vas_improvements.get(role, 0),
                    productivity_improvements.get(role, 0),
                    turnover_reductions.get(role, 0),
                    custom_weights
                )
                role_hiei_values[role] = role_hiei

            # 全体の重み付け合計を計算
            total_weight = sum(role_weights.values())
            if total_weight <= 0:
                raise ValueError("役職の重み係数の合計が0以下です")

            # 正規化された役職の重み
            normalized_weights = {role: weight/total_weight for role, weight in role_weights.items()}

            # 最終的な重み付けHIEI値を計算
            final_hiei = sum(role_hiei_values[role] * normalized_weights[role] for role in role_hiei_values.keys())

            result = {
                'hiei': final_hiei,
                'weighted_by_role': role_hiei_values,
                'role_weights': role_weights,
                'normalized_weights': normalized_weights,
                'industry': industry_name
            }

            self.logger.info(f"役職別重み付けHIEI計算が完了しました。最終HIEI値: {final_hiei:.2f}")
            return result

        except Exception as e:
            self.logger.error(f"役職別重み付けHIEI計算中にエラーが発生しました: {str(e)}")
            raise

    def calculate_team_based_hiei(self,
                                 team_data: pd.DataFrame,
                                 health_metrics: pd.DataFrame,
                                 performance_metrics: pd.DataFrame,
                                 company_data: Dict[str, str],
                                 db_connection) -> Dict[str, float]:
        """
        チーム構成と役職分布を考慮したHIEI値を計算

        Parameters
        ----------
        team_data : pd.DataFrame
            チームメンバーデータ (役職情報を含む)
            必須カラム: 'employee_id', 'position_title'
        health_metrics : pd.DataFrame
            健康指標データ
            必須カラム: 'employee_id', 'vas_before', 'vas_after'
        performance_metrics : pd.DataFrame
            パフォーマンス指標データ
            必須カラム: 'employee_id', 'productivity_before', 'productivity_after'
        company_data : Dict[str, str]
            企業データ（業種などの情報を含む）
            必須キー: 'industry'（業種名）
        db_connection
            PostgreSQLデータベース接続オブジェクト

        Returns
        -------
        Dict[str, float]
            計算結果を含む辞書
        """
        try:
            self.logger.info("チームベースのHIEI計算を開始")

            if 'industry' not in company_data:
                raise ValueError("企業データに業種情報（industry）が含まれていません")

            industry_name = company_data['industry']

            # チームデータと健康/パフォーマンスデータを結合
            merged_data = pd.merge(team_data, health_metrics, on='employee_id')
            merged_data = pd.merge(merged_data, performance_metrics, on='employee_id')

            # 役職ごとにグループ化して平均改善率を計算
            role_improvements = merged_data.groupby('position_title').apply(
                lambda x: {
                    'vas_improvement': (x['vas_after'].mean() - x['vas_before'].mean()) / x['vas_before'].mean() * 100,
                    'productivity_improvement': (x['productivity_after'].mean() - x['productivity_before'].mean()) / x['productivity_before'].mean() * 100,
                    'count': len(x)
                }
            ).to_dict()

            # 離職率のデータがなければ仮のゼロ値を設定
            turnover_reductions = {role: 0 for role in role_improvements.keys()}

            # 改善データを役職ごとに整理
            vas_improvements = {role: data['vas_improvement'] for role, data in role_improvements.items()}
            productivity_improvements = {role: data['productivity_improvement'] for role, data in role_improvements.items()}

            # 役職別重み付けHIEI値を計算
            result = self.calculate_role_weighted_hiei(
                vas_improvements,
                productivity_improvements,
                turnover_reductions,
                company_data,
                db_connection
            )

            # チーム構成情報を追加
            result['team_composition'] = {role: data['count'] for role, data in role_improvements.items()}
            result['total_team_size'] = merged_data['employee_id'].nunique()

            self.logger.info(f"チームベースのHIEI計算が完了しました。最終HIEI値: {result['hiei']:.2f}")
            return result

        except Exception as e:
            self.logger.error(f"チームベースのHIEI計算中にエラーが発生しました: {str(e)}")
            raise