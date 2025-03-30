import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy import stats
from .base import BaseAnalyzer

class FinancialAnalyzer(BaseAnalyzer):
    """
    財務分析モジュール

    投資先企業の包括的な財務状況と成長性を評価するためのクラス
    """

    def __init__(self):
        """
        FinancialAnalyzerの初期化
        """
        super().__init__(analysis_type='financial')
        self.logger.info("FinancialAnalyzer initialized")

    def calculate_burn_rate(self,
                          financial_data: pd.DataFrame,
                          period: str = 'monthly',
                          cash_column: str = 'cash_balance',
                          expense_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        キャッシュバーン率とランウェイ（資金枯渇までの期間）を計算

        Parameters
        ----------
        financial_data : pd.DataFrame
            財務データ（日付インデックス付き）
        period : str, optional
            'monthly'または'quarterly'（デフォルト: 'monthly'）
        cash_column : str, optional
            現金残高のカラム名（デフォルト: 'cash_balance'）
        expense_columns : List[str], optional
            費用項目のカラム名リスト。指定がなければ現金残高の変化から計算

        Returns
        -------
        Dict[str, Any]
            バーン率、ランウェイ、関連指標を含む辞書
        """
        try:
            if not isinstance(financial_data.index, pd.DatetimeIndex):
                self.logger.warning("Financial data index is not DatetimeIndex, attempting to convert")
                try:
                    financial_data = financial_data.set_index(pd.DatetimeIndex(financial_data.index))
                except:
                    raise ValueError("Could not convert index to DatetimeIndex")

            # 期間ごとのデータにリサンプリング
            if period == 'monthly':
                data = financial_data.resample('M').last()
                months_factor = 1
            elif period == 'quarterly':
                data = financial_data.resample('Q').last()
                months_factor = 3
            else:
                raise ValueError("Period must be 'monthly' or 'quarterly'")

            # キャッシュバーン率の計算
            if expense_columns:
                # 費用項目から直接計算
                expenses = data[expense_columns].sum(axis=1)
                burn_rate = expenses.mean()
            else:
                # 現金残高の変化から計算
                cash_changes = data[cash_column].diff().dropna()
                # 負の値（現金減少）だけを抽出して平均を計算
                negative_changes = cash_changes[cash_changes < 0]
                burn_rate = abs(negative_changes.mean()) if not negative_changes.empty else 0

            # 最新の現金残高を取得
            latest_cash = data[cash_column].iloc[-1]

            # ランウェイの計算（月数）
            runway_months = latest_cash / burn_rate if burn_rate > 0 else float('inf')

            # 結果の格納
            result = {
                'burn_rate': burn_rate,
                'runway_months': runway_months,
                'runway_quarters': runway_months / 3,
                'runway_years': runway_months / 12,
                'latest_cash': latest_cash,
                'period': period,
                'calculation_date': datetime.now().strftime('%Y-%m-%d')
            }

            self.logger.info(f"Burn rate calculated: {burn_rate:.2f} per {period} period, runway: {runway_months:.2f} months")
            return result
        except Exception as e:
            self.logger.error(f"Error calculating burn rate: {str(e)}")
            raise

    def compare_burn_rate_to_benchmarks(self,
                                      company_burn_rate: float,
                                      company_runway: float,
                                      industry_benchmarks: Dict[str, Dict[str, float]],
                                      industry: str) -> Dict[str, Any]:
        """
        企業のバーン率を業界ベンチマークと比較

        Parameters
        ----------
        company_burn_rate : float
            企業のバーン率
        company_runway : float
            企業のランウェイ（月数）
        industry_benchmarks : Dict[str, Dict[str, float]]
            業界別のバーン率とランウェイのベンチマーク
        industry : str
            企業の業界

        Returns
        -------
        Dict[str, Any]
            ベンチマーク比較結果
        """
        try:
            if industry not in industry_benchmarks:
                self.logger.warning(f"Industry {industry} not found in benchmarks, using average")
                # すべての業界の平均値を計算
                avg_burn_rate = np.mean([b['burn_rate'] for b in industry_benchmarks.values()])
                avg_runway = np.mean([b['runway'] for b in industry_benchmarks.values()])
                benchmark = {'burn_rate': avg_burn_rate, 'runway': avg_runway}
            else:
                benchmark = industry_benchmarks[industry]

            # 比較結果の計算
            burn_rate_ratio = company_burn_rate / benchmark['burn_rate'] if benchmark['burn_rate'] > 0 else float('inf')
            runway_ratio = company_runway / benchmark['runway'] if benchmark['runway'] > 0 else float('inf')

            # バーン率は低いほど良い、ランウェイは長いほど良い
            burn_rate_performance = -1 * (burn_rate_ratio - 1) * 100  # -100%〜∞% （正の値が良い）
            runway_performance = (runway_ratio - 1) * 100  # -100%〜∞% （正の値が良い）

            # 総合スコアの計算 (0-100スケール)
            score = 50 * (1 + np.tanh(burn_rate_performance / 100)) + 50 * (1 + np.tanh(runway_performance / 100))

            result = {
                'benchmark_burn_rate': benchmark['burn_rate'],
                'benchmark_runway': benchmark['runway'],
                'burn_rate_ratio': burn_rate_ratio,
                'runway_ratio': runway_ratio,
                'burn_rate_performance': burn_rate_performance,
                'runway_performance': runway_performance,
                'overall_score': min(max(score / 2, 0), 100),  # 0-100スケールに正規化
                'industry': industry
            }

            self.logger.info(f"Burn rate benchmark comparison completed for {industry} industry")
            return result
        except Exception as e:
            self.logger.error(f"Error comparing burn rate to benchmarks: {str(e)}")
            raise

    def analyze_unit_economics(self,
                             revenue_data: pd.DataFrame,
                             customer_data: pd.DataFrame,
                             cost_data: pd.DataFrame,
                             customer_id_column: str = 'customer_id',
                             revenue_column: str = 'revenue',
                             acquisition_cost_column: str = 'acquisition_cost',
                             acquisition_date_column: str = 'acquisition_date',
                             churn_date_column: Optional[str] = None) -> Dict[str, Any]:
        """
        LTV/CAC分析を実行

        Parameters
        ----------
        revenue_data : pd.DataFrame
            顧客別収益データ (customer_id, date, revenue)
        customer_data : pd.DataFrame
            顧客データ (customer_id, acquisition_date, [churn_date])
        cost_data : pd.DataFrame
            顧客獲得コストデータ (customer_id, acquisition_cost)
        customer_id_column : str, optional
            顧客IDのカラム名（デフォルト: 'customer_id'）
        revenue_column : str, optional
            収益のカラム名（デフォルト: 'revenue'）
        acquisition_cost_column : str, optional
            獲得コストのカラム名（デフォルト: 'acquisition_cost'）
        acquisition_date_column : str, optional
            獲得日のカラム名（デフォルト: 'acquisition_date'）
        churn_date_column : str, optional
            解約日のカラム名（デフォルト: None）

        Returns
        -------
        Dict[str, Any]
            CAC、LTV、LTV/CAC比率などの指標を含む辞書
        """
        try:
            # CAC（顧客獲得コスト）の計算
            cac = cost_data[acquisition_cost_column].mean()

            # 顧客期間の計算
            if churn_date_column and churn_date_column in customer_data.columns:
                # チャーン日から顧客期間を計算
                customer_data['tenure'] = (
                    customer_data[churn_date_column] - customer_data[acquisition_date_column]
                ).dt.days / 30  # 月数に変換

                # チャーンしていない顧客は現在までの期間を使用
                customer_data.loc[customer_data[churn_date_column].isnull(), 'tenure'] = (
                    datetime.now() - customer_data.loc[customer_data[churn_date_column].isnull(), acquisition_date_column]
                ).dt.days / 30
            else:
                # チャーン日がない場合は獲得日から現在までの期間を使用
                customer_data['tenure'] = (
                    datetime.now() - customer_data[acquisition_date_column]
                ).dt.days / 30

            # 平均顧客期間
            avg_tenure = customer_data['tenure'].mean()

            # 月間平均収益（ARPU）の計算
            customer_revenues = revenue_data.groupby(customer_id_column)[revenue_column].sum()
            customer_data = customer_data.merge(
                customer_revenues.reset_index(),
                on=customer_id_column,
                how='left'
            )
            customer_data['monthly_revenue'] = customer_data[revenue_column] / customer_data['tenure']
            arpu = customer_data['monthly_revenue'].mean()

            # LTV（顧客生涯価値）の計算
            # 簡易版: LTV = ARPU * 平均顧客期間
            ltv_simple = arpu * avg_tenure

            # 割引率を考慮したLTV (使用する場合)
            discount_rate = 0.1  # 年率10%
            monthly_discount_rate = (1 + discount_rate) ** (1/12) - 1

            # DCF法によるLTV計算
            max_months = int(customer_data['tenure'].max()) + 12  # 十分な将来期間
            ltv_dcf = 0
            for month in range(1, max_months + 1):
                # 月ごとの生存率（簡易指数関数）
                survival_rate = np.exp(-month / avg_tenure) if avg_tenure > 0 else 0
                # 割引後の価値
                discounted_value = arpu * survival_rate / ((1 + monthly_discount_rate) ** month)
                ltv_dcf += discounted_value

            # LTV/CAC比率
            ltv_cac_ratio = ltv_dcf / cac if cac > 0 else float('inf')

            # 収益分析
            if 'date' in revenue_data.columns:
                # 月次収益の計算
                revenue_data['year_month'] = revenue_data['date'].dt.to_period('M')
                mrr = revenue_data.groupby('year_month')[revenue_column].sum().reset_index()

                # MRRの変化率
                if len(mrr) >= 2:
                    mrr['growth_rate'] = mrr[revenue_column].pct_change()
                    avg_mrr_growth = mrr['growth_rate'].mean()
                else:
                    avg_mrr_growth = 0

                # ARR (年間経常収益) = 最新のMRR × 12
                arr = mrr[revenue_column].iloc[-1] * 12 if not mrr.empty else 0
            else:
                avg_mrr_growth = 0
                arr = 0

            result = {
                'cac': cac,
                'ltv_simple': ltv_simple,
                'ltv_dcf': ltv_dcf,
                'ltv_cac_ratio': ltv_cac_ratio,
                'arpu': arpu,
                'avg_tenure_months': avg_tenure,
                'mrr': arpu * len(customer_data),  # 推定MRR
                'arr': arr,
                'avg_mrr_growth_rate': avg_mrr_growth,
                'payback_period_months': cac / arpu if arpu > 0 else float('inf')  # 回収期間
            }

            self.logger.info(f"Unit economics analysis completed: LTV/CAC = {ltv_cac_ratio:.2f}")
            return result
        except Exception as e:
            self.logger.error(f"Error analyzing unit economics: {str(e)}")
            raise

    def calculate_growth_metrics(self,
                               financial_data: pd.DataFrame,
                               metric_column: str = 'revenue',
                               benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        成長指標を計算し評価

        Parameters
        ----------
        financial_data : pd.DataFrame
            財務データ（日付インデックス付き）
        metric_column : str, optional
            分析する指標のカラム名（デフォルト: 'revenue'）
        benchmark_data : pd.DataFrame, optional
            業界ベンチマークデータ（デフォルト: None）

        Returns
        -------
        Dict[str, Any]
            成長指標と評価を含む辞書
        """
        try:
            if not isinstance(financial_data.index, pd.DatetimeIndex):
                self.logger.warning("Financial data index is not DatetimeIndex, attempting to convert")
                try:
                    financial_data = financial_data.set_index(pd.DatetimeIndex(financial_data.index))
                except:
                    raise ValueError("Could not convert index to DatetimeIndex")

            # 時系列データの整理
            monthly_data = financial_data.resample('M').last()

            # 成長率の計算
            monthly_data['mom_growth'] = monthly_data[metric_column].pct_change()

            # 四半期データの作成
            quarterly_data = financial_data.resample('Q').last()
            quarterly_data['qoq_growth'] = quarterly_data[metric_column].pct_change()

            # 年次データの作成
            yearly_data = financial_data.resample('Y').last()
            yearly_data['yoy_growth'] = yearly_data[metric_column].pct_change()

            # 平均成長率の計算
            avg_mom_growth = monthly_data['mom_growth'].mean()
            avg_qoq_growth = quarterly_data['qoq_growth'].mean()
            avg_yoy_growth = yearly_data['yoy_growth'].mean()

            # T2D3フレームワーク評価（SaaSの成長基準）
            # T2D3 = 3年で3倍、その後2年連続で2倍、その後2年連続で2倍
            years_data = monthly_data.resample('Y').last()

            t2d3_score = 0
            if len(years_data) >= 2:
                growth_rates = []
                for i in range(1, min(6, len(years_data))):
                    growth_rate = years_data[metric_column].iloc[i] / years_data[metric_column].iloc[i-1]
                    growth_rates.append(growth_rate)

                if len(growth_rates) >= 1:
                    if len(growth_rates) >= 3 and np.prod(growth_rates[:3]) >= 3:
                        t2d3_score += 0.5  # 3年で3倍

                    if len(growth_rates) >= 5:
                        if growth_rates[3] >= 2 and growth_rates[4] >= 2:
                            t2d3_score += 0.25  # その後2年連続で2倍

                        if len(growth_rates) >= 7:
                            if growth_rates[5] >= 2 and growth_rates[6] >= 2:
                                t2d3_score += 0.25  # さらに2年連続で2倍

            # Rule of 40評価（成長率 + 利益率 >= 40%）
            has_profit_data = 'profit_margin' in financial_data.columns

            rule_of_40_score = 0
            if has_profit_data:
                latest_profit_margin = financial_data['profit_margin'].iloc[-1]
                latest_growth_rate = yearly_data['yoy_growth'].iloc[-1] if not yearly_data.empty and len(yearly_data['yoy_growth']) > 0 else 0
                rule_of_40_value = latest_growth_rate * 100 + latest_profit_margin
                rule_of_40_score = rule_of_40_value / 40  # 40%を1.0とするスコア

            # 業界ベンチマークとの比較
            benchmark_comparison = {}
            if benchmark_data is not None:
                for period in ['mom', 'qoq', 'yoy']:
                    if f'{period}_growth' in benchmark_data.columns:
                        company_growth = locals()[f'avg_{period}_growth']
                        benchmark_growth = benchmark_data[f'{period}_growth'].mean()
                        relative_performance = (company_growth - benchmark_growth) / abs(benchmark_growth) if benchmark_growth != 0 else float('inf')
                        benchmark_comparison[f'{period}_vs_benchmark'] = relative_performance

            result = {
                'avg_mom_growth': avg_mom_growth,
                'avg_qoq_growth': avg_qoq_growth,
                'avg_yoy_growth': avg_yoy_growth,
                'latest_mom_growth': monthly_data['mom_growth'].iloc[-1] if not monthly_data.empty and len(monthly_data['mom_growth']) > 0 else 0,
                'latest_qoq_growth': quarterly_data['qoq_growth'].iloc[-1] if not quarterly_data.empty and len(quarterly_data['qoq_growth']) > 0 else 0,
                'latest_yoy_growth': yearly_data['yoy_growth'].iloc[-1] if not yearly_data.empty and len(yearly_data['yoy_growth']) > 0 else 0,
                't2d3_score': t2d3_score,
                'rule_of_40_score': rule_of_40_score
            }

            # ベンチマーク比較結果を追加
            result.update(benchmark_comparison)

            self.logger.info(f"Growth metrics calculation completed. YoY growth: {avg_yoy_growth:.2%}")
            return result
        except Exception as e:
            self.logger.error(f"Error calculating growth metrics: {str(e)}")
            raise

    def analyze_funding_efficiency(self,
                                 funding_data: pd.DataFrame,
                                 valuation_data: pd.DataFrame,
                                 competitor_funding: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        資金調達効率を分析

        Parameters
        ----------
        funding_data : pd.DataFrame
            資金調達データ (date, round, amount, investors)
        valuation_data : pd.DataFrame
            バリュエーションデータ (date, valuation, revenue_multiple)
        competitor_funding : Dict[str, pd.DataFrame], optional
            競合企業の資金調達データ（企業名: データフレーム）

        Returns
        -------
        Dict[str, Any]
            資金調達効率分析結果
        """
        try:
            # 資金調達ラウンドごとのバリュエーション推移
            funding_rounds = funding_data.copy()

            # 各ラウンドにバリュエーションデータを紐付け
            funding_rounds = pd.merge_asof(
                funding_rounds.sort_values('date'),
                valuation_data.sort_values('date'),
                on='date',
                direction='nearest'
            )

            # 希薄化率の計算
            funding_rounds['cumulative_raised'] = funding_rounds['amount'].cumsum()
            funding_rounds['dilution'] = funding_rounds['amount'] / funding_rounds['valuation']

            # 資金調達効率（バリュエーション / 調達額の推移）
            funding_rounds['funding_efficiency'] = funding_rounds['valuation'] / funding_rounds['cumulative_raised']

            # バリュエーション倍率の計算
            if len(funding_rounds) >= 2:
                funding_rounds['valuation_multiple'] = funding_rounds['valuation'].pct_change() + 1
                last_funding_multiple = funding_rounds['valuation_multiple'].iloc[-1]
            else:
                last_funding_multiple = 1.0

            # 競合他社との比較
            competitor_comparison = {}
            if competitor_funding:
                all_efficiency = []
                all_dilution = []

                for competitor, data in competitor_funding.items():
                    # 競合のデータも同様に処理
                    comp_data = data.copy()
                    comp_data['cumulative_raised'] = comp_data['amount'].cumsum()

                    if 'valuation' in comp_data.columns:
                        comp_data['funding_efficiency'] = comp_data['valuation'] / comp_data['cumulative_raised']
                        comp_data['dilution'] = comp_data['amount'] / comp_data['valuation']

                        # 最新の効率と希薄化を記録
                        last_efficiency = comp_data['funding_efficiency'].iloc[-1]
                        last_dilution = comp_data['dilution'].iloc[-1]

                        competitor_comparison[competitor] = {
                            'funding_efficiency': last_efficiency,
                            'dilution': last_dilution
                        }

                        all_efficiency.append(last_efficiency)
                        all_dilution.append(last_dilution)

                # 競合との相対比較
                if all_efficiency and all_dilution:
                    avg_competitor_efficiency = np.mean(all_efficiency)
                    avg_competitor_dilution = np.mean(all_dilution)

                    own_efficiency = funding_rounds['funding_efficiency'].iloc[-1]
                    own_dilution = funding_rounds['dilution'].iloc[-1]

                    efficiency_percentile = stats.percentileofscore(all_efficiency, own_efficiency)
                    dilution_percentile = 100 - stats.percentileofscore(all_dilution, own_dilution)  # 低いほど良い

                    competitor_comparison['summary'] = {
                        'efficiency_vs_avg': own_efficiency / avg_competitor_efficiency if avg_competitor_efficiency > 0 else float('inf'),
                        'dilution_vs_avg': own_dilution / avg_competitor_dilution if avg_competitor_dilution > 0 else float('inf'),
                        'efficiency_percentile': efficiency_percentile,
                        'dilution_percentile': dilution_percentile
                    }

            result = {
                'total_raised': funding_rounds['amount'].sum(),
                'latest_valuation': funding_rounds['valuation'].iloc[-1] if not funding_rounds.empty else 0,
                'latest_funding_efficiency': funding_rounds['funding_efficiency'].iloc[-1] if not funding_rounds.empty else 0,
                'avg_dilution_per_round': funding_rounds['dilution'].mean(),
                'total_dilution': funding_rounds['dilution'].sum(),
                'valuation_growth_multiple': last_funding_multiple,
                'funding_rounds': [
                    {
                        'date': row['date'].strftime('%Y-%m-%d'),
                        'round': row['round'],
                        'amount': row['amount'],
                        'valuation': row['valuation'],
                        'dilution': row['dilution'],
                        'funding_efficiency': row['funding_efficiency']
                    }
                    for _, row in funding_rounds.iterrows()
                ],
                'competitor_comparison': competitor_comparison
            }

            self.logger.info(f"Funding efficiency analysis completed for {len(funding_rounds)} rounds")
            return result
        except Exception as e:
            self.logger.error(f"Error analyzing funding efficiency: {str(e)}")
            raise