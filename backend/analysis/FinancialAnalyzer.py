import pandas as pd
import numpy as np
import gc
import weakref
from typing import Dict, List, Tuple, Optional, Union, Any, ContextManager
from datetime import datetime, timedelta
from contextlib import contextmanager
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
        self._temp_data_refs = weakref.WeakValueDictionary()
        self._plot_resources = weakref.WeakValueDictionary()
        self.logger.info("FinancialAnalyzer initialized")

    def __del__(self):
        """デストラクタ - リソース自動解放"""
        self.release_resources()

    def release_resources(self):
        """明示的なリソース解放メソッド"""
        try:
            # 一時データの解放
            self._temp_data_refs.clear()

            # プロットリソースの解放
            for plot in list(self._plot_resources.values()):
                try:
                    import matplotlib.pyplot as plt
                    plt.close(plot)
                except:
                    pass
            self._plot_resources.clear()

            gc.collect()
            self.logger.info("FinancialAnalyzer resources released")
        except Exception as e:
            self.logger.error(f"リソース解放中にエラーが発生しました: {str(e)}")

    @contextmanager
    def _managed_dataframe(self, df: pd.DataFrame, name: str = "temp_df") -> ContextManager[pd.DataFrame]:
        """
        データフレームのリソース管理用コンテキストマネージャ

        Parameters
        ----------
        df : pd.DataFrame
            管理対象のデータフレーム
        name : str
            データフレームの識別名

        Yields
        ------
        pd.DataFrame
            管理対象のデータフレーム
        """
        try:
            # データフレームを弱参照辞書に登録
            self._temp_data_refs[name] = df
            self.logger.debug(f"Dataframe {name} registered for management")
            yield df
        finally:
            # 明示的に参照を削除
            if name in self._temp_data_refs:
                del self._temp_data_refs[name]
                self.logger.debug(f"Dataframe {name} released from management")
            # 部分的なガベージコレクションを実行
            gc.collect()

    @contextmanager
    def _plot_context(self, name: str = "temp_plot"):
        """
        プロットリソース管理用コンテキストマネージャ

        Parameters
        ----------
        name : str
            プロットの識別名

        Yields
        ------
        int または Figure
            プロットIDまたはFigureオブジェクト
        """
        import matplotlib.pyplot as plt

        try:
            # 新しいフィギュアを作成
            fig = plt.figure()
            self._plot_resources[name] = fig
            self.logger.debug(f"Plot {name} registered for management")
            yield fig
        finally:
            # 明示的にプロットリソースをクリーンアップ
            plt.close(fig)
            if name in self._plot_resources:
                del self._plot_resources[name]
            self.logger.debug(f"Plot {name} released from management")

    def _optimize_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データフレームのデータ型を最適化してメモリ使用量を削減

        Parameters
        ----------
        df : pd.DataFrame
            最適化対象のデータフレーム

        Returns
        -------
        pd.DataFrame
            最適化されたデータフレーム
        """
        try:
            # 数値データ型の最適化
            for col in df.select_dtypes(include=['int64']).columns:
                c_min, c_max = df[col].min(), df[col].max()
                if c_min >= 0:
                    if c_max < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                else:
                    if c_min > -128 and c_max < 127:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > -32768 and c_max < 32767:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > -2147483648 and c_max < 2147483647:
                        df[col] = df[col].astype(np.int32)

            # 浮動小数点の最適化
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype(np.float32)

            return df
        except Exception as e:
            self.logger.warning(f"データ型最適化中にエラーが発生しました: {str(e)}")
            return df  # 元のデータフレームを返す

    def estimate_memory_usage(self, data_rows: int, data_cols: int) -> Dict[str, float]:
        """
        メモリ使用量を推定する

        Parameters
        ----------
        data_rows : int
            データの行数
        data_cols : int
            データの列数

        Returns
        -------
        Dict[str, float]
            推定メモリ使用量(MB)
        """
        try:
            # 基本的なメモリ使用量の推定
            # 1. 入力データフレームのサイズ推定 (8バイト/セル)
            df_size_mb = (data_rows * data_cols * 8) / (1024 * 1024)

            # 2. 計算結果と中間データのサイズ
            results_mb = data_rows * 0.1  # 結果サイズの概算

            # 3. その他のオーバーヘッド
            overhead_mb = 20  # 固定オーバーヘッド

            total_mb = df_size_mb + results_mb + overhead_mb

            return {
                'dataframe_mb': df_size_mb,
                'results_mb': results_mb,
                'overhead_mb': overhead_mb,
                'total_mb': total_mb
            }
        except Exception as e:
            self.logger.error(f"メモリ使用量推定中にエラーが発生しました: {str(e)}")
            return {'total_mb': 50}  # デフォルト値を返す

    def calculate_burn_rate(self,
                          financial_data: pd.DataFrame,
                          period: str = 'monthly',
                          cash_column: str = 'cash_balance',
                          expense_columns: Optional[List[str]] = None,
                          optimize_memory: bool = True) -> Dict[str, Any]:
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
        optimize_memory : bool, optional
            メモリ最適化を行うかどうか (デフォルト: True)

        Returns
        -------
        Dict[str, Any]
            バーン率、ランウェイ、関連指標を含む辞書
        """
        try:
            self.logger.info(f"バーン率計算を開始: データサイズ={len(financial_data)}行, 期間={period}")

            # メモリ使用量の推定
            memory_estimate = self.estimate_memory_usage(len(financial_data), len(financial_data.columns))
            self.logger.debug(f"推定メモリ使用量: {memory_estimate['total_mb']:.2f} MB")

            # 入力データの検証
            if not financial_data.index.is_all_dates and not isinstance(financial_data.index, pd.DatetimeIndex):
                self.logger.warning("インデックスが日付型ではありません。変換を試みます。")
                date_columns = [col for col in financial_data.columns if 'date' in col.lower()]

                # 日付列があれば最初の列をインデックスに使用
                if date_columns and date_columns[0] in financial_data.columns:
                    try:
                        with self._managed_dataframe(financial_data.copy(), "financial_data_original") as df:
                            df[date_columns[0]] = pd.to_datetime(df[date_columns[0]])
                            df = df.set_index(date_columns[0])
                            return self._calculate_burn_rate_internal(
                                df, period, cash_column, expense_columns, optimize_memory
                            )
                    except Exception as date_error:
                        self.logger.error(f"日付列の変換中にエラーが発生しました: {str(date_error)}")
                        raise ValueError(f"日付インデックスへの変換に失敗しました: {str(date_error)}")
                else:
                    raise ValueError("データにはDatetimeIndexまたは日付列が必要です")
            else:
                with self._managed_dataframe(financial_data, "financial_data") as df:
                    return self._calculate_burn_rate_internal(
                        df, period, cash_column, expense_columns, optimize_memory
                    )

        except Exception as e:
            self.logger.error(f"バーン率計算中にエラーが発生しました: {str(e)}")
            raise
        finally:
            # 明示的なメモリ解放
            gc.collect()

    def _calculate_burn_rate_internal(self,
                                    data: pd.DataFrame,
                                    period: str,
                                    cash_column: str,
                                    expense_columns: Optional[List[str]],
                                    optimize_memory: bool) -> Dict[str, Any]:
        """
        バーン率計算の内部実装

        Parameters
        ----------
        同calculate_burn_rateと同様

        Returns
        -------
        同calculate_burn_rateと同様
        """
        try:
            # データ型の最適化（メモリ消費削減）
            if optimize_memory:
                data = self._optimize_dataframe_types(data)

            # 期間ごとのデータにリサンプリング
            if period == 'monthly':
                resampled_data = data.resample('M').last()
                months_factor = 1
            elif period == 'quarterly':
                resampled_data = data.resample('Q').last()
                months_factor = 3
            else:
                raise ValueError("Period must be 'monthly' or 'quarterly'")

            with self._managed_dataframe(resampled_data, "resampled_data") as period_data:
                # 入力データの検証
                if cash_column not in period_data.columns:
                    raise ValueError(f"現金残高カラム '{cash_column}' がデータに見つかりません")

                if expense_columns:
                    # 指定された費用項目を検証
                    missing_columns = [col for col in expense_columns if col not in period_data.columns]
                    if missing_columns:
                        self.logger.warning(f"以下の費用項目がデータに見つかりません: {', '.join(missing_columns)}")
                        expense_columns = [col for col in expense_columns if col in period_data.columns]
                        if not expense_columns:
                            self.logger.warning("有効な費用項目がないため、現金残高の変化からバーン率を計算します")
                            expense_columns = None

                # キャッシュバーン率の計算
                if expense_columns:
                    # 費用項目から直接計算
                    with self._managed_dataframe(period_data[expense_columns], "expenses_data") as expenses_df:
                        expenses = expenses_df.sum(axis=1)
                        burn_rate = expenses.mean()
                        self.logger.info(f"費用項目から計算したバーン率: {burn_rate:.2f}")
                else:
                    # 現金残高の変化から計算
                    cash_changes = period_data[cash_column].diff().dropna()
                    # 負の値（現金減少）だけを抽出して平均を計算
                    negative_changes = cash_changes[cash_changes < 0]
                    if negative_changes.empty:
                        self.logger.warning("現金減少が見つかりません。バーン率をゼロとします。")
                        burn_rate = 0
                    else:
                        burn_rate = abs(negative_changes.mean())
                        self.logger.info(f"現金残高の変化から計算したバーン率: {burn_rate:.2f}")

                # 最新の現金残高を取得
                latest_cash = period_data[cash_column].iloc[-1]

                # ランウェイの計算（月数）
                if burn_rate > 0:
                    runway_months = latest_cash / burn_rate
                    self.logger.info(f"ランウェイ: {runway_months:.2f}ヶ月")
                else:
                    runway_months = float('inf')
                    self.logger.info("バーン率がゼロのため、ランウェイは無限大")

                # 結果の格納
                result = {
                    'burn_rate': burn_rate,
                    'runway_months': runway_months,
                    'runway_quarters': runway_months / 3,
                    'runway_years': runway_months / 12,
                    'latest_cash': latest_cash,
                    'period': period,
                    'calculation_timestamp': datetime.now().isoformat(),
                    'data_points': len(period_data),
                    'calculation_method': 'expense_items' if expense_columns else 'cash_changes',
                    'memory_usage_mb': memory_estimate['total_mb'] if 'memory_estimate' in locals() else None
                }

                return result
        except Exception as e:
            self.logger.error(f"バーン率内部計算中にエラーが発生しました: {str(e)}")
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