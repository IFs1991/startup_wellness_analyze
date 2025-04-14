import pandas as pd
import numpy as np
import gc
import weakref
from typing import Dict, List, Tuple, Optional, Union, ContextManager
from contextlib import contextmanager
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
        self._temp_data_refs = weakref.WeakValueDictionary()
        self._plot_resources = weakref.WeakValueDictionary()
        self.logger.info("HealthInvestmentEffectIndexCalculator initialized")

    def __del__(self):
        """デストラクタ - リソース自動解放"""
        self.release_resources()

    def release_resources(self):
        """明示的なリソース解放メソッド"""
        try:
            # 一時データの解放
            self._temp_data_refs.clear()
            self._plot_resources.clear()
            gc.collect()
            self.logger.info("HealthInvestmentEffectIndexCalculator resources released")
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
        int
            プロットID (実際の値は重要ではない)
        """
        import matplotlib.pyplot as plt

        try:
            # 新しいフィギュアを作成
            fig = plt.figure()
            self._plot_resources[name] = fig
            self.logger.debug(f"Plot {name} registered for management")
            yield 1  # プロットIDは重要ではない
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

    def estimate_memory_usage(self,
                            data_size: int,
                            num_variables: int) -> Dict[str, float]:
        """
        メモリ使用量を推定する

        Parameters
        ----------
        data_size : int
            データ行数
        num_variables : int
            変数の数

        Returns
        -------
        Dict[str, float]
            推定メモリ使用量(MB)
        """
        try:
            # 基本的なメモリ使用量の推定
            # 1. データフレームのサイズ推定 (8バイト/セル)
            df_size_mb = (data_size * num_variables * 8) / (1024 * 1024)

            # 2. 計算結果の推定サイズ
            results_mb = 10  # 基本サイズ

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
        try:
            improvements = []
            common_columns = set(vas_before.columns).intersection(set(vas_after.columns))

            if not common_columns:
                self.logger.warning("VASデータに共通カラムがありません")
                return 0.0

            for column in common_columns:
                try:
                    before_mean = vas_before[column].mean()
                    after_mean = vas_after[column].mean()

                    # 改善率を計算 (VASスケールは値が小さいほど良いと仮定)
                    if before_mean > 0:
                        improvement = (before_mean - after_mean) / before_mean * 100
                        improvements.append(improvement)
                except Exception as column_error:
                    self.logger.warning(f"カラム '{column}' の処理中にエラーが発生しました: {str(column_error)}")
                    continue

            if improvements:
                mean_improvement = np.mean(improvements)
                self.logger.info(f"VAS改善率: {mean_improvement:.2f}%")
                return mean_improvement
            else:
                self.logger.warning("有効な改善率を計算できませんでした")
                return 0.0
        except Exception as e:
            self.logger.error(f"VAS改善率計算中にエラーが発生しました: {str(e)}")
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
        try:
            details = {}
            common_columns = set(vas_before.columns).intersection(set(vas_after.columns))

            if not common_columns:
                self.logger.warning("VASデータに共通カラムがありません")
                return {}

            for column in common_columns:
                try:
                    before_mean = vas_before[column].mean()
                    after_mean = vas_after[column].mean()

                    # 改善率を計算 (VASスケールは値が小さいほど良いと仮定)
                    if before_mean > 0:
                        improvement = (before_mean - after_mean) / before_mean * 100
                        details[column] = improvement
                    else:
                        details[column] = 0.0
                except Exception as column_error:
                    self.logger.warning(f"カラム '{column}' の詳細計算中にエラーが発生しました: {str(column_error)}")
                    details[column] = 0.0

            return details
        except Exception as e:
            self.logger.error(f"VAS改善詳細計算中にエラーが発生しました: {str(e)}")
            return {}

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
            # カラムの存在確認
            required_columns = ['revenue', 'employees']
            for column in required_columns:
                if column not in financial_data_before.columns:
                    self.logger.warning(f"導入前データに必須カラム '{column}' がありません")
                    return 0.0
                if column not in financial_data_after.columns:
                    self.logger.warning(f"導入後データに必須カラム '{column}' がありません")
                    return 0.0

            # NaN値の確認
            for df, label in [(financial_data_before, "導入前"), (financial_data_after, "導入後")]:
                for column in required_columns:
                    if df[column].isna().any():
                        self.logger.warning(f"{label}データの '{column}' カラムにNaN値があります")
                        # NaNを0で置換
                        df[column] = df[column].fillna(0)

            # 従業員1人あたりの収益で生産性を計算
            revenue_before = financial_data_before['revenue'].mean()
            employees_before = financial_data_before['employees'].mean()

            revenue_after = financial_data_after['revenue'].mean()
            employees_after = financial_data_after['employees'].mean()

            # 従業員数が0の場合のエラー処理
            if employees_before <= 0 or employees_after <= 0:
                self.logger.warning("従業員数がゼロまたは負数です。代替生産性指標を使用します")

                if revenue_before <= 0:
                    self.logger.warning("導入前の収益がゼロまたは負数です。生産性向上率を計算できません")
                    return 0.0

                # 代替計算: 単純な収益成長率
                improvement = (revenue_after - revenue_before) / revenue_before * 100
                self.logger.info(f"代替生産性指標(収益成長率): {improvement:.2f}%")
                return improvement

            # 通常の生産性計算
            productivity_before = revenue_before / employees_before
            productivity_after = revenue_after / employees_after

            if productivity_before <= 0:
                self.logger.warning("導入前の生産性がゼロまたは負数です。生産性向上率を計算できません")
                return 0.0

            improvement = (productivity_after - productivity_before) / productivity_before * 100
            self.logger.info(f"生産性向上率: {improvement:.2f}%")
            return improvement

        except Exception as e:
            self.logger.error(f"生産性向上率計算中にエラーが発生しました: {str(e)}")
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
            self.logger.info(f"エコシステム影響度計算を開始: {len(hiei_values)}企業、{network_adjacency.shape}の隣接行列")

            # 入力データの検証
            if not hiei_values:
                self.logger.warning("HIEI値が空です")
                return {}

            if network_adjacency.empty:
                self.logger.warning("ネットワーク隣接行列が空です")
                return hiei_values  # 影響がないので元のHIEI値をそのまま返す

            # 行列のサイズが大きい場合のメモリ最適化
            with self._managed_dataframe(network_adjacency, "network_adjacency") as adj_matrix:
                # データ型の最適化（必要に応じて）
                if adj_matrix.size > 10000:  # 大規模行列の場合
                    self.logger.info("大規模隣接行列のメモリ最適化を実行")
                    # 浮動小数点の精度を下げる
                    for col in adj_matrix.select_dtypes(include=['float64']).columns:
                        adj_matrix[col] = adj_matrix[col].astype(np.float32)

                # 結果格納用の辞書
                ecosystem_impact = {}

                # 進捗ログのための準備
                total_companies = len(hiei_values)
                log_interval = max(1, total_companies // 10)  # 10%ごとに進捗ログ

                # 企業ごとの計算
                for i, company in enumerate(hiei_values.keys()):
                    # 定期的な進捗ログ
                    if i % log_interval == 0:
                        self.logger.info(f"エコシステム影響度計算進捗: {i}/{total_companies}企業 ({i/total_companies*100:.1f}%)")

                    if company in adj_matrix.index and company in adj_matrix.columns:
                        # 当該企業と関連のある企業の重みを取得
                        connected_companies = adj_matrix.loc[company]

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

                self.logger.info(f"エコシステム影響度計算が完了しました: {len(ecosystem_impact)}企業")
                return ecosystem_impact

        except Exception as e:
            self.logger.error(f"エコシステム影響度計算中にエラーが発生しました: {str(e)}")
            # エラー時は元のHIEI値を返す
            return {k: v for k, v in hiei_values.items()}
        finally:
            # 明示的なメモリ解放
            gc.collect()

    def calculate_industry_benchmarks(self,
                                    hiei_values: Dict[str, float],
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
            self.logger.info(f"業界ベンチマーク計算を開始: {len(hiei_values)}企業、{len(industry_mapping)}業界マッピング")

            # 入力データの検証
            if not hiei_values:
                self.logger.warning("HIEI値が空です")
                return {}

            if not industry_mapping:
                self.logger.warning("業界マッピングが空です")
                return {}

            # 業界ごとにHIEI値をグループ化
            industry_hiei = {}
            missing_mapping = 0

            for company, industry in industry_mapping.items():
                if company in hiei_values:
                    if industry not in industry_hiei:
                        industry_hiei[industry] = []

                    industry_hiei[industry].append(hiei_values[company])
                else:
                    missing_mapping += 1

            if missing_mapping > 0:
                self.logger.warning(f"{missing_mapping}企業がHIEI値に見つかりませんでした")

            # 業界ごとの統計量を計算
            industry_benchmarks = {}
            for industry, values in industry_hiei.items():
                if not values:
                    industry_benchmarks[industry] = 0
                    continue

                try:
                    # 平均値の計算
                    mean_value = np.mean(values)

                    # 異常値の確認と処理
                    if not np.isfinite(mean_value):
                        self.logger.warning(f"業界 '{industry}' の平均値が無効です。0を使用します。")
                        mean_value = 0

                    industry_benchmarks[industry] = mean_value

                    # 詳細なログ記録
                    self.logger.debug(f"業界 '{industry}': サンプル数={len(values)}, 平均={mean_value:.2f}, "
                                      f"最小={min(values):.2f}, 最大={max(values):.2f}")

                except Exception as industry_error:
                    self.logger.error(f"業界 '{industry}' のベンチマーク計算中にエラーが発生しました: {str(industry_error)}")
                    industry_benchmarks[industry] = 0

            self.logger.info(f"業界ベンチマーク計算が完了しました: {len(industry_benchmarks)}業界")
            return industry_benchmarks

        except Exception as e:
            self.logger.error(f"業界ベンチマーク計算中にエラーが発生しました: {str(e)}")
            return {}
        finally:
            # 明示的なメモリ解放
            gc.collect()

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

            # 入力データの検証
            if 'industry' not in company_data:
                error_msg = "企業データに業種情報（industry）が含まれていません"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            industry_name = company_data['industry']
            self.logger.info(f"対象業種: {industry_name}")

            # 役職データの検証
            if not vas_improvements:
                self.logger.warning("VAS改善データが空です")
                return {'hiei': 0, 'error': '役職データが不足しています'}

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
            calculation_errors = []

            try:
                # 各役職について処理
                for role in vas_improvements.keys():
                    try:
                        # 役職の健康影響度の重み係数を取得
                        try:
                            weight = HealthImpactWeightUtility.get_health_impact_weight(
                                db_connection, industry_name, role
                            )
                            role_weights[role] = weight
                        except Exception as weight_error:
                            self.logger.warning(f"役職 '{role}' の重み取得中にエラーが発生しました: {str(weight_error)}")
                            # デフォルト値を使用
                            role_weights[role] = 1.0
                            calculation_errors.append(f"役職 '{role}' の重み取得エラー: {str(weight_error)}")

                        # 役職ごとのHIEI値を計算
                        vas_value = vas_improvements.get(role, 0)
                        productivity_value = productivity_improvements.get(role, 0)
                        turnover_value = turnover_reductions.get(role, 0)

                        role_hiei = self.calculate_hiei(
                            vas_value, productivity_value, turnover_value, custom_weights
                        )
                        role_hiei_values[role] = role_hiei

                        self.logger.debug(f"役職 '{role}' のHIEI: {role_hiei:.2f}, 重み: {role_weights[role]}")

                    except Exception as role_error:
                        self.logger.error(f"役職 '{role}' のHIEI計算中にエラーが発生しました: {str(role_error)}")
                        calculation_errors.append(f"役職 '{role}' の計算エラー: {str(role_error)}")
                        role_hiei_values[role] = 0
            except Exception as roles_error:
                self.logger.error(f"役職処理中にエラーが発生しました: {str(roles_error)}")
                raise

            # 全体の重み付け合計を計算
            total_weight = sum(role_weights.values())
            if total_weight <= 0:
                error_msg = "役職の重み係数の合計が0以下です"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

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

            # 計算エラーがあれば追加
            if calculation_errors:
                result['calculation_warnings'] = calculation_errors

            self.logger.info(f"役職別重み付けHIEI計算が完了しました。最終HIEI値: {final_hiei:.2f}")
            return result

        except Exception as e:
            self.logger.error(f"役職別重み付けHIEI計算中にエラーが発生しました: {str(e)}")
            raise
        finally:
            # 明示的なメモリ解放
            gc.collect()

    def calculate_team_based_hiei(self,
                                 team_data: pd.DataFrame,
                                 health_metrics: pd.DataFrame,
                                 performance_metrics: pd.DataFrame,
                                 company_data: Dict[str, str],
                                 db_connection,
                                 optimize_memory: bool = True) -> Dict[str, float]:
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
        optimize_memory : bool, optional
            メモリ最適化を行うかどうか (デフォルト: True)

        Returns
        -------
        Dict[str, float]
            計算結果を含む辞書
        """
        try:
            self.logger.info("チームベースのHIEI計算を開始")

            # 入力データの検証
            if 'industry' not in company_data:
                error_msg = "企業データに業種情報（industry）が含まれていません"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            industry_name = company_data['industry']
            self.logger.info(f"対象業種: {industry_name}, データサイズ: チーム={len(team_data)}行, 健康指標={len(health_metrics)}行, パフォーマンス={len(performance_metrics)}行")

            # 必須カラムの確認
            required_columns = {
                'team_data': ['employee_id', 'position_title'],
                'health_metrics': ['employee_id', 'vas_before', 'vas_after'],
                'performance_metrics': ['employee_id', 'productivity_before', 'productivity_after']
            }

            for df_name, columns in required_columns.items():
                df = eval(df_name)  # 各データフレームを取得
                missing_columns = [col for col in columns if col not in df.columns]
                if missing_columns:
                    error_msg = f"{df_name}に必須カラムがありません: {', '.join(missing_columns)}"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

            # メモリ最適化が必要な場合
            if optimize_memory:
                self.logger.info("データフレームのメモリ最適化を開始")
                with self._managed_dataframe(team_data, "team_data") as opt_team_data, \
                     self._managed_dataframe(health_metrics, "health_metrics") as opt_health_metrics, \
                     self._managed_dataframe(performance_metrics, "performance_metrics") as opt_performance_metrics:

                    # データ型の最適化
                    self._optimize_dataframe_types(opt_team_data)
                    self._optimize_dataframe_types(opt_health_metrics)
                    self._optimize_dataframe_types(opt_performance_metrics)

                    # 計算の実行
                    result = self._execute_team_based_hiei_calculation(
                        opt_team_data, opt_health_metrics, opt_performance_metrics,
                        company_data, db_connection
                    )
            else:
                # 最適化なしでの実行
                result = self._execute_team_based_hiei_calculation(
                    team_data, health_metrics, performance_metrics,
                    company_data, db_connection
                )

            self.logger.info(f"チームベースのHIEI計算が完了しました。最終HIEI値: {result['hiei']:.2f}")
            return result

        except Exception as e:
            self.logger.error(f"チームベースのHIEI計算中にエラーが発生しました: {str(e)}")
            raise
        finally:
            # 明示的なメモリ解放
            gc.collect()

    def _execute_team_based_hiei_calculation(self,
                                           team_data: pd.DataFrame,
                                           health_metrics: pd.DataFrame,
                                           performance_metrics: pd.DataFrame,
                                           company_data: Dict[str, str],
                                           db_connection) -> Dict[str, float]:
        """
        チームベースHIEI計算の実行部分（内部メソッド）

        Parameters
        ----------
        同calculate_team_based_hieiと同様

        Returns
        -------
        Dict[str, float]
            計算結果を含む辞書
        """
        try:
            # チームデータと健康/パフォーマンスデータを結合
            with self._managed_dataframe(pd.merge(team_data, health_metrics, on='employee_id'), "merged_health") as merged_data:
                merged_data = pd.merge(merged_data, performance_metrics, on='employee_id')

                # データ検証 - 結合後に行数が0にならないか確認
                if len(merged_data) == 0:
                    self.logger.warning("結合後のデータが空になりました。マージキーを確認してください。")
                    return {'hiei': 0, 'error': 'データ結合エラー: 結合結果が空です'}

                # 役職ごとにグループ化して平均改善率を計算
                role_data = {}
                for role, group in merged_data.groupby('position_title'):
                    # NaN値のチェックと処理
                    vas_before_mean = group['vas_before'].fillna(0).mean()
                    vas_after_mean = group['vas_after'].fillna(0).mean()
                    prod_before_mean = group['productivity_before'].fillna(0).mean()
                    prod_after_mean = group['productivity_after'].fillna(0).mean()

                    # 改善率の計算 (ゼロ除算を回避)
                    vas_improvement = 0
                    if vas_before_mean > 0:
                        vas_improvement = (vas_after_mean - vas_before_mean) / vas_before_mean * 100

                    prod_improvement = 0
                    if prod_before_mean > 0:
                        prod_improvement = (prod_after_mean - prod_before_mean) / prod_before_mean * 100

                    role_data[role] = {
                        'vas_improvement': vas_improvement,
                        'productivity_improvement': prod_improvement,
                        'count': len(group)
                    }

                # 改善データを役職ごとに整理
                vas_improvements = {role: data['vas_improvement'] for role, data in role_data.items()}
                productivity_improvements = {role: data['productivity_improvement'] for role, data in role_data.items()}

                # 離職率のデータがなければ仮のゼロ値を設定
                turnover_reductions = {role: 0 for role in role_data.keys()}

                # 役職別重み付けHIEI値を計算
                result = self.calculate_role_weighted_hiei(
                    vas_improvements,
                    productivity_improvements,
                    turnover_reductions,
                    company_data,
                    db_connection
                )

                # チーム構成情報を追加
                result['team_composition'] = {role: data['count'] for role, data in role_data.items()}
                result['total_team_size'] = merged_data['employee_id'].nunique()
                result['calculation_details'] = {
                    'vas_improvements': vas_improvements,
                    'productivity_improvements': productivity_improvements
                }

                return result

        except Exception as e:
            self.logger.error(f"チームベースHIEI計算の実行中にエラーが発生しました: {str(e)}")
            raise ValueError(f"チームベースHIEI計算エラー: {str(e)}")