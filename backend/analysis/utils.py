"""
分析モジュールで使用する共通ユーティリティ関数

主な機能:
- 統計計算
- 可視化
- 入出力
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import scipy.stats as stats
from .base import AnalysisError

# ロガーの設定
logger = logging.getLogger(__name__)

class PlotUtility:
    """
    可視化用ユーティリティクラス
    """

    @staticmethod
    def generate_basic_plot(
        x_data: List,
        y_data: List,
        title: str,
        x_label: str,
        y_label: str,
        color: str = 'blue',
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        基本的なプロットを生成

        Args:
            x_data: X軸データ
            y_data: Y軸データ
            title: グラフタイトル
            x_label: X軸ラベル
            y_label: Y軸ラベル
            color: 線の色
            figsize: 図のサイズ

        Returns:
            プロット図

        Raises:
            AnalysisError: プロット生成時にエラーが発生した場合
        """
        try:
            if len(x_data) != len(y_data):
                raise ValueError(f"X軸データ({len(x_data)}個)とY軸データ({len(y_data)}個)の長さが一致しません")

            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(x_data, y_data, color=color)
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, linestyle='--', alpha=0.7)

            return fig
        except Exception as e:
            error_msg = f"基本プロット生成中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    @staticmethod
    def generate_histogram_plot(
        data: np.ndarray,
        title: str,
        x_label: str,
        bins: int = 30,
        figsize: Tuple[int, int] = (10, 6),
        confidence_intervals: Optional[Dict[str, List[float]]] = None
    ) -> plt.Figure:
        """
        ヒストグラムプロットを生成

        Args:
            data: プロットするデータ
            title: グラフタイトル
            x_label: X軸ラベル
            bins: ビンの数
            figsize: 図のサイズ
            confidence_intervals: 信頼区間情報（オプション）

        Returns:
            プロット図

        Raises:
            AnalysisError: プロット生成時にエラーが発生した場合
        """
        try:
            # 入力データの検証
            if data is None or len(data) == 0:
                raise ValueError("プロットするデータが空です")

            # 無効な値の除去
            valid_data = data[~np.isnan(data)]
            if len(valid_data) == 0:
                raise ValueError("有効なデータがありません（すべてNaNまたは無効な値）")

            # binの調整（データ点より多すぎる場合）
            actual_bins = min(bins, len(valid_data) // 2 + 1)
            if actual_bins != bins:
                logger.warning(f"ビン数を{bins}から{actual_bins}に調整しました（データが少なすぎます）")

            fig, ax = plt.subplots(figsize=figsize)

            sns.histplot(valid_data, bins=actual_bins, kde=True, ax=ax)

            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel('頻度')
            ax.grid(True, linestyle='--', alpha=0.7)

            # 信頼区間を表示
            if confidence_intervals:
                if 'percentile' in confidence_intervals:
                    ci_lower, ci_upper = confidence_intervals['percentile']
                    ax.axvline(x=ci_lower, color='r', linestyle='--', alpha=0.8,
                              label=f'{(1-0.95)*100/2:.1f}%パーセンタイル')
                    ax.axvline(x=ci_upper, color='r', linestyle='--', alpha=0.8,
                              label=f'{(1-(1-0.95)*100/2):.1f}%パーセンタイル')

                if 'mean' in confidence_intervals:
                    ax.axvline(x=confidence_intervals['mean'], color='g', linestyle='-', alpha=0.8,
                              label='平均')

                ax.legend()

            return fig
        except Exception as e:
            error_msg = f"ヒストグラムプロット生成中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    @staticmethod
    def generate_residual_plot(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        予測モデルの残差プロットを生成する

        Args:
            y_true: 実際の値
            y_pred: 予測値
            figsize: 図のサイズ

        Returns:
            残差プロット（2x2のサブプロットを含む）

        Raises:
            AnalysisError: プロット生成時にエラーが発生した場合
        """
        try:
            # 入力データの検証
            if len(y_true) != len(y_pred):
                raise ValueError(f"実際の値({len(y_true)}個)と予測値({len(y_pred)}個)の長さが一致しません")

            if len(y_true) == 0:
                raise ValueError("プロットするデータが空です")

            # 残差の計算
            residuals = y_true - y_pred

            # 2x2のサブプロットを作成
            fig, axes = plt.subplots(2, 2, figsize=figsize)

            # 1. 残差のヒストグラム
            sns.histplot(residuals, kde=True, ax=axes[0, 0])
            axes[0, 0].set_title('残差の分布')
            axes[0, 0].set_xlabel('残差')
            axes[0, 0].set_ylabel('頻度')

            # 2. 予測値に対する残差のスキャッタープロット
            axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
            axes[0, 1].axhline(y=0, color='r', linestyle='-')
            axes[0, 1].set_title('予測値 vs 残差')
            axes[0, 1].set_xlabel('予測値')
            axes[0, 1].set_ylabel('残差')

            # 3. 実際の値と予測値の散布図
            axes[1, 0].scatter(y_true, y_pred, alpha=0.5)
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
            axes[1, 0].set_title('実際の値 vs 予測値')
            axes[1, 0].set_xlabel('実際の値')
            axes[1, 0].set_ylabel('予測値')

            # 4. QQプロット（残差の正規性の確認）
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('残差の正規Q-Qプロット')

            # レイアウト調整
            plt.tight_layout()

            return fig
        except Exception as e:
            error_msg = f"残差プロット生成中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    @staticmethod
    def save_plot_to_base64(fig: plt.Figure) -> str:
        """
        プロット図をBase64エンコードされた文字列に変換する

        Args:
            fig: プロット図

        Returns:
            Base64エンコードされた画像文字列

        Raises:
            AnalysisError: エンコード時にエラーが発生した場合
        """
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return img_str
        except Exception as e:
            error_msg = f"プロットのBase64エンコード中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            raise AnalysisError(error_msg) from e


class StatisticsUtility:
    """
    統計計算用ユーティリティクラス
    """

    @staticmethod
    def calculate_confidence_intervals(
        data: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, Union[List[float], float, int]]:
        """
        信頼区間を計算

        Args:
            data: 信頼区間を計算するデータ
            confidence_level: 信頼水準 (0から1の間)

        Returns:
            信頼区間情報を含む辞書

        Raises:
            AnalysisError: 計算中にエラーが発生した場合
        """
        try:
            # パラメータの検証
            if not 0 < confidence_level < 1:
                raise ValueError(f"信頼水準は0から1の間である必要があります (指定値: {confidence_level})")

            # データの検証
            if data is None or len(data) == 0:
                raise ValueError("データが空です")

            # 無効な値の除去
            valid_data = data[~np.isnan(data)]
            if len(valid_data) < 2:
                raise ValueError("有効なデータが2つ未満です（信頼区間を計算できません）")

            alpha = 1 - confidence_level

            # 通常分布に基づく信頼区間
            mean = np.mean(valid_data)
            std = np.std(valid_data)
            n = len(valid_data)

            # 標準誤差
            se = std / np.sqrt(n)

            # 臨界値 (t分布)
            t_critical = stats.t.ppf(1 - alpha/2, n-1)

            # 信頼区間
            ci_lower = mean - t_critical * se
            ci_upper = mean + t_critical * se

            # パーセンタイルに基づく信頼区間 (ノンパラメトリック)
            percentile_lower = np.percentile(valid_data, 100 * alpha/2)
            percentile_upper = np.percentile(valid_data, 100 * (1 - alpha/2))

            return {
                't_distribution': [ci_lower, ci_upper],
                'percentile': [percentile_lower, percentile_upper],
                'mean': mean,
                'std': std,
                'n': n,
                'confidence_level': confidence_level
            }
        except Exception as e:
            error_msg = f"信頼区間計算中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    @staticmethod
    def calculate_skewness(data: np.ndarray) -> float:
        """
        歪度（スキューネス）を計算する

        Args:
            data: データ配列

        Returns:
            歪度

        Raises:
            AnalysisError: 計算中にエラーが発生した場合
        """
        try:
            # データの検証
            if data is None or len(data) == 0:
                raise ValueError("データが空です")

            # 無効な値の除去
            valid_data = data[~np.isnan(data)]
            if len(valid_data) < 3:
                raise ValueError("有効なデータが3つ未満です（歪度を計算できません）")

            n = len(valid_data)
            mean = np.mean(valid_data)
            std = np.std(valid_data, ddof=1)

            if std == 0:
                logger.warning("データの標準偏差が0です（一定値のデータ）。歪度は0として扱います。")
                return 0.0

            m3 = np.sum((valid_data - mean) ** 3) / n
            return m3 / (std ** 3)
        except Exception as e:
            error_msg = f"歪度計算中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    @staticmethod
    def calculate_kurtosis(data: np.ndarray) -> float:
        """
        尖度（クルトシス）を計算する

        Args:
            data: データ配列

        Returns:
            尖度

        Raises:
            AnalysisError: 計算中にエラーが発生した場合
        """
        try:
            # データの検証
            if data is None or len(data) == 0:
                raise ValueError("データが空です")

            # 無効な値の除去
            valid_data = data[~np.isnan(data)]
            if len(valid_data) < 4:
                raise ValueError("有効なデータが4つ未満です（尖度を計算できません）")

            n = len(valid_data)
            mean = np.mean(valid_data)
            std = np.std(valid_data, ddof=1)

            if std == 0:
                logger.warning("データの標準偏差が0です（一定値のデータ）。尖度は0として扱います。")
                return 0.0

            m4 = np.sum((valid_data - mean) ** 4) / n
            return m4 / (std ** 4) - 3  # 正規分布からの差分（正規分布の尖度は3）
        except Exception as e:
            error_msg = f"尖度計算中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            raise AnalysisError(error_msg) from e

class HealthImpactWeightUtility:
    """
    業種・役職別健康影響度重み付け係数に関するユーティリティクラス
    """

    @staticmethod
    def get_health_impact_weight(
        connection,
        industry_name: str,
        position_level: str
    ) -> float:
        """
        業種と役職レベルに基づいて健康影響度の重み係数を取得する

        Parameters:
        -----------
        connection : psycopg2.connection
            PostgreSQLデータベース接続オブジェクト
        industry_name : str
            業種名 (例: 'SaaS・クラウドサービス', '製薬・創薬', 'リテール・小売')
        position_level : str
            役職レベル (例: 'レベル1', 'レベル2', ... 'レベル5')
            または直接役職名も可能 (例: 'C級役員/経営層', '上級管理職')

        Returns:
        --------
        float
            健康影響度の最終重み係数
            該当する係数が見つからない場合はデフォルト値(0.05)を返す

        Raises:
        -------
        AnalysisError
            データベース接続エラーなどの例外が発生した場合
        """
        try:
            # 役職レベルを判定（直接レベル名か役職名か）
            if position_level.startswith('レベル'):
                level_query = "level_name = %s"
                level_param = position_level
            else:
                level_query = "position_title = %s"
                level_param = position_level

            # 重み係数を取得するクエリ
            query = f"""
            SELECT final_weight
            FROM vw_health_impact_weights
            WHERE industry_name = %s AND {level_query}
            """

            with connection.cursor() as cursor:
                cursor.execute(query, (industry_name, level_param))
                result = cursor.fetchone()

            if result:
                return result[0]
            else:
                logger.warning(f"指定された業種 '{industry_name}' と役職 '{position_level}' の重み係数が見つかりませんでした。デフォルト値を使用します。")
                return 0.05  # 一般職員の最小値をデフォルトとする

        except Exception as e:
            logger.error(f"健康影響度重み係数の取得中にエラーが発生しました: {str(e)}")
            raise AnalysisError(f"健康影響度重み係数の取得に失敗しました: {str(e)}")

    @staticmethod
    def get_all_health_impact_weights(connection) -> pd.DataFrame:
        """
        すべての業種・役職レベルの健康影響度重み係数をDataFrameとして取得する

        Parameters:
        -----------
        connection : psycopg2.connection
            PostgreSQLデータベース接続オブジェクト

        Returns:
        --------
        pd.DataFrame
            業種・役職レベル別の健康影響度重み係数のDataFrame

        Raises:
        -------
        AnalysisError
            データベース接続エラーなどの例外が発生した場合
        """
        try:
            query = """
            SELECT
                industry_name,
                level_name,
                position_title,
                base_weight,
                industry_adjustment,
                final_weight
            FROM vw_health_impact_weights
            ORDER BY industry_name, level_name
            """

            with connection.cursor() as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()

            df = pd.DataFrame(results, columns=columns)
            return df

        except Exception as e:
            logger.error(f"健康影響度重み係数一覧の取得中にエラーが発生しました: {str(e)}")
            raise AnalysisError(f"健康影響度重み係数一覧の取得に失敗しました: {str(e)}")

    @staticmethod
    def get_position_level_from_title(connection, position_title: str) -> str:
        """
        役職名からレベル名を取得する

        Parameters:
        -----------
        connection : psycopg2.connection
            PostgreSQLデータベース接続オブジェクト
        position_title : str
            役職名 (例: 'C級役員/経営層', '上級管理職')

        Returns:
        --------
        str
            レベル名 (例: 'レベル1', 'レベル2')
            該当するレベルが見つからない場合は空文字列を返す

        Raises:
        -------
        AnalysisError
            データベース接続エラーなどの例外が発生した場合
        """
        try:
            query = """
            SELECT level_name
            FROM position_levels
            WHERE position_title = %s
            """

            with connection.cursor() as cursor:
                cursor.execute(query, (position_title,))
                result = cursor.fetchone()

            if result:
                return result[0]
            else:
                logger.warning(f"指定された役職名 '{position_title}' に対応するレベルが見つかりませんでした。")
                return ""

        except Exception as e:
            logger.error(f"役職名からレベル名の取得中にエラーが発生しました: {str(e)}")
            raise AnalysisError(f"役職名からレベル名の取得に失敗しました: {str(e)}")