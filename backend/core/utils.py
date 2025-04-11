"""
分析と処理のためのユーティリティ関数を提供します。
Firestoreとの連携や可視化、統計分析のための共通機能を提供します。
"""
from io import StringIO
from typing import Dict, List, Union, Optional, Tuple, Any
import pandas as pd
import numpy as np
from fastapi import HTTPException
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import scipy.stats as stats
from .common_logger import get_logger
from .exceptions import DataProcessingError, AnalysisError

# ロギングの設定
logger = get_logger(__name__)

async def convert_csv_to_dataframe(
    csv_data: str,
    date_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    文字列型のCSVデータをpandasのDataFrameに変換し、
    Firestore互換のデータ型に調整します。

    Args:
        csv_data (str): CSVデータ
        date_columns (Optional[List[str]]): 日付型として処理するカラム名のリスト
        numeric_columns (Optional[List[str]]): 数値型として処理するカラム名のリスト

    Returns:
        pd.DataFrame: 変換後のDataFrame

    Raises:
        HTTPException: CSV変換時のエラー
        DataProcessingError: データ処理時のエラー
    """
    try:
        logger.info("Starting CSV to DataFrame conversion")

        # CSVデータをDataFrameに変換
        df = pd.read_csv(StringIO(csv_data))

        # 日付カラムの処理
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col]).apply(
                        lambda x: x.isoformat() if pd.notnull(x) else None
                    )

        # 数値カラムの処理
        if numeric_columns:
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        # NaN値をNoneに変換（Firestore互換）
        df = df.replace({pd.NA: None, pd.NaT: None})

        logger.info(f"Successfully converted CSV data to DataFrame with {len(df)} rows")
        return df

    except pd.errors.ParserError as e:
        error_msg = f"Invalid CSV data format: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    except Exception as e:
        error_msg = f"Error processing CSV data: {str(e)}"
        logger.error(error_msg)
        raise DataProcessingError(error_msg)

async def prepare_firestore_data(
    df: pd.DataFrame,
    metadata: Optional[Dict] = None
) -> List[Dict]:
    """
    DataFrameをFirestore互換の形式に変換します。

    Args:
        df (pd.DataFrame): 変換対象のDataFrame
        metadata (Optional[Dict]): 追加のメタデータ

    Returns:
        List[Dict]: Firestore互換のデータリスト

    Raises:
        DataProcessingError: データ変換時のエラー
    """
    try:
        logger.info("Starting DataFrame to Firestore format conversion")

        # DataFrameを辞書のリストに変換
        records = df.to_dict('records')

        # Firestore互換のデータ構造に変換
        firestore_data = []
        for record in records:
            # メタデータの追加
            document = {
                'data': record,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
            }

            # オプションのメタデータを追加
            if metadata:
                document['metadata'] = metadata

            firestore_data.append(document)

        logger.info(f"Successfully converted {len(firestore_data)} records to Firestore format")
        return firestore_data

    except Exception as e:
        error_msg = f"Error preparing data for Firestore: {str(e)}"
        logger.error(error_msg)
        raise DataProcessingError(error_msg)

async def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    column_types: Optional[Dict[str, str]] = None
) -> bool:
    """
    DataFrameのスキーマを検証します。

    Args:
        df (pd.DataFrame): 検証対象のDataFrame
        required_columns (List[str]): 必須カラムのリスト
        column_types (Optional[Dict[str, str]]): カラム名と期待される型の辞書

    Returns:
        bool: 検証結果

    Raises:
        HTTPException: スキーマ検証エラー
    """
    try:
        logger.info("Starting DataFrame schema validation")

        # 必須カラムの存在チェック
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}"
            )

        # カラムの型チェック
        if column_types:
            for col, expected_type in column_types.items():
                if col in df.columns:
                    actual_type = df[col].dtype.name
                    if actual_type != expected_type:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Column {col} has type {actual_type}, expected {expected_type}"
                        )

        logger.info("DataFrame schema validation successful")
        return True

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error validating DataFrame schema: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

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
    def generate_scatter_plot(
        x_data: np.ndarray,
        y_data: np.ndarray,
        title: str,
        x_label: str,
        y_label: str,
        hue: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        散布図を生成

        Args:
            x_data: X軸データ
            y_data: Y軸データ
            title: グラフタイトル
            x_label: X軸ラベル
            y_label: Y軸ラベル
            hue: 色分けに使用するデータ（オプション）
            figsize: 図のサイズ

        Returns:
            プロット図

        Raises:
            AnalysisError: プロット生成時にエラーが発生した場合
        """
        try:
            if len(x_data) != len(y_data):
                raise ValueError(f"X軸データ({len(x_data)}個)とY軸データ({len(y_data)}個)の長さが一致しません")

            if hue is not None and len(hue) != len(x_data):
                raise ValueError(f"色分けデータ({len(hue)}個)とX軸データ({len(x_data)}個)の長さが一致しません")

            fig, ax = plt.subplots(figsize=figsize)

            if hue is not None:
                scatter = ax.scatter(x_data, y_data, c=hue, cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, ax=ax, label='値')
            else:
                ax.scatter(x_data, y_data, alpha=0.6)

            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, linestyle='--', alpha=0.7)

            return fig
        except Exception as e:
            error_msg = f"散布図生成中にエラーが発生しました: {str(e)}"
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

    @staticmethod
    def calculate_descriptive_statistics(
        data: np.ndarray,
        include_percentiles: bool = True
    ) -> Dict[str, float]:
        """
        記述統計量を計算する

        Args:
            data: データ配列
            include_percentiles: パーセンタイル値を含めるかどうか

        Returns:
            記述統計量を含む辞書

        Raises:
            AnalysisError: 計算中にエラーが発生した場合
        """
        try:
            # データの検証
            if data is None or len(data) == 0:
                raise ValueError("データが空です")

            # 無効な値の除去
            valid_data = data[~np.isnan(data)]
            if len(valid_data) == 0:
                raise ValueError("有効なデータがありません（すべてNaNまたは無効な値）")

            # 基本統計量の計算
            stats_dict = {
                'count': len(valid_data),
                'mean': float(np.mean(valid_data)),
                'std': float(np.std(valid_data)),
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'median': float(np.median(valid_data)),
                'skewness': StatisticsUtility.calculate_skewness(valid_data),
                'kurtosis': StatisticsUtility.calculate_kurtosis(valid_data)
            }

            # パーセンタイル値の追加（オプション）
            if include_percentiles:
                percentiles = [1, 5, 10, 25, 75, 90, 95, 99]
                for p in percentiles:
                    stats_dict[f'p{p}'] = float(np.percentile(valid_data, p))

            return stats_dict
        except Exception as e:
            error_msg = f"記述統計量計算中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    @staticmethod
    def perform_normality_test(data: np.ndarray) -> Dict[str, Any]:
        """
        正規性検定を実行する

        Args:
            data: データ配列

        Returns:
            検定結果を含む辞書

        Raises:
            AnalysisError: 検定中にエラーが発生した場合
        """
        try:
            # データの検証
            if data is None or len(data) == 0:
                raise ValueError("データが空です")

            # 無効な値の除去
            valid_data = data[~np.isnan(data)]
            if len(valid_data) < 3:
                raise ValueError("有効なデータが不足しています（正規性検定を実行できません）")

            # Shapiro-Wilk検定（サンプルサイズが5000未満の場合）
            if len(valid_data) < 5000:
                shapiro_test = stats.shapiro(valid_data)
                shapiro_result = {
                    'test': 'Shapiro-Wilk',
                    'statistic': float(shapiro_test[0]),
                    'p_value': float(shapiro_test[1]),
                    'is_normal': shapiro_test[1] > 0.05
                }

                # D'Agostino-Pearson検定
                k2_test = stats.normaltest(valid_data)
                k2_result = {
                    'test': "D'Agostino-Pearson",
                    'statistic': float(k2_test[0]),
                    'p_value': float(k2_test[1]),
                    'is_normal': k2_test[1] > 0.05
                }

                return {
                    'shapiro': shapiro_result,
                    'dagostino': k2_result,
                    'is_normal': shapiro_result['is_normal'] or k2_result['is_normal']
                }

            # サンプルサイズが大きい場合はKolmogorov-Smirnov検定
            else:
                ks_test = stats.kstest(valid_data, 'norm', args=(np.mean(valid_data), np.std(valid_data)))
                ks_result = {
                    'test': 'Kolmogorov-Smirnov',
                    'statistic': float(ks_test[0]),
                    'p_value': float(ks_test[1]),
                    'is_normal': ks_test[1] > 0.05
                }

                return {
                    'ks': ks_result,
                    'is_normal': ks_result['is_normal']
                }
        except Exception as e:
            error_msg = f"正規性検定中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            raise AnalysisError(error_msg) from e