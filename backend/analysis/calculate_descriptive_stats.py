# -*- coding: utf-8 -*-

"""
記述統計量の計算

VAS データと損益計算書データの記述統計量を計算します。
BigQueryService を利用した非同期処理に対応しています。
"""

from typing import Optional, Tuple, Any, Dict, List, ContextManager
import pandas as pd
import numpy as np
import statsmodels.api as sm
from dataclasses import dataclass
import contextlib
import logging
import gc
import weakref
from service.bigquery.client import BigQueryService
from service.firestore.client import FirestoreService

@dataclass
class DescriptiveStatsConfig:
    """記述統計量の計算設定を保持するデータクラス"""
    query: str
    target_variable: str
    arima_order: Tuple[int, int, int] = (5, 1, 0)
    columns: Optional[List[str]] = None
    save_results: bool = True
    dataset_id: Optional[str] = None
    table_id: Optional[str] = None
    batch_size: Optional[int] = None  # データバッチ処理サイズ
    max_memory_usage_mb: Optional[int] = None  # 最大メモリ使用量(MB)

class DescriptiveStatsCalculator:
    """
    記述統計量を計算するクラスです。
    BigQueryService を利用して非同期でデータの取得と保存を行います。
    """

    def __init__(self, bq_service: BigQueryService):
        """
        Args:
            bq_service (BigQueryService): BigQuery操作用のサービスインスタンス
        """
        self.bq_service = bq_service
        self.logger = logging.getLogger(__name__)
        self._temp_data = {}
        self._data_refs = weakref.WeakValueDictionary()
        self._finalizer = weakref.finalize(self, self._cleanup_resources)

    def __del__(self):
        """デストラクタ - リソースの解放"""
        self.release_resources()

    def release_resources(self) -> None:
        """すべてのリソースを解放"""
        self._temp_data.clear()
        self._data_refs.clear()
        gc.collect()

    def _cleanup_resources(self) -> None:
        """オブジェクト破棄時の内部クリーンアップ処理"""
        try:
            self.release_resources()
        except Exception as e:
            # すでにロガーが利用できない可能性があるため標準エラーに出力
            import sys
            print(f"リソース解放中にエラーが発生しました: {e}", file=sys.stderr)

    @contextlib.contextmanager
    def _managed_dataframe(self, df: pd.DataFrame, key: str = None) -> ContextManager[pd.DataFrame]:
        """
        データフレームを管理するコンテキストマネージャ

        Args:
            df: 管理するデータフレーム
            key: 追跡用のキー

        Yields:
            pd.DataFrame: 管理されるデータフレーム
        """
        df_id = id(df)
        if key:
            self._temp_data[key] = df
        self._data_refs[df_id] = df

        try:
            yield df
        finally:
            if key and key in self._temp_data:
                del self._temp_data[key]
            if df_id in self._data_refs:
                del self._data_refs[df_id]

    def _validate_data(self, data: pd.DataFrame, target_variable: str) -> Tuple[bool, Optional[str]]:
        """
        データのバリデーションを行います。

        Args:
            data (pd.DataFrame): 検証対象のデータ
            target_variable (str): 分析対象の変数名

        Returns:
            Tuple[bool, Optional[str]]: (検証結果, エラーメッセージ)
        """
        if data.empty:
            return False, "データが空です"

        if target_variable not in data.columns:
            return False, f"指定された変数 '{target_variable}' がデータに存在しません"

        if not pd.api.types.is_numeric_dtype(data[target_variable]):
            return False, f"変数 '{target_variable}' が数値型ではありません"

        if data[target_variable].isnull().any():
            return False, f"変数 '{target_variable}' に欠損値が含まれています"

        if len(data) < 10:  # 時系列分析に最低限必要なデータ数
            return False, "データ数が不足しています（10以上必要）"

        return True, None

    def _calculate_descriptive_stats(self, data: pd.DataFrame, target_variable: str) -> Dict[str, Any]:
        """
        データの記述統計量を計算します。

        Args:
            data (pd.DataFrame): 分析対象のデータ
            target_variable (str): 分析対象の変数名

        Returns:
            Dict[str, Any]: 記述統計量
        """
        # 必要な統計量だけを効率的に計算
        target_data = data[target_variable]

        return {
            "mean": float(target_data.mean()),
            "median": float(target_data.median()),
            "std": float(target_data.std()),
            "min": float(target_data.min()),
            "max": float(target_data.max()),
            "skewness": float(target_data.skew()),
            "kurtosis": float(target_data.kurt())
        }

    def _calculate_arima_metrics(self, data: pd.DataFrame, target_variable: str, arima_order: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        ARIMA モデルの評価指標を計算します。

        Args:
            data (pd.DataFrame): 分析対象のデータ
            target_variable (str): 分析対象の変数名
            arima_order (Tuple[int, int, int]): ARIMA モデルのオーダー

        Returns:
            Dict[str, Any]: ARIMA モデルの評価指標
        """
        try:
            # 必要なデータのみを抽出
            target_series = data[target_variable].copy()

            # メモリ効率化: 計算に必要な最小限のスコープでモデル作成
            model = sm.tsa.ARIMA(target_series, order=arima_order)
            results = model.fit()

            # 必要な結果のみをメモリ効率良く抽出
            metrics = {
                "aic": float(results.aic),
                "bic": float(results.bic),
                "hqic": float(results.hqic),
                "mse": float(results.mse),
                "mae": float(np.mean(np.abs(results.resid))),
                "rmse": float(np.sqrt(results.mse)),
                "residual_std": float(results.resid.std())
            }

            # 明示的にメモリを解放
            del results
            gc.collect()

            return metrics

        except Exception as e:
            self.logger.error(f"ARIMA モデル計算中にエラーが発生しました: {str(e)}")
            # 最低限のメトリクスを返す
            return {
                "error": str(e),
                "aic": None,
                "bic": None,
                "hqic": None,
                "mse": None,
                "mae": None,
                "rmse": None,
                "residual_std": None
            }

    async def _fetch_data_in_batches(self, query: str, batch_size: Optional[int] = None) -> pd.DataFrame:
        """
        データをバッチで取得して結合します。

        Args:
            query: 実行クエリ
            batch_size: バッチサイズ（指定がない場合は一括取得）

        Returns:
            pd.DataFrame: 結合されたデータフレーム
        """
        if not batch_size:
            # バッチ指定がない場合は一括取得
            return await self.bq_service.fetch_data(query)

        try:
            # バッチ取得用のクエリを構築
            batch_query_template = f"{query} LIMIT {batch_size} OFFSET {{}}"

            all_data_frames = []
            offset = 0
            has_more = True

            while has_more:
                batch_query = batch_query_template.format(offset)
                batch_data = await self.bq_service.fetch_data(batch_query)

                if len(batch_data) > 0:
                    all_data_frames.append(batch_data)
                    offset += batch_size

                    # 一時データを格納
                    key = f"batch_{offset}"
                    self._temp_data[key] = batch_data
                else:
                    has_more = False

            # 一時データをクリア
            temp_keys = [k for k in self._temp_data.keys() if k.startswith("batch_")]
            for key in temp_keys:
                del self._temp_data[key]

            # すべてのバッチを連結
            if all_data_frames:
                result = pd.concat(all_data_frames, ignore_index=True)
                del all_data_frames
                gc.collect()
                return result
            else:
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"バッチデータ取得中にエラーが発生しました: {str(e)}")
            raise RuntimeError(f"バッチデータ取得に失敗しました: {str(e)}")

    async def calculate(self, config: DescriptiveStatsConfig) -> Dict[str, Any]:
        """
        記述統計量の計算を実行します。

        Args:
            config: 記述統計量の計算設定

        Returns:
            Dict[str, Any]: 計算結果

        Raises:
            ValueError: データのバリデーションエラー
            RuntimeError: 計算実行中のエラー
        """
        try:
            # データ取得（バッチ処理対応）
            data = await self._fetch_data_in_batches(config.query, config.batch_size)

            # データフレームをコンテキストマネージャで管理
            with self._managed_dataframe(data, key="raw_data") as data:
                # 指定されたカラムのみを抽出
                if config.columns:
                    available_columns = [col for col in config.columns if col in data.columns]
                    if not available_columns:
                        raise ValueError("指定されたカラムが存在しません")

                    # データの複製を避け、必要なカラムだけを参照
                    data = data[available_columns]

                # データバリデーション
                is_valid, error_message = self._validate_data(data, config.target_variable)
                if not is_valid:
                    raise ValueError(error_message)

                # データの型最適化
                data = self._optimize_datatypes(data)

                # 記述統計量の計算
                descriptive_stats = self._calculate_descriptive_stats(data, config.target_variable)

                # ARIMA モデルの評価指標の計算
                arima_metrics = self._calculate_arima_metrics(data, config.target_variable, config.arima_order)

                # 結果の整形
                analysis_results = {
                    "descriptive_stats": descriptive_stats,
                    "arima_metrics": arima_metrics,
                    "metadata": {
                        "target_variable": config.target_variable,
                        "arima_order": config.arima_order,
                        "column_list": data.columns.tolist(),
                        "analysis_timestamp": pd.Timestamp.now().isoformat(),
                        "data_size": len(data)
                    }
                }

                # 結果の保存
                if config.save_results and config.dataset_id and config.table_id:
                    # 結果データフレームを最小限に保つ
                    results_df = pd.DataFrame({
                        "metric": list(analysis_results["arima_metrics"].keys()),
                        "value": list(analysis_results["arima_metrics"].values()),
                        "target_variable": config.target_variable,
                        "timestamp": pd.Timestamp.now()
                    })

                    # 結果保存用のデータフレームをコンテキストマネージャで管理
                    with self._managed_dataframe(results_df, key="results_df") as results_df:
                        await self.bq_service.save_results(
                            results_df,
                            dataset_id=config.dataset_id,
                            table_id=config.table_id
                        )

                return analysis_results

        except ValueError as e:
            self.release_resources()  # エラー発生時も確実にリソースを解放
            raise ValueError(f"データの検証に失敗しました: {str(e)}") from e
        except Exception as e:
            self.release_resources()  # エラー発生時も確実にリソースを解放
            raise RuntimeError(f"記述統計量の計算中にエラーが発生しました: {str(e)}") from e

    def _optimize_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データフレームのメモリ使用量を最適化します。

        Args:
            df: 最適化するデータフレーム

        Returns:
            最適化されたデータフレーム
        """
        if df.empty:
            return df

        # 数値型カラムの最適化
        for col in df.select_dtypes(include=['int']).columns:
            # 値の範囲に基づいて最適な整数型を選択
            col_min, col_max = df[col].min(), df[col].max()

            # 符号なし整数が使用可能か確認
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:
                # 符号付き整数
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype(np.int8)
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype(np.int16)
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype(np.int32)

        # float型の最適化
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = df[col].astype(np.float32)

        # カテゴリ型に変換できるカラムの最適化
        for col in df.select_dtypes(include=['object']).columns:
            # ユニーク値の数が少ない場合にカテゴリ型に変換
            if df[col].nunique() / len(df) < 0.5:  # 50%未満のユニーク比率
                df[col] = df[col].astype('category')

        return df

    def estimate_memory_usage(self, row_count: int, column_count: int) -> Dict[str, Any]:
        """
        予想メモリ使用量を計算し、適切なバッチサイズを推奨する

        Args:
            row_count: 行数
            column_count: 列数

        Returns:
            Dict[str, Any]: メモリ使用量予測
        """
        # 簡易的なメモリ使用量予測（数値はおおよその値）
        bytes_per_value = 8  # 8バイト/値と仮定
        base_memory = 50  # MB
        overhead_factor = 1.5  # オーバーヘッド係数

        df_memory_mb = (row_count * column_count * bytes_per_value) / (1024 * 1024) * overhead_factor
        arima_memory_mb = base_memory + (0.001 * row_count)  # ARIMAのメモリ使用量概算

        total_estimated_memory = base_memory + df_memory_mb + arima_memory_mb

        # 推奨バッチサイズ（最大で5GBまで）
        max_memory_mb = 5000
        if total_estimated_memory > max_memory_mb:
            batch_factor = max_memory_mb / total_estimated_memory
            recommended_batch_size = int(row_count * batch_factor)
            # 最小バッチサイズは1000行
            recommended_batch_size = max(1000, recommended_batch_size)
        else:
            recommended_batch_size = None  # バッチ処理不要

        return {
            'estimated_memory_mb': total_estimated_memory,
            'recommended_batch_size': recommended_batch_size,
            'row_count': row_count,
            'column_count': column_count,
            'base_memory_mb': base_memory,
            'dataframe_memory_mb': df_memory_mb,
            'arima_memory_mb': arima_memory_mb
        }

async def calculate_descriptive_stats(request: Any) -> Tuple[Dict[str, Any], int]:
    """
    Cloud Functions用のエントリーポイント関数

    Args:
        request: Cloud Functionsのリクエストオブジェクト

    Returns:
        Tuple[Dict[str, Any], int]: (レスポンス, ステータスコード)
    """
    calculator = None
    try:
        request_json = request.get_json()

        if not request_json:
            return {'error': 'リクエストデータがありません'}, 400

        # 必須パラメータのバリデーション
        required_params = ['query', 'target_variable']
        for param in required_params:
            if param not in request_json:
                return {'error': f"必須パラメータ '{param}' が指定されていません"}, 400

        # 設定オブジェクトの作成
        config = DescriptiveStatsConfig(
            query=request_json['query'],
            target_variable=request_json['target_variable'],
            arima_order=tuple(request_json.get('arima_order', (5, 1, 0))),
            columns=request_json.get('columns'),
            dataset_id=request_json.get('dataset_id'),
            table_id=request_json.get('table_id'),
            save_results=request_json.get('save_results', True),
            batch_size=request_json.get('batch_size'),
            max_memory_usage_mb=request_json.get('max_memory_usage_mb')
        )

        # サービスの初期化
        bq_service = BigQueryService()
        calculator = DescriptiveStatsCalculator(bq_service)

        # 記述統計量の計算
        results = await calculator.calculate(config)

        # 明示的にリソース解放
        if calculator:
            calculator.release_resources()

        return {
            'status': 'success',
            'results': results
        }, 200

    except ValueError as e:
        # エラー時にリソース解放
        if calculator:
            calculator.release_resources()
        return {
            'status': 'error',
            'type': 'validation_error',
            'message': str(e)
        }, 400
    except Exception as e:
        # エラー時にリソース解放
        if calculator:
            calculator.release_resources()
        return {
            'status': 'error',
            'type': 'internal_error',
            'message': str(e)
        }, 500