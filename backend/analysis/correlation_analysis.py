# -*- coding: utf-8 -*-

"""
相関分析

VAS データと損益計算書データの相関関係を分析します。
BigQueryService を利用した非同期処理に対応しています。
"""

import gc
import weakref
import logging
from typing import List, Optional, Tuple, Any, Dict, ContextManager
from contextlib import contextmanager
import pandas as pd
import numpy as np
from service.bigquery.client import BigQueryService
from service.firestore.client import FirestoreService
from .base import BaseAnalyzer

class CorrelationAnalyzer(BaseAnalyzer):
    def __init__(self, bq_service: BigQueryService):
        """
        コンストラクタ

        Args:
            bq_service (BigQueryService): BigQueryサービスのインスタンス
        """
        super().__init__()
        self.bq_service = bq_service
        self._temp_data_refs = weakref.WeakValueDictionary()
        self.logger.info("CorrelationAnalyzer initialized")

    def __del__(self):
        """デストラクタ - リソースの自動解放"""
        self.release_resources()

    def release_resources(self):
        """明示的なリソース解放メソッド"""
        try:
            self._temp_data_refs.clear()
            gc.collect()
            self.logger.info("CorrelationAnalyzer resources released")
        except Exception as e:
            self.logger.error(f"Error releasing resources: {str(e)}")

    @contextmanager
    def _managed_dataframe(self, df: pd.DataFrame, name: str = "temp_df") -> ContextManager[pd.DataFrame]:
        """
        データフレームのリソース管理用コンテキストマネージャ

        Args:
            df (pd.DataFrame): 管理対象のデータフレーム
            name (str): データフレームの識別名

        Yields:
            pd.DataFrame: 管理対象のデータフレーム
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

    def _validate_data(self,
                      data: pd.DataFrame,
                      vas_variables: List[str],
                      financial_variables: List[str]) -> Tuple[bool, Optional[str]]:
        """
        データのバリデーションを行う

        Args:
            data (pd.DataFrame): 検証対象のデータ
            vas_variables (List[str]): VASデータの変数名リスト
            financial_variables (List[str]): 損益計算書データの変数名リスト

        Returns:
            Tuple[bool, Optional[str]]: (バリデーション結果, エラーメッセージ)
        """
        if data.empty:
            return False, "データが空です"

        # 必要なカラムの存在確認
        required_columns = vas_variables + financial_variables
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"以下のカラムが見つかりません: {', '.join(missing_columns)}"

        # データ型の確認（数値型であることを確認）
        non_numeric_columns = [col for col in required_columns
                             if not pd.api.types.is_numeric_dtype(data[col])]
        if non_numeric_columns:
            return False, f"以下のカラムが数値型ではありません: {', '.join(non_numeric_columns)}"

        return True, None

    def estimate_memory_usage(self,
                            data_size: int,
                            num_variables: int) -> Dict[str, float]:
        """
        メモリ使用量を推定する

        Args:
            data_size (int): データ行数
            num_variables (int): 変数の数

        Returns:
            Dict[str, float]: 推定メモリ使用量(MB)
        """
        try:
            # 基本的なメモリ使用量の推定
            # 1. データフレームのサイズ推定 (8バイト/セル)
            df_size_mb = (data_size * num_variables * 8) / (1024 * 1024)

            # 2. 相関行列のサイズ推定 (8バイト/セル)
            corr_matrix_mb = (num_variables * num_variables * 8) / (1024 * 1024)

            # 3. その他のオーバーヘッド
            overhead_mb = 10  # 固定オーバーヘッド

            total_mb = df_size_mb + corr_matrix_mb + overhead_mb

            return {
                'dataframe_mb': df_size_mb,
                'correlation_matrix_mb': corr_matrix_mb,
                'overhead_mb': overhead_mb,
                'total_mb': total_mb
            }
        except Exception as e:
            self.logger.error(f"メモリ使用量推定中にエラーが発生しました: {str(e)}")
            return {'total_mb': 50}  # デフォルト値を返す

    async def analyze(self,
                     query: str,
                     vas_variables: List[str],
                     financial_variables: List[str],
                     save_results: bool = True,
                     dataset_id: Optional[str] = None,
                     table_id: Optional[str] = None,
                     optimize_dtypes: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        VASデータと損益計算書データの相関関係を分析する

        Args:
            query (str): BigQueryクエリ
            vas_variables (List[str]): VASデータの変数名リスト
            financial_variables (List[str]): 損益計算書データの変数名リスト
            save_results (bool): 結果を保存するかどうか
            dataset_id (Optional[str]): 保存先データセットID
            table_id (Optional[str]): 保存先テーブルID
            optimize_dtypes (bool): データ型を最適化するかどうか

        Returns:
            Tuple[pd.DataFrame, Dict]: (相関行列, メタデータ)

        Raises:
            ValueError: データのバリデーションエラー
            RuntimeError: 分析実行時のエラー
        """
        try:
            # データ取得
            raw_data = await self.bq_service.fetch_data(query)

            # データ型の最適化（メモリ使用量削減）
            if optimize_dtypes:
                self.logger.info("データ型の最適化を開始")
                with self._managed_dataframe(raw_data, "original_data") as data:
                    # 数値データ型の最適化
                    for col in data.select_dtypes(include=['int64']).columns:
                        c_min, c_max = data[col].min(), data[col].max()
                        if c_min >= 0:
                            if c_max < 255:
                                data[col] = data[col].astype(np.uint8)
                            elif c_max < 65535:
                                data[col] = data[col].astype(np.uint16)
                            elif c_max < 4294967295:
                                data[col] = data[col].astype(np.uint32)
                        else:
                            if c_min > -128 and c_max < 127:
                                data[col] = data[col].astype(np.int8)
                            elif c_min > -32768 and c_max < 32767:
                                data[col] = data[col].astype(np.int16)
                            elif c_min > -2147483648 and c_max < 2147483647:
                                data[col] = data[col].astype(np.int32)

                    # 浮動小数点の最適化
                    for col in data.select_dtypes(include=['float64']).columns:
                        data[col] = data[col].astype(np.float32)

                    self.logger.info(f"データ型最適化後のメモリ使用量: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

                    # データバリデーション
                    is_valid, error_message = self._validate_data(data, vas_variables, financial_variables)
                    if not is_valid:
                        raise ValueError(error_message)

                    # 相関分析の実行 - 必要なカラムのみを選択
                    with self._managed_dataframe(data[vas_variables + financial_variables], "selected_data") as selected_data:
                        correlation_matrix = selected_data.corr()

                        # 結果の保存
                        if save_results and dataset_id and table_id:
                            await self.bq_service.save_results(
                                correlation_matrix,
                                dataset_id=dataset_id,
                                table_id=table_id
                            )

                        # メタデータの作成
                        metadata = {
                            'row_count': len(data),
                            'vas_variables': vas_variables,
                            'financial_variables': financial_variables,
                            'correlation_pairs': len(vas_variables) * len(financial_variables),
                            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
                        }

                        return correlation_matrix, metadata
            else:
                # 最適化なしの処理パス
                with self._managed_dataframe(raw_data, "raw_data") as data:
                    # データバリデーション
                    is_valid, error_message = self._validate_data(data, vas_variables, financial_variables)
                    if not is_valid:
                        raise ValueError(error_message)

                    # 相関分析の実行
                    with self._managed_dataframe(data[vas_variables + financial_variables], "selected_data") as selected_data:
                        correlation_matrix = selected_data.corr()

                        # 結果の保存
                        if save_results and dataset_id and table_id:
                            await self.bq_service.save_results(
                                correlation_matrix,
                                dataset_id=dataset_id,
                                table_id=table_id
                            )

                        # メタデータの作成
                        metadata = {
                            'row_count': len(data),
                            'vas_variables': vas_variables,
                            'financial_variables': financial_variables,
                            'correlation_pairs': len(vas_variables) * len(financial_variables)
                        }

                        return correlation_matrix, metadata

        except Exception as e:
            self.logger.error(f"相関分析の実行中にエラーが発生しました: {str(e)}")
            raise RuntimeError(f"相関分析の実行中にエラーが発生しました: {str(e)}")
        finally:
            # 明示的なメモリ解放
            gc.collect()


async def analyze_correlation(request: Any) -> Tuple[Dict, int]:
    """
    Cloud Functions用のエントリーポイント関数

    Args:
        request: Cloud Functionsのリクエストオブジェクト

    Returns:
        Tuple[Dict, int]: (レスポンス, ステータスコード)
    """
    bq_service = None
    analyzer = None

    try:
        request_json = request.get_json()

        if not request_json:
            return {'error': 'リクエストデータがありません'}, 400

        # 必須パラメータの確認
        required_params = ['query', 'vas_variables', 'financial_variables']
        missing_params = [param for param in required_params if param not in request_json]
        if missing_params:
            return {
                'error': f'必須パラメータが不足しています: {", ".join(missing_params)}'
            }, 400

        # パラメータの取得
        query = request_json['query']
        vas_variables = request_json['vas_variables']
        financial_variables = request_json['financial_variables']
        dataset_id = request_json.get('dataset_id')
        table_id = request_json.get('table_id')
        optimize_dtypes = request_json.get('optimize_dtypes', True)

        # サービスの初期化
        bq_service = BigQueryService()
        analyzer = CorrelationAnalyzer(bq_service)

        # メモリ使用量推定
        memory_estimate = analyzer.estimate_memory_usage(
            request_json.get('estimated_rows', 10000),
            len(vas_variables) + len(financial_variables)
        )

        # 分析実行
        correlation_matrix, metadata = await analyzer.analyze(
            query=query,
            vas_variables=vas_variables,
            financial_variables=financial_variables,
            save_results=bool(dataset_id and table_id),
            dataset_id=dataset_id,
            table_id=table_id,
            optimize_dtypes=optimize_dtypes
        )

        # メタデータにメモリ使用量推定値を追加
        metadata['memory_estimate'] = memory_estimate

        return {
            'status': 'success',
            'results': correlation_matrix.to_dict('records'),
            'metadata': metadata
        }, 200

    except Exception as e:
        logging.error(f"Correlation analysis request failed: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }, 500
    finally:
        # リソース解放
        if analyzer:
            analyzer.release_resources()
        # 部分的なガベージコレクション
        gc.collect()