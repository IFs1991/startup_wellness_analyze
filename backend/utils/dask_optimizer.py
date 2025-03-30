"""
Daskを活用した大規模データ処理最適化ユーティリティ

このモジュールは、Daskを利用して大規模データの並列処理を最適化する
ユーティリティ関数を提供します。メモリ使用量の削減と処理速度の向上を実現します。
"""

import os
import logging
from typing import Optional, Dict, Any, Union, List, Tuple, Callable
import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import psutil
import warnings

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DaskOptimizer:
    """
    Daskを活用してデータ処理を最適化するクラス
    """

    def __init__(
        self,
        n_workers: Optional[int] = None,
        threads_per_worker: int = 2,
        memory_limit: Optional[str] = None,
        temporary_directory: Optional[str] = None
    ):
        """
        Dask最適化環境を初期化します。

        Args:
            n_workers: ワーカー数（Noneの場合はCPUコア数の半分を使用）
            threads_per_worker: ワーカーあたりのスレッド数
            memory_limit: ワーカーあたりのメモリ制限（例: '4GB'）
            temporary_directory: 一時ファイル用ディレクトリ
        """
        self.n_workers = n_workers or max(1, psutil.cpu_count(logical=False) // 2)
        self.threads_per_worker = threads_per_worker

        # メモリ制限が指定されていない場合、利用可能メモリの80%をワーカー数で分割
        if memory_limit is None:
            available_memory = psutil.virtual_memory().available
            memory_per_worker = int(available_memory * 0.8 / self.n_workers)
            self.memory_limit = f"{memory_per_worker // (1024 ** 2)}MB"
        else:
            self.memory_limit = memory_limit

        self.temporary_directory = temporary_directory
        self.client = None
        self.cluster = None

    def __enter__(self):
        """
        コンテキストマネージャの開始時に呼ばれる。
        Daskクライアントを初期化します。
        """
        self.start_client()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        コンテキストマネージャの終了時に呼ばれる。
        Daskクライアントをクローズします。
        """
        self.close_client()

    def start_client(self) -> None:
        """
        Daskクライアントを開始します。
        """
        if self.client is None:
            logger.info(f"Daskクラスターを開始: ワーカー数={self.n_workers}, "
                       f"ワーカーあたりスレッド数={self.threads_per_worker}, "
                       f"メモリ制限={self.memory_limit}")

            self.cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=self.threads_per_worker,
                memory_limit=self.memory_limit,
                local_directory=self.temporary_directory
            )

            self.client = Client(self.cluster)
            logger.info(f"Daskクライアント開始: {self.client}")

    def close_client(self) -> None:
        """
        Daskクライアントを終了します。
        """
        if self.client is not None:
            logger.info("Daskクライアントを終了します")
            self.client.close()
            self.client = None

        if self.cluster is not None:
            self.cluster.close()
            self.cluster = None

    @staticmethod
    def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrameのデータ型を最適化してメモリ使用量を削減します。

        Args:
            df: 最適化するDataFrame

        Returns:
            最適化されたDataFrame
        """
        result = df.copy()

        # 整数列の最適化
        int_cols = result.select_dtypes(include=['int']).columns
        for col in int_cols:
            col_min, col_max = result[col].min(), result[col].max()

            # 最適な整数データ型を選択
            if col_min >= 0:
                if col_max < 2**8:
                    result[col] = result[col].astype(np.uint8)
                elif col_max < 2**16:
                    result[col] = result[col].astype(np.uint16)
                elif col_max < 2**32:
                    result[col] = result[col].astype(np.uint32)
                else:
                    result[col] = result[col].astype(np.uint64)
            else:
                if col_min > -2**7 and col_max < 2**7:
                    result[col] = result[col].astype(np.int8)
                elif col_min > -2**15 and col_max < 2**15:
                    result[col] = result[col].astype(np.int16)
                elif col_min > -2**31 and col_max < 2**31:
                    result[col] = result[col].astype(np.int32)
                else:
                    result[col] = result[col].astype(np.int64)

        # 浮動小数点列の最適化
        float_cols = result.select_dtypes(include=['float']).columns
        for col in float_cols:
            # 単精度浮動小数点数で十分な場合
            result[col] = result[col].astype(np.float32)

        # カテゴリ列の最適化
        obj_cols = result.select_dtypes(include=['object']).columns
        for col in obj_cols:
            num_unique = result[col].nunique()
            # ユニーク値が少ない場合はカテゴリ型に変換
            if num_unique < len(result) * 0.5:  # 50%未満がユニーク値の場合
                result[col] = result[col].astype('category')

        return result

    def parallelize_dataframe(
        self,
        df: pd.DataFrame,
        func: Callable,
        partition_size: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        DataFrameを分割して並列処理します。

        Args:
            df: 処理するDataFrame
            func: 各パーティションに適用する関数
            partition_size: パーティションサイズ
            **kwargs: 関数に渡す追加引数

        Returns:
            処理結果を結合したDataFrame
        """
        self.start_client()

        # 推奨パーティションサイズの計算（約100MB/パーティション）
        if partition_size is None:
            df_size_bytes = df.memory_usage(deep=True).sum()
            partition_size = max(100, min(len(df), int(len(df) / (df_size_bytes / (100 * 1024 * 1024)))))

        # DataFrame全体がパーティションサイズより小さい場合は調整
        partition_size = min(partition_size, len(df))

        # Dask DataFrameに変換
        ddf = dd.from_pandas(df, npartitions=len(df) // partition_size + 1)

        # 各パーティションに関数を適用
        result = ddf.map_partitions(func, **kwargs).compute()

        return result

    def read_csv_optimized(
        self,
        filepath_or_buffer: Union[str, List[str]],
        memory_efficient: bool = True,
        dtype_backend: str = 'numpy',
        **kwargs
    ) -> pd.DataFrame:
        """
        メモリ効率の良いCSV読み込みを行います。

        Args:
            filepath_or_buffer: CSVファイルパスまたはパスのリスト
            memory_efficient: メモリ効率化を行うかどうか
            dtype_backend: データ型バックエンド ('numpy' または 'pyarrow')
            **kwargs: pandas.read_csv または dask.dataframe.read_csv に渡す追加引数

        Returns:
            読み込まれたDataFrame
        """
        self.start_client()

        # Daskを使用してCSVを読み込む
        ddf = dd.read_csv(filepath_or_buffer, **kwargs)

        # 結果を計算
        df = ddf.compute()

        # データ型の最適化
        if memory_efficient:
            df = self.optimize_dataframe_dtypes(df)

        return df

    def process_large_dataset(
        self,
        df: pd.DataFrame,
        processing_func: Callable,
        partition_size: Optional[int] = None,
        optimize_memory: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        大規模データセットを効率的に処理します。

        Args:
            df: 処理するDataFrame
            processing_func: 各パーティションに適用する処理関数
            partition_size: パーティションサイズ
            optimize_memory: メモリ最適化を行うかどうか
            **kwargs: 処理関数に渡す追加引数

        Returns:
            処理結果のDataFrame
        """
        # メモリ最適化
        if optimize_memory:
            df = self.optimize_dataframe_dtypes(df)

        # 並列処理
        result = self.parallelize_dataframe(df, processing_func, partition_size, **kwargs)

        return result

    @staticmethod
    def get_dask_diagnostic_info() -> Dict[str, Any]:
        """
        現在のDask診断情報を取得します。

        Returns:
            診断情報の辞書
        """
        from dask.distributed import get_worker
        try:
            worker = get_worker()
            info = {
                'memory': worker.memory_manager.memory,
                'memory_limit': worker.memory_manager.memory_limit,
                'memory_usage': worker.memory_monitor.process.memory_info().rss,
                'cpu_usage': worker.monitor.cpu,
                'executing_count': len(worker.executing),
                'ready_count': len(worker.ready),
                'in_flight_count': len(worker.in_flight),
                'in_memory_count': len(worker.data)
            }
            return info
        except ValueError:
            # ワーカーコンテキスト外で実行された場合
            return {
                'error': 'Daskワーカーコンテキスト外で実行されました'
            }

    def monitor_task_progress(self, task):
        """
        Daskタスクの進行状況をモニタリングします。

        Args:
            task: モニタリングするDaskタスク
        """
        from dask.distributed import wait

        task = self.client.persist(task)
        progress = self.client.get_task_stream(plot=False)

        # プログレスバーを表示（Jupyterの場合）
        try:
            from dask.diagnostics import ProgressBar
            with ProgressBar():
                result = self.client.compute(task)
                wait(result)
        except:
            # プログレスバーが使用できない場合、単に結果を計算
            result = self.client.compute(task)
            wait(result)

        return result