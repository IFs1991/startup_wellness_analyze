#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
スタートアップウェルネス分析プラットフォーム
メモリ使用量最適化スクリプト

このスクリプトはpandas DataFrameやNumPy配列などのメモリ使用量を
分析し、最適化するための機能を提供します。
特にGCP東京リージョンの割引時間帯外での実行時に効果的です。
"""

import os
import sys
import gc
import logging
import psutil
import numpy as np
import pandas as pd
import datetime
from typing import Dict, List, Union, Optional, Tuple, Any

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('memory_optimizer')

class MemoryOptimizer:
    """
    メモリ使用量を最適化するためのクラス
    """

    def __init__(self, verbose: bool = True):
        """
        初期化

        Args:
            verbose: 詳細なログを出力するかどうか
        """
        self.verbose = verbose
        self.memory_usage_log: List[Dict[str, Any]] = []
        self.process = psutil.Process(os.getpid())

        # 東京リージョンの割引時間帯設定
        self.discount_hours_start = int(os.environ.get('DISCOUNT_HOURS_START', 22))
        self.discount_hours_end = int(os.environ.get('DISCOUNT_HOURS_END', 8))
        self.weekend_discount = os.environ.get('WEEKEND_DISCOUNT', 'true').lower() == 'true'

        if self.verbose:
            logger.info(f"メモリ最適化クラスを初期化しました")
            logger.info(f"東京リージョン割引時間帯設定: {self.discount_hours_start}:00-{self.discount_hours_end}:00")
            logger.info(f"週末割引設定: {self.weekend_discount}")
            self.log_current_memory_usage()

    def is_discount_time(self) -> bool:
        """
        現在が割引時間帯かどうかを判定

        Returns:
            bool: 割引時間帯の場合はTrue
        """
        now = datetime.datetime.now()
        current_hour = now.hour
        is_weekend = now.weekday() >= 5  # 5=土曜, 6=日曜

        # 週末の場合
        if is_weekend and self.weekend_discount:
            return True

        # 22:00-08:00の判定
        if current_hour >= self.discount_hours_start or current_hour < self.discount_hours_end:
            return True

        return False

    def log_current_memory_usage(self) -> Dict[str, Union[float, str]]:
        """
        現在のメモリ使用量をログに記録

        Returns:
            Dict: メモリ使用量情報
        """
        memory_info = self.process.memory_info()
        memory_usage = {
            'timestamp': datetime.datetime.now().isoformat(),
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': self.process.memory_percent(),
            'is_discount_time': self.is_discount_time()
        }

        self.memory_usage_log.append(memory_usage)

        if self.verbose:
            logger.info(f"メモリ使用量: {memory_usage['rss_mb']:.2f} MB (RSS), "
                      f"{memory_usage['percent']:.2f}% 使用中")

        return memory_usage

    def get_df_memory_usage(self, df: pd.DataFrame, deep: bool = True) -> pd.Series:
        """
        DataFrame各列のメモリ使用量を計算

        Args:
            df: 対象DataFrameオブジェクト
            deep: オブジェクトの参照サイズも計算するかどうか

        Returns:
            Series: 各列のメモリ使用量
        """
        memory_usage = df.memory_usage(deep=deep)
        memory_usage_mb = memory_usage / (1024 * 1024)

        if self.verbose:
            logger.info(f"DataFrame全体のメモリ使用量: {memory_usage_mb.sum():.2f} MB")
            logger.info(f"行数: {len(df)}, 列数: {len(df.columns)}")

        return memory_usage_mb

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrameのデータ型を最適化

        Args:
            df: 対象DataFrameオブジェクト

        Returns:
            DataFrame: 最適化後のDataFrame
        """
        original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        result = df.copy()

        # メモリ使用量を表示
        if self.verbose:
            logger.info(f"最適化前メモリ使用量: {original_memory:.2f} MB")
            logger.info("データ型最適化を開始します...")

        # 数値列の最適化
        for col in result.select_dtypes(include=['int']).columns:
            # 最小値と最大値を取得
            col_min = result[col].min()
            col_max = result[col].max()

            # 適切なデータ型に変換
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
        for col in result.select_dtypes(include=['float']).columns:
            # 32ビットに変換（精度が許容できるなら）
            result[col] = result[col].astype(np.float32)

        # オブジェクト列の最適化（カテゴリカルデータに変換）
        for col in result.select_dtypes(include=['object']).columns:
            num_unique = result[col].nunique()
            num_total = len(result)

            # ユニーク値の比率が30%未満の場合はカテゴリカルに変換
            if num_unique / num_total < 0.3:
                result[col] = result[col].astype('category')

        # メモリ使用量の変化を計算
        optimized_memory = result.memory_usage(deep=True).sum() / (1024 * 1024)
        saved_memory = original_memory - optimized_memory
        saved_percent = (saved_memory / original_memory) * 100

        if self.verbose:
            logger.info(f"最適化後メモリ使用量: {optimized_memory:.2f} MB")
            logger.info(f"節約されたメモリ: {saved_memory:.2f} MB ({saved_percent:.2f}%)")

        return result

    def reduce_dataframe_columns(self, df: pd.DataFrame,
                                required_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        必要な列のみを残したDataFrameを作成

        Args:
            df: 対象DataFrameオブジェクト
            required_columns: 保持する列名のリスト

        Returns:
            DataFrame: 必要な列のみを含むDataFrame
        """
        if required_columns is None:
            logger.warning("必要な列が指定されていないため、元のDataFrameをそのまま返します")
            return df

        # 存在する列のみをフィルタリング
        valid_columns = [col for col in required_columns if col in df.columns]

        if len(valid_columns) != len(required_columns):
            missing = set(required_columns) - set(valid_columns)
            logger.warning(f"以下の列が見つかりませんでした: {missing}")

        # 必要な列のみを選択
        result = df[valid_columns].copy()

        if self.verbose:
            original_size = df.memory_usage(deep=True).sum() / (1024 * 1024)
            reduced_size = result.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(f"元のDataFrame列数: {len(df.columns)}, "
                      f"サイズ: {original_size:.2f} MB")
            logger.info(f"削減後DataFrame列数: {len(result.columns)}, "
                      f"サイズ: {reduced_size:.2f} MB")
            logger.info(f"削減率: {(1 - reduced_size/original_size) * 100:.2f}%")

        return result

    def optimize_dataframe(self, df: pd.DataFrame,
                         required_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        DataFrameを完全に最適化（列削減＋データ型最適化）

        Args:
            df: 対象DataFrameオブジェクト
            required_columns: 保持する列名のリスト

        Returns:
            DataFrame: 最適化されたDataFrame
        """
        # メモリ使用量をログ
        self.log_current_memory_usage()

        # 現在が割引時間帯でない場合は最大限の最適化を実施
        aggressive_optimization = not self.is_discount_time()

        if self.verbose:
            if aggressive_optimization:
                logger.info("非割引時間帯: 積極的な最適化を実施します")
            else:
                logger.info("割引時間帯: 通常の最適化を実施します")

        # 列の削減
        if required_columns is not None:
            df = self.reduce_dataframe_columns(df, required_columns)

        # データ型の最適化
        df = self.optimize_dtypes(df)

        # 非割引時間帯は追加の最適化
        if aggressive_optimization:
            # カテゴリカル列の更なる最適化
            for col in df.select_dtypes(include=['category']).columns:
                # カテゴリコードのデータ型を最適化
                if df[col].cat.categories.size < 2**8:
                    df[col] = df[col].cat.set_categories(
                        df[col].cat.categories,
                        fastpath=True
                    )

        # ガベージコレクションを実行
        gc.collect()

        # 最終メモリ使用量をログ
        self.log_current_memory_usage()

        return df

    def reduce_precision(self, arr: np.ndarray) -> np.ndarray:
        """
        NumPy配列の精度を下げてメモリ使用量を削減

        Args:
            arr: 対象NumPy配列

        Returns:
            ndarray: 精度を下げた配列
        """
        original_size = arr.nbytes / (1024 * 1024)

        # データ型に基づいて精度を下げる
        if arr.dtype == np.float64:
            result = arr.astype(np.float32)
        elif arr.dtype == np.float32:
            # 非割引時間帯ではさらに精度を下げる
            if not self.is_discount_time():
                result = arr.astype(np.float16)
            else:
                result = arr
        elif arr.dtype == np.int64:
            if np.min(arr) >= 0:
                if np.max(arr) < 2**32:
                    result = arr.astype(np.uint32)
                else:
                    result = arr.astype(np.uint64)
            else:
                if np.min(arr) > -2**31 and np.max(arr) < 2**31:
                    result = arr.astype(np.int32)
                else:
                    result = arr
        elif arr.dtype == np.int32:
            if np.min(arr) >= 0:
                if np.max(arr) < 2**16:
                    result = arr.astype(np.uint16)
                else:
                    result = arr.astype(np.uint32)
            else:
                if np.min(arr) > -2**15 and np.max(arr) < 2**15:
                    result = arr.astype(np.int16)
                else:
                    result = arr
        else:
            # 他のデータ型はそのまま
            result = arr

        new_size = result.nbytes / (1024 * 1024)

        if self.verbose:
            logger.info(f"NumPy配列精度削減: {arr.dtype} -> {result.dtype}")
            logger.info(f"サイズ変化: {original_size:.2f} MB -> {new_size:.2f} MB "
                      f"({(1 - new_size/original_size) * 100:.2f}% 削減)")

        return result

    def optimize_large_dataframe_with_dask(self,
                                         df_path: str,
                                         output_path: Optional[str] = None,
                                         optimize_dtypes: bool = True) -> str:
        """
        大きなCSVファイルをDaskを使って最適化

        Args:
            df_path: 入力CSVファイルパス
            output_path: 出力CSVファイルパス（Noneの場合は自動生成）
            optimize_dtypes: データ型を最適化するかどうか

        Returns:
            str: 最適化されたCSVファイルのパス
        """
        import dask.dataframe as dd

        # 入出力パスの処理
        if output_path is None:
            base, ext = os.path.splitext(df_path)
            output_path = f"{base}_optimized{ext}"

        logger.info(f"大規模DataFrameをDaskで最適化中: {df_path}")

        # Daskでファイルを読み込み
        ddf = dd.read_csv(df_path)

        # データ型の最適化
        if optimize_dtypes:
            # 数値列の最適化
            for col in ddf.select_dtypes(include=['int64']).columns:
                # サンプルからデータ型を推定
                sample = ddf[col].head(1000)
                col_min = sample.min()
                col_max = sample.max()

                # 適切なデータ型を選択
                if col_min >= 0:
                    if col_max < 2**8:
                        ddf[col] = ddf[col].astype(np.uint8)
                    elif col_max < 2**16:
                        ddf[col] = ddf[col].astype(np.uint16)
                    elif col_max < 2**32:
                        ddf[col] = ddf[col].astype(np.uint32)
                else:
                    if col_min > -2**7 and col_max < 2**7:
                        ddf[col] = ddf[col].astype(np.int8)
                    elif col_min > -2**15 and col_max < 2**15:
                        ddf[col] = ddf[col].astype(np.int16)
                    elif col_min > -2**31 and col_max < 2**31:
                        ddf[col] = ddf[col].astype(np.int32)

            # 浮動小数点列の最適化
            for col in ddf.select_dtypes(include=['float64']).columns:
                ddf[col] = ddf[col].astype(np.float32)

        # 最適化されたDataFrameを書き出し
        logger.info(f"最適化されたデータをCSVに保存中: {output_path}")
        ddf.to_csv(output_path, single_file=True, index=False)

        return output_path

    def report_memory_history(self) -> Dict[str, Any]:
        """
        メモリ使用量の履歴を報告

        Returns:
            Dict: メモリ使用量の統計情報
        """
        if not self.memory_usage_log:
            logger.warning("メモリ使用量の履歴がありません")
            return {}

        # 履歴からデータを抽出
        rss_values = [log['rss_mb'] for log in self.memory_usage_log]
        percent_values = [log['percent'] for log in self.memory_usage_log]
        discount_time_values = [log['is_discount_time'] for log in self.memory_usage_log]

        # 統計情報を計算
        stats = {
            'start_time': self.memory_usage_log[0]['timestamp'],
            'end_time': self.memory_usage_log[-1]['timestamp'],
            'points': len(self.memory_usage_log),
            'max_memory_mb': max(rss_values),
            'min_memory_mb': min(rss_values),
            'avg_memory_mb': sum(rss_values) / len(rss_values),
            'max_percent': max(percent_values),
            'discount_time_ratio': sum(discount_time_values) / len(discount_time_values)
        }

        # 最大・最小メモリ使用のインデックスを取得
        max_idx = rss_values.index(stats['max_memory_mb'])
        min_idx = rss_values.index(stats['min_memory_mb'])

        # 詳細情報
        stats['max_memory_time'] = self.memory_usage_log[max_idx]['timestamp']
        stats['min_memory_time'] = self.memory_usage_log[min_idx]['timestamp']
        stats['memory_change_mb'] = rss_values[-1] - rss_values[0]
        stats['memory_change_percent'] = ((rss_values[-1] / rss_values[0]) - 1) * 100 if rss_values[0] > 0 else 0

        if self.verbose:
            logger.info(f"メモリ使用量履歴レポート ({stats['start_time']} - {stats['end_time']})")
            logger.info(f"最大メモリ使用量: {stats['max_memory_mb']:.2f} MB ({stats['max_memory_time']})")
            logger.info(f"最小メモリ使用量: {stats['min_memory_mb']:.2f} MB ({stats['min_memory_time']})")
            logger.info(f"平均メモリ使用量: {stats['avg_memory_mb']:.2f} MB")
            logger.info(f"メモリ使用量変化: {stats['memory_change_mb']:.2f} MB ({stats['memory_change_percent']:.2f}%)")

        return stats


def get_optimal_chunk_size(total_size: int, available_memory: Optional[int] = None) -> int:
    """
    大規模データ処理のための最適なチャンクサイズを計算

    Args:
        total_size: 処理するデータの総サイズ
        available_memory: 利用可能なメモリ量（バイト）

    Returns:
        int: 最適なチャンクサイズ
    """
    # 利用可能なメモリが指定されていない場合は自動計算
    if available_memory is None:
        # 現在の空きメモリの80%を使用
        available_memory = int(psutil.virtual_memory().available * 0.8)

    # 最適なチャンクサイズの計算（利用可能メモリの25%を使用）
    chunk_size = int(available_memory * 0.25)

    # チャンクサイズが総サイズより大きい場合は総サイズを使用
    if chunk_size > total_size:
        return total_size

    # 最低1MBのチャンクサイズを保証
    min_chunk_size = 1 * 1024 * 1024  # 1MB
    if chunk_size < min_chunk_size:
        chunk_size = min_chunk_size

    logger.info(f"計算された最適チャンクサイズ: {chunk_size / (1024 * 1024):.2f} MB")
    return chunk_size


def main():
    """
    メイン関数
    """
    # 引数の解析
    if len(sys.argv) < 2:
        print("使用方法: python optimize_memory.py <csv_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    # メモリ最適化インスタンスの作成
    optimizer = MemoryOptimizer(verbose=True)

    # 現在の割引時間帯状態を表示
    if optimizer.is_discount_time():
        print("現在は割引時間帯です。通常の最適化を実行します。")
    else:
        print("現在は割引時間帯外です。積極的な最適化を実行します。")

    # CSVファイルの最適化
    optimized_file = optimizer.optimize_large_dataframe_with_dask(
        input_file,
        output_file,
        optimize_dtypes=True
    )

    print(f"ファイルを最適化しました: {optimized_file}")

    # メモリ使用量の履歴レポート
    optimizer.report_memory_history()


if __name__ == "__main__":
    main()