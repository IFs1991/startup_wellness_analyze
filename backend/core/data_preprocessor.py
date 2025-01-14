# -*- coding: utf-8 -*-
"""
データ前処理モジュール
Firestoreから取得したデータの前処理と整形を行います。
"""
from typing import Dict, List, Optional, Union, Any, Literal, cast, TypeVar, overload
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pandas import DataFrame, Series
from pandas._libs.missing import NAType
from pandas.api.types import is_numeric_dtype

# 型変数の定義
T = TypeVar('T')
FillMethod = Literal['backfill', 'bfill', 'ffill', 'pad']

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class DataPreprocessingError(Exception):
    """データ前処理に関するエラー"""
    pass

class DataPreprocessor:
    """
    Firestoreから取得したデータを前処理するためのクラス
    """
    def __init__(self):
        """
        DataPreprocessorの初期化
        """
        self.required_columns = {
            'vas_data': ['startup_id', 'timestamp', 'score'],
            'financial_data': ['startup_id', 'timestamp', 'revenue']
        }

    def preprocess_firestore_data(
        self,
        data: List[Dict[str, Any]],
        data_type: Literal['vas_data', 'financial_data'],
        options: Optional[Dict[str, Any]] = None
    ) -> DataFrame:
        """
        Firestoreから取得したデータを前処理します

        Args:
            data: Firestoreから取得した生データ
            data_type: データタイプ ('vas_data' または 'financial_data')
            options: 前処理オプション

        Returns:
            前処理済みのデータフレーム
        """
        try:
            logger.info(f"Starting preprocessing for {data_type}")

            df = pd.DataFrame(data)
            self._validate_columns(df, data_type)
            df = self._process_timestamp(df)
            df = self._convert_data_types(df, data_type)
            df = self._handle_missing_values(df, options)
            df = self._handle_outliers(df, options)
            df = cast(DataFrame, df.sort_values('timestamp'))

            logger.info(f"Successfully preprocessed {len(df)} records for {data_type}")
            return df

        except Exception as e:
            error_msg = f"Error preprocessing {data_type}: {str(e)}"
            logger.error(error_msg)
            raise DataPreprocessingError(error_msg) from e

    def _validate_columns(self, df: DataFrame, data_type: str) -> None:
        """必須カラムの存在を確認します"""
        missing_columns = set(self.required_columns[data_type]) - set(df.columns)
        if missing_columns:
            raise DataPreprocessingError(f"Missing required columns: {missing_columns}")

    def _process_timestamp(self, df: DataFrame) -> DataFrame:
        """タイムスタンプを処理します"""
        try:
            if isinstance(df['timestamp'].iloc[0], datetime):
                return df

            result_df = df.copy()
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
            return result_df
        except Exception as e:
            raise DataPreprocessingError(f"Error processing timestamp: {str(e)}")

    def _convert_data_types(self, df: DataFrame, data_type: str) -> DataFrame:
        """データ型を適切な型に変換します"""
        try:
            result_df = df.copy()
            if data_type == 'vas_data':
                result_df['score'] = pd.to_numeric(result_df['score'], errors='coerce')
            elif data_type == 'financial_data':
                result_df['revenue'] = pd.to_numeric(result_df['revenue'], errors='coerce')

            result_df['startup_id'] = result_df['startup_id'].astype(str)
            return result_df
        except Exception as e:
            raise DataPreprocessingError(f"Error converting data types: {str(e)}")

    def _apply_fill_method(
        self,
        series: Series,
        method: FillMethod
    ) -> Series:
        """
        個別のSeriesに対して補完メソッドを適用します
        """
        kwargs: Dict[str, Any] = {'method': method}
        return series.fillna(**kwargs)

    def _forward_fill(self, df: DataFrame) -> DataFrame:
        """前方向に欠損値を補完します"""
        try:
            result_df = df.copy()
            for col in result_df.columns:
                if is_numeric_dtype(result_df[col]):
                    result_df[col] = self._apply_fill_method(result_df[col], 'ffill')
            return result_df
        except Exception as e:
            raise DataPreprocessingError(f"Error in forward fill: {str(e)}")

    def _backward_fill(self, df: DataFrame) -> DataFrame:
        """後方向に欠損値を補完します"""
        try:
            result_df = df.copy()
            for col in result_df.columns:
                if is_numeric_dtype(result_df[col]):
                    result_df[col] = self._apply_fill_method(result_df[col], 'bfill')
            return result_df
        except Exception as e:
            raise DataPreprocessingError(f"Error in backward fill: {str(e)}")

    def _handle_missing_values(
        self,
        df: DataFrame,
        options: Optional[Dict[str, Any]] = None
    ) -> DataFrame:
        """
        欠損値を処理します
        """
        try:
            if not options or 'missing_value_strategy' not in options:
                # デフォルトの処理: 前方補完→後方補完
                result_df = self._forward_fill(df)
                result_df = self._backward_fill(result_df)
                return result_df

            strategy = options['missing_value_strategy']
            result_df = df.copy()

            if strategy == 'drop':
                return cast(DataFrame, result_df.dropna())
            elif strategy == 'mean':
                numeric_cols = result_df.select_dtypes(include=[np.number]).columns
                means = result_df[numeric_cols].mean()
                for col in numeric_cols:
                    result_df[col] = result_df[col].fillna(value=means[col])
                return result_df
            elif strategy == 'median':
                numeric_cols = result_df.select_dtypes(include=[np.number]).columns
                medians = result_df[numeric_cols].median()
                for col in numeric_cols:
                    result_df[col] = result_df[col].fillna(value=medians[col])
                return result_df
            else:
                result_df = self._forward_fill(result_df)
                result_df = self._backward_fill(result_df)
                return result_df

        except Exception as e:
            raise DataPreprocessingError(f"Error handling missing values: {str(e)}")

    def _handle_outliers(
        self,
        df: DataFrame,
        options: Optional[Dict[str, Any]] = None
    ) -> DataFrame:
        """
        異常値を処理します
        """
        try:
            if not options or 'outlier_strategy' not in options:
                return df

            strategy = options['outlier_strategy']
            result_df = df.copy()
            numeric_columns = result_df.select_dtypes(include=[np.number]).columns

            if strategy == 'iqr':
                for col in numeric_columns:
                    Q1 = result_df[col].quantile(0.25)
                    Q3 = result_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    result_df[col] = result_df[col].clip(lower=lower_bound, upper=upper_bound)

            elif strategy == 'zscore':
                for col in numeric_columns:
                    mean = result_df[col].mean()
                    std = result_df[col].std()
                    result_df[col] = result_df[col].clip(
                        lower=mean - 3*std,
                        upper=mean + 3*std
                    )

            return result_df

        except Exception as e:
            raise DataPreprocessingError(f"Error handling outliers: {str(e)}")

    def merge_datasets(
        self,
        vas_df: DataFrame,
        financial_df: DataFrame,
        merge_on: Optional[List[str]] = None
    ) -> DataFrame:
        """
        VASデータと財務データを結合します

        Args:
            vas_df: VASデータ
            financial_df: 財務データ
            merge_on: 結合キーのリスト (デフォルト: ['startup_id', 'timestamp'])

        Returns:
            結合されたデータフレーム
        """
        try:
            if merge_on is None:
                merge_on = ['startup_id', 'timestamp']

            # データの結合
            result_df = pd.merge(
                vas_df,
                financial_df,
                on=merge_on,
                how='outer',
                suffixes=('_vas', '_fin')
            )

            return result_df

        except Exception as e:
            raise DataPreprocessingError(f"データの結合中にエラーが発生: {str(e)}")