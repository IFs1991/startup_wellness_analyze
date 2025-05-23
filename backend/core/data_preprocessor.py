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
from .common_logger import get_logger
from .exceptions import DataPreprocessingError
from .patterns import Singleton, LazyImport

# 型変数の定義
T = TypeVar('T')
FillMethod = Literal['backfill', 'bfill', 'ffill', 'pad']

# ロギングの設定
logger = get_logger(__name__)

# Firebaseクライアント用の遅延インポート
firebase_client_module = LazyImport('core.firebase_client', 'get_firebase_client')

@Singleton
class DataPreprocessor:
    """
    データの前処理と整形を行うクラス
    このクラスはシングルトンパターンで実装されており、アプリケーション全体で一貫したインスタンスを提供します。
    """
    def __init__(self):
        """初期化処理"""
        self._firestore_service = None
        self.required_columns = {
            'vas_data': ['startup_id', 'timestamp', 'score', 'category'],
            'financial_data': ['startup_id', 'timestamp', 'revenue', 'expenses']
        }
        logger.info("DataPreprocessorを初期化しました")

    def _get_firestore_service(self):
        """FirestoreServiceを遅延ロードして返す"""
        if self._firestore_service is None:
            # 新しいFirebaseクライアントを使用
            self._firestore_service = firebase_client_module()
            logger.info("FirebaseClientを初期化しました")
        return self._firestore_service

    async def get_data(
        self,
        collection_name: str,
        conditions: Optional[List[Dict[str, Any]]] = None,
        as_dataframe: bool = True
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Firestoreからデータを取得し、必要に応じてDataFrameに変換します。

        Args:
            collection_name: 取得するコレクション名
            conditions: クエリ条件のリスト
            as_dataframe: DataFrameに変換するかどうか

        Returns:
            DataFrame または ドキュメントのリスト
        """
        try:
            # FirestoreServiceを遅延ロード
            firebase_client = self._get_firestore_service()

            # クエリ条件を新しいフォーマットに変換
            filters = None
            if conditions:
                filters = [
                    {
                        "field": cond.get("field"),
                        "op": cond.get("operator", "=="),
                        "value": cond.get("value")
                    }
                    for cond in conditions
                ]

            # Firestoreからデータを取得
            results = await firebase_client.query_documents(
                collection=collection_name,
                filters=filters
            )

            if not results:
                logger.warning(f"コレクション {collection_name} からデータが見つかりませんでした。")
                return pd.DataFrame() if as_dataframe else []

            if as_dataframe:
                return pd.DataFrame(results)
            else:
                return results

        except Exception as e:
            error_msg = f"データ取得中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            raise DataPreprocessingError(error_msg) from e

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
            merge_on: 結合キーのリスト

        Returns:
            結合されたデータフレーム
        """
        try:
            if merge_on is None:
                merge_on = ['startup_id', 'timestamp']

            result_df = pd.merge(
                vas_df,
                financial_df,
                on=merge_on,
                how='outer',
                suffixes=('_vas', '_fin')
            )

            logger.info(f"Successfully merged datasets with {len(result_df)} records")
            return cast(DataFrame, result_df)

        except Exception as e:
            error_msg = f"Error merging datasets: {str(e)}"
            logger.error(error_msg)
            raise DataPreprocessingError(error_msg) from e