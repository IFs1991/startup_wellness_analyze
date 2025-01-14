# -*- coding: utf-8 -*-
"""
時系列分析サービス
VASデータと損益計算書データの経時変化を分析し、結果をFirestoreに保存します。
"""
from typing import Dict, Any, Optional, List, Union, TypeVar, cast
import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
from datetime import datetime
import asyncio
import firebase_admin
from firebase_admin import firestore
from google.cloud.firestore_v1.client import Client
from google.cloud.firestore_v1.document import DocumentReference
from google.cloud.firestore_v1.query import Query

# 型変数の定義
T = TypeVar('T')

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TimeSeriesAnalysisError(Exception):
    """時系列分析に関するエラー"""
    pass

class ValidationError(Exception):
    """データバリデーションに関するエラー"""
    pass

class TimeSeriesAnalyzer:
    """
    時系列データを分析し、結果をFirestoreに保存するためのクラスです。
    """
    def __init__(self, db: Union[Client, Any]):
        """
        初期化

        Args:
            db (Union[Client, Any]): Firestoreクライアントインスタンス
        """
        self.db = db
        self.collection_name = 'time_series_analysis'
        logger.info("TimeSeriesAnalyzer initialized successfully")

    def _safe_get(self, data: Optional[Dict[str, T]], key: str, default: T) -> T:
        """
        安全にディクショナリからデータを取得します。

        Args:
            data (Optional[Dict[str, T]]): データディクショナリ
            key (str): キー
            default (T): デフォルト値

        Returns:
            T: 取得した値またはデフォルト値
        """
        if data is None:
            return default
        return data.get(key, default)

    async def analyze_and_save(
        self,
        data: pd.DataFrame,
        target_variable: str,
        analysis_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        時系列分析を実行し、結果をFirestoreに保存します。

        Args:
            data (pd.DataFrame): 時系列データ
            target_variable (str): 分析対象の変数名
            analysis_metadata (Optional[Dict[str, Any]]): 分析に関する追加メタデータ
            user_id (Optional[str]): 分析を実行したユーザーのID

        Returns:
            Dict[str, Any]: 分析結果とドキュメントID

        Raises:
            TimeSeriesAnalysisError: 分析処理中にエラーが発生した場合
            ValidationError: データが無効な場合
        """
        try:
            logger.info(f"Starting time series analysis for variable: {target_variable}")

            # データバリデーション
            await self._validate_data(data, target_variable)

            # 分析実行
            analysis_results = await self._perform_analysis(data, target_variable)

            if analysis_results is None:
                raise TimeSeriesAnalysisError("Analysis failed to produce results")

            # 保存用データの準備
            document_data = {
                'target_variable': target_variable,
                'analysis_results': analysis_results,
                'data_points': len(data),
                'created_at': datetime.now(),
                'user_id': user_id if user_id is not None else 'anonymous',
                'metadata': analysis_metadata or {}
            }

            # Firestoreに保存
            doc_ref = await self._save_to_firestore(document_data)

            if doc_ref is None:
                raise TimeSeriesAnalysisError("Failed to save to Firestore")

            # 結果の作成
            result = {
                'document_id': doc_ref.id,
                'analysis_results': analysis_results,
                'metadata': document_data
            }

            logger.info(f"Analysis completed and saved with document ID: {doc_ref.id}")
            return result

        except ValidationError as e:
            error_msg = f"Validation error: {str(e)}"
            logger.error(error_msg)
            raise ValidationError(error_msg) from e
        except Exception as e:
            error_msg = f"Error in analyze_and_save: {str(e)}"
            logger.error(error_msg)
            raise TimeSeriesAnalysisError(error_msg) from e

    async def get_analysis_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        過去の分析結果を取得します。

        Args:
            user_id (Optional[str]): 特定ユーザーの分析結果のみを取得
            limit (int): 取得する結果の最大数

        Returns:
            List[Dict[str, Any]]: 過去の分析結果のリスト

        Raises:
            TimeSeriesAnalysisError: データ取得中にエラーが発生した場合
        """
        try:
            query = self.db.collection(self.collection_name)

            if user_id:
                query = query.where('user_id', '==', user_id)

            query = query.order_by('created_at', direction='DESCENDING').limit(limit)

            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, query.get)

            results = []
            if docs is not None:
                for doc in docs:
                    if doc is not None and doc.exists:
                        data = doc.to_dict()
                        if data is not None:
                            data['id'] = doc.id
                            results.append(data)

            return results

        except Exception as e:
            error_msg = f"Error retrieving analysis history: {str(e)}"
            logger.error(error_msg)
            raise TimeSeriesAnalysisError(error_msg) from e

    async def _validate_data(self, data: pd.DataFrame, target_variable: str) -> None:
        """
        入力データのバリデーションを行います。

        Args:
            data (pd.DataFrame): 検証するデータフレーム
            target_variable (str): 検証する変数名

        Raises:
            ValidationError: バリデーションエラー
        """
        if data is None:
            raise ValidationError("Data frame is None")

        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")

        if target_variable not in data.columns:
            raise ValidationError(f"Target variable '{target_variable}' not found in data")

        if data[target_variable].isnull().any():
            raise ValidationError(f"Missing values found in target variable '{target_variable}'")

        if len(data) < 10:  # 最小データポイント数の確認
            raise ValidationError("Insufficient data points for analysis (minimum 10 required)")

    async def _perform_analysis(
        self,
        data: pd.DataFrame,
        target_variable: str
    ) -> Dict[str, Any]:
        """
        時系列分析を実行します。

        Args:
            data (pd.DataFrame): 分析するデータ
            target_variable (str): 分析対象の変数名

        Returns:
            Dict[str, Any]: 分析結果

        Raises:
            TimeSeriesAnalysisError: 分析中にエラーが発生した場合
        """
        try:
            loop = asyncio.get_event_loop()

            # 非同期で時系列分析を実行
            model = await loop.run_in_executor(
                None,
                lambda: sm.tsa.ARIMA(data[target_variable], order=(5,1,0)).fit()
            )

            if model is None:
                raise TimeSeriesAnalysisError("Failed to fit ARIMA model")

            # 予測の実行
            forecast = model.forecast(5)
            if forecast is None:
                raise TimeSeriesAnalysisError("Failed to generate forecast")

            # 結果の整形
            analysis_results = {
                'aic': float(model.aic),
                'bic': float(model.bic),
                'forecast': forecast.tolist(),
                'model_parameters': {k: float(v) for k, v in model.params.items()},
                'residuals_std': float(model.resid.std())
            }

            return analysis_results

        except Exception as e:
            error_msg = f"Error performing time series analysis: {str(e)}"
            logger.error(error_msg)
            raise TimeSeriesAnalysisError(error_msg) from e

    async def _save_to_firestore(
        self,
        document_data: Dict[str, Any]
    ) -> Optional[DocumentReference]:
        """
        分析結果をFirestoreに保存します。

        Args:
            document_data (Dict[str, Any]): 保存するデータ

        Returns:
            Optional[DocumentReference]: 保存されたドキュメントの参照

        Raises:
            TimeSeriesAnalysisError: 保存中にエラーが発生した場合
        """
        try:
            if document_data is None:
                raise ValueError("Document data is None")

            doc_ref = self.db.collection(self.collection_name).document()

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: doc_ref.set(document_data))

            return doc_ref

        except Exception as e:
            error_msg = f"Error saving to Firestore: {str(e)}"
            logger.error(error_msg)
            raise TimeSeriesAnalysisError(error_msg) from e