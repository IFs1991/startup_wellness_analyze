# -*- coding: utf-8 -*-
"""
記述統計量 計算サービス
基本的な統計量 (平均, 中央値, 標準偏差など) を算出し、
結果をFirestoreに保存します。
"""
from typing import Dict, Any, Optional
import pandas as pd
import logging
from datetime import datetime
from firebase_admin import firestore
from typing import Optional, Dict, Any, Union
import asyncio
from pandas.core.frame import DataFrame
from google.cloud.firestore_v1.client import Client as FirestoreClient

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class CalculationError(Exception):
    """統計量計算に関するエラー"""
    pass

class DescriptiveStatsCalculator:
    """
    記述統計量を計算し、結果をFirestoreに保存するクラスです。
    """
    def __init__(self, db: FirestoreClient):
        """
        初期化メソッド
        Args:
            db (FirestoreClient): Firestoreクライアントインスタンス
        """
        self.db = db
        self.stats_collection = 'descriptive_stats'
        logger.info("DescriptiveStatsCalculator initialized")

    def calculate(self, data: DataFrame) -> Dict[str, Any]:
        """
        記述統計量を計算します。（同期バージョン）

        Args:
            data (DataFrame): 分析対象データ

        Returns:
            Dict[str, Any]: 計算結果

        Raises:
            CalculationError: 計算時のエラー
        """
        try:
            # 基本的な記述統計量の計算
            stats_df = data.describe()

            # 追加の統計量計算
            additional_stats = {
                'skewness': data.skew().to_dict(),
                'kurtosis': data.kurtosis().to_dict(),
                'missing_values': data.isnull().sum().to_dict()
            }

            # 結果の整形
            stats_result = {
                'basic_stats': stats_df.to_dict(),
                'additional_stats': additional_stats,
                'column_names': list(data.columns),
                'row_count': len(data)
            }

            return stats_result

        except Exception as e:
            error_msg = f"Error calculating descriptive statistics: {str(e)}"
            logger.error(error_msg)
            raise CalculationError(error_msg) from e

    async def calculate_and_save(
        self,
        data: DataFrame,
        analysis_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        記述統計量を計算し、結果をFirestoreに保存します。

        Args:
            data (DataFrame): 分析対象データ
            analysis_id (str): 分析ID
            metadata (Optional[Dict[str, Any]]): メタデータ

        Returns:
            Dict[str, Any]: 計算結果と保存されたドキュメントID

        Raises:
            CalculationError: 計算またはデータ保存時のエラー
        """
        try:
            logger.info(f"Starting statistical analysis for analysis_id: {analysis_id}")

            # 統計量の計算（同期的に実行）
            stats_result = self.calculate(data)

            # メタデータとタイムスタンプの追加
            stats_result.update({
                'analysis_id': analysis_id,
                'created_at': datetime.now(),
                'metadata': metadata or {}
            })

            # Firestoreに保存（非同期処理）
            loop = asyncio.get_event_loop()
            doc_ref = self.db.collection(self.stats_collection).document(analysis_id)
            await loop.run_in_executor(None, lambda: doc_ref.set(stats_result))

            logger.info(f"Successfully saved statistical analysis results for analysis_id: {analysis_id}")
            return stats_result

        except Exception as e:
            error_msg = f"Error in calculate_and_save: {str(e)}"
            logger.error(error_msg)
            raise CalculationError(error_msg) from e

    async def get_analysis_results(
        self,
        analysis_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        保存された分析結果を取得します。

        Args:
            analysis_id (str): 分析ID

        Returns:
            Optional[Dict[str, Any]]: 分析結果。存在しない場合はNone

        Raises:
            CalculationError: データ取得時のエラー
        """
        try:
            logger.info(f"Fetching analysis results for analysis_id: {analysis_id}")

            # 非同期でドキュメント取得
            loop = asyncio.get_event_loop()
            doc_ref = self.db.collection(self.stats_collection).document(analysis_id)
            doc = await loop.run_in_executor(None, doc_ref.get)

            if doc.exists:
                logger.info(f"Successfully retrieved analysis results for analysis_id: {analysis_id}")
                return doc.to_dict()

            logger.info(f"No analysis results found for analysis_id: {analysis_id}")
            return None

        except Exception as e:
            error_msg = f"Error retrieving analysis results: {str(e)}"
            logger.error(error_msg)
            raise CalculationError(error_msg) from e

    async def delete_analysis_results(
        self,
        analysis_id: str
    ) -> None:
        """
        保存された分析結果を削除します。

        Args:
            analysis_id (str): 分析ID

        Raises:
            CalculationError: データ削除時のエラー
        """
        try:
            logger.info(f"Deleting analysis results for analysis_id: {analysis_id}")

            # 非同期でドキュメント削除
            loop = asyncio.get_event_loop()
            doc_ref = self.db.collection(self.stats_collection).document(analysis_id)
            await loop.run_in_executor(None, doc_ref.delete)

            logger.info(f"Successfully deleted analysis results for analysis_id: {analysis_id}")

        except Exception as e:
            error_msg = f"Error deleting analysis results: {str(e)}"
            logger.error(error_msg)
            raise CalculationError(error_msg) from e