# -*- coding: utf-8 -*-
"""
生存時間分析
Startup Wellness プログラム導入前後における、従業員の離職までの時間を比較分析します。
Firestoreと統合された非同期処理対応バージョン。
"""
from typing import Dict, Any, Optional, List, cast
import pandas as pd
from lifelines import KaplanMeierFitter
import logging
from datetime import datetime
import asyncio
from firebase_admin import firestore
from google.cloud.firestore_v1.document import DocumentReference
from backend.service.firestore.client import get_firestore_client

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class SurvivalAnalysisError(Exception):
    """生存時間分析に関するエラー"""
    pass

class SurvivalAnalyzer:
    """
    Firestore統合された生存時間分析を実行するためのクラス
    """
    def __init__(self):
        """
        Firestoreクライアントを初期化
        """
        try:
            self.db = get_firestore_client()
            self.collection_name = 'survival_analysis_results'
            logger.info("SurvivalAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SurvivalAnalyzer: {str(e)}")
            raise SurvivalAnalysisError("Initialization failed") from e

    async def analyze(
        self,
        data: pd.DataFrame,
        duration_col: str,
        event_col: str,
        analysis_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生存時間分析を実行し、結果をFirestoreに保存します

        Args:
            data (pd.DataFrame): 分析対象データ
            duration_col (str): イベント発生までの時間を表すカラム名
            event_col (str): イベント発生を表すカラム名 (例: 離職 = 1, 在職中 = 0)
            analysis_id (Optional[str]): 分析結果の識別子
            metadata (Optional[Dict[str, Any]]): 追加のメタデータ

        Returns:
            Dict[str, Any]: 分析結果とドキュメントID
        """
        try:
            logger.info("Starting survival analysis")

            # データバリデーション
            self._validate_input_data(data, duration_col, event_col)

            # Kaplan-Meier分析の実行
            kmf = KaplanMeierFitter()
            kmf.fit(data[duration_col], data[event_col])

            # 分析結果の整形
            survival_curve = kmf.survival_function_.reset_index().to_dict('records')
            median_survival = kmf.median_survival_time_

            # 基本的な分析結果
            analysis_results = {
                "survival_curve": survival_curve,
                "median_survival_time": float(median_survival),
                "total_subjects": len(data),
                "event_observed": int(data[event_col].sum()),
                "censored_subjects": int(len(data) - data[event_col].sum()),
                "analysis_timestamp": datetime.now(),
                "duration_column": duration_col,
                "event_column": event_col
            }

            # メタデータの追加（オプション）
            if metadata:
                analysis_results["metadata"] = metadata

            # Firestoreへの保存
            doc_ref = (self.db.collection(self.collection_name).document(analysis_id)
                      if analysis_id
                      else self.db.collection(self.collection_name).document())

            await self._save_to_firestore(doc_ref, analysis_results)

            # 結果の返却
            return {
                "document_id": doc_ref.id,
                "analysis_results": analysis_results
            }

        except Exception as e:
            error_msg = f"Error during survival analysis: {str(e)}"
            logger.error(error_msg)
            raise SurvivalAnalysisError(error_msg) from e

    async def get_analysis_results(
        self,
        analysis_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        特定の分析結果をFirestoreから取得します

        Args:
            analysis_id (str): 分析結果のドキュメントID

        Returns:
            Optional[Dict[str, Any]]: 分析結果
        """
        try:
            doc_ref = self.db.collection(self.collection_name).document(analysis_id)
            doc = doc_ref.get()

            if doc.exists:
                result_dict = doc.to_dict()
                if result_dict is not None:
                    return {"id": doc.id, **result_dict}
                return None
            else:
                logger.warning(f"No analysis results found for ID: {analysis_id}")
                return None

        except Exception as e:
            error_msg = f"Error retrieving analysis results: {str(e)}"
            logger.error(error_msg)
            raise SurvivalAnalysisError(error_msg) from e

    async def list_analyses(
        self,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        実行済みの分析結果一覧を取得します

        Args:
            limit (int): 取得する結果の最大数
            metadata_filter (Optional[Dict[str, Any]]): メタデータによるフィルタリング条件

        Returns:
            List[Dict[str, Any]]: 分析結果のリスト
        """
        try:
            query = self.db.collection(self.collection_name)

            if metadata_filter:
                for key, value in metadata_filter.items():
                    query = query.where(f"metadata.{key}", "==", value)

            docs = query.limit(limit).get()
            results: List[Dict[str, Any]] = []

            for doc in docs:
                doc_dict = doc.to_dict()
                if doc_dict is not None:
                    results.append({"id": doc.id, **doc_dict})

            return results

        except Exception as e:
            error_msg = f"Error listing analyses: {str(e)}"
            logger.error(error_msg)
            raise SurvivalAnalysisError(error_msg) from e

    def _validate_input_data(
        self,
        data: pd.DataFrame,
        duration_col: str,
        event_col: str
    ) -> None:
        """
        入力データのバリデーションを行います
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")

        if duration_col not in data.columns:
            raise ValueError(f"Duration column '{duration_col}' not found in data")

        if event_col not in data.columns:
            raise ValueError(f"Event column '{event_col}' not found in data")

        if data[duration_col].isnull().any():
            raise ValueError("Duration column contains null values")

        if event_col not in data.columns:
            raise ValueError("Event column contains null values")

    async def _save_to_firestore(
        self,
        doc_ref: DocumentReference,
        data: Dict[str, Any]
    ) -> None:
        """
        分析結果をFirestoreに保存します
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, doc_ref.set, data)
            logger.info(f"Successfully saved analysis results to document: {doc_ref.id}")
        except Exception as e:
            error_msg = f"Error saving to Firestore: {str(e)}"
            logger.error(error_msg)
            raise SurvivalAnalysisError(error_msg) from e