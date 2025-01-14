# -*- coding: utf-8 -*-
"""
PCA（主成分分析）Firestore Service
従業員の健康状態データに対する主成分分析を実行し、
結果をFirestoreに保存する機能を提供します。
"""
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import logging
from datetime import datetime
from backend.service.firestore.client import FirestoreService, StorageError

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class PCAAnalysisError(Exception):
    """PCA分析に関するエラー"""
    pass

class FirestorePCAAnalyzer:
    """
    主成分分析を実行し、結果をFirestoreに保存するクラスです。
    """
    def __init__(self) -> None:
        """
        Firestoreサービスとの接続を初期化します。

        Raises:
            StorageError: Firestore接続の初期化に失敗した場合
        """
        try:
            self.firestore_service = FirestoreService()
            logger.info("FirestorePCAAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FirestorePCAAnalyzer: {str(e)}")
            raise StorageError(f"Failed to initialize FirestorePCAAnalyzer: {str(e)}") from e

    async def analyze_and_save(
        self,
        data: pd.DataFrame,
        n_components: int,
        analysis_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        主成分分析を実行し、結果をFirestoreに保存します。

        Args:
            data (pd.DataFrame): 分析対象データ
            n_components (int): 抽出する主成分の数
            analysis_metadata (Optional[Dict[str, Any]], optional): 分析に関するメタデータ
            user_id (Optional[str], optional): 分析を実行したユーザーのID

        Returns:
            Tuple[pd.DataFrame, str]: (主成分分析結果のDataFrame, FirestoreのドキュメントID)

        Raises:
            PCAAnalysisError: 分析処理中にエラーが発生した場合
            StorageError: Firestoreへの保存時にエラーが発生した場合
            ValueError: 入力データが不正な場合
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")

        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer")

        try:
            logger.info(f"Starting PCA analysis with {n_components} components")

            # 入力データの検証
            if data.empty:
                raise PCAAnalysisError("Input data is empty")

            if n_components > min(data.shape):
                raise PCAAnalysisError(
                    f"Number of components ({n_components}) cannot exceed min(n_samples, n_features) = {min(data.shape)}"
                )

            # PCA実行
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(data)

            # 主成分スコアをデータフレームに変換
            principal_df = pd.DataFrame(
                data=principal_components,
                columns=[f'principal_component_{i+1}' for i in range(n_components)]
            )

            # 分析結果の準備
            analysis_result: Dict[str, Any] = {
                'timestamp': datetime.now(),
                'n_components': n_components,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'input_shape': list(data.shape),  # tupleをlistに変換
                'component_names': principal_df.columns.tolist(),
                'metadata': analysis_metadata or {},
            }

            if user_id is not None:
                analysis_result['user_id'] = user_id

            # 結果をFirestoreに保存
            doc_ids = await self.firestore_service.save_results(
                results=[analysis_result],
                collection_name='pca_analyses'
            )

            if not doc_ids:
                raise StorageError("Failed to save analysis results to Firestore")

            logger.info(f"PCA analysis completed and saved with document ID: {doc_ids[0]}")

            # 元のデータと主成分スコアを結合
            result_df = pd.concat([data, principal_df], axis=1)

            return result_df, doc_ids[0]

        except Exception as e:
            error_msg = f"Error during PCA analysis: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (PCAAnalysisError, StorageError, ValueError)):
                raise
            raise PCAAnalysisError(error_msg) from e

    async def get_analysis_history(
        self,
        limit: int = 10,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        過去のPCA分析結果を取得します。

        Args:
            limit (int): 取得する結果の最大数
            user_id (Optional[str]): 特定ユーザーの結果のみを取得する場合のユーザーID

        Returns:
            List[Dict[str, Any]]: 分析結果の履歴

        Raises:
            StorageError: Firestoreからのデータ取得に失敗した場合
            ValueError: 不正なパラメータが指定された場合
        """
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")

        try:
            conditions = None
            if user_id is not None:
                conditions = [{'field': 'user_id', 'operator': '==', 'value': user_id}]

            results = await self.firestore_service.fetch_documents(
                collection_name='pca_analyses',
                conditions=conditions,
                limit=limit,
                order_by='timestamp',
                direction='desc'
            )

            if results is None:
                return []

            return results

        except Exception as e:
            error_msg = f"Error fetching PCA analysis history: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, StorageError):
                raise
            raise StorageError(error_msg) from e

    async def close(self) -> None:
        """
        リソースを解放します。

        Raises:
            StorageError: リソースの解放に失敗した場合
        """
        try:
            await self.firestore_service.close()
            logger.info("FirestorePCAAnalyzer closed successfully")
        except Exception as e:
            error_msg = f"Error closing FirestorePCAAnalyzer: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

def create_pca_analyzer() -> FirestorePCAAnalyzer:
    """
    FirestorePCAAnalyzerのインスタンスを作成します。

    Returns:
        FirestorePCAAnalyzer: 初期化済みのアナライザーインスタンス

    Raises:
        StorageError: アナライザーの初期化に失敗した場合
    """
    return FirestorePCAAnalyzer()