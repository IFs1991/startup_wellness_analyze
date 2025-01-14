# -*- coding: utf-8 -*-
"""
相関分析
VAS データと損益計算書データの相関関係を分析します。
"""
from backend.service.firestore.client import FirestoreService
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """
    変数間の相関関係を分析するためのクラスです。
    """
    def __init__(self):
        """Initialize the CorrelationAnalyzer with Firestore service"""
        self.firestore_service = FirestoreService()
        self.collection_name = 'correlation_analysis'

    async def analyze(
        self,
        data: pd.DataFrame,
        variables: List[str],
        user_id: str,
        analysis_name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        相関分析を実行し、結果をFirestoreに保存します。

        Args:
            data (pd.DataFrame): 分析対象のデータ
            variables (List[str]): 分析対象の変数名のリスト
            user_id (str): 分析を実行したユーザーのID
            analysis_name (Optional[str]): 分析の名前
            metadata (Optional[Dict]): 追加のメタデータ

        Returns:
            pd.DataFrame: 相関行列
        """
        try:
            # 相関分析の実行
            correlation_matrix = data[variables].corr()

            # 結果の整形とNaN値の処理
            correlation_dict = correlation_matrix.replace({np.nan: None}).to_dict('index')

            # 保存用データの準備
            analysis_result = {
                'analysis_type': 'correlation',
                'variables': variables,
                'user_id': user_id,
                'analysis_name': analysis_name or f"Correlation Analysis {datetime.now().isoformat()}",
                'correlation_matrix': correlation_dict,
                'data_shape': data[variables].shape,
                'variable_statistics': {
                    var: {
                        'mean': float(data[var].mean()),
                        'std': float(data[var].std()),
                        'count': int(data[var].count())
                    } for var in variables
                },
                'metadata': metadata or {},
                'created_at': datetime.now()
            }

            # Firestoreに保存
            doc_ids = await self.firestore_service.save_results(
                results=[analysis_result],
                collection_name=self.collection_name
            )

            logger.info(f"Successfully saved correlation analysis results with ID: {doc_ids[0]}")

            # 相関行列にIDを付与（メタデータとして）
            correlation_matrix.attrs['analysis_id'] = doc_ids[0]

            return correlation_matrix

        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            raise

    async def get_analysis_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        過去の相関分析結果を取得します。

        Args:
            user_id (Optional[str]): 特定ユーザーの結果のみを取得する場合に指定
            limit (int): 取得する結果の最大数

        Returns:
            List[Dict]: 分析結果のリスト
        """
        try:
            conditions = []
            if user_id:
                conditions.append({
                    'field': 'user_id',
                    'operator': '==',
                    'value': user_id
                })

            results = await self.firestore_service.fetch_documents(
                collection_name=self.collection_name,
                conditions=conditions,
                limit=limit,
                order_by='created_at',
                direction='desc'
            )

            logger.info(f"Retrieved {len(results)} correlation analysis results")
            return results

        except Exception as e:
            logger.error(f"Error retrieving analysis history: {str(e)}")
            raise

    async def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict]:
        """
        特定の分析結果を取得します。

        Args:
            analysis_id (str): 取得する分析のID

        Returns:
            Optional[Dict]: 分析結果
        """
        try:
            conditions = [{
                'field': 'id',
                'operator': '==',
                'value': analysis_id
            }]

            results = await self.firestore_service.fetch_documents(
                collection_name=self.collection_name,
                conditions=conditions,
                limit=1
            )

            return results[0] if results else None

        except Exception as e:
            logger.error(f"Error retrieving analysis {analysis_id}: {str(e)}")
            raise

    async def update_analysis_metadata(
        self,
        analysis_id: str,
        metadata: Dict
    ) -> None:
        """
        分析結果のメタデータを更新します。

        Args:
            analysis_id (str): 更新する分析のID
            metadata (Dict): 新しいメタデータ
        """
        try:
            await self.firestore_service.update_document(
                collection_name=self.collection_name,
                document_id=analysis_id,
                data={'metadata': metadata}
            )
            logger.info(f"Updated metadata for analysis {analysis_id}")

        except Exception as e:
            logger.error(f"Error updating analysis metadata: {str(e)}")
            raise