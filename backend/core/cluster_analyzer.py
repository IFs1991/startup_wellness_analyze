# backend/core/cluster_analyzer.py
from backend.service.firestore.client import FirestoreService
from sklearn.cluster import KMeans
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ClusterAnalyzer:
    def __init__(self):
        """Initialize the ClusterAnalyzer with Firestore service"""
        self.firestore_service = FirestoreService()
        self.collection_name = 'cluster_analysis'

    async def analyze(
        self,
        data: pd.DataFrame,
        n_clusters: int,
        user_id: str,
        metadata: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        クラスタ分析を実行し、結果をFirestoreに保存

        Args:
            data: 分析対象のデータ
            n_clusters: クラスタ数
            user_id: 分析を実行したユーザーのID
            metadata: 追加のメタデータ
        """
        try:
            # クラスタリング実行
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            data['cluster'] = kmeans.fit_predict(data)

            # 保存用データの準備
            analysis_result = {
                'model_type': 'kmeans',
                'n_clusters': n_clusters,
                'user_id': user_id,
                'data_shape': data.shape,
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'cluster_labels': data['cluster'].tolist(),
                'feature_names': data.columns.tolist(),
                'inertia': float(kmeans.inertia_),
                'metadata': metadata or {},
                'created_at': datetime.now(),
                'results': data.to_dict('records')
            }

            # Firestoreに保存
            doc_ids = await self.firestore_service.save_results(
                results=[analysis_result],
                collection_name=self.collection_name
            )

            # 分析IDを結果に追加
            data['analysis_id'] = doc_ids[0]

            logger.info(f"Successfully saved cluster analysis results with ID: {doc_ids[0]}")
            return data

        except Exception as e:
            logger.error(f"Error in cluster analysis: {str(e)}")
            raise

    async def get_analysis_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        過去の分析結果を取得

        Args:
            user_id: 特定ユーザーの結果のみを取得する場合に指定
            limit: 取得する結果の最大数
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

            logger.info(f"Retrieved {len(results)} analysis results")
            return results

        except Exception as e:
            logger.error(f"Error retrieving analysis history: {str(e)}")
            raise

    async def update_analysis_metadata(
        self,
        analysis_id: str,
        metadata: Dict
    ) -> None:
        """
        分析結果のメタデータを更新
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

    async def delete_analysis(self, analysis_id: str) -> None:
        """
        分析結果を削除
        """
        try:
            await self.firestore_service.delete_document(
                collection_name=self.collection_name,
                document_id=analysis_id
            )
            logger.info(f"Deleted analysis {analysis_id}")

        except Exception as e:
            logger.error(f"Error deleting analysis: {str(e)}")
            raise