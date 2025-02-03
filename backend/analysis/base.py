from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import pandas as pd
from backend.src.database.firestore.client import get_firestore_client

class AnalysisError(Exception):
    """分析処理に関するエラー"""
    pass

class BaseAnalyzer:
    """分析モジュールの基底クラス"""

    def __init__(self, analysis_type: str, firestore_client=None):
        """
        初期化メソッド

        Args:
            analysis_type (str): 分析タイプ（例: 'correlation', 'pca', 'clustering'）
            firestore_client: Firestoreクライアントのインスタンス（テスト用）
        """
        self.analysis_type = analysis_type
        self.firestore_client = firestore_client if firestore_client is not None else get_firestore_client()
        self.logger = logging.getLogger(f"{__name__}.{analysis_type}")

    async def fetch_data(
        self,
        collection: str,
        filters: Optional[List[Dict[str, Any]]] = None,
        order_by: Optional[tuple] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Firestoreからデータを取得してDataFrameに変換

        Args:
            collection (str): コレクション名
            filters (Optional[List[Dict[str, Any]]]): フィルター条件
            order_by (Optional[tuple]): ソート条件
            limit (Optional[int]): 取得件数

        Returns:
            pd.DataFrame: 取得したデータ
        """
        try:
            docs = await self.firestore_client.query_documents(
                collection=collection,
                filters=filters,
                order_by=order_by,
                limit=limit
            )
            return pd.DataFrame(docs)
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            raise AnalysisError(f"データの取得に失敗しました: {str(e)}")

    async def save_results(
        self,
        results: Dict[str, Any],
        collection: str = 'analysis_results'
    ) -> str:
        """
        分析結果を保存

        Args:
            results (Dict[str, Any]): 分析結果
            collection (str): 保存先コレクション名

        Returns:
            str: 保存したドキュメントのID
        """
        try:
            # メタデータを追加
            results.update({
                'analysis_type': self.analysis_type,
                'created_at': datetime.utcnow(),
                'status': 'completed'
            })

            # ドキュメントIDを生成して保存
            doc_id = f"{self.analysis_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            await self.firestore_client.create_document(collection, doc_id, results)
            return doc_id

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise AnalysisError(f"結果の保存に失敗しました: {str(e)}")

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        データのバリデーション

        Args:
            data (pd.DataFrame): 検証対象のデータ

        Returns:
            bool: バリデーション結果
        """
        if data.empty:
            raise AnalysisError("データが空です")
        return True

    async def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        分析を実行（サブクラスで実装）

        Args:
            data (pd.DataFrame): 分析対象データ
            **kwargs: 追加のパラメータ

        Returns:
            Dict[str, Any]: 分析結果
        """
        raise NotImplementedError("このメソッドはサブクラスで実装する必要があります")

    async def analyze_and_save(
        self,
        data: pd.DataFrame,
        save_results: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        分析を実行して結果を保存

        Args:
            data (pd.DataFrame): 分析対象データ
            save_results (bool): 結果を保存するかどうか
            **kwargs: 追加のパラメータ

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            # データのバリデーション
            self.validate_data(data)

            # 分析の実行
            results = await self.analyze(data, **kwargs)

            # 結果の保存
            if save_results:
                doc_id = await self.save_results(results)
                results['document_id'] = doc_id

            return results

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise AnalysisError(f"分析に失敗しました: {str(e)}")