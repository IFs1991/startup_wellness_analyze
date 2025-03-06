from typing import List, Optional, Dict, Any, Union, Callable
from datetime import datetime
from dataclasses import dataclass
import logging
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from .base import BaseAnalyzer, AnalysisError
from src.database.firestore.client import get_firestore_client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class AnalysisConfig:
    """分析設定を表すデータクラス"""
    collection_name: str
    target_fields: List[str]
    filters: Optional[List[tuple]] = None
    order_by: Optional[tuple] = None
    limit: Optional[int] = None

class AssociationAnalyzer(BaseAnalyzer):
    """アソシエーション分析を行うクラス"""

    def __init__(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.5,
        min_lift: float = 1.0,
        firestore_client=None
    ):
        """
        初期化メソッド

        Args:
            min_support (float): 最小サポート値
            min_confidence (float): 最小確信度
            min_lift (float): 最小リフト値
            firestore_client: Firestoreクライアントのインスタンス（テスト用）
        """
        super().__init__("association_analysis", firestore_client)
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        データのバリデーション

        Args:
            data (pd.DataFrame): 検証対象のデータ

        Returns:
            bool: バリデーション結果
        """
        if data.empty:
            raise AnalysisError("データが空です")

        if not all(data.dtypes == bool):
            raise AnalysisError("すべての列がブール型である必要があります")

        return True

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        データの前処理

        Args:
            data (pd.DataFrame): 前処理対象のデータ

        Returns:
            pd.DataFrame: 前処理済みデータ
        """
        # 数値データをブール型に変換
        binary_data = data.astype(bool)
        return binary_data

    async def get_analysis_data(self, config: AnalysisConfig) -> pd.DataFrame:
        """
        Firestoreからデータを取得してDataFrameに変換

        Args:
            config (AnalysisConfig): 分析設定

        Returns:
            pd.DataFrame: 分析用データフレーム
        """
        try:
            # Firestoreからデータを取得
            docs = await self.firestore_client.query_documents(
                collection=config.collection_name,
                filters=config.filters,
                order_by=config.order_by,
                limit=config.limit
            )

            # DataFrameに変換
            if not docs:
                return pd.DataFrame()

            # 指定されたフィールドのみを抽出
            data = [{field: doc.get(field) for field in config.target_fields} for doc in docs]
            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Error fetching data from Firestore: {str(e)}")
            raise

    async def analyze_and_store(self, config: AnalysisConfig, target_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        分析を実行し、結果をFirestoreに保存

        Args:
            config (AnalysisConfig): 分析設定
            target_columns (Optional[List[str]], optional): 分析対象の列名リスト

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            # データ取得
            data = await self.get_analysis_data(config)
            if data.empty:
                raise ValueError("No data available for analysis")

            # 分析実行
            result = await self.analyze(data, target_columns=target_columns)

            # 結果をFirestoreに保存
            analysis_doc = {
                "type": "association_analysis",
                "config": {
                    "min_support": self.min_support,
                    "min_confidence": self.min_confidence,
                    "min_lift": self.min_lift,
                    "collection_analyzed": config.collection_name,
                    "fields_analyzed": config.target_fields,
                    "target_columns": target_columns
                },
                "results": result,
                "created_at": datetime.utcnow(),
            }

            # 分析結果の保存
            await self.firestore_client.create_document(
                collection="analytics",
                doc_id=f"assoc_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                data=analysis_doc
            )

            return result

        except Exception as e:
            logger.error(f"Error in association analysis and storage: {str(e)}")
            raise

    async def analyze(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        アソシエーション分析を実行

        Args:
            data (pd.DataFrame): 分析対象データ
            target_columns (Optional[List[str]]): 分析対象の列名リスト
            **kwargs: 追加のパラメータ

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            # 対象列の選択
            if target_columns:
                data = data[target_columns]

            # データの前処理
            binary_data = self._prepare_data(data)

            # 頻出アイテムセットの抽出
            frequent_itemsets = apriori(
                binary_data,
                min_support=self.min_support,
                use_colnames=True
            )

            # アソシエーションルールの生成
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence
            )

            # リフトによるフィルタリング
            filtered_rules = rules[rules['lift'] >= self.min_lift]

            # 結果の整形
            result = {
                'frequent_itemsets': frequent_itemsets.to_dict('records'),
                'rules': filtered_rules.to_dict('records'),
                'summary': {
                    'total_itemsets': len(frequent_itemsets),
                    'total_rules': len(filtered_rules),
                    'parameters': {
                        'min_support': self.min_support,
                        'min_confidence': self.min_confidence,
                        'min_lift': self.min_lift
                    }
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in association analysis: {str(e)}")
            raise AnalysisError(f"アソシエーション分析に失敗しました: {str(e)}")

async def analyze_associations(
    collection: str,
    target_columns: List[str],
    min_support: float = 0.1,
    min_confidence: float = 0.5,
    min_lift: float = 1.0,
    filters: Optional[List[Dict[str, Any]]] = None,
    firestore_client=None
) -> Dict[str, Any]:
    """
    アソシエーション分析を実行するヘルパー関数

    Args:
        collection (str): 分析対象のコレクション名
        target_columns (List[str]): 分析対象の列名リスト
        min_support (float): 最小サポート値
        min_confidence (float): 最小確信度
        min_lift (float): 最小リフト値
        filters (Optional[List[Dict[str, Any]]]): データ取得時のフィルター条件
        firestore_client: Firestoreクライアントのインスタンス（テスト用）

    Returns:
        Dict[str, Any]: 分析結果
    """
    try:
        analyzer = AssociationAnalyzer(
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            firestore_client=firestore_client
        )

        # AnalysisConfigオブジェクトの作成
        config = AnalysisConfig(
            collection_name=collection,
            target_fields=target_columns,
            filters=filters
        )

        # 分析の実行と結果の保存
        results = await analyzer.analyze_and_store(
            config=config,
            target_columns=target_columns
        )

        return results

    except Exception as e:
        raise AnalysisError(f"アソシエーション分析の実行に失敗しました: {str(e)}")