from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from dataclasses import dataclass
import logging
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class AggregationConfig:
    """集計設定を表すデータクラス"""
    table_name: str
    group_by_columns: List[str]
    agg_functions: Dict[str, str]
    conditions: Optional[str] = None

class AssociationAnalyzer:
    """
    アソシエーション分析を行うクラス
    """
    def __init__(self):
        """
        コンストラクタ
        """
        logger.info("AssociationAnalyzerが初期化されました")

    def analyze(self,
                data: pd.DataFrame,
                min_support: float = 0.01,
                min_confidence: float = 0.5,
                min_lift: float = 1.0) -> Dict[str, Any]:
        """
        アソシエーション分析を実行します

        Args:
            data (pd.DataFrame): 分析対象データ
            min_support (float): 最小サポート値
            min_confidence (float): 最小確信度
            min_lift (float): 最小リフト値

        Returns:
            Dict[str, Any]: 分析結果

        Raises:
            ValueError: パラメータが無効な場合
        """
        if data.empty:
            raise ValueError("入力データが空です")

        try:
            # アソシエーション分析の実行
            logger.info(f"アソシエーション分析を実行します: サンプル数={len(data)}")

            # 頻出アイテムセットの抽出
            frequent_itemsets = apriori(
                data,
                min_support=min_support,
                use_colnames=True
            )

            # アソシエーションルールの生成
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=min_confidence
            )

            # リフトでフィルタリング
            rules = rules[rules['lift'] >= min_lift]

            # 結果を整形
            result = {
                'frequent_itemsets': frequent_itemsets.to_dict('records'),
                'rules': rules.to_dict('records'),
                'stats': {
                    'itemset_count': len(frequent_itemsets),
                    'rule_count': len(rules),
                    'min_support': min_support,
                    'min_confidence': min_confidence,
                    'min_lift': min_lift
                }
            }

            logger.info(f"アソシエーション分析が完了しました: ルール数={len(rules)}")

            return result

        except Exception as e:
            logger.error(f"アソシエーション分析でエラーが発生しました: {str(e)}")
            raise RuntimeError(f"アソシエーション分析の実行に失敗しました: {str(e)}")

class DataQueries:
    """データ取得用のクエリビルダー"""

    @staticmethod
    def build_analysis_query(
        table_name: str,
        columns: List[str],
        conditions: Optional[str] = None
    ) -> str:
        """
        分析用のクエリを構築します

        Args:
            table_name (str): テーブル名
            columns (List[str]): 取得するカラム
            conditions (Optional[str]): WHERE句の条件

        Returns:
            str: 構築されたクエリ

        Raises:
            ValueError: パラメータが無効な場合
        """
        if not table_name:
            raise ValueError("Table name must not be empty")
        if not columns:
            raise ValueError("Columns list must not be empty")

        safe_columns = [DataQueries._sanitize_identifier(col) for col in columns]
        safe_table = DataQueries._sanitize_identifier(table_name)

        query = f"""
            SELECT {', '.join(safe_columns)}
            FROM {safe_table}
            WHERE 1=1
        """
        if conditions:
            query += f"\nAND {conditions}"
        return query

    @staticmethod
    def build_aggregation_query(
        config: AggregationConfig
    ) -> str:
        """
        集計クエリを構築します

        Args:
            config (AggregationConfig): 集計設定

        Returns:
            str: 構築されたクエリ

        Raises:
            ValueError: パラメータが無効な場合
        """
        if not config.table_name:
            raise ValueError("Table name must not be empty")
        if not config.group_by_columns:
            raise ValueError("Group by columns must not be empty")
        if not config.agg_functions:
            raise ValueError("Aggregation functions must not be empty")

        safe_table = DataQueries._sanitize_identifier(config.table_name)
        safe_group_by = [
            DataQueries._sanitize_identifier(col)
            for col in config.group_by_columns
        ]

        agg_cols = [
            f"{func}({DataQueries._sanitize_identifier(col)}) as {DataQueries._sanitize_identifier(f'{col}_{func}')}"
            for col, func in config.agg_functions.items()
        ]

        query = f"""
            SELECT
                {', '.join(safe_group_by)},
                {', '.join(agg_cols)}
            FROM {safe_table}
            WHERE 1=1
        """

        if config.conditions:
            query += f"\nAND {config.conditions}"

        query += f"\nGROUP BY {', '.join(safe_group_by)}"
        return query

    @staticmethod
    def _sanitize_identifier(identifier: str) -> str:
        """
        SQL識別子をサニタイズします

        Args:
            identifier (str): サニタイズする識別子

        Returns:
            str: サニタイズされた識別子
        """
        # 基本的なサニタイズ
        sanitized = identifier.replace('"', '').replace('`', '')
        return f"`{sanitized}`"