from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class AggregationConfig:
    """集計設定を表すデータクラス"""
    table_name: str
    group_by_columns: List[str]
    agg_functions: Dict[str, str]
    conditions: Optional[str] = None

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