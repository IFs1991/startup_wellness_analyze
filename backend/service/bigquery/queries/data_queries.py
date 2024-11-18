from typing import List, Optional, Dict
from datetime import datetime

class DataQueries:
    """データ取得用のクエリビルダー"""

    @staticmethod
    def build_analysis_query(
        table_name: str,
        columns: List[str],
        conditions: Optional[str] = None
    ) -> str:
        """分析用のクエリを構築"""
        query = f"""
            SELECT {', '.join(columns)}
            FROM `{table_name}`
            WHERE 1=1
        """

        if conditions:
            query += f"\nAND {conditions}"

        return query

    @staticmethod
    def build_aggregation_query(
        table_name: str,
        group_by_columns: List[str],
        agg_functions: Dict[str, str],
        conditions: Optional[str] = None
    ) -> str:
        """集計クエリを構築"""
        agg_cols = [
            f"{func}({col}) as {col}_{func}"
            for col, func in agg_functions.items()
        ]

        query = f"""
            SELECT
                {', '.join(group_by_columns)},
                {', '.join(agg_cols)}
            FROM `{table_name}`
            WHERE 1=1
        """

        if conditions:
            query += f"\nAND {conditions}"

        query += f"\nGROUP BY {', '.join(group_by_columns)}"

        return query