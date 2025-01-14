from typing import List, Optional
from dataclasses import dataclass
from google.cloud import bigquery

@dataclass
class AnalysisTableSchema:
    """分析結果テーブルのスキーマ定義"""
    name: str
    fields: List[bigquery.SchemaField]
    partition_field: Optional[str] = None

# 分析結果用のスキーマ
ANALYSIS_RESULTS_SCHEMA = AnalysisTableSchema(
    name="analysis_results",
    fields=[
        bigquery.SchemaField("analysis_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("results", "JSON", mode="REQUIRED"),
        bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
    ],
    partition_field="timestamp"
)