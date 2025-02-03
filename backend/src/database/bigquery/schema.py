"""
BigQueryテーブルスキーマの定義
分析結果やレポートデータのスキーマを定義します。
"""

from typing import Dict, List

# 分析結果テーブルのスキーマ
ANALYSIS_RESULTS_SCHEMA: List[Dict[str, str]] = [
    {"name": "analysis_id", "type": "STRING", "mode": "REQUIRED"},
    {"name": "company_id", "type": "STRING", "mode": "REQUIRED"},
    {"name": "analysis_type", "type": "STRING", "mode": "REQUIRED"},
    {"name": "result_data", "type": "JSON", "mode": "REQUIRED"},
    {"name": "created_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
    {"name": "updated_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
    {"name": "status", "type": "STRING", "mode": "REQUIRED"},
    {"name": "metadata", "type": "JSON", "mode": "NULLABLE"}
]

# レポートデータテーブルのスキーマ
REPORT_DATA_SCHEMA: List[Dict[str, str]] = [
    {"name": "report_id", "type": "STRING", "mode": "REQUIRED"},
    {"name": "company_id", "type": "STRING", "mode": "REQUIRED"},
    {"name": "report_type", "type": "STRING", "mode": "REQUIRED"},
    {"name": "report_data", "type": "JSON", "mode": "REQUIRED"},
    {"name": "generated_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
    {"name": "status", "type": "STRING", "mode": "REQUIRED"},
    {"name": "metadata", "type": "JSON", "mode": "NULLABLE"}
]

# 時系列データテーブルのスキーマ
TIME_SERIES_DATA_SCHEMA: List[Dict[str, str]] = [
    {"name": "series_id", "type": "STRING", "mode": "REQUIRED"},
    {"name": "company_id", "type": "STRING", "mode": "REQUIRED"},
    {"name": "metric_name", "type": "STRING", "mode": "REQUIRED"},
    {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
    {"name": "value", "type": "FLOAT", "mode": "REQUIRED"},
    {"name": "metadata", "type": "JSON", "mode": "NULLABLE"}
]