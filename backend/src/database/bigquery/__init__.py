"""
BigQueryデータベースモジュール
データの分析、集計、レポート生成のための機能を提供します。
"""

from typing import Optional, Dict, Any, List
from google.cloud import bigquery
from google.cloud.bigquery import Client, QueryJobConfig, LoadJobConfig
from google.cloud.bigquery.table import RowIterator
import os
import logging
from datetime import datetime

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BigQueryClient:
    """BigQueryクライアントクラス"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BigQueryClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """シングルトンパターンでBigQueryクライアントを初期化"""
        if self._initialized:
            return

        try:
            # 認証情報の設定
            if os.getenv("ENVIRONMENT") == "development":
                credentials_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    'credentials',
                    'bigquery-credentials.json'
                )
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

            self.client = bigquery.Client()
            self.project = self.client.project
            self.dataset_id = os.getenv("BIGQUERY_DATASET_ID", "analysis_results")
            self._initialized = True
            logger.info("BigQuery client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {str(e)}")
            raise

    @property
    def dataset_ref(self):
        """データセットの参照を取得"""
        return self.client.dataset(self.dataset_id)

    def get_table_ref(self, table_name: str):
        """テーブルの参照を取得"""
        return self.dataset_ref.table(table_name)

def get_bigquery_client() -> BigQueryClient:
    """BigQueryクライアントのインスタンスを取得"""
    return BigQueryClient()