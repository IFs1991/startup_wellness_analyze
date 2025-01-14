from pathlib import Path
import os
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
from google.cloud import bigquery
from google.api_core import retry
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BigQueryService:
    """BigQuery操作用のサービスクラス"""

    def __init__(
        self,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None
    ):
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        self.client = bigquery.Client(project=project_id)
        self.logger = logging.getLogger(__name__)
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def fetch_data(self, query: str) -> pd.DataFrame:
        """クエリを非同期で実行しデータを取得"""
        try:
            self.logger.debug(f"Executing query: {query}")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                self._execute_query_with_retry,
                query
            )
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            raise

    @retry.Retry(predicate=retry.if_transient_error)
    def _execute_query_with_retry(self, query: str) -> pd.DataFrame:
        """リトライ機能付きでクエリを実行"""
        query_job = self.client.query(query)
        return query_job.result().to_dataframe()

    async def save_results(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        table_id: str,
        if_exists: str = 'replace'
    ) -> None:
        """分析結果を非同期で保存"""
        try:
            self.logger.debug(f"Saving results to {dataset_id}.{table_id}")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self._save_data_with_retry,
                df,
                dataset_id,
                table_id,
                if_exists
            )
        except Exception as e:
            self.logger.error(f"Data save failed: {str(e)}")
            raise

    @retry.Retry(predicate=retry.if_transient_error)
    def _save_data_with_retry(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        table_id: str,
        if_exists: str
    ) -> None:
        """リトライ機能付きでデータを保存"""
        job_config = bigquery.LoadJobConfig()
        if if_exists == 'replace':
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        else:
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND

        table_ref = f"{dataset_id}.{table_id}"
        job = self.client.load_table_from_dataframe(
            df,
            table_ref,
            job_config=job_config
        )
        job.result()

    def __del__(self):
        """クリーンアップ処理"""
        self._executor.shutdown(wait=True)