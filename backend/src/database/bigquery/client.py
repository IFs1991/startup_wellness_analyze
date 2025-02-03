"""
BigQueryクライアントの実装
クエリの実行、データのロード、テーブル操作などの機能を提供します。
"""

from typing import Optional, Dict, Any, List, Union
from google.cloud import bigquery
from google.cloud.bigquery import QueryJobConfig, LoadJobConfig
from google.cloud.bigquery.table import RowIterator, Table
from google.cloud.bigquery.job import QueryJob
import pandas as pd
import logging
from datetime import datetime
from . import BigQueryClient

logger = logging.getLogger(__name__)

class BigQueryService:
    """BigQueryサービスクラス"""

    def __init__(self, client: Optional[BigQueryClient] = None):
        """
        Args:
            client (Optional[BigQueryClient]): BigQueryクライアント
        """
        self.client = client or BigQueryClient()

    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> RowIterator:
        """クエリを実行する

        Args:
            query (str): 実行するクエリ
            params (Optional[Dict[str, Any]]): クエリパラメータ
            timeout (Optional[float]): タイムアウト時間（秒）

        Returns:
            RowIterator: クエリ結果
        """
        try:
            job_config = QueryJobConfig()
            if params:
                job_config.query_parameters = [
                    bigquery.ScalarQueryParameter(k, 'STRING', v)
                    for k, v in params.items()
                ]

            query_job = self.client.client.query(
                query,
                job_config=job_config,
                timeout=timeout
            )
            return query_job.result()

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    async def load_data_from_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: Optional[List[Dict[str, str]]] = None,
        write_disposition: str = 'WRITE_TRUNCATE'
    ) -> None:
        """DataFrameからデータをロードする

        Args:
            df (pd.DataFrame): ロードするデータ
            table_name (str): テーブル名
            schema (Optional[List[Dict[str, str]]]): テーブルスキーマ
            write_disposition (str): 書き込みモード
        """
        try:
            job_config = LoadJobConfig()
            if schema:
                job_config.schema = [
                    bigquery.SchemaField(
                        field['name'],
                        field['type'],
                        mode=field.get('mode', 'NULLABLE')
                    )
                    for field in schema
                ]
            job_config.write_disposition = write_disposition

            table_ref = self.client.get_table_ref(table_name)
            job = self.client.client.load_table_from_dataframe(
                df,
                table_ref,
                job_config=job_config
            )
            job.result()  # 完了を待機

            logger.info(f"Loaded {len(df)} rows into {table_name}")

        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    async def create_table(
        self,
        table_name: str,
        schema: List[Dict[str, str]],
        partition_field: Optional[str] = None,
        cluster_fields: Optional[List[str]] = None
    ) -> Table:
        """テーブルを作成する

        Args:
            table_name (str): テーブル名
            schema (List[Dict[str, str]]): テーブルスキーマ
            partition_field (Optional[str]): パーティションフィールド
            cluster_fields (Optional[List[str]]): クラスタリングフィールド

        Returns:
            Table: 作成されたテーブル
        """
        try:
            table_ref = self.client.get_table_ref(table_name)
            schema_fields = [
                bigquery.SchemaField(
                    field['name'],
                    field['type'],
                    mode=field.get('mode', 'NULLABLE')
                )
                for field in schema
            ]

            table = bigquery.Table(table_ref, schema=schema_fields)

            if partition_field:
                table.time_partitioning = bigquery.TimePartitioning(
                    field=partition_field
                )

            if cluster_fields:
                table.clustering_fields = cluster_fields

            return self.client.client.create_table(table)

        except Exception as e:
            logger.error(f"Table creation failed: {str(e)}")
            raise

    async def delete_table(self, table_name: str) -> None:
        """テーブルを削除する

        Args:
            table_name (str): テーブル名
        """
        try:
            table_ref = self.client.get_table_ref(table_name)
            self.client.client.delete_table(table_ref)
            logger.info(f"Table {table_name} deleted successfully")

        except Exception as e:
            logger.error(f"Table deletion failed: {str(e)}")
            raise