# -*- coding: utf-8 -*-
"""
Cluster Analysis with BigQueryService Integration

This code implements data analysis functionality with BigQueryService integration,
supporting asynchronous processing and Cloud Functions deployment.
"""

from typing import Optional, Tuple, Any, Dict
import pandas as pd
from service.bigquery.client import BigQueryService
from service.firestore.client import FirestoreService
class DataAnalyzer:
    def __init__(self, bq_service: BigQueryService):
        """
        Initialize the DataAnalyzer with BigQueryService.

        Args:
            bq_service: Initialized BigQueryService instance
        """
        self.bq_service = bq_service

    def _validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validate the input data.

        Args:
            data: Input DataFrame to validate

        Returns:
            Tuple containing validation result and error message if any
        """
        if data.empty:
            return False, "データが空です"

        # 必要に応じて追加のバリデーションルールをここに実装
        return True, None

    async def analyze(self,
                     query: str,
                     save_results: bool = True,
                     dataset_id: Optional[str] = None,
                     table_id: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform data analysis using BigQuery.

        Args:
            query: BigQuery query to fetch data
            save_results: Flag to control result saving
            dataset_id: Target dataset ID for saving results
            table_id: Target table ID for saving results

        Returns:
            Tuple containing analysis results DataFrame and metadata dictionary

        Raises:
            ValueError: If input data validation fails
            RuntimeError: If analysis execution fails
        """
        try:
            # BigQueryからデータ取得
            data = await self.bq_service.fetch_data(query)

            # データバリデーション
            is_valid, error_message = self._validate_data(data)
            if not is_valid:
                raise ValueError(error_message)

            # 基本的な統計分析の実行
            results = data.describe()

            # オプションで追加の分析を実装可能
            # 例: クラスタリング、回帰分析など

            # 分析結果の保存
            if save_results and dataset_id and table_id:
                await self.bq_service.save_results(
                    results,
                    dataset_id=dataset_id,
                    table_id=table_id
                )

            # メタデータの作成
            metadata = {
                'row_count': len(data),
                'columns_analyzed': list(data.columns),
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'query_executed': query
            }

            return results, metadata

        except Exception as e:
            raise RuntimeError(f"分析実行中にエラーが発生しました: {str(e)}")

async def analyze_data(request: Any) -> Tuple[Dict, int]:
    """
    Cloud Functions entry point for data analysis.

    Args:
        request: Cloud Functions request object containing analysis parameters

    Returns:
        Tuple containing response dictionary and HTTP status code

    Response format:
        Success (200):
        {
            'status': 'success',
            'results': [Analysis results as records],
            'metadata': {Analysis metadata}
        }

        Error (400/500):
        {
            'status': 'error',
            'message': 'Error description'
        }
    """
    try:
        # リクエストの検証
        request_json = request.get_json()
        if not request_json:
            return {'status': 'error', 'message': 'リクエストデータがありません'}, 400

        # 必須パラメータの検証
        required_params = ['query']
        missing_params = [param for param in required_params if param not in request_json]
        if missing_params:
            return {
                'status': 'error',
                'message': f'必須パラメータが不足しています: {", ".join(missing_params)}'
            }, 400

        # サービスの初期化
        bq_service = BigQueryService()
        analyzer = DataAnalyzer(bq_service)

        # パラメータの取得
        query = request_json['query']
        dataset_id = request_json.get('dataset_id')
        table_id = request_json.get('table_id')
        save_results = request_json.get('save_results', True)

        # 分析の実行
        results, metadata = await analyzer.analyze(
            query=query,
            save_results=save_results,
            dataset_id=dataset_id,
            table_id=table_id
        )

        # レスポンスの作成
        return {
            'status': 'success',
            'results': results.to_dict('records'),
            'metadata': metadata
        }, 200

    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }, 500