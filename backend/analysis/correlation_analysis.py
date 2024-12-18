# -*- coding: utf-8 -*-

"""
相関分析

VAS データと損益計算書データの相関関係を分析します。
BigQueryService を利用した非同期処理に対応しています。
"""

from typing import List, Optional, Tuple, Any, Dict
import pandas as pd
from service.bigquery.client import BigQueryService
from service.firestore.client import FirestoreService

class CorrelationAnalyzer:
    def __init__(self, bq_service: BigQueryService):
        """
        コンストラクタ

        Args:
            bq_service (BigQueryService): BigQueryサービスのインスタンス
        """
        self.bq_service = bq_service

    def _validate_data(self,
                      data: pd.DataFrame,
                      vas_variables: List[str],
                      financial_variables: List[str]) -> Tuple[bool, Optional[str]]:
        """
        データのバリデーションを行う

        Args:
            data (pd.DataFrame): 検証対象のデータ
            vas_variables (List[str]): VASデータの変数名リスト
            financial_variables (List[str]): 損益計算書データの変数名リスト

        Returns:
            Tuple[bool, Optional[str]]: (バリデーション結果, エラーメッセージ)
        """
        if data.empty:
            return False, "データが空です"

        # 必要なカラムの存在確認
        required_columns = vas_variables + financial_variables
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"以下のカラムが見つかりません: {', '.join(missing_columns)}"

        # データ型の確認（数値型であることを確認）
        non_numeric_columns = [col for col in required_columns
                             if not pd.api.types.is_numeric_dtype(data[col])]
        if non_numeric_columns:
            return False, f"以下のカラムが数値型ではありません: {', '.join(non_numeric_columns)}"

        return True, None

    async def analyze(self,
                     query: str,
                     vas_variables: List[str],
                     financial_variables: List[str],
                     save_results: bool = True,
                     dataset_id: Optional[str] = None,
                     table_id: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        VASデータと損益計算書データの相関関係を分析する

        Args:
            query (str): BigQueryクエリ
            vas_variables (List[str]): VASデータの変数名リスト
            financial_variables (List[str]): 損益計算書データの変数名リスト
            save_results (bool): 結果を保存するかどうか
            dataset_id (Optional[str]): 保存先データセットID
            table_id (Optional[str]): 保存先テーブルID

        Returns:
            Tuple[pd.DataFrame, Dict]: (相関行列, メタデータ)

        Raises:
            ValueError: データのバリデーションエラー
            RuntimeError: 分析実行時のエラー
        """
        try:
            # データ取得
            data = await self.bq_service.fetch_data(query)

            # データバリデーション
            is_valid, error_message = self._validate_data(data, vas_variables, financial_variables)
            if not is_valid:
                raise ValueError(error_message)

            # 相関分析の実行
            selected_data = data[vas_variables + financial_variables]
            correlation_matrix = selected_data.corr()

            # 結果の保存
            if save_results and dataset_id and table_id:
                await self.bq_service.save_results(
                    correlation_matrix,
                    dataset_id=dataset_id,
                    table_id=table_id
                )

            # メタデータの作成
            metadata = {
                'row_count': len(data),
                'vas_variables': vas_variables,
                'financial_variables': financial_variables,
                'correlation_pairs': len(vas_variables) * len(financial_variables)
            }

            return correlation_matrix, metadata

        except Exception as e:
            raise RuntimeError(f"相関分析の実行中にエラーが発生しました: {str(e)}")


async def analyze_correlation(request: Any) -> Tuple[Dict, int]:
    """
    Cloud Functions用のエントリーポイント関数

    Args:
        request: Cloud Functionsのリクエストオブジェクト

    Returns:
        Tuple[Dict, int]: (レスポンス, ステータスコード)
    """
    try:
        request_json = request.get_json()

        if not request_json:
            return {'error': 'リクエストデータがありません'}, 400

        # 必須パラメータの確認
        required_params = ['query', 'vas_variables', 'financial_variables']
        missing_params = [param for param in required_params if param not in request_json]
        if missing_params:
            return {
                'error': f'必須パラメータが不足しています: {", ".join(missing_params)}'
            }, 400

        # パラメータの取得
        query = request_json['query']
        vas_variables = request_json['vas_variables']
        financial_variables = request_json['financial_variables']
        dataset_id = request_json.get('dataset_id')
        table_id = request_json.get('table_id')

        # サービスの初期化
        bq_service = BigQueryService()
        analyzer = CorrelationAnalyzer(bq_service)

        # 分析実行
        correlation_matrix, metadata = await analyzer.analyze(
            query=query,
            vas_variables=vas_variables,
            financial_variables=financial_variables,
            save_results=bool(dataset_id and table_id),
            dataset_id=dataset_id,
            table_id=table_id
        )

        return {
            'status': 'success',
            'results': correlation_matrix.to_dict('records'),
            'metadata': metadata
        }, 200

    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }, 500