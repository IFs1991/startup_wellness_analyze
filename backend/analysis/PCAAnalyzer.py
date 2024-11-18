# -*- coding: utf-8 -*-

"""
主成分分析

多数のVAS項目から、従業員の健康状態を代表する少数の主成分を抽出します。
BigQueryServiceを利用した非同期処理に対応しています。
"""

from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from service.bigquery.client import BigQueryService
from service.firestore.client import FirestoreService
class PCAAnalyzer:
    """
    BigQueryServiceを利用して主成分分析を行うクラスです。
    """

    def __init__(self, bq_service: BigQueryService):
        """
        初期化メソッド

        Args:
            bq_service (BigQueryService): BigQueryとの接続を管理するサービス
        """
        self.bq_service = bq_service

    def _validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        分析対象データのバリデーションを行います。

        Args:
            data (pd.DataFrame): 検証対象のデータフレーム

        Returns:
            Tuple[bool, Optional[str]]: (バリデーション結果, エラーメッセージ)
        """
        if data.empty:
            return False, "データが空です"

        # 数値型のカラムのみを抽出
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return False, "数値型のカラムが存在しません"

        # 欠損値のチェック
        if data[numeric_cols].isnull().any().any():
            return False, "数値データに欠損値が含まれています"

        return True, None

    async def analyze(
        self,
        query: str,
        n_components: int,
        save_results: bool = True,
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        主成分分析を実行します。

        Args:
            query (str): 分析対象データを取得するBigQueryクエリ
            n_components (int): 抽出する主成分の数
            save_results (bool): 結果を保存するかどうか
            dataset_id (Optional[str]): 保存先データセットID
            table_id (Optional[str]): 保存先テーブルID

        Returns:
            Tuple[pd.DataFrame, Dict]: (主成分スコアが付与されたデータ, メタデータ)

        Raises:
            ValueError: パラメータが不正な場合
            RuntimeError: 分析実行中にエラーが発生した場合
        """
        try:
            # パラメータの検証
            if n_components < 1:
                raise ValueError("主成分の数は1以上である必要があります")

            # データ取得
            data = await self.bq_service.fetch_data(query)

            # データのバリデーション
            is_valid, error_message = self._validate_data(data)
            if not is_valid:
                raise ValueError(error_message)

            # 数値型のカラムのみを抽出
            numeric_data = data.select_dtypes(include=[np.number])

            # 主成分分析の実行
            pca = PCA(n_components=min(n_components, len(numeric_data.columns)))
            principal_components = pca.fit_transform(numeric_data)

            # 結果のデータフレーム作成
            principal_df = pd.DataFrame(
                data=principal_components,
                columns=[f'principal_component_{i+1}' for i in range(pca.n_components_)]
            )

            # 元データと主成分スコアの結合
            results = pd.concat([data, principal_df], axis=1)

            # メタデータの作成
            metadata = {
                'row_count': len(data),
                'original_columns': list(numeric_data.columns),
                'n_components': pca.n_components_,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist()
            }

            # 結果の保存
            if save_results and dataset_id and table_id:
                await self.bq_service.save_results(
                    results,
                    dataset_id=dataset_id,
                    table_id=table_id
                )

            return results, metadata

        except Exception as e:
            raise RuntimeError(f"主成分分析の実行中にエラーが発生しました: {str(e)}")

async def analyze_pca(request: Any) -> Tuple[Dict, int]:
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

        # 必須パラメータの検証
        required_params = ['query', 'n_components']
        for param in required_params:
            if param not in request_json:
                return {'error': f'必須パラメータ {param} が指定されていません'}, 400

        # パラメータの取得
        query = request_json['query']
        n_components = int(request_json['n_components'])
        dataset_id = request_json.get('dataset_id')
        table_id = request_json.get('table_id')

        # サービスの初期化とPCA実行
        bq_service = BigQueryService()
        analyzer = PCAAnalyzer(bq_service)

        results, metadata = await analyzer.analyze(
            query=query,
            n_components=n_components,
            save_results=bool(dataset_id and table_id),
            dataset_id=dataset_id,
            table_id=table_id
        )

        return {
            'status': 'success',
            'results': results.to_dict('records'),
            'metadata': metadata
        }, 200

    except ValueError as e:
        return {
            'status': 'error',
            'message': f'パラメータエラー: {str(e)}'
        }, 400

    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }, 500