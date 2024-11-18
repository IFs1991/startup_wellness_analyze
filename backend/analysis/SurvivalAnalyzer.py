# -*- coding: utf-8 -*-

"""
生存時間分析

Startup Wellness プログラム導入前後における、従業員の離職までの時間を比較分析します。
BigQueryServiceを利用した非同期処理に対応しています。
"""

from typing import Optional, Tuple, Dict, Any
import pandas as pd
from lifelines import KaplanMeierFitter
from service.bigquery.client import BigQueryService
from service.firestore.client import FirestoreService
class SurvivalAnalyzer:
    """
    生存時間分析を行うクラスです。
    BigQueryServiceを利用して、データの取得と保存を行います。
    """

    def __init__(self, bq_service: BigQueryService):
        """
        Args:
            bq_service (BigQueryService): BigQueryServiceのインスタンス
        """
        self.bq_service = bq_service

    def _validate_data(self,
                      data: pd.DataFrame,
                      duration_col: str,
                      event_col: str) -> Tuple[bool, Optional[str]]:
        """
        分析対象データのバリデーションを行います。

        Args:
            data (pd.DataFrame): 検証対象のデータフレーム
            duration_col (str): イベント発生までの時間を表すカラム名
            event_col (str): イベント発生を表すカラム名

        Returns:
            Tuple[bool, Optional[str]]: (検証結果, エラーメッセージ)
        """
        if data.empty:
            return False, "データが空です"

        required_cols = [duration_col, event_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return False, f"必要なカラムが存在しません: {', '.join(missing_cols)}"

        if data[duration_col].isnull().any():
            return False, f"{duration_col}に欠損値が含まれています"

        if data[event_col].isnull().any():
            return False, f"{event_col}に欠損値が含まれています"

        if not all(data[event_col].isin([0, 1])):
            return False, f"{event_col}は0または1である必要があります"

        if (data[duration_col] < 0).any():
            return False, f"{duration_col}に負の値が含まれています"

        return True, None

    async def analyze(self,
                     query: str,
                     duration_col: str,
                     event_col: str,
                     save_results: bool = True,
                     dataset_id: Optional[str] = None,
                     table_id: Optional[str] = None) -> Tuple[Dict, Dict]:
        """
        生存時間分析を実行します。

        Args:
            query (str): 分析対象データを取得するBigQueryクエリ
            duration_col (str): イベント発生までの時間を表すカラム名
            event_col (str): イベント発生を表すカラム名 (例: 離職 = 1, 在職中 = 0)
            save_results (bool): 結果を保存するかどうか
            dataset_id (Optional[str]): 保存先データセットID
            table_id (Optional[str]): 保存先テーブルID

        Returns:
            Tuple[Dict, Dict]: (分析結果, メタデータ)

        Raises:
            ValueError: データのバリデーションエラー
            RuntimeError: 分析実行時のエラー
        """
        try:
            # データ取得
            data = await self.bq_service.fetch_data(query)

            # データバリデーション
            is_valid, error_message = self._validate_data(data, duration_col, event_col)
            if not is_valid:
                raise ValueError(error_message)

            # Kaplan-Meier 生存時間分析
            kmf = KaplanMeierFitter()
            kmf.fit(data[duration_col], data[event_col])

            # 生存曲線をデータフレームに変換
            survival_df = kmf.survival_function_.reset_index()
            survival_df.columns = ['timeline', 'survival_probability']

            # イベント発生までの平均時間を計算
            mean_survival_time = kmf.median_survival_time_

            # 分析結果を辞書に格納
            results = {
                "survival_curve": survival_df.to_dict(orient='records'),
                "mean_survival_time": mean_survival_time
            }

            # メタデータの作成
            metadata = {
                'row_count': len(data),
                'event_count': data[event_col].sum(),
                'censored_count': len(data) - data[event_col].sum(),
                'max_duration': data[duration_col].max(),
                'min_duration': data[duration_col].min()
            }

            # 結果の保存
            if save_results and dataset_id and table_id:
                results_df = pd.DataFrame(results["survival_curve"])
                await self.bq_service.save_results(
                    results_df,
                    dataset_id=dataset_id,
                    table_id=table_id
                )

            return results, metadata

        except Exception as e:
            raise RuntimeError(f"生存時間分析の実行中にエラーが発生しました: {str(e)}")

async def analyze_survival(request: Any) -> Tuple[Dict, int]:
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
        required_params = ['query', 'duration_col', 'event_col']
        missing_params = [param for param in required_params if param not in request_json]
        if missing_params:
            return {
                'error': f'必須パラメータが不足しています: {", ".join(missing_params)}'
            }, 400

        # サービスの初期化
        bq_service = BigQueryService()
        analyzer = SurvivalAnalyzer(bq_service)

        # パラメータの取得
        query = request_json['query']
        duration_col = request_json['duration_col']
        event_col = request_json['event_col']
        dataset_id = request_json.get('dataset_id')
        table_id = request_json.get('table_id')

        # 分析実行
        results, metadata = await analyzer.analyze(
            query=query,
            duration_col=duration_col,
            event_col=event_col,
            save_results=bool(dataset_id and table_id),
            dataset_id=dataset_id,
            table_id=table_id
        )

        return {
            'status': 'success',
            'results': results,
            'metadata': metadata
        }, 200

    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }, 500