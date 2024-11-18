# -*- coding: utf-8 -*-

"""
時系列分析

VAS データと損益計算書データの経時変化を分析します。
BigQueryService を利用した非同期処理に対応しています。
"""

from typing import Optional, Tuple, Dict, Any
import pandas as pd
import statsmodels.api as sm
from service.bigquery.client import BigQueryService
from service.firestore.client import FirestoreService
class TimeSeriesAnalyzer:
    """
    時系列分析を行うクラスです。
    BigQueryService を利用して非同期でデータの取得と保存を行います。
    """

    def __init__(self, bq_service: BigQueryService):
        """
        Args:
            bq_service (BigQueryService): BigQuery操作用のサービスインスタンス
        """
        self.bq_service = bq_service

    def _validate_data(self, data: pd.DataFrame, target_variable: str) -> Tuple[bool, Optional[str]]:
        """
        データのバリデーションを行います。

        Args:
            data (pd.DataFrame): 検証対象のデータ
            target_variable (str): 分析対象の変数名

        Returns:
            Tuple[bool, Optional[str]]: (検証結果, エラーメッセージ)
        """
        if data.empty:
            return False, "データが空です"

        if target_variable not in data.columns:
            return False, f"指定された変数 '{target_variable}' がデータに存在しません"

        if data[target_variable].isnull().any():
            return False, f"変数 '{target_variable}' に欠損値が含まれています"

        if len(data) < 10:  # 時系列分析に最低限必要なデータ数
            return False, "データ数が不足しています（10以上必要）"

        return True, None

    async def analyze(self,
                     query: str,
                     target_variable: str,
                     save_results: bool = True,
                     dataset_id: Optional[str] = None,
                     table_id: Optional[str] = None,
                     arima_order: Tuple[int, int, int] = (5, 1, 0)) -> Dict[str, Any]:
        """
        時系列分析を実行します。

        Args:
            query (str): データ取得用のBigQueryクエリ
            target_variable (str): 分析対象の変数名
            save_results (bool): 結果を保存するかどうか
            dataset_id (Optional[str]): 保存先データセットID
            table_id (Optional[str]): 保存先テーブルID
            arima_order (Tuple[int, int, int]): ARIMAモデルのパラメータ (p, d, q)

        Returns:
            Dict[str, Any]: 分析結果

        Raises:
            ValueError: データのバリデーションエラー
            RuntimeError: 分析実行中のエラー
        """
        try:
            # データ取得
            data = await self.bq_service.fetch_data(query)

            # データバリデーション
            is_valid, error_message = self._validate_data(data, target_variable)
            if not is_valid:
                raise ValueError(error_message)

            # 時系列分析 (ARIMAモデル)
            model = sm.tsa.ARIMA(data[target_variable], order=arima_order)
            results = model.fit()

            # 分析結果の整形
            analysis_results = {
                "summary": results.summary().tables[1].as_html(),
                "aic": float(results.aic),  # numpy.float64 を Python float に変換
                "bic": float(results.bic),
                "hqic": float(results.hqic),
                "metadata": {
                    "row_count": len(data),
                    "target_variable": target_variable,
                    "arima_order": arima_order,
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            }

            # 結果の保存
            if save_results and dataset_id and table_id:
                # 保存用にDataFrame形式に変換
                results_df = pd.DataFrame({
                    "metric": ["aic", "bic", "hqic"],
                    "value": [analysis_results["aic"],
                             analysis_results["bic"],
                             analysis_results["hqic"]],
                    "target_variable": target_variable,
                    "timestamp": pd.Timestamp.now()
                })

                await self.bq_service.save_results(
                    results_df,
                    dataset_id=dataset_id,
                    table_id=table_id
                )

            return analysis_results

        except ValueError as e:
            raise ValueError(f"データの検証に失敗しました: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"時系列分析の実行中にエラーが発生しました: {str(e)}")

async def analyze_timeseries(request: Any) -> Tuple[Dict[str, Any], int]:
    """
    Cloud Functions用のエントリーポイント関数

    Args:
        request: Cloud Functionsのリクエストオブジェクト

    Returns:
        Tuple[Dict[str, Any], int]: (レスポンス, ステータスコード)
    """
    try:
        request_json = request.get_json()

        if not request_json:
            return {'error': 'リクエストデータがありません'}, 400

        # 必須パラメータのバリデーション
        required_params = ['query', 'target_variable']
        for param in required_params:
            if param not in request_json:
                return {'error': f"必須パラメータ '{param}' が指定されていません"}, 400

        # オプションパラメータの取得
        arima_order = tuple(request_json.get('arima_order', (5, 1, 0)))
        dataset_id = request_json.get('dataset_id')
        table_id = request_json.get('table_id')
        save_results = request_json.get('save_results', True)

        # サービスの初期化
        bq_service = BigQueryService()
        analyzer = TimeSeriesAnalyzer(bq_service)

        # 分析実行
        results = await analyzer.analyze(
            query=request_json['query'],
            target_variable=request_json['target_variable'],
            save_results=save_results,
            dataset_id=dataset_id,
            table_id=table_id,
            arima_order=arima_order
        )

        return {
            'status': 'success',
            'results': results
        }, 200

    except ValueError as e:
        return {
            'status': 'error',
            'type': 'validation_error',
            'message': str(e)
        }, 400
    except Exception as e:
        return {
            'status': 'error',
            'type': 'internal_error',
            'message': str(e)
        }, 500