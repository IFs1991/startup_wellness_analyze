# -*- coding: utf-8 -*-

"""
時系列分析

VAS データと損益計算書データの経時変化を分析します。
BigQueryService を利用した非同期処理に対応しています。
"""

from typing import Optional, Tuple, Dict, Any, List, Union
import pandas as pd
import numpy as np
import statsmodels.api as sm
from service.bigquery.client import BigQueryService
from service.firestore.client import FirestoreService
import traceback
import logging
import gc
import contextlib
import time

# ロガーの設定
logger = logging.getLogger(__name__)


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
        self.logger = logging.getLogger(__name__)
        self._temp_resources = []  # 一時リソース追跡用

    def __del__(self):
        """
        デストラクタ：リソースの自動解放
        """
        self.release_resources()

    def release_resources(self):
        """
        使用したリソースを解放する
        """
        try:
            # 一時リソースのクリーンアップ
            self._temp_resources.clear()

            # 明示的にガベージコレクションを実行
            gc.collect()
        except Exception as e:
            self.logger.error(f"リソース解放中にエラーが発生しました: {str(e)}")

    def _validate_data(self, data: pd.DataFrame, target_variable: str) -> Tuple[bool, Optional[str]]:
        """
        データのバリデーションを行います。

        Args:
            data (pd.DataFrame): 検証対象のデータ
            target_variable (str): 分析対象の変数名

        Returns:
            Tuple[bool, Optional[str]]: (検証結果, エラーメッセージ)
        """
        # データが空かどうかを確認
        if data is None or data.empty:
            return False, "データが空です"

        # 必要な列が存在するかを確認
        if target_variable not in data.columns:
            return False, f"指定された変数 '{target_variable}' がデータに存在しません"

        # 欠損値の確認
        if data[target_variable].isnull().any():
            return False, f"変数 '{target_variable}' に欠損値が含まれています"

        # 十分なデータ数があるかを確認
        if len(data) < 10:  # 時系列分析に最低限必要なデータ数
            return False, "データ数が不足しています（10以上必要）"

        # 全てのチェックをパスした場合
        return True, None

    @contextlib.contextmanager
    def _managed_dataframe(self, df: pd.DataFrame, copy: bool = False):
        """
        データフレームのリソース管理を行うコンテキストマネージャ

        Args:
            df (pd.DataFrame): 管理するデータフレーム
            copy (bool): データをコピーするかどうか

        Yields:
            pd.DataFrame: 管理対象のデータフレーム
        """
        try:
            if copy:
                df_copy = df.copy()
                yield df_copy
            else:
                yield df
        finally:
            # 明示的なクリーンアップ
            if copy:
                del df_copy

            # メモリ使用状況のログ（デバッグレベル）
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("データフレーム管理コンテキスト終了")

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
        start_time = time.time()
        self.logger.info(f"時系列分析を開始します: 対象変数={target_variable}")

        try:
            # データ取得
            data = await self._fetch_data(query)

            # データバリデーション
            is_valid, error_message = self._validate_data(data, target_variable)
            if not is_valid:
                raise ValueError(error_message)

            # 時系列モデルの構築と分析
            with self._managed_dataframe(data) as managed_data:
                # ARIMAモデルの構築と学習
                model_results = self._build_and_fit_arima_model(
                    managed_data[target_variable],
                    arima_order
                )

                # 分析結果の整形
                analysis_results = self._format_analysis_results(
                    model_results,
                    len(managed_data),
                    target_variable,
                    arima_order
                )

                # 結果の保存（指定がある場合）
                if save_results and dataset_id and table_id:
                    await self._save_analysis_results(
                        analysis_results,
                        target_variable,
                        dataset_id,
                        table_id
                    )

            elapsed_time = time.time() - start_time
            self.logger.info(f"時系列分析が完了しました: 対象変数={target_variable}, 実行時間={elapsed_time:.2f}秒")
            return analysis_results

        except ValueError as e:
            self.logger.error(f"データバリデーションエラー: {str(e)}")
            raise ValueError(f"データの検証に失敗しました: {str(e)}")
        except Exception as e:
            self.logger.error(f"時系列分析中にエラーが発生しました: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise RuntimeError(f"時系列分析の実行中にエラーが発生しました: {str(e)}")

    async def _fetch_data(self, query: str) -> pd.DataFrame:
        """
        BigQueryからデータを取得します。

        Args:
            query (str): 実行するSQLクエリ

        Returns:
            pd.DataFrame: 取得したデータ

        Raises:
            RuntimeError: データ取得に失敗した場合
        """
        try:
            self.logger.info("BigQueryからデータを取得しています...")
            data = await self.bq_service.fetch_data(query)
            self.logger.info(f"データ取得完了: {len(data)}行")
            return data
        except Exception as e:
            self.logger.error(f"データ取得中にエラーが発生しました: {str(e)}")
            raise RuntimeError(f"BigQueryからのデータ取得に失敗しました: {str(e)}")

    def _build_and_fit_arima_model(self, series: pd.Series, arima_order: Tuple[int, int, int]):
        """
        ARIMAモデルを構築し、フィッティングします。

        Args:
            series (pd.Series): モデル化する時系列データ
            arima_order (Tuple[int, int, int]): ARIMAモデルのパラメータ (p, d, q)

        Returns:
            モデルフィッティング結果
        """
        try:
            self.logger.info(f"ARIMAモデル ({arima_order[0]}, {arima_order[1]}, {arima_order[2]}) の構築を開始...")
            model = sm.tsa.ARIMA(series, order=arima_order)
            results = model.fit()
            self.logger.info("ARIMAモデルのフィッティングが完了しました")
            return results
        except Exception as e:
            self.logger.error(f"ARIMAモデルのフィッティング中にエラーが発生しました: {str(e)}")
            raise

    def _format_analysis_results(self, model_results, data_length: int,
                               target_variable: str, arima_order: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        モデル結果を整形して返します。

        Args:
            model_results: ARIMAモデルのフィッティング結果
            data_length (int): データの行数
            target_variable (str): 分析対象の変数名
            arima_order (Tuple[int, int, int]): ARIMAモデルのパラメータ

        Returns:
            Dict[str, Any]: 整形された分析結果
        """
        # 主要な統計指標をPython標準の型に変換
        aic = float(model_results.aic)
        bic = float(model_results.bic)
        hqic = float(model_results.hqic)

        # 結果の整形
        analysis_results = {
            "summary": model_results.summary().tables[1].as_html(),
            "aic": aic,
            "bic": bic,
            "hqic": hqic,
            "metadata": {
                "row_count": data_length,
                "target_variable": target_variable,
                "arima_order": arima_order,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        }

        return analysis_results

    async def _save_analysis_results(self, analysis_results: Dict[str, Any],
                                   target_variable: str, dataset_id: str, table_id: str):
        """
        分析結果をBigQueryに保存します。

        Args:
            analysis_results (Dict[str, Any]): 保存する分析結果
            target_variable (str): 分析対象の変数名
            dataset_id (str): 保存先データセットID
            table_id (str): 保存先テーブルID
        """
        try:
            # 保存用にDataFrame形式に変換
            results_df = pd.DataFrame({
                "metric": ["aic", "bic", "hqic"],
                "value": [
                    analysis_results["aic"],
                    analysis_results["bic"],
                    analysis_results["hqic"]
                ],
                "target_variable": target_variable,
                "timestamp": pd.Timestamp.now()
            })

            self.logger.info(f"分析結果をBigQueryに保存しています: {dataset_id}.{table_id}")
            await self.bq_service.save_results(
                results_df,
                dataset_id=dataset_id,
                table_id=table_id
            )
            self.logger.info("分析結果が正常に保存されました")
        except Exception as e:
            self.logger.error(f"結果保存中にエラーが発生しました: {str(e)}")
            self.logger.debug(traceback.format_exc())
            # 保存失敗はエラーとして伝播せず、警告として記録するのみ


async def analyze_timeseries(request: Any) -> Tuple[Dict[str, Any], int]:
    """
    Cloud Functions用のエントリーポイント関数

    Args:
        request: Cloud Functionsのリクエストオブジェクト

    Returns:
        Tuple[Dict[str, Any], int]: (レスポンス, ステータスコード)
    """
    try:
        # リクエストのバリデーション
        request_json = request.get_json()
        if not request_json:
            return {'error': 'リクエストデータがありません'}, 400

        # 必須パラメータの検証
        required_params = ['query', 'target_variable']
        missing_params = [param for param in required_params if param not in request_json]
        if missing_params:
            return {
                'error': f"必須パラメータが指定されていません: {', '.join(missing_params)}"
            }, 400

        # オプションパラメータの取得
        arima_order = tuple(request_json.get('arima_order', (5, 1, 0)))
        dataset_id = request_json.get('dataset_id')
        table_id = request_json.get('table_id')
        save_results = request_json.get('save_results', True)

        # サービスの初期化と分析実行
        bq_service = BigQueryService()
        analyzer = TimeSeriesAnalyzer(bq_service)

        try:
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
        finally:
            # リソース解放
            analyzer.release_resources()

    except ValueError as e:
        # バリデーションエラー
        logger.error(f"バリデーションエラー: {str(e)}")
        return {
            'status': 'error',
            'type': 'validation_error',
            'message': str(e)
        }, 400
    except Exception as e:
        # 予期せぬエラー
        logger.error(f"時系列分析中に予期せぬエラーが発生しました: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            'status': 'error',
            'type': 'internal_error',
            'message': str(e)
        }, 500