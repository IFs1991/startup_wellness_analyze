# -*- coding: utf-8 -*-

"""
生存時間分析

Startup Wellness プログラム導入前後における、従業員の離職までの時間を比較分析します。
BigQueryServiceを利用した非同期処理に対応しています。
"""

from typing import Optional, Tuple, Dict, Any, List, Iterator, ContextManager
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from service.bigquery.client import BigQueryService
from service.firestore.client import FirestoreService
from contextlib import contextmanager
import gc
import weakref
import logging
import asyncio

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
        self.logger = logging.getLogger(__name__)
        self._temp_data = weakref.WeakValueDictionary()  # 一時データを弱参照で管理
        self._kmf_models = weakref.WeakValueDictionary()  # KaplanMeierFitterモデルを弱参照で管理

    @contextmanager
    def _managed_dataframe(self, df: pd.DataFrame, name: str = "default") -> Iterator[pd.DataFrame]:
        """
        データフレームのライフサイクルを管理するコンテキストマネージャー

        Args:
            df (pd.DataFrame): 管理対象のデータフレーム
            name (str): データフレームの識別名

        Yields:
            Iterator[pd.DataFrame]: 管理されたデータフレーム
        """
        try:
            # データフレームを弱参照辞書に登録
            self._temp_data[name] = df
            yield df
        finally:
            # コンテキスト終了時にデータフレーム参照を削除
            if name in self._temp_data:
                del self._temp_data[name]
            # 明示的なガベージコレクション
            gc.collect()

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
                     table_id: Optional[str] = None,
                     storage_mode: str = "memory",
                     batch_size: int = 10000) -> Tuple[Dict, Dict]:
        """
        生存時間分析を実行します。

        Args:
            query (str): 分析対象データを取得するBigQueryクエリ
            duration_col (str): イベント発生までの時間を表すカラム名
            event_col (str): イベント発生を表すカラム名 (例: 離職 = 1, 在職中 = 0)
            save_results (bool): 結果を保存するかどうか
            dataset_id (Optional[str]): 保存先データセットID
            table_id (Optional[str]): 保存先テーブルID
            storage_mode (str): データの保存モード ("memory", "disk", "hybrid")
            batch_size (int): 大規模データ処理用のバッチサイズ

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

            # データサイズに基づくストレージモードの自動調整
            if storage_mode == "auto":
                if len(data) > batch_size:
                    storage_mode = "hybrid"
                    self.logger.info(f"大規模データセット検出: {len(data)}行。ハイブリッドモードを使用します。")
                else:
                    storage_mode = "memory"
                    self.logger.info(f"標準データセット検出: {len(data)}行。メモリモードを使用します。")

            # データフレームをコンテキストマネージャで管理
            with self._managed_dataframe(data, "survival_data") as managed_data:
                # 大規模データの場合はバッチ処理
                if len(managed_data) > batch_size and storage_mode != "memory":
                    self.logger.info(f"大規模データセット検出: {len(managed_data)}行。バッチ処理を実行します。")
                    return await self._analyze_in_batches(managed_data, duration_col, event_col,
                                                        save_results, dataset_id, table_id,
                                                        storage_mode, batch_size)

                # Kaplan-Meier 生存時間分析
                kmf = KaplanMeierFitter()

                # モデルを弱参照で管理
                self._kmf_models["current_model"] = kmf

                try:
                    # モデルに適合
                    kmf.fit(managed_data[duration_col], managed_data[event_col])

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
                        'row_count': len(managed_data),
                        'event_count': managed_data[event_col].sum(),
                        'censored_count': len(managed_data) - managed_data[event_col].sum(),
                        'max_duration': managed_data[duration_col].max(),
                        'min_duration': managed_data[duration_col].min()
                    }

                    # 結果の保存
                    if save_results and dataset_id and table_id:
                        # 保存用のデータフレームを作成して保存
                        with self._managed_dataframe(pd.DataFrame(results["survival_curve"]), "results_df") as results_df:
                            await self.bq_service.save_results(
                                results_df,
                                dataset_id=dataset_id,
                                table_id=table_id
                            )

                    return results, metadata

                except Exception as e:
                    self.logger.error(f"Kaplan-Meier解析中にエラーが発生しました: {str(e)}")
                    # フォールバック - より堅牢な計算方法を試行
                    self.logger.info("フォールバックメソッドを試行します...")
                    return await self._fallback_analysis(managed_data, duration_col, event_col,
                                                       save_results, dataset_id, table_id)

        except Exception as e:
            self.logger.error(f"生存時間分析の実行中にエラーが発生しました: {str(e)}")
            self._cleanup_on_error()
            raise RuntimeError(f"生存時間分析の実行中にエラーが発生しました: {str(e)}")

    async def _analyze_in_batches(self,
                                 data: pd.DataFrame,
                                 duration_col: str,
                                 event_col: str,
                                 save_results: bool,
                                 dataset_id: Optional[str],
                                 table_id: Optional[str],
                                 storage_mode: str,
                                 batch_size: int) -> Tuple[Dict, Dict]:
        """
        大規模データセットをバッチで処理して生存時間分析を実行します。

        Args:
            data (pd.DataFrame): 分析対象データ
            duration_col (str): イベント発生までの時間を表すカラム名
            event_col (str): イベント発生を表すカラム名
            save_results (bool): 結果を保存するかどうか
            dataset_id (Optional[str]): 保存先データセットID
            table_id (Optional[str]): 保存先テーブルID
            storage_mode (str): データの保存モード
            batch_size (int): バッチサイズ

        Returns:
            Tuple[Dict, Dict]: (分析結果, メタデータ)
        """
        # Kaplan-Meier 分析用の初期データ準備
        durations = data[duration_col]
        events = data[event_col]

        # メタデータの計算
        metadata = {
            'row_count': len(data),
            'event_count': events.sum(),
            'censored_count': len(data) - events.sum(),
            'max_duration': durations.max(),
            'min_duration': durations.min(),
            'batch_processing': True
        }

        # Kaplan-Meier 生存時間分析
        kmf = KaplanMeierFitter()
        self._kmf_models["batch_model"] = kmf

        # ディスクモードとハイブリッドモードはバッチ処理が必要
        if storage_mode in ["disk", "hybrid"]:
            self.logger.info(f"{storage_mode}モードで生存時間分析を実行しています...")

            # バッチごとに部分適合を行う場合の適合方法
            try:
                # 全データに対して一度に適合
                kmf.fit(durations, events)

                # 生存曲線をデータフレームに変換
                survival_df = kmf.survival_function_.reset_index()
                survival_df.columns = ['timeline', 'survival_probability']

                # ハイブリッドモードでは結果をメモリに保持し、中間計算結果を解放
                results = {
                    "survival_curve": survival_df.to_dict(orient='records'),
                    "mean_survival_time": kmf.median_survival_time_
                }

                # 大規模結果の場合、バッチで保存
                if save_results and dataset_id and table_id:
                    if len(survival_df) > batch_size:
                        self.logger.info(f"結果({len(survival_df)}行)をバッチで保存しています...")
                        save_batches = [survival_df[i:i+batch_size] for i in range(0, len(survival_df), batch_size)]
                        for i, save_batch in enumerate(save_batches):
                            self.logger.info(f"結果バッチ {i+1}/{len(save_batches)} を保存中...")
                            batch_table_id = f"{table_id}_batch_{i+1}" if i > 0 else table_id
                            await self.bq_service.save_results(
                                save_batch,
                                dataset_id=dataset_id,
                                table_id=batch_table_id
                            )
                    else:
                        await self.bq_service.save_results(
                            survival_df,
                            dataset_id=dataset_id,
                            table_id=table_id
                        )

                return results, metadata

            except Exception as e:
                self.logger.error(f"バッチ処理中にエラーが発生しました: {str(e)}")
                # フォールバック - より堅牢な計算方法
                return await self._fallback_analysis(data, duration_col, event_col,
                                                   save_results, dataset_id, table_id)

    async def _fallback_analysis(self,
                                data: pd.DataFrame,
                                duration_col: str,
                                event_col: str,
                                save_results: bool,
                                dataset_id: Optional[str],
                                table_id: Optional[str]) -> Tuple[Dict, Dict]:
        """
        エラー発生時のフォールバック分析を実行します。

        Args:
            data (pd.DataFrame): 分析対象データ
            duration_col (str): イベント発生までの時間を表すカラム名
            event_col (str): イベント発生を表すカラム名
            save_results (bool): 結果を保存するかどうか
            dataset_id (Optional[str]): 保存先データセットID
            table_id (Optional[str]): 保存先テーブルID

        Returns:
            Tuple[Dict, Dict]: (分析結果, メタデータ)
        """
        try:
            self.logger.info("簡易生存時間分析を実行しています...")
            # データをソート
            sorted_data = data.sort_values(by=duration_col)
            durations = sorted_data[duration_col].values
            events = sorted_data[event_col].values

            # 時間点での生存確率を手動で計算
            unique_times = np.unique(durations)
            survival_probs = []
            at_risk = len(durations)
            survival_prob = 1.0

            for t in unique_times:
                # この時点でのイベント数を計算
                events_at_t = np.sum((durations == t) & (events == 1))
                # 生存確率を更新
                if at_risk > 0:
                    survival_prob *= (1 - events_at_t / at_risk)
                # リスク集合のサイズを更新
                at_risk -= np.sum(durations == t)
                # 結果を格納
                survival_probs.append(survival_prob)

            # 結果をデータフレームに変換
            survival_df = pd.DataFrame({
                'timeline': unique_times,
                'survival_probability': survival_probs
            })

            # 中央生存時間の計算（生存確率が0.5以下になる最初の時間）
            median_idx = np.where(np.array(survival_probs) <= 0.5)[0]
            mean_survival_time = unique_times[median_idx[0]] if len(median_idx) > 0 else np.max(unique_times)

            # 分析結果を辞書に格納
            results = {
                "survival_curve": survival_df.to_dict(orient='records'),
                "mean_survival_time": float(mean_survival_time),
                "fallback_method": True
            }

            # メタデータの作成
            metadata = {
                'row_count': len(data),
                'event_count': data[event_col].sum(),
                'censored_count': len(data) - data[event_col].sum(),
                'max_duration': data[duration_col].max(),
                'min_duration': data[duration_col].min(),
                'fallback_method': True
            }

            # 結果の保存
            if save_results and dataset_id and table_id:
                await self.bq_service.save_results(
                    survival_df,
                    dataset_id=dataset_id,
                    table_id=table_id
                )

            return results, metadata

        except Exception as e:
            self.logger.error(f"フォールバック分析中にエラーが発生しました: {str(e)}")
            # 最終的なフォールバック - 非常に基本的な結果のみを返す
            empty_results = {
                "survival_curve": [],
                "mean_survival_time": None,
                "error": str(e),
                "error_fallback": True
            }
            basic_metadata = {
                'row_count': len(data),
                'error': str(e),
                'error_fallback': True
            }
            return empty_results, basic_metadata

    def release_resources(self) -> None:
        """
        メモリリソースを解放します。
        """
        try:
            # 一時データの削除
            self._temp_data.clear()

            # モデルの削除
            self._kmf_models.clear()

            # 明示的なガベージコレクション
            gc.collect()
            self.logger.info("リソースが正常に解放されました")
        except Exception as e:
            self.logger.error(f"リソース解放中にエラーが発生しました: {str(e)}")

    def _cleanup_on_error(self) -> None:
        """
        エラー発生時のクリーンアップ処理
        """
        try:
            self.release_resources()
        except Exception as e:
            self.logger.error(f"エラークリーンアップ中に問題が発生しました: {str(e)}")

    def __del__(self):
        """
        デストラクタ - リソース解放を保証
        """
        self.release_resources()

    def __enter__(self):
        """
        コンテキストマネージャのエントリーポイント
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        コンテキストマネージャの終了処理
        """
        self.release_resources()
        return False  # 例外を伝播させる

    def estimate_memory_usage(self, row_count: int, col_count: int = 10) -> Dict[str, float]:
        """
        メモリ使用量を推定します。

        Args:
            row_count (int): 行数
            col_count (int): 列数

        Returns:
            Dict[str, float]: 各ストレージモードでの推定メモリ使用量 (MB)
        """
        # 基本的なメモリ使用量の推定（概算）
        bytes_per_value = 8  # double型の値
        dataframe_bytes = row_count * col_count * bytes_per_value

        # KMFモデルサイズの概算
        kmf_model_bytes = row_count * 3 * bytes_per_value

        # 結果の推定サイズ
        result_size_bytes = row_count * 2 * bytes_per_value

        # 各モードでの推定メモリ使用量
        memory_mode = (dataframe_bytes + kmf_model_bytes + result_size_bytes) / (1024 * 1024)
        hybrid_mode = (kmf_model_bytes + result_size_bytes) / (1024 * 1024)
        disk_mode = result_size_bytes / (1024 * 1024)

        return {
            "memory_mode_mb": memory_mode,
            "hybrid_mode_mb": hybrid_mode,
            "disk_mode_mb": disk_mode,
            "recommended_mode": "memory" if memory_mode < 1000 else "hybrid" if hybrid_mode < 500 else "disk"
        }

async def analyze_survival(request: Any) -> Tuple[Dict, int]:
    """
    Cloud Functions用のエントリーポイント関数

    Args:
        request: Cloud Functionsのリクエストオブジェクト

    Returns:
        Tuple[Dict, int]: (レスポンス, ステータスコード)
    """
    logger = logging.getLogger(__name__)

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

        # 追加パラメータの取得
        storage_mode = request_json.get('storage_mode', 'auto')
        batch_size = int(request_json.get('batch_size', 10000))

        # パラメータの取得
        query = request_json['query']
        duration_col = request_json['duration_col']
        event_col = request_json['event_col']
        dataset_id = request_json.get('dataset_id')
        table_id = request_json.get('table_id')

        # コンテキストマネージャとしてアナライザーを使用
        async with contextmanager(lambda: SurvivalAnalyzer(bq_service))() as analyzer:
            # メモリ使用量の推定
            if 'estimate_memory' in request_json and request_json['estimate_memory']:
                row_estimate = int(request_json.get('estimated_rows', 10000))
                col_estimate = int(request_json.get('estimated_cols', 10))
                memory_estimate = analyzer.estimate_memory_usage(row_estimate, col_estimate)
                return {
                    'status': 'success',
                    'memory_estimate': memory_estimate
                }, 200

            # 分析実行
            results, metadata = await analyzer.analyze(
                query=query,
                duration_col=duration_col,
                event_col=event_col,
                save_results=bool(dataset_id and table_id),
                dataset_id=dataset_id,
                table_id=table_id,
                storage_mode=storage_mode,
                batch_size=batch_size
            )

            return {
                'status': 'success',
                'results': results,
                'metadata': metadata
            }, 200

    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }, 500