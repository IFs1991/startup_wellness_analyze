# -*- coding: utf-8 -*-

"""
主成分分析

多数のVAS項目から、従業員の健康状態を代表する少数の主成分を抽出します。
BigQueryServiceを利用した非同期処理に対応しています。
"""

from typing import Optional, Tuple, Dict, Any, List, Iterator, ContextManager
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from service.bigquery.client import BigQueryService
from service.firestore.client import FirestoreService
from contextlib import contextmanager
import gc
import weakref
import logging
import asyncio

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
        self.logger = logging.getLogger(__name__)
        self._temp_data = weakref.WeakValueDictionary()  # 一時データを弱参照で管理
        self._pca_models = weakref.WeakValueDictionary()  # PCAモデルを弱参照で管理

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
        table_id: Optional[str] = None,
        batch_size: int = 10000
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        主成分分析を実行します。

        Args:
            query (str): 分析対象データを取得するBigQueryクエリ
            n_components (int): 抽出する主成分の数
            save_results (bool): 結果を保存するかどうか
            dataset_id (Optional[str]): 保存先データセットID
            table_id (Optional[str]): 保存先テーブルID
            batch_size (int): 大規模データ処理のバッチサイズ

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

            # データフレームをコンテキストマネージャで管理
            with self._managed_dataframe(data, "original_data") as managed_data:
                # 数値型のカラムのみを抽出
                numeric_data = managed_data.select_dtypes(include=[np.number])

                # 大規模データの場合はバッチ処理
                if len(numeric_data) > batch_size:
                    self.logger.info(f"大規模データセット検出: {len(numeric_data)}行。バッチ処理を実行します。")
                    return await self._analyze_in_batches(managed_data, numeric_data, n_components,
                                                        save_results, dataset_id, table_id, batch_size)

                # 通常の処理（小・中規模データ）
                # 主成分分析の実行
                pca = PCA(n_components=min(n_components, len(numeric_data.columns)))

                # モデルを弱参照で管理
                self._pca_models["current_model"] = pca
                principal_components = pca.fit_transform(numeric_data)

                # 結果のデータフレーム作成
                principal_df = pd.DataFrame(
                    data=principal_components,
                    columns=[f'principal_component_{i+1}' for i in range(pca.n_components_)]
                )

                # 元データと主成分スコアの結合
                results = pd.concat([managed_data, principal_df], axis=1)

                # メタデータの作成
                metadata = {
                    'row_count': len(managed_data),
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
            self.logger.error(f"主成分分析の実行中にエラーが発生しました: {str(e)}")
            self._cleanup_on_error()
            raise RuntimeError(f"主成分分析の実行中にエラーが発生しました: {str(e)}")

    async def _analyze_in_batches(self,
                               full_data: pd.DataFrame,
                               numeric_data: pd.DataFrame,
                               n_components: int,
                               save_results: bool,
                               dataset_id: Optional[str],
                               table_id: Optional[str],
                               batch_size: int) -> Tuple[pd.DataFrame, Dict]:
        """
        大規模データセットをバッチで処理して主成分分析を実行します。

        Args:
            full_data (pd.DataFrame): 元の完全なデータフレーム
            numeric_data (pd.DataFrame): 数値カラムのみのデータフレーム
            n_components (int): 抽出する主成分の数
            save_results (bool): 結果を保存するかどうか
            dataset_id (Optional[str]): 保存先データセットID
            table_id (Optional[str]): 保存先テーブルID
            batch_size (int): バッチサイズ

        Returns:
            Tuple[pd.DataFrame, Dict]: (主成分スコアが付与されたデータ, メタデータ)
        """
        # 初期化フェーズ - データのサブセットで適合
        sample_size = min(batch_size, len(numeric_data))
        sample_data = numeric_data.sample(n=sample_size, random_state=42)

        # 主成分分析モデルの初期化
        pca = PCA(n_components=min(n_components, len(numeric_data.columns)))

        # サンプルデータでモデルを適合
        self.logger.info(f"サンプルデータ({sample_size}行)でPCAモデルを初期化しています...")
        pca.fit(sample_data)

        # 弱参照で管理
        self._pca_models["batch_model"] = pca

        # 変換フェーズ - バッチごとに処理
        all_components = []
        batches = [numeric_data.iloc[i:i+batch_size] for i in range(0, len(numeric_data), batch_size)]

        self.logger.info(f"データを{len(batches)}バッチに分割して処理しています...")
        for i, batch in enumerate(batches):
            self.logger.info(f"バッチ {i+1}/{len(batches)} を処理中...")
            batch_components = pca.transform(batch)
            all_components.append(batch_components)

            # 使用済みバッチのメモリ解放
            del batch
            # 50バッチごとにガベージコレクション
            if (i+1) % 50 == 0:
                gc.collect()

        # 全バッチの結果を結合
        principal_components = np.vstack(all_components)

        # 結果のデータフレーム作成
        principal_df = pd.DataFrame(
            data=principal_components,
            columns=[f'principal_component_{i+1}' for i in range(pca.n_components_)]
        )

        # 元データと主成分スコアの結合
        with self._managed_dataframe(principal_df, "principal_components") as managed_principal_df:
            results = pd.concat([full_data, managed_principal_df], axis=1)

            # メタデータの作成
            metadata = {
                'row_count': len(full_data),
                'original_columns': list(numeric_data.columns),
                'n_components': pca.n_components_,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'batch_processing': True,
                'batch_count': len(batches)
            }

            # 結果の保存
            if save_results and dataset_id and table_id:
                # 大きなデータセットはバッチで保存
                if len(results) > batch_size:
                    self.logger.info(f"結果をバッチで保存しています...")
                    save_batches = [results.iloc[i:i+batch_size] for i in range(0, len(results), batch_size)]
                    for j, save_batch in enumerate(save_batches):
                        self.logger.info(f"結果バッチ {j+1}/{len(save_batches)} を保存中...")
                        batch_table_id = f"{table_id}_batch_{j+1}" if j > 0 else table_id
                        await self.bq_service.save_results(
                            save_batch,
                            dataset_id=dataset_id,
                            table_id=batch_table_id
                        )
                else:
                    await self.bq_service.save_results(
                        results,
                        dataset_id=dataset_id,
                        table_id=table_id
                    )

            return results, metadata

    def release_resources(self) -> None:
        """
        メモリリソースを解放します。
        """
        try:
            # 一時データの削除
            self._temp_data.clear()

            # PCAモデルの削除
            self._pca_models.clear()

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

async def analyze_pca(request: Any) -> Tuple[Dict, int]:
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
        batch_size = int(request_json.get('batch_size', 10000))

        # サービスの初期化とPCA実行
        bq_service = BigQueryService()

        # コンテキストマネージャとしてアナライザーを使用
        async with contextmanager(lambda: PCAAnalyzer(bq_service))() as analyzer:
            results, metadata = await analyzer.analyze(
                query=query,
                n_components=n_components,
                save_results=bool(dataset_id and table_id),
                dataset_id=dataset_id,
                table_id=table_id,
                batch_size=batch_size
            )

            return {
                'status': 'success',
                'results': results.to_dict('records'),
                'metadata': metadata
            }, 200

    except ValueError as e:
        logger.error(f"パラメータエラー: {str(e)}")
        return {
            'status': 'error',
            'message': f'パラメータエラー: {str(e)}'
        }, 400

    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }, 500