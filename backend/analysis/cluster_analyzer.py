# -*- coding: utf-8 -*-
"""
Cluster Analysis with BigQueryService Integration

This code implements clustering analysis functionality with BigQueryService integration,
supporting asynchronous processing and Cloud Functions deployment.
"""

from typing import Optional, Tuple, Any, Dict, List
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from backend.src.database.bigquery.client import BigQueryService
from backend.src.database.firestore.client import FirestoreClient
from . import BaseAnalyzer, AnalysisError

class ClusterAnalyzer(BaseAnalyzer):
    """クラスタリング分析を行うクラス"""

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        max_clusters: int = 10,
        random_state: int = 42
    ):
        """
        初期化メソッド

        Args:
            n_clusters (Optional[int]): クラスター数（Noneの場合は最適なクラスター数を自動決定）
            max_clusters (int): クラスター数の最大値（自動決定時に使用）
            random_state (int): 乱数シード
        """
        super().__init__("cluster_analysis")
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        データのバリデーション

        Args:
            data (pd.DataFrame): 検証対象のデータ

        Returns:
            bool: バリデーション結果
        """
        if data.empty:
            raise AnalysisError("データが空です")

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            raise AnalysisError("数値型の列が2つ以上必要です")

        if data.isnull().any().any():
            raise AnalysisError("欠損値が含まれています")

        return True

    def _prepare_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        データの前処理

        Args:
            data (pd.DataFrame): 前処理対象のデータ

        Returns:
            np.ndarray: 前処理済みデータ
        """
        # 数値型のカラムのみを抽出
        numeric_data = data.select_dtypes(include=[np.number])

        # 標準化
        scaled_data = self.scaler.fit_transform(numeric_data)
        return scaled_data

    def _determine_optimal_clusters(self, data: np.ndarray) -> int:
        """
        最適なクラスター数を決定

        Args:
            data (np.ndarray): 分析対象データ

        Returns:
            int: 最適なクラスター数
        """
        if self.n_clusters is not None:
            return self.n_clusters

        # シルエット係数とCalinski-Harabasz指標を使用して最適なクラスター数を決定
        silhouette_scores = []
        ch_scores = []
        k_range = range(2, min(self.max_clusters + 1, len(data)))

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(data)
            silhouette_scores.append(silhouette_score(data, labels))
            ch_scores.append(calinski_harabasz_score(data, labels))

        # スコアを正規化して組み合わせる
        normalized_silhouette = np.array(silhouette_scores) / np.max(silhouette_scores)
        normalized_ch = np.array(ch_scores) / np.max(ch_scores)
        combined_scores = (normalized_silhouette + normalized_ch) / 2

        optimal_k = k_range[np.argmax(combined_scores)]
        return optimal_k

    def _calculate_cluster_statistics(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        numeric_columns: List[str]
    ) -> List[Dict[str, Any]]:
        """
        各クラスターの統計情報を計算

        Args:
            data (pd.DataFrame): 元データ
            labels (np.ndarray): クラスターラベル
            numeric_columns (List[str]): 数値型カラムのリスト

        Returns:
            List[Dict[str, Any]]: クラスター統計情報
        """
        cluster_stats = []
        for cluster_id in range(len(np.unique(labels))):
            cluster_data = data[labels == cluster_id]
            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': float(len(cluster_data) / len(data) * 100),
                'statistics': {
                    column: {
                        'mean': float(cluster_data[column].mean()),
                        'std': float(cluster_data[column].std()),
                        'min': float(cluster_data[column].min()),
                        'max': float(cluster_data[column].max()),
                        'median': float(cluster_data[column].median())
                    }
                    for column in numeric_columns
                }
            }
            cluster_stats.append(stats)
        return cluster_stats

    async def analyze(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        クラスター分析を実行

        Args:
            data (pd.DataFrame): 分析対象データ
            target_columns (Optional[List[str]]): 分析対象の列名リスト
            **kwargs: 追加のパラメータ

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            # 対象列の選択
            if target_columns:
                data = data[target_columns]

            # データの前処理
            preprocessed_data = self._prepare_data(data)
            numeric_columns = data.select_dtypes(include=[np.number]).columns

            # クラスター数の決定
            n_clusters = self._determine_optimal_clusters(preprocessed_data)

            # クラスタリングの実行
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state
            )
            labels = kmeans.fit_predict(preprocessed_data)

            # クラスター中心の変換（元のスケールに戻す）
            cluster_centers = self.scaler.inverse_transform(kmeans.cluster_centers_)

            # 評価指標の計算
            silhouette = silhouette_score(preprocessed_data, labels)
            ch_score = calinski_harabasz_score(preprocessed_data, labels)

            # クラスター統計の計算
            cluster_stats = self._calculate_cluster_statistics(
                data,
                labels,
                numeric_columns
            )

            # 結果の整形
            result = {
                'n_clusters': n_clusters,
                'cluster_centers': [
                    {
                        'cluster_id': i,
                        'center': {
                            feature: float(value)
                            for feature, value in zip(numeric_columns, center)
                        }
                    }
                    for i, center in enumerate(cluster_centers)
                ],
                'cluster_labels': labels.tolist(),
                'evaluation_metrics': {
                    'silhouette_score': float(silhouette),
                    'calinski_harabasz_score': float(ch_score)
                },
                'cluster_statistics': cluster_stats,
                'summary': {
                    'total_samples': len(data),
                    'features_analyzed': list(numeric_columns),
                    'parameters': {
                        'n_clusters': n_clusters,
                        'random_state': self.random_state
                    }
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in cluster analysis: {str(e)}")
            raise AnalysisError(f"クラスター分析に失敗しました: {str(e)}")

async def analyze_clusters(
    collection: str,
    target_columns: List[str],
    n_clusters: Optional[int] = None,
    max_clusters: int = 10,
    random_state: int = 42,
    filters: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    クラスター分析を実行するヘルパー関数

    Args:
        collection (str): 分析対象のコレクション名
        target_columns (List[str]): 分析対象の列名リスト
        n_clusters (Optional[int]): クラスター数
        max_clusters (int): クラスター数の最大値
        random_state (int): 乱数シード
        filters (Optional[List[Dict[str, Any]]]): データ取得時のフィルター条件

    Returns:
        Dict[str, Any]: 分析結果
    """
    try:
        analyzer = ClusterAnalyzer(
            n_clusters=n_clusters,
            max_clusters=max_clusters,
            random_state=random_state
        )

        # データの取得
        data = await analyzer.fetch_data(
            collection=collection,
            filters=filters
        )

        # 分析の実行と結果の保存
        results = await analyzer.analyze_and_save(
            data=data,
            target_columns=target_columns
        )

        return results

    except Exception as e:
        raise AnalysisError(f"クラスター分析の実行に失敗しました: {str(e)}")

async def analyze_clusters_request(request: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    クラスター分析のリクエストを処理

    Args:
        request (Dict[str, Any]): リクエストデータ

    Returns:
        Tuple[Dict[str, Any], int]: (レスポンス, ステータスコード)
    """
    try:
        # 必須パラメータの確認
        required_params = ['collection', 'target_columns']
        missing_params = [param for param in required_params if param not in request]
        if missing_params:
            return {
                'status': 'error',
                'message': f'必須パラメータが不足しています: {", ".join(missing_params)}'
            }, 400

        # サービスの初期化
        firestore_client = FirestoreClient()
        analyzer = ClusterAnalyzer(
            n_clusters=request.get('n_clusters'),
            max_clusters=request.get('max_clusters', 10),
            random_state=request.get('random_state', 42)
        )

        # データの取得
        data = await analyzer.fetch_data(
            collection=request['collection'],
            filters=request.get('filters')
        )

        # 分析の実行と結果の保存
        results = await analyzer.analyze_and_save(
            data=data,
            target_columns=request['target_columns']
        )

        return {
            'status': 'success',
            'results': results
        }, 200

    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }, 500