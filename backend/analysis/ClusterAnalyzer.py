# -*- coding: utf-8 -*-
"""
Cluster Analysis with BigQueryService Integration

This code implements data analysis functionality with BigQueryService integration,
supporting asynchronous processing and Cloud Functions deployment.
"""

from typing import Optional, Tuple, Any, Dict, List, Union
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import logging
import networkx as nx
import asyncio
from service.bigquery.client import BigQueryService
from service.firestore.client import FirestoreService
from .base import BaseAnalyzer, AnalysisError
from .CausalStructureAnalyzer import CausalStructureAnalyzer

class DataAnalyzer:
    def __init__(self, bq_service: BigQueryService):
        """
        Initialize the DataAnalyzer with BigQueryService.

        Args:
            bq_service: Initialized BigQueryService instance
        """
        self.bq_service = bq_service
        self.logger = logging.getLogger(__name__)
        # 因果構造分析用のアナライザー
        self.causal_analyzer = CausalStructureAnalyzer()

    def _validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validate the input data.

        Args:
            data: Input DataFrame to validate

        Returns:
            Tuple containing validation result and error message if any
        """
        if data is None or data.empty:
            return False, "データが空です"

        # 少なくとも数値列が2つ以上あることを確認
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) < 2:
            return False, "クラスタリングには少なくとも2つの数値列が必要です"

        return True, None

    async def analyze(self,
                     query: str,
                     save_results: bool = True,
                     dataset_id: Optional[str] = None,
                     table_id: Optional[str] = None,
                     algorithm: str = 'kmeans',
                     n_clusters: int = 3,
                     **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform data analysis using BigQuery.

        Args:
            query: BigQuery query to fetch data
            save_results: Flag to control result saving
            dataset_id: Target dataset ID for saving results
            table_id: Target table ID for saving results
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
            n_clusters: Number of clusters
            **kwargs: Additional parameters

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

            # 数値列の抽出
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
            features = data[numeric_cols].copy()

            # 欠損値の処理
            features = features.fillna(features.mean())

            # 特徴量のスケーリング
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            # クラスタリングの実行
            if algorithm == 'kmeans':
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            elif algorithm == 'dbscan':
                eps = kwargs.get('eps', 0.5)
                min_samples = kwargs.get('min_samples', 5)
                clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            elif algorithm == 'hierarchical':
                linkage = kwargs.get('linkage', 'ward')
                clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            else:
                raise ValueError(f"未対応のアルゴリズムです: {algorithm}")

            # クラスタリング実行
            cluster_labels = clusterer.fit_predict(scaled_features)

            # 結果をデータフレームに追加
            data['cluster'] = cluster_labels

            # クラスタの基本統計量
            cluster_stats = {}
            for cluster_id in np.unique(cluster_labels):
                cluster_data = data[data['cluster'] == cluster_id]
                cluster_stats[int(cluster_id)] = {
                    'count': len(cluster_data),
                    'mean': dict(cluster_data[numeric_cols].mean()),
                    'std': dict(cluster_data[numeric_cols].std())
                }

            # PCAで2次元に次元削減して可視化
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_features)

            # 可視化用データ
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Cluster')
            plt.title('クラスタリング結果 (PCA)')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.grid(True, linestyle='--', alpha=0.7)

            # 画像をBase64エンコード
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            # 分析結果の作成
            analysis_result = {
                'algorithm': algorithm,
                'n_clusters': n_clusters,
                'cluster_stats': cluster_stats,
                'visualization': img_str,
                'timestamp': datetime.now().isoformat()
            }

            # 分析結果の保存
            if save_results and dataset_id and table_id:
                await self.bq_service.save_results(
                    data,
                    dataset_id=dataset_id,
                    table_id=table_id,
                    write_disposition='WRITE_TRUNCATE'
                )

            # メタデータの作成
            metadata = {
                'row_count': len(data),
                'columns_analyzed': list(data.columns),
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'query_executed': query
            }

            return data, analysis_result

        except Exception as e:
            raise RuntimeError(f"分析実行中にエラーが発生しました: {str(e)}")

    async def causal_cluster_analysis(self,
                                    data: pd.DataFrame,
                                    n_clusters: int = 3,
                                    method: str = 'pc',
                                    **kwargs) -> Dict[str, Any]:
        """
        クラスタごとに因果構造分析を実行

        Args:
            data (pd.DataFrame): 分析対象のデータ
            n_clusters (int): クラスタ数
            method (str): 因果構造学習手法 ('pc', 'hill_climb', 'expert')
            **kwargs: 追加のパラメータ
                - significance_level (float): 有意水準
                - max_cond_vars (int): 条件付き独立性テストの最大変数数
                - variables (List[str]): 分析対象の変数リスト

        Returns:
            Dict[str, Any]: クラスタごとの因果分析結果
        """
        # データの検証
        is_valid, error_msg = self._validate_data(data)
        if not is_valid:
            self.logger.error(f"データ検証エラー: {error_msg}")
            raise AnalysisError(error_msg)

        # 数値列の抽出
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

        # 分析対象の変数を制限（オプション）
        variables = kwargs.get('variables', list(numeric_cols))
        features = data[variables].copy()

        # 欠損値の処理
        features = features.fillna(features.mean())

        # 特徴量のスケーリング
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # クラスタリングの実行
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(scaled_features)

        # 結果をデータフレームに追加
        features_with_cluster = features.copy()
        features_with_cluster['cluster'] = cluster_labels

        # クラスタごとの因果分析
        cluster_causal_results = {}
        cluster_graphs = {}

        # クラスタごとに因果分析を実行
        for cluster_id in np.unique(cluster_labels):
            cluster_data = features_with_cluster[features_with_cluster['cluster'] == cluster_id].drop(columns=['cluster'])

            # クラスタのサンプル数が少ない場合はスキップ
            if len(cluster_data) < 10:
                self.logger.warning(f"クラスタ {cluster_id} のサンプル数が不足しています ({len(cluster_data)} < 10)。スキップします。")
                continue

            # 因果構造分析
            try:
                causal_results = await self.causal_analyzer.analyze(
                    data=cluster_data,
                    method=method,
                    **kwargs
                )
                cluster_causal_results[int(cluster_id)] = causal_results

                # グラフの構築
                graph = nx.DiGraph()
                graph.add_nodes_from(causal_results['nodes'])
                for edge in causal_results['edges']:
                    graph.add_edge(edge['from'], edge['to'])
                cluster_graphs[int(cluster_id)] = graph

            except Exception as e:
                self.logger.error(f"クラスタ {cluster_id} の因果分析中にエラーが発生しました: {str(e)}")
                cluster_causal_results[int(cluster_id)] = {"error": str(e)}

        # クラスタ間の共通構造分析
        common_structures = self._analyze_common_structures(cluster_graphs)

        # クラスタの可視化
        visualization = self._visualize_clusters_with_causal_structures(
            data=scaled_features,
            cluster_labels=cluster_labels,
            cluster_graphs=cluster_graphs
        )

        # 結果の整形
        result = {
            'n_clusters': n_clusters,
            'method': method,
            'cluster_sizes': {int(cluster_id): sum(cluster_labels == cluster_id) for cluster_id in np.unique(cluster_labels)},
            'cluster_causal_results': cluster_causal_results,
            'common_structures': common_structures,
            'visualization': visualization,
            'timestamp': datetime.now().isoformat()
        }

        return result

    def _analyze_common_structures(self, cluster_graphs: Dict[int, nx.DiGraph]) -> Dict[str, Any]:
        """
        クラスタ間の共通因果構造を分析

        Args:
            cluster_graphs (Dict[int, nx.DiGraph]): クラスタごとの因果グラフ

        Returns:
            Dict[str, Any]: 共通構造の分析結果
        """
        if not cluster_graphs:
            return {"error": "有効なグラフがありません"}

        # 全てのグラフで共通するノードを抽出
        common_nodes = set.intersection(
            *[set(graph.nodes()) for graph in cluster_graphs.values()]
        )

        # 共通エッジを抽出
        common_edges = []
        for edge in list(next(iter(cluster_graphs.values())).edges()):
            if all(edge in graph.edges() for graph in cluster_graphs.values()):
                common_edges.append(edge)

        # クラスタごとの特有エッジ
        unique_edges = {}
        for cluster_id, graph in cluster_graphs.items():
            edges = list(graph.edges())
            unique = [edge for edge in edges if edge not in common_edges]
            unique_edges[cluster_id] = unique

        # 中心性の高いノード
        centrality_scores = {}
        for cluster_id, graph in cluster_graphs.items():
            try:
                centrality = nx.betweenness_centrality(graph)
                centrality_scores[cluster_id] = {
                    k: v for k, v in sorted(
                        centrality.items(),
                        key=lambda item: item[1],
                        reverse=True
                    )[:5]  # 上位5つのノード
                }
            except Exception:
                centrality_scores[cluster_id] = {}

        return {
            "common_nodes": list(common_nodes),
            "common_edges": common_edges,
            "unique_edges": unique_edges,
            "top_centrality": centrality_scores
        }

    def _visualize_clusters_with_causal_structures(self,
                                              data: np.ndarray,
                                              cluster_labels: np.ndarray,
                                              cluster_graphs: Dict[int, nx.DiGraph]) -> str:
        """
        クラスタと因果構造を可視化

        Args:
            data (np.ndarray): スケーリング済みの特徴量データ
            cluster_labels (np.ndarray): クラスタラベル
            cluster_graphs (Dict[int, nx.DiGraph]): クラスタごとの因果グラフ

        Returns:
            str: Base64エンコードされた画像
        """
        # PCAで2次元に次元削減
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data)

        # サブプロットの設定
        n_clusters = len(set(cluster_labels))
        fig, axs = plt.subplots(1, n_clusters+1, figsize=(5*(n_clusters+1), 5))

        # 全体のクラスタプロット
        scatter = axs[0].scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
        axs[0].set_title('全クラスタ (PCA)')
        axs[0].set_xlabel('PC1')
        axs[0].set_ylabel('PC2')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        plt.colorbar(scatter, ax=axs[0], label='クラスタID')

        # クラスタごとの因果グラフプロット
        for i, cluster_id in enumerate(sorted(cluster_graphs.keys()), 1):
            if i < len(axs):
                graph = cluster_graphs[cluster_id]
                pos = nx.spring_layout(graph, seed=42)
                nx.draw_networkx(
                    graph,
                    pos=pos,
                    ax=axs[i],
                    with_labels=True,
                    node_color='lightblue',
                    node_size=500,
                    edge_color='gray',
                    arrows=True,
                    font_size=8,
                    font_weight='bold'
                )
                axs[i].set_title(f'クラスタ {cluster_id} の因果構造')
                axs[i].axis('off')

        plt.tight_layout()

        # 画像をBase64エンコード
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return img_str

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
        algorithm = request_json.get('algorithm', 'kmeans')
        n_clusters = request_json.get('n_clusters', 3)

        # 分析の実行
        results, metadata = await analyzer.analyze(
            query=query,
            save_results=save_results,
            dataset_id=dataset_id,
            table_id=table_id,
            algorithm=algorithm,
            n_clusters=n_clusters,
            **request_json
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

async def analyze_causal_clusters(request: Any) -> Tuple[Dict, int]:
    """
    クラスタごとの因果構造分析APIエンドポイント

    Args:
        request: リクエストオブジェクト

    Returns:
        Tuple[Dict, int]: (レスポンス, ステータスコード)
    """
    try:
        # BigQueryサービスの初期化
        bq_service = BigQueryService()

        # データアナライザーの初期化
        analyzer = DataAnalyzer(bq_service)

        # リクエストパラメータの取得
        query = request.get('query')
        n_clusters = request.get('n_clusters', 3)
        method = request.get('method', 'pc')

        # データの取得
        data = await bq_service.execute_query(query)

        # クラスタごとの因果分析の実行
        result = await analyzer.causal_cluster_analysis(
            data=data,
            n_clusters=n_clusters,
            method=method,
            **request
        )

        return {"result": result, "status": "success"}, 200
    except AnalysisError as e:
        return {"error": str(e), "status": "error"}, 400
    except Exception as e:
        logging.error(f"因果クラスタ分析中に予期しないエラーが発生しました: {str(e)}")
        return {"error": f"予期しないエラー: {str(e)}", "status": "error"}, 500