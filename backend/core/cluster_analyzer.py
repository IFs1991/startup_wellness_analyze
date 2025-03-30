# backend/core/cluster_analyzer.py
from service.firestore.client import FirestoreService, StorageError
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from .utils import PlotUtility, StatisticsUtility, AnalysisError

logger = logging.getLogger(__name__)

class ClusterAnalysisError(AnalysisError):
    """クラスタリング分析に関するエラー"""
    pass

class ClusterAnalyzer:
    def __init__(self):
        """Initialize the ClusterAnalyzer with Firestore service"""
        try:
            self.firestore_service = FirestoreService()
            self.collection_name = 'cluster_analysis'
            logger.info("ClusterAnalyzer initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize ClusterAnalyzer: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

    async def analyze(
        self,
        data: pd.DataFrame,
        n_clusters: Optional[int] = None,
        method: str = 'kmeans',
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        standardize: bool = True,
        generate_plots: bool = True
    ) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
        """
        クラスタ分析を実行し、結果をFirestoreに保存

        Args:
            data: 分析対象のデータ
            n_clusters: クラスタ数（kmeansとagglomerativeの場合は必須）
            method: クラスタリング手法 ('kmeans', 'dbscan', 'agglomerative')
            user_id: 分析を実行したユーザーのID
            metadata: 追加のメタデータ
            standardize: データを標準化するかどうか
            generate_plots: 可視化を生成するかどうか

        Returns:
            Tuple[pd.DataFrame, str, Dict[str, Any]]:
                (クラスタリング結果のDataFrame, FirestoreのドキュメントID, 分析結果の詳細)

        Raises:
            ClusterAnalysisError: 分析処理中にエラーが発生した場合
            StorageError: Firestoreへの保存時にエラーが発生した場合
            ValueError: 入力データやパラメータが不正な場合
        """
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("データはpandas DataFrameである必要があります")

            if data.empty:
                raise ValueError("データが空です")

            # 数値データのみを抽出
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                raise ClusterAnalysisError("数値データが見つかりませんでした")

            # 欠損値の処理
            if numeric_data.isnull().values.any():
                logger.warning("入力データに欠損値が含まれています。列の平均値で補完します。")
                numeric_data = numeric_data.fillna(numeric_data.mean())

            # データの標準化
            if standardize:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
            else:
                scaled_data = numeric_data.values

            # クラスタリングモデルの選択と実行
            if method == 'kmeans':
                if n_clusters is None or n_clusters < 2:
                    raise ValueError("kmeansには正の整数のクラスタ数が必要です")
                model = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = model.fit_predict(scaled_data)
                model_params = {'n_clusters': n_clusters}
                model_attributes = {'inertia': float(model.inertia_), 'centers': model.cluster_centers_.tolist()}

            elif method == 'dbscan':
                # DBSCANのパラメータを自動設定
                eps = self._estimate_eps(scaled_data) if 'eps' not in metadata else metadata.get('eps')
                min_samples = int(np.log(len(scaled_data))) if 'min_samples' not in metadata else metadata.get('min_samples')
                model = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = model.fit_predict(scaled_data)
                model_params = {'eps': eps, 'min_samples': min_samples}
                model_attributes = {'n_clusters': len(set([x for x in cluster_labels if x >= 0]))}

            elif method == 'agglomerative':
                if n_clusters is None or n_clusters < 2:
                    raise ValueError("agglomerativeクラスタリングには正の整数のクラスタ数が必要です")
                linkage = metadata.get('linkage', 'ward')
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                cluster_labels = model.fit_predict(scaled_data)
                model_params = {'n_clusters': n_clusters, 'linkage': linkage}
                model_attributes = {}

            else:
                raise ValueError(f"未サポートのクラスタリング手法です: {method}")

            # クラスタラベルをDataFrameに追加
            result_df = data.copy()
            result_df['cluster'] = cluster_labels

            # クラスタリング評価指標
            eval_metrics = {}
            try:
                if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
                    eval_metrics['silhouette_score'] = float(silhouette_score(scaled_data, cluster_labels))
                    eval_metrics['davies_bouldin_score'] = float(davies_bouldin_score(scaled_data, cluster_labels))
                    eval_metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(scaled_data, cluster_labels))
            except Exception as e:
                logger.warning(f"評価指標の計算中にエラーが発生しました: {str(e)}")

            # クラスタごとの統計情報
            cluster_stats = self._calculate_cluster_statistics(result_df, numeric_data.columns)

            # 可視化の生成（オプション）
            plots = {}
            if generate_plots:
                plots = self._generate_cluster_visualizations(scaled_data, cluster_labels, method, numeric_data.columns)

            # 保存用データの準備
            analysis_result = {
                'model_type': method,
                'model_params': model_params,
                'model_attributes': model_attributes,
                'user_id': user_id,
                'data_shape': list(data.shape),
                'feature_names': numeric_data.columns.tolist(),
                'cluster_labels': cluster_labels.tolist(),
                'cluster_counts': {str(k): int(v) for k, v in pd.Series(cluster_labels).value_counts().items()},
                'evaluation_metrics': eval_metrics,
                'cluster_statistics': cluster_stats,
                'metadata': metadata or {},
                'created_at': datetime.now().isoformat(),
            }

            # 可視化データを追加
            if plots:
                analysis_result['plots'] = plots

            # Firestoreに保存
            doc_ids = await self.firestore_service.save_results(
                results=[analysis_result],
                collection_name=self.collection_name
            )

            if not doc_ids:
                raise StorageError("結果をFirestoreに保存できませんでした")

            doc_id = doc_ids[0]

            # 分析IDを結果に追加
            result_df['analysis_id'] = doc_id

            logger.info(f"Successfully saved cluster analysis results with ID: {doc_id}")
            return result_df, doc_id, analysis_result

        except Exception as e:
            error_msg = f"クラスタ分析中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (ClusterAnalysisError, StorageError, ValueError)):
                raise
            raise ClusterAnalysisError(error_msg) from e

    def _estimate_eps(self, data: np.ndarray) -> float:
        """
        DBSCANのeps値を自動推定する

        Args:
            data: スケーリング済みデータ

        Returns:
            float: 推定されたeps値
        """
        try:
            from sklearn.neighbors import NearestNeighbors

            # knnクラスで近傍点間の距離を計算
            k = min(len(data) - 1, 5)  # 少なくとも2、データサイズが小さい場合は調整
            nbrs = NearestNeighbors(n_neighbors=k).fit(data)
            distances, _ = nbrs.kneighbors(data)

            # k番目の距離を取得してソート
            k_dist = np.sort(distances[:, -1])

            # エルボー法によるeps値の推定
            # 変化率が急激に変わる点を見つける簡易的な方法
            acceleration = np.diff(np.diff(k_dist))
            acceleration_idx = np.argmax(acceleration) + 1
            eps = k_dist[acceleration_idx]

            logger.info(f"Estimated eps value: {eps}")
            return eps
        except Exception as e:
            logger.warning(f"eps値の自動推定に失敗しました: {str(e)}")
            return 0.5  # デフォルト値

    def _calculate_cluster_statistics(
        self,
        data: pd.DataFrame,
        feature_columns: List[str]
    ) -> Dict[str, Any]:
        """
        クラスタごとの統計情報を計算する

        Args:
            data: クラスタラベル付きデータフレーム
            feature_columns: 特徴量列のリスト

        Returns:
            Dict[str, Any]: クラスタごとの統計情報
        """
        try:
            cluster_stats = {}

            for cluster_id in data['cluster'].unique():
                cluster_data = data[data['cluster'] == cluster_id]

                # クラスタのサイズ
                cluster_size = len(cluster_data)

                # 特徴量ごとの統計量を計算
                feature_stats = {}
                for feature in feature_columns:
                    feature_values = cluster_data[feature].values
                    stats = StatisticsUtility.calculate_descriptive_statistics(
                        feature_values, include_percentiles=False
                    )
                    feature_stats[feature] = stats

                cluster_stats[str(cluster_id)] = {
                    'size': cluster_size,
                    'percentage': cluster_size / len(data) * 100,
                    'feature_statistics': feature_stats
                }

            return cluster_stats
        except Exception as e:
            logger.warning(f"クラスタ統計情報の計算中にエラーが発生しました: {str(e)}")
            return {}

    def _generate_cluster_visualizations(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        method: str,
        feature_names: List[str]
    ) -> Dict[str, str]:
        """
        クラスタリング結果の可視化を生成する

        Args:
            data: スケーリング済みデータ
            labels: クラスタラベル
            method: クラスタリング手法
            feature_names: 特徴量名

        Returns:
            Dict[str, str]: Base64エンコードされた可視化画像
        """
        plots = {}

        try:
            # クラスタサイズの分布
            fig, ax = plt.subplots(figsize=(10, 6))
            cluster_counts = pd.Series(labels).value_counts().sort_index()

            # 異常値クラスタ(-1)は別の色で表示
            if -1 in cluster_counts.index:
                normal_counts = cluster_counts[cluster_counts.index != -1]
                ax.bar(normal_counts.index, normal_counts.values, color='skyblue')
                ax.bar(-1, cluster_counts.get(-1, 0), color='salmon')
            else:
                ax.bar(cluster_counts.index, cluster_counts.values, color='skyblue')

            ax.set_xlabel('クラスタID')
            ax.set_ylabel('データ数')
            ax.set_title(f'{method.upper()}クラスタリング - クラスタサイズ分布')

            # メソッド名でラベルを綺麗に整形
            method_labels = {
                'kmeans': 'K-means',
                'dbscan': 'DBSCAN',
                'agglomerative': '階層的クラスタリング'
            }
            method_name = method_labels.get(method, method.upper())

            ax.set_xticks(np.sort(list(cluster_counts.index)))
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            plots['cluster_size_distribution'] = PlotUtility.save_plot_to_base64(fig)

            # 次元削減して2Dプロット（データ点が多い場合）
            if data.shape[1] > 2:
                # PCAで2次元に削減
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(data)

                # 2次元散布図
                fig = PlotUtility.generate_scatter_plot(
                    x_data=reduced_data[:, 0],
                    y_data=reduced_data[:, 1],
                    title=f'{method_name}クラスタリング (PCA次元削減)',
                    x_label=f'第1主成分 ({pca.explained_variance_ratio_[0]:.2%})',
                    y_label=f'第2主成分 ({pca.explained_variance_ratio_[1]:.2%})',
                    hue=labels
                )
                plots['cluster_pca_2d'] = PlotUtility.save_plot_to_base64(fig)

            # データが2次元または3次元の場合は元の空間でプロット
            elif data.shape[1] == 2:
                fig = PlotUtility.generate_scatter_plot(
                    x_data=data[:, 0],
                    y_data=data[:, 1],
                    title=f'{method_name}クラスタリング',
                    x_label=feature_names[0] if len(feature_names) > 0 else 'Feature 1',
                    y_label=feature_names[1] if len(feature_names) > 1 else 'Feature 2',
                    hue=labels
                )
                plots['cluster_original_2d'] = PlotUtility.save_plot_to_base64(fig)

            # K-means専用の重心プロット
            if method == 'kmeans' and data.shape[1] == 2:
                # モデルを再構築して重心を取得
                kmeans = KMeans(n_clusters=len(set(labels)), random_state=42)
                kmeans.fit(data)
                centers = kmeans.cluster_centers_

                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
                ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')

                ax.set_title('K-means クラスタリング - クラスタと重心')
                ax.set_xlabel(feature_names[0] if len(feature_names) > 0 else 'Feature 1')
                ax.set_ylabel(feature_names[1] if len(feature_names) > 1 else 'Feature 2')
                ax.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()

                plots['kmeans_centroids'] = PlotUtility.save_plot_to_base64(fig)

            return plots

        except Exception as e:
            logger.warning(f"クラスタ可視化の生成中にエラーが発生しました: {str(e)}")
            return {}

    async def get_analysis_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        過去の分析結果を取得

        Args:
            user_id: 特定ユーザーの結果のみを取得する場合に指定
            limit: 取得する結果の最大数

        Returns:
            List[Dict]: 分析結果の履歴

        Raises:
            StorageError: Firestoreからのデータ取得に失敗した場合
        """
        try:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError("limitは正の整数である必要があります")

            conditions = []
            if user_id:
                conditions.append({
                    'field': 'user_id',
                    'operator': '==',
                    'value': user_id
                })

            results = await self.firestore_service.fetch_documents(
                collection_name=self.collection_name,
                conditions=conditions,
                limit=limit,
                order_by='created_at',
                direction='desc'
            )

            if results is None:
                return []

            logger.info(f"Retrieved {len(results)} analysis results")
            return results

        except Exception as e:
            error_msg = f"分析履歴の取得中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, StorageError):
                raise
            raise StorageError(error_msg) from e

    async def update_analysis_metadata(
        self,
        analysis_id: str,
        metadata: Dict
    ) -> None:
        """
        分析結果のメタデータを更新

        Args:
            analysis_id: 分析結果のID
            metadata: 更新するメタデータ

        Raises:
            StorageError: Firestoreへの更新に失敗した場合
        """
        try:
            if not analysis_id or not isinstance(analysis_id, str):
                raise ValueError("有効な分析IDを指定してください")

            if not metadata or not isinstance(metadata, dict):
                raise ValueError("メタデータは辞書形式で指定してください")

            await self.firestore_service.update_document(
                collection_name=self.collection_name,
                document_id=analysis_id,
                data={'metadata': metadata}
            )
            logger.info(f"Updated metadata for analysis {analysis_id}")

        except Exception as e:
            error_msg = f"メタデータの更新中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, StorageError):
                raise
            raise StorageError(error_msg) from e

    async def delete_analysis(self, analysis_id: str) -> None:
        """
        分析結果を削除

        Args:
            analysis_id: 削除する分析結果のID

        Raises:
            StorageError: Firestoreからの削除に失敗した場合
        """
        try:
            if not analysis_id or not isinstance(analysis_id, str):
                raise ValueError("有効な分析IDを指定してください")

            await self.firestore_service.delete_document(
                collection_name=self.collection_name,
                document_id=analysis_id
            )
            logger.info(f"Deleted analysis {analysis_id}")

        except Exception as e:
            error_msg = f"分析結果の削除中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, StorageError):
                raise
            raise StorageError(error_msg) from e