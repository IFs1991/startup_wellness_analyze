# -*- coding: utf-8 -*-
"""
主成分分析モジュール
多次元データを低次元に変換し、重要な特徴を抽出します。
Firestoreと統合したバージョン。
"""
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from service.firestore.client import FirestoreService, StorageError
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import PlotUtility, StatisticsUtility, AnalysisError

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class PCAAnalysisError(AnalysisError):
    """PCA分析に関するエラー"""
    pass

class FirestorePCAAnalyzer:
    """
    主成分分析を実行し、結果をFirestoreに保存するクラスです。
    """
    def __init__(self) -> None:
        """
        Firestoreサービスとの接続を初期化します。

        Raises:
            StorageError: Firestore接続の初期化に失敗した場合
        """
        try:
            self.firestore_service = FirestoreService()
            logger.info("FirestorePCAAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FirestorePCAAnalyzer: {str(e)}")
            raise StorageError(f"Failed to initialize FirestorePCAAnalyzer: {str(e)}") from e

    async def analyze_and_save(
        self,
        data: pd.DataFrame,
        n_components: int,
        analysis_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        standardize: bool = True,
        generate_plots: bool = True
    ) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
        """
        主成分分析を実行し、結果をFirestoreに保存します。

        Args:
            data (pd.DataFrame): 分析対象データ
            n_components (int): 抽出する主成分の数
            analysis_metadata (Optional[Dict[str, Any]], optional): 分析に関するメタデータ
            user_id (Optional[str], optional): 分析を実行したユーザーのID
            standardize (bool, optional): データを標準化するかどうか
            generate_plots (bool, optional): 可視化を生成するかどうか

        Returns:
            Tuple[pd.DataFrame, str, Dict[str, Any]]:
                (主成分分析結果のDataFrame, FirestoreのドキュメントID, 分析結果の詳細)

        Raises:
            PCAAnalysisError: 分析処理中にエラーが発生した場合
            StorageError: Firestoreへの保存時にエラーが発生した場合
            ValueError: 入力データが不正な場合
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")

        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer")

        try:
            logger.info(f"Starting PCA analysis with {n_components} components")

            # 入力データの検証
            if data.empty:
                raise PCAAnalysisError("Input data is empty")

            # 数値データのみを抽出
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                raise PCAAnalysisError("No numeric data found in the input DataFrame")

            if n_components > min(numeric_data.shape):
                logger.warning(
                    f"Requested {n_components} components, but data has only {min(numeric_data.shape)} dimensions. "
                    f"Adjusting n_components to {min(numeric_data.shape)}."
                )
                n_components = min(numeric_data.shape)

            # 欠損値の処理
            if numeric_data.isnull().values.any():
                logger.warning("Input data contains NaN values. They will be replaced with column means.")
                numeric_data = numeric_data.fillna(numeric_data.mean())

            # データの標準化
            if standardize:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
            else:
                scaled_data = numeric_data.values

            # PCA実行
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(scaled_data)

            # 主成分スコアをデータフレームに変換
            principal_df = pd.DataFrame(
                data=principal_components,
                columns=[f'principal_component_{i+1}' for i in range(n_components)]
            )

            # 可視化の生成（オプション）
            plots = {}
            if generate_plots:
                plots = self._generate_pca_visualizations(pca, principal_components, numeric_data.columns)

            # 特徴量重要度（成分負荷量）を計算
            feature_importance = self._calculate_feature_importance(pca, numeric_data.columns)

            # 分析結果の準備
            analysis_result: Dict[str, Any] = {
                'timestamp': datetime.now(),
                'n_components': n_components,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'input_shape': list(numeric_data.shape),  # tupleをlistに変換
                'component_names': principal_df.columns.tolist(),
                'feature_importance': feature_importance,
                'metadata': analysis_metadata or {},
            }

            # 可視化データを追加
            if plots:
                analysis_result['plots'] = plots

            if user_id is not None:
                analysis_result['user_id'] = user_id

            # 結果をFirestoreに保存
            doc_ids = await self.firestore_service.save_results(
                results=[analysis_result],
                collection_name='pca_analyses'
            )

            if not doc_ids:
                raise StorageError("Failed to save analysis results to Firestore")

            logger.info(f"PCA analysis completed and saved with document ID: {doc_ids[0]}")

            # 元のデータと主成分スコアを結合
            result_df = pd.concat([data, principal_df], axis=1)

            return result_df, doc_ids[0], analysis_result

        except Exception as e:
            error_msg = f"Error during PCA analysis: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, (PCAAnalysisError, StorageError, ValueError)):
                raise
            raise PCAAnalysisError(error_msg) from e

    def _calculate_feature_importance(
        self,
        pca: PCA,
        feature_names: List[str]
    ) -> Dict[str, List[float]]:
        """
        各特徴量の主成分に対する寄与度（成分負荷量）を計算します。

        Args:
            pca: 学習済みのPCAモデル
            feature_names: 特徴量名のリスト

        Returns:
            Dict[str, List[float]]: 特徴量寄与度のデータ
        """
        try:
            # 成分負荷量（各特徴量の各主成分への寄与度）を計算
            loadings = pca.components_.T

            # 特徴量ごとの主成分への寄与度
            feature_importances = {}
            for i, feature in enumerate(feature_names):
                feature_importances[feature] = loadings[i].tolist()

            return feature_importances
        except Exception as e:
            logger.warning(f"Error calculating feature importance: {str(e)}")
            return {}

    def _generate_pca_visualizations(
        self,
        pca: PCA,
        principal_components: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, str]:
        """
        PCA分析結果の可視化を生成します。

        Args:
            pca: 学習済みのPCAモデル
            principal_components: 主成分データ
            feature_names: 特徴量名のリスト

        Returns:
            Dict[str, str]: Base64エンコードされた可視化画像
        """
        plots = {}

        try:
            # 1. 寄与率の棒グラフ＋累積寄与率の折れ線グラフ
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # 寄与率の棒グラフ
            variance_ratio = pca.explained_variance_ratio_
            x = np.arange(len(variance_ratio)) + 1
            ax1.bar(x, variance_ratio, alpha=0.7, color='skyblue', label='Explained Variance')
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Explained Variance Ratio')
            ax1.set_xticks(x)

            # 累積寄与率の折れ線グラフ
            ax2 = ax1.twinx()
            cumulative_variance = np.cumsum(variance_ratio)
            ax2.plot(x, cumulative_variance, 'r-', marker='o', label='Cumulative Variance')
            ax2.set_ylabel('Cumulative Explained Variance')
            ax2.axhline(y=0.8, color='grey', linestyle='--', alpha=0.7, label='80% Threshold')

            # 凡例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            plt.title('Explained Variance by Principal Components')
            plt.tight_layout()

            # Base64エンコード
            plots['variance_plot'] = PlotUtility.save_plot_to_base64(fig)

            # 2. 最初の2つの主成分のスキャッタープロット（データが十分ある場合）
            if principal_components.shape[1] >= 2:
                fig = PlotUtility.generate_scatter_plot(
                    x_data=principal_components[:, 0],
                    y_data=principal_components[:, 1],
                    title='First Two Principal Components',
                    x_label='Principal Component 1',
                    y_label='Principal Component 2'
                )
                plots['pc_scatter_plot'] = PlotUtility.save_plot_to_base64(fig)

            # 3. バイプロット（最初の2つの主成分に対する特徴量の寄与度）
            if pca.components_.shape[0] >= 2 and pca.components_.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(12, 10))

                # スケールを合わせるための正規化
                pca_components = pca.components_[:2, :]
                scaling = np.max(np.abs(principal_components[:, :2])) / np.max(np.abs(pca_components))

                # 散布図（最初の2つの主成分）
                ax.scatter(
                    principal_components[:, 0],
                    principal_components[:, 1],
                    alpha=0.3,
                    label='Samples'
                )

                # 特徴量ベクトル
                for i, (feature, color) in enumerate(zip(feature_names, plt.cm.tab10.colors)):
                    ax.arrow(
                        0, 0,  # 原点
                        pca.components_[0, i] * scaling * 0.8,  # X方向
                        pca.components_[1, i] * scaling * 0.8,  # Y方向
                        head_width=scaling * 0.05,
                        head_length=scaling * 0.08,
                        fc=color,
                        ec=color
                    )
                    ax.text(
                        pca.components_[0, i] * scaling * 0.85,
                        pca.components_[1, i] * scaling * 0.85,
                        feature,
                        color=color,
                        ha='center',
                        va='center'
                    )

                # 軸ラベルに寄与率を追加
                ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
                ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
                ax.set_title('PCA Biplot - Feature Contributions to Principal Components')

                # 原点に十字線を追加
                ax.axhline(y=0, color='grey', linestyle='--', alpha=0.3)
                ax.axvline(x=0, color='grey', linestyle='--', alpha=0.3)

                plt.tight_layout()
                plots['biplot'] = PlotUtility.save_plot_to_base64(fig)

            return plots

        except Exception as e:
            logger.warning(f"Error generating PCA visualizations: {str(e)}")
            return {}

    async def get_analysis_history(
        self,
        limit: int = 10,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        過去のPCA分析結果を取得します。

        Args:
            limit (int): 取得する結果の最大数
            user_id (Optional[str]): 特定ユーザーの結果のみを取得する場合のユーザーID

        Returns:
            List[Dict[str, Any]]: 分析結果の履歴

        Raises:
            StorageError: Firestoreからのデータ取得に失敗した場合
            ValueError: 不正なパラメータが指定された場合
        """
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")

        try:
            conditions = None
            if user_id is not None:
                conditions = [{'field': 'user_id', 'operator': '==', 'value': user_id}]

            results = await self.firestore_service.fetch_documents(
                collection_name='pca_analyses',
                conditions=conditions,
                limit=limit,
                order_by='timestamp',
                direction='desc'
            )

            if results is None:
                return []

            return results

        except Exception as e:
            error_msg = f"Error fetching PCA analysis history: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, StorageError):
                raise
            raise StorageError(error_msg) from e

    async def close(self) -> None:
        """
        リソースを解放します。

        Raises:
            StorageError: リソースの解放に失敗した場合
        """
        try:
            await self.firestore_service.close()
            logger.info("FirestorePCAAnalyzer closed successfully")
        except Exception as e:
            error_msg = f"Error closing FirestorePCAAnalyzer: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e

def create_pca_analyzer() -> FirestorePCAAnalyzer:
    """
    FirestorePCAAnalyzerのインスタンスを作成します。

    Returns:
        FirestorePCAAnalyzer: 初期化済みのアナライザーインスタンス

    Raises:
        StorageError: アナライザーの初期化に失敗した場合
    """
    return FirestorePCAAnalyzer()

# クラスの別名を定義
PCAAnalyzer = FirestorePCAAnalyzer