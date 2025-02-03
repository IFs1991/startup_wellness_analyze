# -*- coding: utf-8 -*-

"""
主成分分析モジュール
多次元データの次元削減と主要な特徴の抽出を行います。
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from . import BaseAnalyzer, AnalysisError

class PCAAnalyzer(BaseAnalyzer):
    """主成分分析を行うクラス"""

    def __init__(
        self,
        n_components: Optional[int] = None,
        explained_variance_ratio_threshold: float = 0.95
    ):
        """
        初期化メソッド

        Args:
            n_components (Optional[int]): 主成分数（Noneの場合は分散説明率に基づいて自動決定）
            explained_variance_ratio_threshold (float): 累積寄与率の閾値
        """
        super().__init__("pca_analysis")
        self.n_components = n_components
        self.explained_variance_ratio_threshold = explained_variance_ratio_threshold
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

    def _determine_n_components(self, data: np.ndarray) -> int:
        """
        最適な主成分数を決定

        Args:
            data (np.ndarray): 分析対象データ

        Returns:
            int: 主成分数
        """
        if self.n_components is not None:
            return min(self.n_components, data.shape[1])

        # 累積寄与率に基づいて主成分数を決定
        pca_full = PCA()
        pca_full.fit(data)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= self.explained_variance_ratio_threshold) + 1

        return n_components

    async def analyze(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        主成分分析を実行

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

            # 主成分数の決定
            n_components = self._determine_n_components(preprocessed_data)

            # PCAの実行
            pca = PCA(n_components=n_components)
            pc_scores = pca.fit_transform(preprocessed_data)

            # 結果の整形
            result = {
                'n_components': n_components,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'components': [
                    {
                        'pc_number': i + 1,
                        'loadings': {
                            feature: float(loading)
                            for feature, loading in zip(numeric_columns, pca.components_[i])
                        }
                    }
                    for i in range(n_components)
                ],
                'pc_scores': [
                    {
                        'index': i,
                        'scores': {
                            f'PC{j+1}': float(score)
                            for j, score in enumerate(scores)
                        }
                    }
                    for i, scores in enumerate(pc_scores)
                ],
                'summary': {
                    'total_features': len(numeric_columns),
                    'features_analyzed': list(numeric_columns),
                    'total_variance_explained': float(sum(pca.explained_variance_ratio_)),
                    'parameters': {
                        'n_components': n_components,
                        'explained_variance_ratio_threshold': self.explained_variance_ratio_threshold
                    }
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in PCA analysis: {str(e)}")
            raise AnalysisError(f"主成分分析に失敗しました: {str(e)}")

async def analyze_pca(
    collection: str,
    target_columns: List[str],
    n_components: Optional[int] = None,
    explained_variance_ratio_threshold: float = 0.95,
    filters: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    主成分分析を実行するヘルパー関数

    Args:
        collection (str): 分析対象のコレクション名
        target_columns (List[str]): 分析対象の列名リスト
        n_components (Optional[int]): 主成分数
        explained_variance_ratio_threshold (float): 累積寄与率の閾値
        filters (Optional[List[Dict[str, Any]]]): データ取得時のフィルター条件

    Returns:
        Dict[str, Any]: 分析結果
    """
    try:
        analyzer = PCAAnalyzer(
            n_components=n_components,
            explained_variance_ratio_threshold=explained_variance_ratio_threshold
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
        raise AnalysisError(f"主成分分析の実行に失敗しました: {str(e)}")

async def analyze_pca_request(request: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    主成分分析のリクエストを処理

    Args:
        request (Dict[str, Any]): リクエストデータ

    Returns:
        Tuple[Dict[str, Any], int]: (レスポンス, ステータスコード)
    """
    try:
        # 必須パラメータの確認
        required_params = ['collection', 'target_columns']
        for param in required_params:
            if param not in request:
                return {
                    'status': 'error',
                    'message': f'必須パラメータ {param} が指定されていません'
                }, 400

        # オプションパラメータの取得
        n_components = request.get('n_components')
        if n_components is not None:
            n_components = int(n_components)
        explained_variance_ratio_threshold = float(request.get('explained_variance_ratio_threshold', 0.95))
        filters = request.get('filters')

        # 分析の実行
        results = await analyze_pca(
            collection=request['collection'],
            target_columns=request['target_columns'],
            n_components=n_components,
            explained_variance_ratio_threshold=explained_variance_ratio_threshold,
            filters=filters
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