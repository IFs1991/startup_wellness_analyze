# -*- coding: utf-8 -*-

"""
相関分析モジュール
データセット内の変数間の相関関係を分析します。
"""

from typing import List, Optional, Tuple, Any, Dict
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from . import BaseAnalyzer, AnalysisError

class CorrelationAnalyzer(BaseAnalyzer):
    """相関分析を行うクラス"""

    def __init__(
        self,
        method: str = 'pearson',
        threshold: float = 0.3,
        p_value_threshold: float = 0.05
    ):
        """
        初期化メソッド

        Args:
            method (str): 相関係数の計算方法（'pearson', 'spearman', 'kendall'）
            threshold (float): 相関係数の閾値
            p_value_threshold (float): p値の閾値
        """
        super().__init__("correlation_analysis")
        self.method = method
        self.threshold = threshold
        self.p_value_threshold = p_value_threshold

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

        return True

    def _calculate_correlation_matrix(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        相関行列とp値行列を計算

        Args:
            data (pd.DataFrame): 分析対象データ

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (相関行列, p値行列)
        """
        numeric_data = data.select_dtypes(include=[np.number])
        n = len(numeric_data.columns)
        p_values = pd.DataFrame(np.zeros((n, n)), columns=numeric_data.columns, index=numeric_data.columns)

        for i in range(n):
            for j in range(i+1, n):
                if self.method == 'pearson':
                    corr, p_value = stats.pearsonr(numeric_data.iloc[:, i], numeric_data.iloc[:, j])
                elif self.method == 'spearman':
                    corr, p_value = stats.spearmanr(numeric_data.iloc[:, i], numeric_data.iloc[:, j])
                else:  # kendall
                    corr, p_value = stats.kendalltau(numeric_data.iloc[:, i], numeric_data.iloc[:, j])

                p_values.iloc[i, j] = p_value
                p_values.iloc[j, i] = p_value

        corr_matrix = numeric_data.corr(method=self.method)
        return corr_matrix, p_values

    async def analyze(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        相関分析を実行

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

            # 数値型のカラムのみを抽出
            numeric_data = data.select_dtypes(include=[np.number])

            # 相関行列とp値行列の計算
            corr_matrix, p_values = self._calculate_correlation_matrix(numeric_data)

            # 有意な相関関係の抽出
            significant_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr = corr_matrix.iloc[i, j]
                    p_value = p_values.iloc[i, j]

                    if abs(corr) >= self.threshold and p_value <= self.p_value_threshold:
                        significant_correlations.append({
                            'variable1': corr_matrix.columns[i],
                            'variable2': corr_matrix.columns[j],
                            'correlation': float(corr),
                            'p_value': float(p_value)
                        })

            # 結果の整形
            result = {
                'correlation_matrix': corr_matrix.to_dict('records'),
                'p_value_matrix': p_values.to_dict('records'),
                'significant_correlations': significant_correlations,
                'summary': {
                    'total_variables': len(numeric_data.columns),
                    'significant_pairs': len(significant_correlations),
                    'method': self.method,
                    'parameters': {
                        'correlation_threshold': self.threshold,
                        'p_value_threshold': self.p_value_threshold
                    },
                    'variables_analyzed': list(numeric_data.columns)
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {str(e)}")
            raise AnalysisError(f"相関分析に失敗しました: {str(e)}")

async def analyze_correlations(
    collection: str,
    target_columns: List[str],
    method: str = 'pearson',
    threshold: float = 0.3,
    p_value_threshold: float = 0.05,
    filters: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    相関分析を実行するヘルパー関数

    Args:
        collection (str): 分析対象のコレクション名
        target_columns (List[str]): 分析対象の列名リスト
        method (str): 相関係数の計算方法
        threshold (float): 相関係数の閾値
        p_value_threshold (float): p値の閾値
        filters (Optional[List[Dict[str, Any]]]): データ取得時のフィルター条件

    Returns:
        Dict[str, Any]: 分析結果
    """
    try:
        analyzer = CorrelationAnalyzer(
            method=method,
            threshold=threshold,
            p_value_threshold=p_value_threshold
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
        raise AnalysisError(f"相関分析の実行に失敗しました: {str(e)}")

async def analyze_correlation_request(request: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    相関分析のリクエストを処理

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
        method = request.get('method', 'pearson')
        threshold = float(request.get('threshold', 0.3))
        p_value_threshold = float(request.get('p_value_threshold', 0.05))
        filters = request.get('filters')

        # 分析の実行
        results = await analyze_correlations(
            collection=request['collection'],
            target_columns=request['target_columns'],
            method=method,
            threshold=threshold,
            p_value_threshold=p_value_threshold,
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