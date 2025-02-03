# -*- coding: utf-8 -*-

"""
回帰分析モジュール
変数間の関係性を分析し、予測モデルを構築します。
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from . import BaseAnalyzer, AnalysisError

class RegressionAnalyzer(BaseAnalyzer):
    """回帰分析を行うクラス"""

    def __init__(
        self,
        target_column: str,
        model_type: str = 'linear',
        test_size: float = 0.2,
        random_state: int = 42,
        **model_params
    ):
        """
        初期化メソッド

        Args:
            target_column (str): 目的変数の列名
            model_type (str): モデルの種類（'linear', 'ridge', 'lasso'）
            test_size (float): テストデータの割合
            random_state (int): 乱数シード
            **model_params: モデル固有のパラメータ
        """
        super().__init__("regression_analysis")
        self.target_column = target_column
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.model_params = model_params
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

        if self.target_column not in data.columns:
            raise AnalysisError(f"目的変数の列 {self.target_column} が存在しません")

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            raise AnalysisError("数値型の列が2つ以上必要です")

        if data.isnull().any().any():
            raise AnalysisError("欠損値が含まれています")

        return True

    def _prepare_data(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        データの前処理

        Args:
            data (pd.DataFrame): 前処理対象のデータ
            feature_columns (Optional[List[str]]): 説明変数の列名リスト

        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: 説明変数、目的変数、選択された特徴量名
        """
        # 説明変数の選択
        if feature_columns is None:
            feature_columns = [col for col in data.select_dtypes(include=[np.number]).columns
                             if col != self.target_column]

        X = data[feature_columns].values
        y = data[self.target_column].values

        # 標準化
        X = self.scaler.fit_transform(X)

        return X, y, feature_columns

    def _create_model(self) -> Any:
        """
        回帰モデルの作成

        Returns:
            Any: 作成されたモデル
        """
        if self.model_type == 'ridge':
            return Ridge(random_state=self.random_state, **self.model_params)
        elif self.model_type == 'lasso':
            return Lasso(random_state=self.random_state, **self.model_params)
        else:  # linear
            return LinearRegression(**self.model_params)

    def _calculate_feature_importance(
        self,
        model: Any,
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        特徴量の重要度を計算

        Args:
            model: 学習済みモデル
            feature_names (List[str]): 特徴量名のリスト

        Returns:
            List[Dict[str, Any]]: 特徴量の重要度情報
        """
        coefficients = model.coef_
        importance = []

        for name, coef in zip(feature_names, coefficients):
            importance.append({
                'feature': name,
                'coefficient': float(coef),
                'absolute_importance': float(abs(coef))
            })

        # 重要度順にソート
        importance.sort(key=lambda x: x['absolute_importance'], reverse=True)
        return importance

    async def analyze(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        回帰分析を実行

        Args:
            data (pd.DataFrame): 分析対象データ
            feature_columns (Optional[List[str]]): 説明変数の列名リスト
            **kwargs: 追加のパラメータ

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            # データの前処理
            X, y, selected_features = self._prepare_data(data, feature_columns)

            # データの分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state
            )

            # モデルの作成と学習
            model = self._create_model()
            model.fit(X_train, y_train)

            # 予測と評価
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # クロスバリデーションの実行
            cv_scores = cross_val_score(
                model, X, y,
                cv=5,
                scoring='r2'
            )

            # 特徴量の重要度計算
            feature_importance = self._calculate_feature_importance(
                model,
                selected_features
            )

            # 結果の整形
            result = {
                'model_performance': {
                    'train': {
                        'r2_score': float(r2_score(y_train, y_pred_train)),
                        'mse': float(mean_squared_error(y_train, y_pred_train)),
                        'mae': float(mean_absolute_error(y_train, y_pred_train)),
                        'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
                    },
                    'test': {
                        'r2_score': float(r2_score(y_test, y_pred_test)),
                        'mse': float(mean_squared_error(y_test, y_pred_test)),
                        'mae': float(mean_absolute_error(y_test, y_pred_test)),
                        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
                    }
                },
                'cross_validation': {
                    'mean_r2': float(cv_scores.mean()),
                    'std_r2': float(cv_scores.std()),
                    'scores': cv_scores.tolist()
                },
                'feature_importance': feature_importance,
                'model_parameters': {
                    'intercept': float(model.intercept_),
                    'type': self.model_type,
                    **self.model_params
                },
                'summary': {
                    'total_samples': len(data),
                    'train_samples': len(y_train),
                    'test_samples': len(y_test),
                    'features_used': selected_features,
                    'target_variable': self.target_column
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in regression analysis: {str(e)}")
            raise AnalysisError(f"回帰分析に失敗しました: {str(e)}")

async def analyze_regression(
    collection: str,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    model_type: str = 'linear',
    test_size: float = 0.2,
    random_state: int = 42,
    filters: Optional[List[Dict[str, Any]]] = None,
    **model_params
) -> Dict[str, Any]:
    """
    回帰分析を実行するヘルパー関数

    Args:
        collection (str): 分析対象のコレクション名
        target_column (str): 目的変数の列名
        feature_columns (Optional[List[str]]): 説明変数の列名リスト
        model_type (str): モデルの種類
        test_size (float): テストデータの割合
        random_state (int): 乱数シード
        filters (Optional[List[Dict[str, Any]]]): データ取得時のフィルター条件
        **model_params: モデル固有のパラメータ

    Returns:
        Dict[str, Any]: 分析結果
    """
    try:
        analyzer = RegressionAnalyzer(
            target_column=target_column,
            model_type=model_type,
            test_size=test_size,
            random_state=random_state,
            **model_params
        )

        # データの取得
        data = await analyzer.fetch_data(
            collection=collection,
            filters=filters
        )

        # 分析の実行と結果の保存
        results = await analyzer.analyze_and_save(
            data=data,
            feature_columns=feature_columns
        )

        return results

    except Exception as e:
        raise AnalysisError(f"回帰分析の実行に失敗しました: {str(e)}")

async def analyze_regression_request(request: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    回帰分析のリクエストを処理

    Args:
        request (Dict[str, Any]): リクエストデータ

    Returns:
        Tuple[Dict[str, Any], int]: (レスポンス, ステータスコード)
    """
    try:
        # 必須パラメータの確認
        required_params = ['collection', 'target_column']
        for param in required_params:
            if param not in request:
                return {
                    'status': 'error',
                    'message': f'必須パラメータ {param} が指定されていません'
                }, 400

        # オプションパラメータの取得
        feature_columns = request.get('feature_columns')
        model_type = request.get('model_type', 'linear')
        test_size = float(request.get('test_size', 0.2))
        random_state = int(request.get('random_state', 42))
        filters = request.get('filters')
        model_params = request.get('model_params', {})

        # 分析の実行
        results = await analyze_regression(
            collection=request['collection'],
            target_column=request['target_column'],
            feature_columns=feature_columns,
            model_type=model_type,
            test_size=test_size,
            random_state=random_state,
            filters=filters,
            **model_params
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