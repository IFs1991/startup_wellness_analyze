import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
from .base import BaseAnalyzer, AnalysisError
from .utils import PlotUtility, StatisticsUtility
from scipy import stats

class PredictiveModelAnalyzer(BaseAnalyzer):
    """
    スタートアップの将来予測のための機械学習モデルを構築・評価・活用するクラス

    機能:
    - 時系列データや横断的データに基づく予測モデルの構築
    - 複数のアルゴリズム比較と最適モデル選定
    - 特徴量の重要度分析
    - 予測結果の評価と可視化
    - モデルの保存と読み込み
    """

    def __init__(self, db=None):
        """
        初期化メソッド

        Parameters:
        -----------
        db : データベース接続オブジェクト（オプション）
        """
        super().__init__(analysis_type="predictive_model", firestore_client=db)
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'elastic_net': ElasticNet(),
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'svr': SVR(),
            'xgboost': XGBRegressor(random_state=42),
            'lightgbm': LGBMRegressor(random_state=42)
        }
        self.best_model = None
        self.feature_importance = None
        self.model_metrics = None
        self.feature_pipeline = None
        self.logger = logging.getLogger(__name__)

    def preprocess_data(self,
                        data: pd.DataFrame,
                        target_column: str,
                        categorical_features: List[str] = None,
                        numerical_features: List[str] = None,
                        test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        データの前処理を行い、学習用とテスト用に分割する

        Parameters:
        -----------
        data : pd.DataFrame
            予測に使用するデータフレーム
        target_column : str
            予測対象の列名
        categorical_features : List[str], optional
            カテゴリカル特徴量の列名リスト
        numerical_features : List[str], optional
            数値特徴量の列名リスト
        test_size : float, default=0.2
            テストデータの割合

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (X_train, X_test, y_train, y_test)
        """
        if categorical_features is None:
            categorical_features = []

        if numerical_features is None:
            numerical_features = [col for col in data.columns if col != target_column
                                  and col not in categorical_features
                                  and pd.api.types.is_numeric_dtype(data[col])]

        features = categorical_features + numerical_features

        # 特徴量と目的変数を分離
        X = data[features]
        y = data[target_column]

        # 学習データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # 前処理パイプラインの構築
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='drop'
        )

        # 前処理の適用
        self.feature_pipeline = preprocessor.fit(X_train)
        X_train_processed = self.feature_pipeline.transform(X_train)
        X_test_processed = self.feature_pipeline.transform(X_test)

        return X_train_processed, X_test_processed, y_train, y_test, X_train, X_test

    def train_model(self,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    model_type: str = 'xgboost',
                    hyperparams: Dict = None) -> Any:
        """
        指定されたモデルタイプでモデルを学習する

        Parameters:
        -----------
        X_train : np.ndarray
            学習用特徴量
        y_train : np.ndarray
            学習用目的変数
        model_type : str, default='xgboost'
            使用するモデルの種類
        hyperparams : Dict, optional
            ハイパーパラメータ

        Returns:
        --------
        Any
            学習済みモデル
        """
        if model_type not in self.models:
            raise ValueError(f"指定されたモデルタイプ '{model_type}' は利用できません。利用可能なモデル: {list(self.models.keys())}")

        model = self.models[model_type]

        if hyperparams:
            # ハイパーパラメータの設定
            model.set_params(**hyperparams)

        # モデルの学習
        model.fit(X_train, y_train)

        return model

    def find_best_model(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        model_types: List[str] = None,
                        cv: int = 5) -> Dict[str, Any]:
        """
        複数のモデルを比較して最適なモデルを選択する

        Parameters:
        -----------
        X_train : np.ndarray
            学習用特徴量
        y_train : np.ndarray
            学習用目的変数
        X_test : np.ndarray
            テスト用特徴量
        y_test : np.ndarray
            テスト用目的変数
        model_types : List[str], optional
            比較するモデルタイプのリスト
        cv : int, default=5
            クロスバリデーションの分割数

        Returns:
        --------
        Dict[str, Any]
            モデル評価の結果
        """
        if model_types is None:
            model_types = list(self.models.keys())

        results = {}
        best_model_name = None
        best_score = float('inf')  # RMSEを最小化するため

        for model_type in model_types:
            print(f"モデル '{model_type}' の評価を開始...")
            model = self.train_model(X_train, y_train, model_type)

            # テストデータで予測
            y_pred = model.predict(X_test)

            # 評価指標の計算
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[model_type] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }

            print(f"モデル '{model_type}' の評価結果: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

            # 最良モデルの更新（RMSEが最小のモデル）
            if rmse < best_score:
                best_score = rmse
                best_model_name = model_type

        print(f"最適なモデル: '{best_model_name}' (RMSE={best_score:.4f})")
        self.best_model = results[best_model_name]['model']
        self.model_metrics = results

        # 特徴量の重要度を取得（利用可能な場合）
        self._extract_feature_importance(self.best_model, X_train)

        return results

    def _extract_feature_importance(self, model, X_train):
        """
        特徴量の重要度を抽出する（利用可能な場合）
        """
        feature_importance = None

        # モデルタイプに応じた特徴量重要度の抽出
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_)

        if feature_importance is not None:
            # 前処理パイプラインから特徴量名を抽出
            feature_names = []

            if hasattr(self.feature_pipeline, 'get_feature_names_out'):
                feature_names = self.feature_pipeline.get_feature_names_out()
            else:
                # 単純な特徴量名（列インデックス）
                feature_names = [f'feature_{i}' for i in range(len(feature_importance))]

            # 特徴量重要度の辞書を作成
            self.feature_importance = dict(zip(feature_names, feature_importance))

    def tune_hyperparameters(self,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             model_type: str,
                             param_grid: Dict[str, List],
                             cv: int = 5) -> Any:
        """
        グリッドサーチによるハイパーパラメータチューニングを行う

        Parameters:
        -----------
        X_train : np.ndarray
            学習用特徴量
        y_train : np.ndarray
            学習用目的変数
        model_type : str
            チューニングするモデルタイプ
        param_grid : Dict[str, List]
            グリッドサーチのパラメータグリッド
        cv : int, default=5
            クロスバリデーションの分割数

        Returns:
        --------
        Any
            チューニング済みの最適モデル
        """
        if model_type not in self.models:
            raise ValueError(f"指定されたモデルタイプ '{model_type}' は利用できません")

        model = self.models[model_type]
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_root_mean_squared_error')
        grid_search.fit(X_train, y_train)

        print(f"最適パラメータ: {grid_search.best_params_}")
        print(f"最適スコア: {-grid_search.best_score_:.4f} (RMSE)")

        # 最適モデルを更新
        self.best_model = grid_search.best_estimator_

        return grid_search.best_estimator_

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        新しいデータに対して予測を行う

        Parameters:
        -----------
        X : Union[pd.DataFrame, np.ndarray]
            予測する特徴量

        Returns:
        --------
        np.ndarray
            予測結果
        """
        if self.best_model is None:
            raise ValueError("予測を行う前にモデルをトレーニングしてください")

        # DataFrameの場合は前処理パイプラインを適用
        if isinstance(X, pd.DataFrame) and self.feature_pipeline is not None:
            X = self.feature_pipeline.transform(X)

        return self.best_model.predict(X)

    def visualize_feature_importance(self, top_n: int = 10) -> Dict[str, Any]:
        """
        特徴量の重要度を可視化する

        Parameters:
        -----------
        top_n : int, default=10
            表示する特徴量の数

        Returns:
        --------
        Dict[str, Any]
            特徴量重要度のデータと可視化結果
        """
        if self.feature_importance is None:
            if hasattr(self.best_model, 'feature_importances_'):
                self.feature_importance = self.best_model.feature_importances_
            elif hasattr(self.best_model, 'coef_'):
                self.feature_importance = np.abs(self.best_model.coef_)
            else:
                error_msg = "このモデルは特徴量の重要度を提供していません"
                self.logger.error(error_msg)
                raise AnalysisError(error_msg)

        try:
            # 特徴量名を取得
            if hasattr(self.feature_pipeline, 'get_feature_names_out'):
                feature_names = self.feature_pipeline.get_feature_names_out()
            else:
                feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]

            # 特徴量の重要度をソート
            indices = np.argsort(self.feature_importance)[::-1]
            top_indices = indices[:top_n]

            top_features = [feature_names[i] for i in top_indices]
            top_importances = [float(self.feature_importance[i]) for i in top_indices]

            # 正規化（最大値を1にする）
            max_importance = max(top_importances)
            normalized_importances = [imp / max_importance for imp in top_importances]

            # プロットの作成
            fig, ax = plt.subplots(figsize=(10, 8))

            # 横向きの棒グラフを作成
            bars = ax.barh(top_features[::-1], normalized_importances[::-1], color='skyblue')

            # 値を表示
            for i, bar in enumerate(bars):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{top_importances[::-1][i]:.4f}', va='center')

            ax.set_title('特徴量の重要度')
            ax.set_xlabel('正規化された重要度')
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            # 結果を返す
            result = {
                'feature_names': top_features,
                'importance_values': top_importances,
                'normalized_values': normalized_importances,
                'plot_base64': PlotUtility.save_plot_to_base64(fig)
            }

            return result

        except Exception as e:
            error_msg = f"特徴量重要度の可視化中にエラーが発生しました: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    def visualize_predictions(self, X_test, y_test) -> Dict[str, Any]:
        """
        テストデータに対する予測結果を可視化する

        Parameters:
        -----------
        X_test : 特徴量のテストデータ
        y_test : 目的変数のテストデータ

        Returns:
        --------
        Dict[str, Any]
            予測結果の可視化データ
        """
        if self.best_model is None:
            error_msg = "モデルが訓練されていません"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg)

        try:
            # 予測
            y_pred = self.best_model.predict(X_test)

            # メトリクスの計算
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # 散布図の作成
            fig, ax = plt.subplots(figsize=(10, 8))

            # 実測値と予測値の散布図
            ax.scatter(y_test, y_pred, alpha=0.5)

            # 完全一致の線
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')

            # プロット設定
            ax.set_title('実測値 vs 予測値')
            ax.set_xlabel('実測値')
            ax.set_ylabel('予測値')

            # メトリクス表示
            metrics_text = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}'
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            # 結果を返す
            result = {
                'metrics': {
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2': float(r2)
                },
                'predictions': {
                    'y_test': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
                    'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
                },
                'plot_base64': PlotUtility.save_plot_to_base64(fig)
            }

            return result

        except Exception as e:
            error_msg = f"予測結果の可視化中にエラーが発生しました: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    def visualize_residuals(self, X_test, y_test) -> Dict[str, Any]:
        """
        残差プロットを生成して予測モデルの診断を行う

        Parameters:
        -----------
        X_test : テスト用特徴量データ
        y_test : テスト用目的変数

        Returns:
        --------
        Dict[str, Any]
            残差分析の結果と可視化データ
        """
        if self.best_model is None:
            error_msg = "モデルが訓練されていません"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg)

        try:
            # 予測
            y_pred = self.best_model.predict(X_test)

            # 残差計算
            residuals = y_test - y_pred

            # 残差の統計量計算
            residual_stats = {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'min': float(np.min(residuals)),
                'max': float(np.max(residuals)),
                'median': float(np.median(residuals))
            }

            # 正規性検定
            shapiro_test = stats.shapiro(residuals)
            normality_test = {
                'test_name': 'Shapiro-Wilk',
                'statistic': float(shapiro_test[0]),
                'p_value': float(shapiro_test[1]),
                'is_normal': shapiro_test[1] > 0.05
            }

            # 残差プロット生成
            fig = PlotUtility.generate_residual_plot(y_test, y_pred)

            # 結果を返す
            result = {
                'residual_stats': residual_stats,
                'normality_test': normality_test,
                'plot_base64': PlotUtility.save_plot_to_base64(fig)
            }

            return result

        except Exception as e:
            error_msg = f"残差分析中にエラーが発生しました: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    def save_model(self, file_path: str) -> None:
        """
        学習済みモデルをファイルに保存する

        Parameters:
        -----------
        file_path : str
            保存先のファイルパス
        """
        if self.best_model is None:
            raise ValueError("保存するモデルがありません")

        model_data = {
            'model': self.best_model,
            'feature_pipeline': self.feature_pipeline,
            'feature_importance': self.feature_importance,
            'model_metrics': self.model_metrics,
            'timestamp': datetime.now().isoformat()
        }

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"モデルを '{file_path}' に保存しました")

    def load_model(self, file_path: str) -> None:
        """
        保存済みモデルをファイルから読み込む

        Parameters:
        -----------
        file_path : str
            読み込むファイルパス
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)

        self.best_model = model_data['model']
        self.feature_pipeline = model_data['feature_pipeline']
        self.feature_importance = model_data['feature_importance']
        self.model_metrics = model_data['model_metrics']

        print(f"モデルを '{file_path}' から読み込みました (保存日時: {model_data.get('timestamp', '不明')})")

    def forecast_time_series(self,
                             time_series_data: pd.DataFrame,
                             date_column: str,
                             target_column: str,
                             horizon: int = 12,
                             seasonal_period: int = None,
                             include_features: List[str] = None) -> Dict[str, Any]:
        """
        時系列データの将来予測を行う

        Parameters:
        -----------
        time_series_data : pd.DataFrame
            時系列データ
        date_column : str
            日付列の名前
        target_column : str
            予測対象の列名
        horizon : int, default=12
            予測期間（将来の期間数）
        seasonal_period : int, optional
            季節性の周期（月次データなら12など）
        include_features : List[str], optional
            予測に含める追加特徴量のリスト

        Returns:
        --------
        Dict[str, Any]
            予測結果
        """
        # 日付列をインデックスに設定
        df = time_series_data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
        df = df.sort_index()

        # 特徴量エンジニアリング
        df['lag_1'] = df[target_column].shift(1)
        df['lag_2'] = df[target_column].shift(2)
        df['lag_3'] = df[target_column].shift(3)

        # 差分特徴量
        df['diff_1'] = df[target_column].diff(1)

        # 移動平均特徴量
        df['rolling_mean_3'] = df[target_column].rolling(window=3).mean()
        df['rolling_mean_6'] = df[target_column].rolling(window=6).mean()

        # 季節性特徴量（指定されている場合）
        if seasonal_period:
            df['lag_seasonal'] = df[target_column].shift(seasonal_period)
            # 季節性差分
            df['seasonal_diff'] = df[target_column].diff(seasonal_period)

        # 時間関連特徴量の追加
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year

        # 追加特徴量の組み込み
        if include_features:
            feature_cols = [col for col in include_features if col in df.columns]
        else:
            feature_cols = []

        # 基本特徴量リスト
        base_features = ['lag_1', 'lag_2', 'lag_3', 'diff_1',
                         'rolling_mean_3', 'rolling_mean_6',
                         'month', 'quarter', 'year']

        if seasonal_period:
            base_features.extend(['lag_seasonal', 'seasonal_diff'])

        # 最終的な特徴量リスト
        features = base_features + feature_cols

        # 欠損値の除去
        df = df.dropna()

        # 学習データとテストデータの分割（直近のデータをテストデータとする）
        train_size = int(len(df) * 0.8)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]

        # 特徴量と目的変数の分割
        X_train = train_data[features]
        y_train = train_data[target_column]
        X_test = test_data[features]
        y_test = test_data[target_column]

        # モデルの学習（ランダムフォレストを使用）
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # テストデータでの予測
        y_pred_test = model.predict(X_test)

        # 評価メトリクスの計算
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)

        # 将来予測のための特徴量作成
        future_dates = pd.date_range(start=df.index[-1], periods=horizon+1, freq='M')[1:]
        future_df = pd.DataFrame(index=future_dates)
        future_df['month'] = future_df.index.month
        future_df['quarter'] = future_df.index.quarter
        future_df['year'] = future_df.index.year

        # 将来予測
        forecasts = []
        last_data = df.iloc[-horizon:].copy()

        for i in range(horizon):
            # 最新の特徴量を取得
            next_date = future_dates[i]
            next_features = pd.DataFrame(index=[next_date])

            next_features['month'] = next_date.month
            next_features['quarter'] = next_date.quarter
            next_features['year'] = next_date.year

            # ラグ特徴量の計算
            next_features['lag_1'] = last_data[target_column].iloc[-1]
            next_features['lag_2'] = last_data[target_column].iloc[-2] if len(last_data) >= 2 else last_data[target_column].iloc[-1]
            next_features['lag_3'] = last_data[target_column].iloc[-3] if len(last_data) >= 3 else last_data[target_column].iloc[-1]

            # 差分特徴量
            next_features['diff_1'] = last_data[target_column].iloc[-1] - last_data[target_column].iloc[-2] if len(last_data) >= 2 else 0

            # 移動平均特徴量
            next_features['rolling_mean_3'] = last_data[target_column].iloc[-3:].mean() if len(last_data) >= 3 else last_data[target_column].mean()
            next_features['rolling_mean_6'] = last_data[target_column].iloc[-6:].mean() if len(last_data) >= 6 else last_data[target_column].mean()

            # 季節性特徴量
            if seasonal_period:
                seasonal_idx = -seasonal_period if len(last_data) >= seasonal_period else -1
                next_features['lag_seasonal'] = last_data[target_column].iloc[seasonal_idx]
                next_features['seasonal_diff'] = last_data[target_column].iloc[-1] - last_data[target_column].iloc[seasonal_idx]

            # 追加特徴量（もし利用可能なら）
            for feat in feature_cols:
                if feat in last_data.columns:
                    next_features[feat] = last_data[feat].iloc[-1]

            # 予測
            next_prediction = model.predict(next_features[features])
            forecasts.append(next_prediction[0])

            # 予測結果を使って次回の予測のために最新データを更新
            new_row = last_data.iloc[-1:].copy()
            new_row.index = [next_date]
            new_row[target_column] = next_prediction[0]
            last_data = pd.concat([last_data.iloc[1:], new_row])

        # 結果の整形
        forecast_result = {
            'forecast_dates': future_dates.strftime('%Y-%m-%d').tolist(),
            'forecast_values': forecasts,
            'model_metrics': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'test_predictions': {
                'dates': test_data.index.strftime('%Y-%m-%d').tolist(),
                'actual': y_test.tolist(),
                'predicted': y_pred_test.tolist()
            }
        }

        return forecast_result

# 使用例
if __name__ == "__main__":
    # サンプルデータセットの生成
    import numpy as np
    from sklearn.datasets import make_regression

    # 回帰データを生成
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

    # DataFrameに変換
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    data['target'] = y

    # 分析器のインスタンス化
    analyzer = PredictiveModelAnalyzer()

    # データの前処理
    X_train_processed, X_test_processed, y_train, y_test, _, _ = analyzer.preprocess_data(
        data=data,
        target_column='target'
    )

    # 複数モデルの評価と最適モデルの選択
    results = analyzer.find_best_model(
        X_train=X_train_processed,
        y_train=y_train,
        X_test=X_test_processed,
        y_test=y_test,
        model_types=['linear', 'ridge', 'random_forest', 'xgboost']
    )

    # 特徴量重要度の可視化
    importance_result = analyzer.visualize_feature_importance()

    # 予測結果の可視化
    prediction_result = analyzer.visualize_predictions(X_test_processed, y_test)

    print(f"最良モデルの性能: RMSE={prediction_result['metrics']['rmse']:.4f}, R2={prediction_result['metrics']['r2']:.4f}")