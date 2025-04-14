import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Iterator, Callable
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, learning_curve, validation_curve
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
from sklearn.base import BaseEstimator
import shap
import traceback
import joblib
import tempfile
import gc
import weakref
import contextlib
import time
import psutil
from functools import lru_cache
import warnings
import math
from multiprocessing import cpu_count

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

    def __init__(self, db=None, storage_mode: str = 'memory'):
        """
        初期化メソッド

        Parameters:
        -----------
        db : データベース接続オブジェクト（オプション）
        storage_mode : str, default='memory'
            データ保存モード ('memory', 'disk', 'hybrid')
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
        self._temp_files = []  # 一時ファイルの追跡用
        self._storage_mode = storage_mode
        self._plot_resources = weakref.WeakValueDictionary()  # プロットリソースの追跡
        self._model_cache = weakref.WeakValueDictionary()  # モデルの弱参照キャッシュ

        # ストレージモードの検証
        valid_modes = ['memory', 'disk', 'hybrid']
        if self._storage_mode not in valid_modes:
            self.logger.warning(f"無効なストレージモード '{storage_mode}'。'memory'に設定します。")
            self._storage_mode = 'memory'

        self.logger.info(f"PredictiveModelAnalyzer初期化: ストレージモード='{self._storage_mode}'")

    @contextlib.contextmanager
    def _managed_dataframe(self, df: pd.DataFrame, copy: bool = False):
        """データフレームのライフサイクルを管理するコンテキストマネージャー"""
        try:
            if copy:
                df_copy = df.copy()
                yield df_copy
            else:
                yield df
        finally:
            # 明示的な参照解除
            if copy:
                del df_copy
            # メモリ使用状況のログ記録（デバッグ用）
            if self.logger.isEnabledFor(logging.DEBUG):
                process = psutil.Process(os.getpid())
                self.logger.debug(f"現在のメモリ使用量: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    @contextlib.contextmanager
    def _plot_context(self):
        """プロットリソースを管理するコンテキストマネージャー"""
        plot_id = id(plt.gcf())
        self._plot_resources[plot_id] = plt.gcf()
        try:
            yield
        finally:
            plt.close()
            if plot_id in self._plot_resources:
                del self._plot_resources[plot_id]

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

    def find_best_model(self, data: pd.DataFrame, target_column: str,
                        features: List[str] = None, model_types: List[str] = None,
                        cv_folds: int = 5, scoring: str = 'neg_mean_squared_error',
                        random_state: int = 42) -> Dict:
        """
        与えられたデータセットに対して最適なモデルを見つける

        Parameters:
        -----------
        data : pd.DataFrame
            予測モデルを構築するためのデータフレーム
        target_column : str
            予測対象の列名
        features : List[str], optional
            使用する特徴量のリスト。Noneの場合は自動選択
        model_types : List[str], optional
            評価するモデルタイプのリスト。Noneの場合はすべてのモデルを評価
        cv_folds : int, default=5
            交差検証で使用するフォールド数
        scoring : str, default='neg_mean_squared_error'
            モデル評価に使用するスコアリング指標
        random_state : int, default=42
            再現性のための乱数シード

        Returns:
        --------
        Dict
            最適なモデル、そのパラメータ、性能メトリクスを含む辞書
        """
        try:
            # ターゲットと特徴量の準備
            if target_column not in data.columns:
                raise ValueError(f"ターゲット列 '{target_column}' がデータに存在しません")

            X, y = self._prepare_features_and_target(data, target_column, features)

            # モデルタイプの選択
            model_candidates = self._get_model_candidates(model_types, random_state)

            # 最適なモデルの検索
            best_model_info = self._find_optimal_model(X, y, model_candidates, cv_folds, scoring)

            return best_model_info

        except Exception as e:
            self._handle_error("最適モデルの検索", e)

    def _prepare_features_and_target(self, data: pd.DataFrame, target_column: str,
                                    features: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        特徴量と目的変数を準備する

        Parameters:
        -----------
        data : pd.DataFrame
            入力データ
        target_column : str
            予測対象の列名
        features : List[str], optional
            使用する特徴量のリスト、Noneの場合は自動選択

        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            (特徴量データフレーム, 目的変数シリーズ)

        Raises:
        -------
        ValueError
            データ検証エラー発生時
        """
        try:
            # データの基本検証
            if data.empty:
                raise ValueError("入力データが空です")

            if target_column not in data.columns:
                raise ValueError(f"ターゲット列 '{target_column}' がデータに存在しません")

            # 参照渡しで効率化（不要なコピーを回避）
            with self._managed_dataframe(data) as managed_data:
                # 特徴量の選択
                if features is None:
                    # 自動特徴量選択（ターゲット以外の数値列）
                    features = [col for col in managed_data.columns
                               if col != target_column
                               and pd.api.types.is_numeric_dtype(managed_data[col])
                               and not managed_data[col].isnull().all()]  # すべて欠損の列は除外

                    self.logger.info(f"自動特徴量選択: {len(features)}個の特徴量を選択")

                # 特徴量の検証
                if not features:
                    raise ValueError("有効な特徴量がありません")

                # 存在しない特徴量をフィルタリング
                valid_features = [f for f in features if f in managed_data.columns]
                if len(valid_features) < len(features):
                    missing_features = set(features) - set(valid_features)
                    self.logger.warning(f"一部の特徴量がデータに存在しません: {missing_features}")
                    features = valid_features

                if not features:
                    raise ValueError("指定された特徴量がデータに存在しません")

                # 欠損値のチェックと警告
                missing_counts = managed_data[features].isnull().sum()
                features_with_missing = missing_counts[missing_counts > 0]
                if not features_with_missing.empty:
                    self.logger.warning(f"以下の特徴量に欠損値があります:\n{features_with_missing}")

                # バメモリ効率を考慮した特徴量と目的変数の取得（参照渡し）
                X = managed_data[features]
                y = managed_data[target_column]

                return X, y

        except Exception as e:
            self.logger.error(f"特徴量と目的変数の準備中にエラーが発生しました: {str(e)}")
            raise ValueError(f"特徴量と目的変数の準備に失敗しました: {str(e)}")

    def _get_model_candidates(self, model_types: List[str] = None,
                             random_state: int = 42) -> List[Tuple[str, BaseEstimator, Dict]]:
        """
        評価するモデル候補のリストを作成

        Parameters:
        -----------
        model_types : List[str], optional
            評価するモデルタイプのリスト
        random_state : int
            再現性のための乱数シード

        Returns:
        --------
        List[Tuple[str, BaseEstimator, Dict]]
            モデル名、モデルインスタンス、ハイパーパラメータグリッドのタプルのリスト
        """
        all_models = [
            (
                "線形回帰",
                LinearRegression(),
                {}
            ),
            (
                "リッジ回帰",
                Ridge(random_state=random_state),
                {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                }
            ),
            (
                "ラッソ回帰",
                Lasso(random_state=random_state),
                {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                }
            ),
            (
                "ランダムフォレスト",
                RandomForestRegressor(random_state=random_state),
                {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            ),
            (
                "勾配ブースティング",
                GradientBoostingRegressor(random_state=random_state),
                {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            ),
            (
                "XGBoost",
                XGBRegressor(random_state=random_state),
                {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'gamma': [0, 0.1, 0.2]
                }
            )
        ]

        if model_types:
            # 指定されたモデルタイプだけを選択
            return [model for model in all_models if model[0] in model_types]

        return all_models

    def _find_optimal_model(self, X: pd.DataFrame, y: pd.Series,
                           model_candidates: List[Tuple[str, BaseEstimator, Dict]],
                           cv_folds: int, scoring: str) -> Dict:
        """
        最適なモデルを見つけるための内部メソッド

        Parameters:
        -----------
        X : pd.DataFrame
            特徴量データ
        y : pd.Series
            目的変数
        model_candidates : List[Tuple[str, BaseEstimator, Dict]]
            評価するモデルと名前のタプルリスト
        cv_folds : int
            交差検証のフォールド数
        scoring : str
            評価指標

        Returns:
        --------
        Dict
            最適なモデルとその性能指標を含む辞書
        """
        best_score = float('-inf')
        best_model_info = None
        all_results = {}

        # 進捗表示の準備
        total_models = len(model_candidates)
        self.logger.info(f"{total_models}個のモデル候補を評価します...")

        # メモリ効率のために各モデルをローカルスコープで処理
        for i, (model_name, model, params) in enumerate(model_candidates):
            try:
                start_time = time.time()
                self.logger.info(f"[{i+1}/{total_models}] '{model_name}'モデルを評価中...")

                # 効率的なモデル評価関数（ローカルスコープで定義）
                def evaluate_model():
                    # 交差検証を使用してモデルを評価
                    scores = cross_val_score(
                        model, X, y,
                        cv=cv_folds,
                        scoring=scoring,
                        n_jobs=min(cv_folds, max(1, cpu_count() - 1))  # 使用可能なCPUコアに基づいて設定
                    )

                    # 評価指標の取得
                    avg_score = scores.mean()
                    std_score = scores.std()

                    # 予測値の取得（メモリ効率のためバッチサイズに基づいて）
                    if len(X) > 5000:
                        # 大規模データセット用のバッチ予測
                        batch_size = min(5000, max(100, len(X) // 10))
                        y_pred = self._cross_val_predict_in_batches(model, X, y, cv_folds, batch_size)
                    else:
                        # 小規模データセットは標準の交差検証予測
                        y_pred = cross_val_predict(model, X, y, cv=cv_folds)

                    # 各種評価指標の計算
                    mse = mean_squared_error(y, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y, y_pred)
                    r2 = r2_score(y, y_pred)

                    return {
                        'avg_score': avg_score,
                        'std_score': std_score,
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'y_pred': y_pred,  # 予測値も保存
                        'params': params
                    }

                # モデル評価の実行
                model_results = evaluate_model()
                elapsed_time = time.time() - start_time

                # 結果と実行時間のログ記録
                self.logger.info(f"モデル '{model_name}' の評価完了: "
                                 f"スコア={model_results['avg_score']:.4f} "
                                 f"(± {model_results['std_score']:.4f}), "
                                 f"R²={model_results['r2']:.4f}, "
                                 f"実行時間={elapsed_time:.2f}秒")

                # 結果の保存
                all_results[model_name] = model_results

                # 最良モデルの更新
                if model_results['avg_score'] > best_score:
                    best_score = model_results['avg_score']
                    # 最良モデルのクローンを作成して保存（参照の問題を避けるため）
                    best_model = pickle.loads(pickle.dumps(model))
                    best_model_info = {
                        'name': model_name,
                        'model': best_model,
                        'score': best_score,
                        'metrics': {k: v for k, v in model_results.items() if k != 'y_pred'},
                        'params': params
                    }

                # 明示的にガベージコレクションを促す（大規模モデル評価後）
                if hasattr(model, 'n_estimators') and getattr(model, 'n_estimators', 0) > 100:
                    gc.collect()

            except Exception as e:
                # モデル評価中のエラーをログに記録し、次のモデルに進む
                self.logger.error(f"モデル '{model_name}' の評価中にエラーが発生しました: {str(e)}")
                self.logger.debug(f"エラーの詳細:\n{traceback.format_exc()}")
                all_results[model_name] = {'error': str(e)}

        # すべてのモデルの結果を含む最終的な結果辞書
        result = {
            'best_model_info': best_model_info,
            'all_results': all_results
        }

        return result

    def _cross_val_predict_in_batches(self, model, X: pd.DataFrame, y: pd.Series, cv: int, batch_size: int = 5000):
        """
        大規模データセット用のバッチ処理による交差検証予測

        Parameters:
        -----------
        model : BaseEstimator
            評価するモデル
        X : pd.DataFrame
            特徴量データ
        y : pd.Series
            目的変数
        cv : int
            交差検証のフォールド数
        batch_size : int, default=5000
            バッチサイズ

        Returns:
        --------
        np.ndarray
            予測値の配列
        """
        from sklearn.model_selection import KFold

        # 予測結果格納用の配列
        y_pred = np.zeros_like(y, dtype=float)

        # 交差検証分割の準備
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        # 各分割について
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train = y.iloc[train_index]

            # モデルの学習
            model.fit(X_train, y_train)

            # バッチ処理による予測
            if len(X_test) > batch_size:
                # バッチ単位で予測
                for i in range(0, len(X_test), batch_size):
                    batch_end = min(i + batch_size, len(X_test))
                    X_batch = X_test.iloc[i:batch_end]
                    y_pred[test_index[i:batch_end]] = model.predict(X_batch)
            else:
                # 一度に予測
                y_pred[test_index] = model.predict(X_test)

        return y_pred

    def _extract_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        モデルから特徴量重要度を抽出する

        Parameters:
        -----------
        model : BaseEstimator
            学習済みモデル
        feature_names : List[str]
            特徴量名のリスト

        Returns:
        --------
        pd.DataFrame
            特徴量名と重要度を含むDataFrame

        Raises:
        -------
        ValueError
            モデルが特徴量重要度をサポートしていない場合
        """
        try:
            # モデルタイプに基づいて特徴量重要度を抽出
            if hasattr(model, 'feature_importances_'):
                # ツリーベースのモデル（RandomForest、GradientBoosting、XGBoostなど）
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # 線形モデル（LinearRegression、Ridge、Lassoなど）
                importance = np.abs(model.coef_)
                if importance.ndim > 1:
                    # 多変量モデルの場合は平均を取る
                    importance = np.mean(importance, axis=0)
            else:
                self.logger.warning("モデルが特徴量重要度をサポートしていません")
                # 代替として一様な重要度を返す
                importance = np.ones(len(feature_names)) / len(feature_names)

            # 特徴量名の数と重要度の数が一致しない場合の対応
            if len(importance) != len(feature_names):
                self.logger.warning(
                    f"特徴量重要度の次元 ({len(importance)}) と特徴量名の数 ({len(feature_names)}) が一致しません。"
                    "重要度の次元に合わせてデータを調整します。"
                )
                # 不一致の場合は短い方に合わせる
                min_len = min(len(importance), len(feature_names))
                importance = importance[:min_len]
                feature_names = feature_names[:min_len]

            # メモリ効率の良いDataFrame作成（必要最小限のメモリ割り当て）
            df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })

            # 重要度の降順でソート
            df = df.sort_values('importance', ascending=False).reset_index(drop=True)

            # メモリ効率化のために適切なデータ型を設定
            if df['importance'].min() >= 0 and df['importance'].max() <= 1:
                # 0-1の範囲であればfloat32に変換（メモリ使用量半減）
                df['importance'] = df['importance'].astype(np.float32)

            return df

        except Exception as e:
            self.logger.error(f"特徴量重要度の抽出中にエラーが発生しました: {str(e)}")
            self.logger.debug(f"エラーの詳細:\n{traceback.format_exc()}")
            # エラー時は空のDataFrameを返す代わりに例外を再発生
            raise ValueError(f"特徴量重要度の抽出に失敗しました: {str(e)}")

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

        # リソースの効率的な利用のために予測を小さなバッチで実行
        try:
            # DataFrameの場合は前処理パイプラインを適用
            if isinstance(X, pd.DataFrame) and self.feature_pipeline is not None:
                # 大きなデータセットの場合はバッチ処理を考慮
                if X.shape[0] > 10000:  # 10000行を超える大きなデータセット
                    return self._predict_in_batches(X)
                else:
                    # 中小規模のデータセットは一度に処理
                    X_transformed = self.feature_pipeline.transform(X)
                    predictions = self.best_model.predict(X_transformed)
                    # 中間データを解放
                    del X_transformed
                    gc.collect()
                    return predictions
            else:
                # ndarrayの場合は直接予測
                return self.best_model.predict(X)
        except Exception as e:
            self._handle_error("予測の実行", e)

    def _predict_in_batches(self, X: pd.DataFrame, batch_size: int = 5000) -> np.ndarray:
        """
        大きなデータセットをバッチに分けて予測を行う

        Parameters:
        -----------
        X : pd.DataFrame
            予測する特徴量データフレーム
        batch_size : int, default=5000
            バッチサイズ

        Returns:
        --------
        np.ndarray
            全バッチの予測結果を結合したもの
        """
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size  # 切り上げ除算

        # 結果を格納するためのリスト
        predictions_list = []

        for i in range(n_batches):
            # バッチの境界を計算
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            # バッチデータを取得（参照のみ、コピーしない）
            batch_X = X.iloc[start_idx:end_idx]

            # 前処理と予測
            batch_X_transformed = self.feature_pipeline.transform(batch_X)
            batch_predictions = self.best_model.predict(batch_X_transformed)

            # 結果をリストに追加
            predictions_list.append(batch_predictions)

            # 中間データを解放
            del batch_X, batch_X_transformed

            # 進捗を記録
            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                self.logger.info(f"予測進捗: {i+1}/{n_batches} バッチ完了")

        # 全バッチの結果を結合
        all_predictions = np.concatenate(predictions_list)

        # リストを解放
        del predictions_list
        gc.collect()

        return all_predictions

    def predict_with_model(self, model, X_data, return_proba: bool = False):
        """
        モデルを使用して予測を行う共通メソッド

        Parameters:
        -----------
        model : 学習済みモデル
            予測に使用するモデル
        X_data : array-like
            予測する特徴量データ
        return_proba : bool, default=False
            確率を返すかどうか（分類問題の場合）

        Returns:
        --------
        array-like
            予測結果
        """
        try:
            # 大きなデータセットかどうかを判断
            is_large_dataset = False
            if hasattr(X_data, 'shape') and hasattr(X_data.shape, '__len__') and len(X_data.shape) > 0:
                is_large_dataset = X_data.shape[0] > 10000

            # バッチ処理の適用
            if is_large_dataset and isinstance(X_data, pd.DataFrame):
                # バッチサイズの決定（モデルとデータに基づいて調整可能）
                batch_size = 5000

                if return_proba and hasattr(model, 'predict_proba'):
                    return self._predict_proba_in_batches(model, X_data, batch_size)
                else:
                    return self._model_predict_in_batches(model, X_data, batch_size)
            else:
                # 小さいデータセットは一度に処理
                if return_proba and hasattr(model, 'predict_proba'):
                    return model.predict_proba(X_data)
                return model.predict(X_data)
        except Exception as e:
            self._handle_error("モデルでの予測", e)

    def _model_predict_in_batches(self, model, X: pd.DataFrame, batch_size: int) -> np.ndarray:
        """
        モデルの予測をバッチに分けて実行する

        Parameters:
        -----------
        model : 学習済みモデル
            予測に使用するモデル
        X : pd.DataFrame
            予測する特徴量データ
        batch_size : int
            バッチサイズ

        Returns:
        --------
        np.ndarray
            予測結果
        """
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        # 結果を格納するためのリスト
        predictions_list = []

        for i in range(n_batches):
            # バッチの範囲を計算
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            # バッチデータを取得
            batch_X = X.iloc[start_idx:end_idx]

            # 予測を実行
            batch_predictions = model.predict(batch_X)

            # 結果をリストに追加
            predictions_list.append(batch_predictions)

            # 不要なデータを解放
            del batch_X

            # 10バッチごとにGCを実行
            if (i + 1) % 10 == 0:
                gc.collect()

        # 全バッチの結果を結合
        all_predictions = np.concatenate(predictions_list)

        # リソース解放
        del predictions_list
        gc.collect()

        return all_predictions

    def _predict_proba_in_batches(self, model, X: pd.DataFrame, batch_size: int) -> np.ndarray:
        """
        確率予測をバッチに分けて実行する

        Parameters:
        -----------
        model : 学習済みモデル
            予測に使用するモデル
        X : pd.DataFrame
            予測する特徴量データ
        batch_size : int
            バッチサイズ

        Returns:
        --------
        np.ndarray
            確率予測結果
        """
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        # 最初のバッチで出力の形状を特定
        first_batch = X.iloc[:min(batch_size, n_samples)]
        first_result = model.predict_proba(first_batch)
        n_classes = first_result.shape[1]

        # 結果を格納するための配列を初期化
        all_proba = np.zeros((n_samples, n_classes), dtype=np.float32)
        all_proba[:min(batch_size, n_samples)] = first_result

        # 残りのバッチを処理
        for i in range(1, n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            batch_X = X.iloc[start_idx:end_idx]
            batch_proba = model.predict_proba(batch_X)

            all_proba[start_idx:end_idx] = batch_proba

            del batch_X, batch_proba

            # メモリ管理
            if (i + 1) % 10 == 0:
                gc.collect()

        return all_proba

    def predict_outcome(self, model, data, target_column, features=None):
        """予測されたアウトカムを返す"""
        try:
            if features is None:
                features = self.get_model_features(model)

            if set(features).issubset(set(data.columns)):
                X = data[features]

                # 欠損値をチェック
                if X.isnull().any().any():
                    self.logger.warning("予測データに欠損値があります。これは予測精度に影響を与える可能性があります。")
                    X = X.fillna(X.mean())

                # 予測の実行
                predictions = self.predict_with_model(model, X)

                # 結果をDataFrameとして返す
                result = data.copy()
                result['predicted'] = predictions

                # 実際の値がデータに含まれている場合、ターゲット列名も返す
                if target_column in data.columns:
                    result['actual'] = data[target_column]

                return result
            else:
                missing_features = set(features) - set(data.columns)
                error_msg = f"予測に必要な特徴量がデータにありません: {missing_features}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        except Exception as e:
            self._handle_error("予測結果の計算", e)

    def visualize_predictions(self, actual_values, predicted_values, title='予測 vs 実績',
                             xlabel='実績', ylabel='予測', figsize=(10, 6)):
        """予測値と実際値の散布図を作成"""
        try:
            plt.figure(figsize=figsize)

            # 散布図
            plt.scatter(actual_values, predicted_values, alpha=0.6)

            # 対角線（完全予測ライン）
            min_val = min(min(actual_values), min(predicted_values))
            max_val = max(max(actual_values), max(predicted_values))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')

            # グラフの装飾
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)

            # メトリクスの計算と表示
            metrics = self.calculate_model_metrics(actual_values, predicted_values)
            metrics_text = (f"RMSE: {metrics['rmse']:.4f}\n"
                           f"MAE: {metrics['mae']:.4f}\n"
                           f"R²: {metrics['r2']:.4f}")

            plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

            plt.tight_layout()
            return plt.gcf()

        except Exception as e:
            self._handle_error("予測の可視化", e)

    def generate_residual_plot(self, y_true, y_pred, figsize=(12, 8)):
        """モデル診断のための残差プロットを生成"""
        try:
            residuals = y_true - y_pred

            fig, axes = plt.subplots(2, 2, figsize=figsize)

            # 残差のヒストグラム
            axes[0, 0].hist(residuals, bins=20, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('残差分布')
            axes[0, 0].set_xlabel('残差')
            axes[0, 0].set_ylabel('頻度')

            # Q-Qプロット
            stats.probplot(residuals, plot=axes[0, 1])
            axes[0, 1].set_title('残差Q-Qプロット')

            # 予測値vs残差
            axes[1, 0].scatter(y_pred, residuals, alpha=0.6)
            axes[1, 0].axhline(y=0, color='r', linestyle='--')
            axes[1, 0].set_title('予測値 vs 残差')
            axes[1, 0].set_xlabel('予測値')
            axes[1, 0].set_ylabel('残差')

            # 実際値vs残差
            axes[1, 1].scatter(y_true, residuals, alpha=0.6)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_title('実際値 vs 残差')
            axes[1, 1].set_xlabel('実際値')
            axes[1, 1].set_ylabel('残差')

            # 統計量の計算と表示
            stats_text = (f"平均残差: {np.mean(residuals):.4f}\n"
                         f"残差標準偏差: {np.std(residuals):.4f}\n"
                         f"歪度: {stats.skew(residuals):.4f}\n"
                         f"尖度: {stats.kurtosis(residuals):.4f}")

            fig.text(0.5, 0.01, stats_text, ha='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            return fig

        except Exception as e:
            self._handle_error("残差分析", e)

    def save_model(self, file_path: str) -> None:
        """
        学習したモデルを保存する

        Parameters:
        -----------
        file_path : str
            モデルを保存するファイルパス

        Raises:
        -------
        ValueError
            モデルが存在しないか保存に失敗した場合
        """
        if self.best_model is None:
            raise ValueError("保存するモデルがありません。モデルを学習してから保存してください。")

        try:
            self.logger.info(f"モデルを保存しています: {file_path}")

            # 保存先ディレクトリが存在しない場合は作成
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            # モデルのサイズに基づいて保存方法を決定
            if self._is_large_model(self.best_model):
                self._save_large_model(file_path)
            else:
                # 通常のモデル保存（メタデータも含める）
                save_data = {
                    'model': self.best_model,
                    'feature_importance': self.feature_importance,
                    'model_metrics': self.model_metrics,
                    'feature_pipeline': self.feature_pipeline,
                    'metadata': {
                        'saved_at': datetime.now().isoformat(),
                        'model_type': type(self.best_model).__name__,
                        'version': '1.0'
                    }
                }

                # 拡張子に基づいて保存形式を判定
                if file_path.endswith('.joblib'):
                    joblib.dump(save_data, file_path, compress=3, protocol=4)
                else:
                    with open(file_path, 'wb') as f:
                        pickle.dump(save_data, f, protocol=4)

            self.logger.info(f"モデルを正常に保存しました: {file_path}")

        except Exception as e:
            self.logger.error(f"モデル保存中にエラーが発生しました: {str(e)}")
            self.logger.debug(f"エラーの詳細:\n{traceback.format_exc()}")
            raise ValueError(f"モデルの保存に失敗しました: {str(e)}")

    def _is_large_model(self, model) -> bool:
        """
        モデルが大規模かどうかを判定する

        Parameters:
        -----------
        model : BaseEstimator
            判定するモデル

        Returns:
        --------
        bool
            大規模モデルならTrue、そうでなければFalse
        """
        # ツリーベースのモデルの場合、木の数に基づいて判定
        if hasattr(model, 'n_estimators'):
            return model.n_estimators > 100

        # モデルをメモリで予測的に評価
        try:
            # 一時シリアライズしてサイズを測定
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                pickle.dump(model, tmp, protocol=4)
                tmp.flush()
                size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
                return size_mb > 100  # 100MB以上を大規模と判定
        except Exception:
            # 評価に失敗した場合は安全側に倒して大規模と判定
            return True

    def _save_large_model(self, file_path: str) -> None:
        """
        大規模モデルを効率的に保存する

        Parameters:
        -----------
        file_path : str
            保存先ファイルパス
        """
        try:
            # メタデータとモデルを分離して保存
            base_path, ext = os.path.splitext(file_path)
            model_path = f"{base_path}_model{ext}"
            meta_path = f"{base_path}_metadata{ext}"

            # モデル本体をjoblibで保存（高圧縮オプション）
            self.logger.info("大規模モデルを圧縮保存しています...")
            joblib.dump(self.best_model, model_path, compress=3, protocol=4)

            # 付随情報を別ファイルに保存
            metadata = {
                'feature_importance': self.feature_importance,
                'model_metrics': self.model_metrics,
                'feature_pipeline': self.feature_pipeline,
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'model_type': type(self.best_model).__name__,
                    'version': '1.0',
                    'is_large_model': True,
                    'model_file': os.path.basename(model_path)
                }
            }

            joblib.dump(metadata, meta_path, compress=3, protocol=4)

            # メインファイルとして参照情報を保存
            reference = {
                'is_reference': True,
                'model_path': model_path,
                'metadata_path': meta_path
            }

            with open(file_path, 'wb') as f:
                pickle.dump(reference, f, protocol=4)

            # 一時ファイルとして記録（削除対象）
            self._temp_files.extend([file_path, model_path, meta_path])

            self.logger.info(f"大規模モデルを正常に保存しました: {file_path}")

        except Exception as e:
            self.logger.error(f"大規模モデル保存中にエラーが発生しました: {str(e)}")
            raise

    def load_model(self, file_path: str) -> None:
        """
        保存したモデルを読み込む

        Parameters:
        -----------
        file_path : str
            読み込むモデルファイルのパス

        Raises:
        -------
        ValueError
            ファイルが存在しないか読み込みに失敗した場合
        """
        if not os.path.exists(file_path):
            raise ValueError(f"指定されたファイル {file_path} が存在しません")

        try:
            self.logger.info(f"モデルを読み込んでいます: {file_path}")

            # ファイル拡張子に基づいて読み込み方法を判定
            if file_path.endswith('.joblib'):
                loaded_data = joblib.load(file_path)
            else:
                with open(file_path, 'rb') as f:
                    loaded_data = pickle.load(f)

            # 分割保存された大規模モデルかチェック
            if isinstance(loaded_data, dict) and loaded_data.get('is_reference', False):
                self.logger.info("分割保存された大規模モデルを読み込んでいます...")
                model_path = loaded_data.get('model_path')
                metadata_path = loaded_data.get('metadata_path')

                if not os.path.exists(model_path):
                    raise ValueError(f"モデルファイル {model_path} が見つかりません")
                if not os.path.exists(metadata_path):
                    raise ValueError(f"メタデータファイル {metadata_path} が見つかりません")

                # モデル本体を読み込み
                self.best_model = joblib.load(model_path)

                # メタデータを読み込み
                metadata = joblib.load(metadata_path)
                self.feature_importance = metadata.get('feature_importance')
                self.model_metrics = metadata.get('model_metrics')
                self.feature_pipeline = metadata.get('feature_pipeline')

                self.logger.info("大規模モデルの読み込みが完了しました")
            else:
                # 通常の一括保存されたモデル
                self.best_model = loaded_data.get('model')
                self.feature_importance = loaded_data.get('feature_importance')
                self.model_metrics = loaded_data.get('model_metrics')
                self.feature_pipeline = loaded_data.get('feature_pipeline')

            # メモリ最適化のためにガベージコレクションを実行
            gc.collect()

            self.logger.info("モデルを正常に読み込みました")

        except Exception as e:
            self.logger.error(f"モデル読み込み中にエラーが発生しました: {str(e)}")
            self.logger.debug(f"エラーの詳細:\n{traceback.format_exc()}")
            raise ValueError(f"モデルの読み込みに失敗しました: {str(e)}")

    def release_resources(self) -> None:
        """
        使用したリソースを解放する

        メモリリークを防止するために、明示的にリソースを解放します。
        大きなモデルを扱った後や複数の分析を連続して行う場合に推奨されます。
        """
        try:
            # 一時ファイルの削除
            for file_path in self._temp_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        self.logger.debug(f"一時ファイルを削除しました: {file_path}")
                    except OSError as e:
                        self.logger.warning(f"一時ファイル {file_path} の削除に失敗しました: {str(e)}")

            # 一時ファイルリストをクリア
            self._temp_files.clear()

            # プロットリソースをクリア
            for fig_id in list(self._plot_resources.keys()):
                plt.close(self._plot_resources[fig_id])
            self._plot_resources.clear()

            # 大きなオブジェクト参照の解放
            self.best_model = None
            self.feature_importance = None
            self.model_metrics = None

            # モデルキャッシュのクリア
            self._model_cache.clear()

            # メモリ回収を促す
            gc.collect()

            self.logger.info("リソースが正常に解放されました")

        except Exception as e:
            self.logger.error(f"リソース解放中にエラーが発生しました: {str(e)}")
            self.logger.debug(f"エラーの詳細:\n{traceback.format_exc()}")

    def __del__(self):
        """
        デストラクタメソッド

        インスタンスが破棄される際に自動的にリソースを解放します。
        """
        try:
            self.release_resources()
        except Exception as e:
            # デストラクタ内での例外はシステムクラッシュの原因になるため抑制
            pass

    def __enter__(self):
        """
        コンテキストマネージャーとしての振る舞いをサポート

        with文で使用できるようにします:
        with PredictiveModelAnalyzer() as analyzer:
            analyzer.analyze(...)
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        withブロック終了時の処理

        リソースを自動的に解放します。
        """
        self.release_resources()
        return False  # 例外を伝播させる

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
        try:
            # データ準備と特徴量エンジニアリング
            df, features = self._prepare_time_series_features(
                time_series_data,
                date_column,
                target_column,
                seasonal_period,
                include_features
            )

            # 学習データとテストデータの分割
            train_data, test_data, X_train, y_train, X_test, y_test = self._split_time_series_data(
                df, features, target_column
            )

            # モデルの学習と評価
            model, y_pred_test, metrics = self._train_time_series_model(X_train, y_train, X_test, y_test)

            # 将来予測の実行
            future_dates, forecasts = self._generate_time_series_forecast(
                df, model, features, target_column, horizon, seasonal_period, include_features
            )

            # 結果の整形
            forecast_result = {
                'forecast_dates': future_dates.strftime('%Y-%m-%d').tolist(),
                'forecast_values': forecasts,
                'model_metrics': metrics,
                'test_predictions': {
                    'dates': test_data.index.strftime('%Y-%m-%d').tolist(),
                    'actual': y_test.tolist(),
                    'predicted': y_pred_test.tolist()
                }
            }

            return forecast_result

        except Exception as e:
            error_msg = f"時系列予測中にエラーが発生しました: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    def _prepare_time_series_features(self,
                                     time_series_data: pd.DataFrame,
                                     date_column: str,
                                     target_column: str,
                                     seasonal_period: int = None,
                                     include_features: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """時系列データの前処理と特徴量エンジニアリングを行う"""
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

        return df, features

    def _split_time_series_data(self,
                               df: pd.DataFrame,
                               features: List[str],
                               target_column: str,
                               train_ratio: float = 0.8) -> Tuple:
        """時系列データを学習用とテスト用に分割する"""
        # 学習データとテストデータの分割（直近のデータをテストデータとする）
        train_size = int(len(df) * train_ratio)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]

        # 特徴量と目的変数の分割
        X_train = train_data[features]
        y_train = train_data[target_column]
        X_test = test_data[features]
        y_test = test_data[target_column]

        return train_data, test_data, X_train, y_train, X_test, y_test

    def _train_time_series_model(self,
                                X_train,
                                y_train,
                                X_test,
                                y_test) -> Tuple[RandomForestRegressor, np.ndarray, Dict[str, float]]:
        """時系列予測モデルをトレーニングして評価する"""
        # モデルの学習（ランダムフォレストを使用）
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # テストデータでの予測
        y_pred_test = model.predict(X_test)

        # 評価メトリクスの計算
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            'mae': float(mean_absolute_error(y_test, y_pred_test)),
            'r2': float(r2_score(y_test, y_pred_test))
        }

        return model, y_pred_test, metrics

    def _generate_time_series_forecast(self,
                                      df: pd.DataFrame,
                                      model,
                                      features: List[str],
                                      target_column: str,
                                      horizon: int,
                                      seasonal_period: int = None,
                                      include_features: List[str] = None) -> Tuple[pd.DatetimeIndex, List[float]]:
        """将来の時系列予測を生成する"""
        # 将来予測のための特徴量作成
        future_dates = pd.date_range(start=df.index[-1], periods=horizon+1, freq='M')[1:]

        # 予測の準備
        forecasts = []
        last_data = df.iloc[-horizon:].copy()
        feature_cols = include_features if include_features else []

        for i in range(horizon):
            # 次の予測のための特徴量を作成
            next_date = future_dates[i]
            next_features = self._create_forecast_features(
                last_data, next_date, target_column,
                seasonal_period, feature_cols
            )

            # 予測
            next_prediction = model.predict(next_features[features])
            forecasts.append(float(next_prediction[0]))

            # 予測結果を使って次回の予測のために最新データを更新
            new_row = last_data.iloc[-1:].copy()
            new_row.index = [next_date]
            new_row[target_column] = float(next_prediction[0])
            last_data = pd.concat([last_data.iloc[1:], new_row])

        return future_dates, forecasts

    def _create_forecast_features(self,
                                 last_data: pd.DataFrame,
                                 next_date: pd.Timestamp,
                                 target_column: str,
                                 seasonal_period: int = None,
                                 feature_cols: List[str] = None) -> pd.DataFrame:
        """時系列予測のための次の時点の特徴量を作成する"""
        next_features = pd.DataFrame(index=[next_date])

        # 時間関連特徴量
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
        if feature_cols:
            for feat in feature_cols:
                if feat in last_data.columns:
                    next_features[feat] = last_data[feat].iloc[-1]

        return next_features

    def analyze_feature_importance(self, model, feature_names: List[str],
                                 plot: bool = True, n_top_features: int = 10) -> pd.DataFrame:
        """
        モデルの特徴量重要度を分析する

        Parameters:
        -----------
        model : 学習済みモデル
            分析対象のモデル
        feature_names : List[str]
            特徴量名のリスト
        plot : bool, default=True
            重要度のプロットを生成するかどうか
        n_top_features : int, default=10
            表示する上位特徴量の数

        Returns:
        --------
        pd.DataFrame
            特徴量重要度のデータフレーム
        """
        try:
            importance_df = self._extract_feature_importance(model, feature_names)

            # 重要度の高い順にソート
            importance_df = importance_df.sort_values('importance', ascending=False)

            # 上位n個の特徴量を選択
            top_features = importance_df.head(n_top_features)

            if plot:
                self._plot_feature_importance(top_features)

            return importance_df

        except Exception as e:
            self._handle_error("特徴量重要度分析", e)

    def _plot_feature_importance(self, importance_df: pd.DataFrame) -> None:
        """
        特徴量重要度をプロットする

        Parameters:
        -----------
        importance_df : pd.DataFrame
            特徴量重要度のデータフレーム
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title('特徴量重要度')
        plt.xlabel('重要度')
        plt.ylabel('特徴量')
        plt.tight_layout()

    def generate_partial_dependence_plots(self, model, X: pd.DataFrame,
                                        features: List[str] = None, n_cols: int = 2) -> None:
        """
        モデルの部分依存プロットを生成する

        Parameters:
        -----------
        model : 学習済みモデル
            分析対象のモデル
        X : pd.DataFrame
            特徴量データ
        features : List[str], optional
            プロットする特徴量のリスト（Noneの場合は重要度上位の特徴量）
        n_cols : int, default=2
            プロット配置の列数
        """
        try:
            if features is None:
                # 重要度から上位特徴量を選択
                if hasattr(model, 'feature_importances_'):
                    indices = np.argsort(model.feature_importances_)[-6:]
                    features = [X.columns[i] for i in indices]
                else:
                    # 上位6個の特徴量を使用
                    features = X.columns[:6].tolist()

            # 特徴量数に基づいて行数を計算
            n_features = len(features)
            n_rows = (n_features + n_cols - 1) // n_cols

            # プロットの作成
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            axes = axes.flatten() if n_features > 1 else [axes]

            for i, feature in enumerate(features):
                if i < len(axes):
                    self._create_single_pdp(model, X, feature, ax=axes[i])

            # 不要なサブプロットを非表示
            for j in range(n_features, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()

        except Exception as e:
            self._handle_error("部分依存プロット生成", e)

    def _create_single_pdp(self, model, X: pd.DataFrame, feature: str, ax=None) -> None:
        """
        単一特徴量の部分依存プロットを作成する

        Parameters:
        -----------
        model : 学習済みモデル
            分析対象のモデル
        X : pd.DataFrame
            特徴量データ
        feature : str
            プロットする特徴量
        ax : matplotlib.axes, optional
            プロット対象の軸
        """
        # 特徴量の値の範囲を取得
        x_values = np.linspace(X[feature].min(), X[feature].max(), num=50)

        # 予測値を格納するリスト
        y_values = []

        # 各特徴量値に対する予測値を計算
        for x_val in x_values:
            X_copy = X.copy()
            X_copy[feature] = x_val
            predictions = model.predict(X_copy)
            y_values.append(np.mean(predictions))

        # プロット
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(x_values, y_values)
        ax.set_xlabel(feature)
        ax.set_ylabel('予測値の平均')
        ax.set_title(f'{feature}の部分依存プロット')
        ax.grid(True, linestyle='--', alpha=0.7)

    def generate_shap_analysis(self, model, X: pd.DataFrame,
                             plot_type: str = 'summary', sample_size: int = 100) -> None:
        """
        SHAPを使用してモデルの予測を解釈する

        Parameters:
        -----------
        model : 学習済みモデル
            分析対象のモデル
        X : pd.DataFrame
            特徴量データ
        plot_type : str, default='summary'
            生成するプロットタイプ ('summary', 'waterfall', 'dependence')
        sample_size : int, default=100
            大きなデータセットの場合に使用するサンプルサイズ
        """
        try:
            # モデルタイプに基づいて適切なSHAP explainerを選択
            if X.shape[0] > sample_size:
                self.logger.info(f"データサイズが大きいためサンプル {sample_size} 行を使用します")
                X_sample = X.sample(sample_size, random_state=42)
            else:
                X_sample = X

            # scikit-learn API検出
            if hasattr(model, 'predict'):
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample)
            else:
                self.logger.warning("このモデルタイプはSHAP解析に対応していない可能性があります")
                return

            # プロットタイプに基づいて適切なプロットを生成
            if plot_type == 'summary':
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, plot_size=(10, 8))
                plt.title('SHAP値の要約')

            elif plot_type == 'waterfall':
                # 最初のインスタンスについてウォーターフォールプロットを生成
                plt.figure(figsize=(10, 8))
                shap.plots.waterfall(shap_values[0])
                plt.title('予測説明（最初のインスタンス）')

            elif plot_type == 'dependence':
                # 最も重要な特徴量の依存性プロット
                if hasattr(model, 'feature_importances_'):
                    top_idx = np.argmax(model.feature_importances_)
                    top_feature = X.columns[top_idx]
                else:
                    top_feature = X.columns[0]

                plt.figure(figsize=(10, 8))
                shap.dependence_plot(top_feature, shap_values.values, X_sample,
                                   feature_names=X.columns)
                plt.title(f'{top_feature}のSHAP依存性プロット')

            else:
                self.logger.warning(f"未知のプロットタイプ: {plot_type}")

        except Exception as e:
            self._handle_error("SHAP解析", e)

    def visualize_cv_results(self, cv_results: Dict, metric: str = 'r2', n_top: int = 5) -> None:
        """
        クロスバリデーション結果を視覚化する

        Parameters:
        -----------
        cv_results : Dict
            GridSearchCVまたはRandomizedSearchCVの結果
        metric : str, default='r2'
            表示するメトリック ('r2', 'mse', 'mae', 'rmse' など)
        n_top : int, default=5
            表示する上位モデル数
        """
        try:
            # 結果をデータフレームに変換
            results_df = pd.DataFrame(cv_results)

            # 評価指標のカラム名を取得
            if f'mean_test_{metric}' in results_df.columns:
                metric_col = f'mean_test_{metric}'
            elif metric == 'r2' and 'mean_test_score' in results_df.columns:
                metric_col = 'mean_test_score'
            else:
                raise ValueError(f"メトリック '{metric}' が結果に見つかりません")

            # ソート方向を決定（r2は高いほど良い、エラー指標は低いほど良い）
            ascending = metric not in ['r2', 'score', 'accuracy', 'precision', 'recall', 'f1']

            # 結果をソート
            sorted_results = results_df.sort_values(by=metric_col, ascending=ascending)

            # 上位n_top個のモデルを選択
            top_results = sorted_results.head(n_top)

            # パラメータを見やすい形式に整形
            param_cols = [col for col in top_results.columns if col.startswith('param_')]
            top_params = top_results[param_cols].copy()

            # パラメータ値を文字列に変換
            for col in param_cols:
                top_params[col] = top_params[col].astype(str)

            # モデル名を構築
            param_names = [col.replace('param_', '') for col in param_cols]
            model_names = []

            for _, row in top_params.iterrows():
                param_str = ", ".join([f"{param}={row[f'param_{param}']}"
                                    for param in param_names
                                    if not pd.isna(row[f'param_{param}']) and row[f'param_{param}'] != 'None'])
                model_names.append(param_str)

            # プロットデータを準備
            plot_df = top_results[[metric_col, f'std_test_{metric}' if f'std_test_{metric}' in top_results.columns else 'std_test_score']].copy()
            plot_df.index = model_names
            plot_df.columns = ['Mean', 'Std']

            # バープロットを作成
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(plot_df)), plot_df['Mean'], yerr=plot_df['Std'],
                        align='center', alpha=0.7, ecolor='black', capsize=10)

            # プロットを装飾
            plt.xticks(range(len(plot_df)), model_names, rotation=45, ha='right')
            plt.title(f'上位{n_top}モデルの{metric}スコア比較')
            plt.ylabel(f'{metric}スコア')
            plt.xlabel('モデルパラメータ')
            plt.grid(axis='y', linestyle='--', alpha=0.7)

        except Exception as e:
            self._handle_error("クロスバリデーション結果の可視化", e)

    def evaluate_model_performance(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                 metrics: List[str] = None, plot: bool = True) -> Dict:
        """
        テストデータでモデルのパフォーマンスを評価する

        Parameters:
        -----------
        model : 学習済みモデル
            評価対象のモデル
        X_test : pd.DataFrame
            テスト用特徴量
        y_test : pd.Series
            テスト用ターゲット
        metrics : List[str], optional
            計算するメトリックのリスト（デフォルトは['r2', 'mse', 'mae', 'rmse']）
        plot : bool, default=True
            パフォーマンスプロットを生成するかどうか

        Returns:
        --------
        Dict
            計算されたメトリックの辞書
        """
        if metrics is None:
            metrics = ['r2', 'mse', 'mae', 'rmse']

        try:
            # 予測を実行
            y_pred = model.predict(X_test)

            # メトリックを計算
            results = {}
            for metric in metrics:
                if metric.lower() == 'r2':
                    results['r2'] = r2_score(y_test, y_pred)
                elif metric.lower() == 'mse':
                    results['mse'] = mean_squared_error(y_test, y_pred)
                elif metric.lower() == 'mae':
                    results['mae'] = mean_absolute_error(y_test, y_pred)
                elif metric.lower() == 'rmse':
                    results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
                elif metric.lower() == 'mape' and not np.any(y_test == 0):
                    results['mape'] = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                elif metric.lower() == 'medae':
                    results['medae'] = np.median(np.abs(y_test - y_pred))
                else:
                    self.logger.warning(f"未対応のメトリック: {metric}")

            # メトリック結果をログに記録
            self.logger.info("モデル評価結果:")
            for metric_name, value in results.items():
                self.logger.info(f"  {metric_name.upper()}: {value:.4f}")

            # プロットの生成
            if plot:
                self._create_performance_plots(y_test, y_pred)

            return results

        except Exception as e:
            self._handle_error("モデルパフォーマンス評価", e)

    def _create_performance_plots(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        モデルのパフォーマンスプロットを生成する

        Parameters:
        -----------
        y_true : np.ndarray
            実際の値
        y_pred : np.ndarray
            予測値
        """
        # 予測値と実際の値の散布図
        plt.figure(figsize=(12, 10))

        # サブプロット1: 予測値 vs 実際の値
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)

        # 完全一致の線を追加
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.xlabel('実際の値')
        plt.ylabel('予測値')
        plt.title('予測値 vs 実際の値')
        plt.grid(True, linestyle='--', alpha=0.7)

        # サブプロット2: 残差プロット
        plt.subplot(2, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('予測値')
        plt.ylabel('残差')
        plt.title('残差プロット')
        plt.grid(True, linestyle='--', alpha=0.7)

        # サブプロット3: 残差のヒストグラム
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('残差')
        plt.ylabel('頻度')
        plt.title('残差分布')
        plt.grid(True, linestyle='--', alpha=0.7)

        # サブプロット4: Q-Qプロット
        plt.subplot(2, 2, 4)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('残差のQ-Qプロット')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

    def plot_learning_curve(self, estimator, X: pd.DataFrame, y: pd.Series,
                          cv: int = 5, n_jobs: int = -1, train_sizes: np.ndarray = None,
                          scoring: str = 'r2', title: str = '学習曲線') -> None:
        """
        モデルの学習曲線をプロットして、過学習/過少学習を分析する

        Parameters:
        -----------
        estimator : 学習器
            学習曲線を生成するモデル
        X : pd.DataFrame
            特徴量データ
        y : pd.Series
            ターゲットデータ
        cv : int, default=5
            クロスバリデーションの分割数
        n_jobs : int, default=-1
            並列処理に使用するCPUコア数
        train_sizes : np.ndarray, optional
            トレーニングサイズの配列（デフォルトは linspace(0.1, 1.0, 10)）
        scoring : str, default='r2'
            使用するスコアリング指標
        title : str, default='学習曲線'
            プロットのタイトル
        """
        try:
            if train_sizes is None:
                train_sizes = np.linspace(0.1, 1.0, 10)

            self.logger.info(f"学習曲線の計算を開始: {type(estimator).__name__}")

            # 学習曲線の計算
            train_sizes, train_scores, test_scores = learning_curve(
                estimator, X, y, train_sizes=train_sizes, cv=cv,
                scoring=scoring, n_jobs=n_jobs, shuffle=True, random_state=42
            )

            # スコアの平均と標準偏差を計算
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            # プロットの作成
            plt.figure(figsize=(10, 6))

            # トレーニングスコアとテストスコアをプロット
            plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label='トレーニングスコア')
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1, color='b')

            plt.plot(train_sizes, test_scores_mean, 'o-', color='r', label='検証スコア')
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color='r')

            # プロットの装飾
            plt.title(title)
            plt.xlabel('トレーニングサンプル数')
            plt.ylabel(f'{scoring}スコア')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best')

            # 分析結果とアドバイスを表示
            gap = train_scores_mean[-1] - test_scores_mean[-1]
            if gap > 0.2:  # 閾値は調整可能
                plt.figtext(0.5, 0.01, f"過学習の兆候: トレーニングとテストのスコア差 = {gap:.2f}",
                          ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
                self.logger.warning(f"過学習の兆候: トレーニングとテストのスコア差 = {gap:.2f}")
            elif test_scores_mean[-1] < 0.7 and train_scores_mean[-1] < 0.7:  # 閾値は調整可能
                plt.figtext(0.5, 0.01, f"過少学習の可能性: 最終テストスコア = {test_scores_mean[-1]:.2f}",
                          ha="center", fontsize=12, bbox={"facecolor":"yellow", "alpha":0.2, "pad":5})
                self.logger.warning(f"過少学習の可能性: 最終テストスコア = {test_scores_mean[-1]:.2f}")

            self.logger.info(f"学習曲線分析完了 - 最終トレーニングスコア: {train_scores_mean[-1]:.4f}, "
                          f"最終検証スコア: {test_scores_mean[-1]:.4f}")

            plt.tight_layout()

        except Exception as e:
            self._handle_error("学習曲線のプロット", e)

    def plot_validation_curve(self, estimator, X: pd.DataFrame, y: pd.Series,
                            param_name: str, param_range: List,
                            cv: int = 5, scoring: str = 'r2', log_scale: bool = False) -> None:
        """
        パラメータ検証曲線をプロットして最適なハイパーパラメータ値を視覚化する

        Parameters:
        -----------
        estimator : 学習器
            検証曲線を生成するモデル
        X : pd.DataFrame
            特徴量データ
        y : pd.Series
            ターゲットデータ
        param_name : str
            評価するパラメータ名（例: 'max_depth', 'C', 'alpha'）
        param_range : List
            テストするパラメータ値の範囲
        cv : int, default=5
            クロスバリデーションの分割数
        scoring : str, default='r2'
            使用するスコアリング指標
        log_scale : bool, default=False
            x軸を対数スケールで表示するかどうか
        """
        try:
            self.logger.info(f"検証曲線の計算を開始: パラメータ = {param_name}")

            # 検証曲線の計算
            train_scores, test_scores = validation_curve(
                estimator, X, y, param_name=param_name, param_range=param_range,
                cv=cv, scoring=scoring, n_jobs=-1
            )

            # スコアの平均と標準偏差を計算
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            # 最適なパラメータ値を見つける
            best_idx = np.argmax(test_scores_mean)
            best_param = param_range[best_idx]
            best_score = test_scores_mean[best_idx]

            # プロットの作成
            plt.figure(figsize=(10, 6))

            # トレーニングスコアとテストスコアをプロット
            plt.plot(param_range, train_scores_mean, 'o-', color='b', label='トレーニングスコア')
            plt.fill_between(param_range, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1, color='b')

            plt.plot(param_range, test_scores_mean, 'o-', color='r', label='検証スコア')
            plt.fill_between(param_range, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color='r')

            # 最適値をハイライト
            plt.axvline(x=best_param, color='g', linestyle='--',
                       label=f'最適値: {best_param} (スコア: {best_score:.4f})')

            # プロットの装飾
            plt.title(f'パラメータ検証曲線 ({param_name})')
            plt.xlabel(param_name)
            plt.ylabel(f'{scoring}スコア')

            if log_scale and all(p > 0 for p in param_range):
                plt.xscale('log')

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best')

            self.logger.info(f"検証曲線分析完了 - 最適な{param_name}値: {best_param} "
                          f"(検証スコア: {best_score:.4f})")

            plt.tight_layout()

        except Exception as e:
            self._handle_error(f"検証曲線のプロット ({param_name})", e)

    def plot_feature_correlations(self, X: pd.DataFrame, y: pd.Series = None,
                                method: str = 'pearson', figsize: Tuple[int, int] = (12, 10),
                                cmap: str = 'coolwarm', annotate: bool = True,
                                top_n_features: int = None) -> pd.DataFrame:
        """
        特徴量間の相関を計算してヒートマップでプロットする

        Parameters:
        -----------
        X : pd.DataFrame
            特徴量データ
        y : pd.Series, optional
            ターゲット変数（含める場合）
        method : str, default='pearson'
            相関係数の計算方法（'pearson', 'kendall', 'spearman'）
        figsize : Tuple[int, int], default=(12, 10)
            プロットのサイズ
        cmap : str, default='coolwarm'
            ヒートマップのカラーマップ
        annotate : bool, default=True
            相関係数の値をアノテーションとして表示するかどうか
        top_n_features : int, optional
            表示する上位の特徴量数。指定しない場合は全て表示

        Returns:
        --------
        pd.DataFrame
            計算された相関行列
        """
        try:
            # yが提供された場合、それをXに含める
            data = X.copy()
            if y is not None:
                data = pd.concat([X, y.rename('target')], axis=1)

            # 相関行列を計算
            corr_matrix = data.corr(method=method)

            # 上位N個の特徴量に絞る（指定された場合）
            if top_n_features and y is not None and top_n_features < len(X.columns):
                # ターゲットとの相関に基づいて特徴量をソート
                target_corrs = abs(corr_matrix['target']).sort_values(ascending=False)
                top_features = target_corrs.iloc[1:top_n_features+1].index.tolist()  # targetを除く
                selected_cols = top_features + ['target']
                corr_matrix = corr_matrix.loc[selected_cols, selected_cols]

            # プロットの作成
            plt.figure(figsize=figsize)
            mask = np.zeros_like(corr_matrix, dtype=bool)
            mask[np.triu_indices_from(mask)] = True

            # ヒートマップのプロット
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, annot=annotate,
                       fmt='.2f', square=True, linewidths=.5, center=0)

            plt.title('特徴量間の相関係数')
            plt.tight_layout()

            # 強い相関のある特徴量を特定
            if len(corr_matrix) > 1:  # 少なくとも2つの特徴量がある場合
                # 対角成分と上三角部分を除外
                corr_no_diag = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool))
                corr_no_diag = corr_no_diag.where(np.triu(np.ones(corr_no_diag.shape), k=1).astype(bool))

                # 絶対値が0.8以上の相関を持つ特徴量ペアを特定
                strong_corrs = corr_no_diag.unstack().dropna()
                strong_corrs = strong_corrs[abs(strong_corrs) >= 0.8]

                if not strong_corrs.empty:
                    self.logger.info("強い相関を持つ特徴量ペア (|r| >= 0.8):")
                    for idx, val in strong_corrs.items():
                        self.logger.info(f"  {idx[0]} -- {idx[1]}: {val:.3f}")

            return corr_matrix

        except Exception as e:
            self._handle_error("特徴量相関分析", e)

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
        data=data,
        target_column='target'
    )

    # 特徴量重要度の可視化
    importance_result = analyzer.visualize_feature_importance()

    # 予測結果の可視化
    prediction_result = analyzer.visualize_predictions(X_test_processed, y_test)

    print(f"最良モデルの性能: RMSE={prediction_result['metrics']['rmse']:.4f}, R2={prediction_result['metrics']['r2']:.4f}")