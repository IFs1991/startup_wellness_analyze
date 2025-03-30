# -*- coding: utf-8 -*-
"""
パフォーマンス予測モジュール
VASデータを基に企業パフォーマンスを予測します。
機械学習モデルとFirestoreを組み合わせた実装です。
"""
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestRegressor
from service.firestore.client import FirestoreService
from sklearn.linear_model import LinearRegression
import asyncio

# 連合学習モジュールのインポート - エラーを回避するためにフラグのみ設定
FEDERATED_LEARNING_AVAILABLE = False
import warnings
warnings.warn("Federated learning module is not available. Using standard prediction only.")

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class PredictionError(Exception):
    """予測処理に関するエラー"""
    pass

class PerformancePredictor:
    """
    機械学習モデルを使用して将来のパフォーマンスを予測し、
    結果をFirestoreに保存するクラスです。
    """
    def __init__(self, use_federated_learning: bool = False):
        """
        予測モデルとFirestoreサービスを初期化します。

        Args:
            use_federated_learning: 連合学習を使用するかどうか
        """
        # Firestoreサービスの初期化
        self.firestore = FirestoreService()

        # 予測モデルの初期化
        self.models = {
            'regression': LinearRegression(),
            'rf_regressor': RandomForestRegressor(n_estimators=100, random_state=42)
        }

        # 連合学習の設定 - 無効化して標準予測のみを使用
        self.use_federated_learning = False
        self.fl_integration = None
        logger.info("連合学習モジュールは使用しません")

        self.last_training_date = None

    async def predict_and_save(
        self,
        data: pd.DataFrame,
        target_variable: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        use_federated: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        将来パフォーマンスを予測し、結果をFirestoreに保存します。

        Args:
            data (pd.DataFrame): 予測に使用するデータ
            target_variable (str): 予測対象の変数名
            user_id (str): 予測を実行したユーザーのID
            metadata (Optional[Dict[str, Any]]): 予測に関する追加メタデータ
            use_federated (Optional[bool]): 連合学習を使用するかどうか

        Returns:
            Dict[str, Any]: 予測結果とドキュメントID
        """
        # 連合学習使用フラグの設定
        use_federated_learning = self.use_federated_learning if use_federated is None else use_federated

        try:
            logger.info(f"Starting prediction for target variable: {target_variable}")

            # データのバリデーション
            if target_variable not in data.columns:
                raise PredictionError(f"Target variable '{target_variable}' not found in data")

            # 予測の実行
            X = data.drop(columns=[target_variable])
            y = data[target_variable]

            self.models['regression'].fit(X, y)
            predictions = self.models['regression'].predict(X)

            # 予測結果の整形
            prediction_results = pd.DataFrame({
                'actual': y,
                'predicted': predictions,
                'difference': predictions - y
            })

            # 連合学習による予測強化
            fl_prediction_data = {}
            if use_federated_learning and self.fl_integration:
                try:
                    # 業界情報の取得（メタデータから）
                    industry_type = metadata.get('industry_type') if metadata else None

                    # 連合学習による予測拡張
                    fl_result = self.fl_integration.integrate_with_performance_predictor(
                        self, data, industry_type
                    )

                    # 連合学習の予測を通常の予測と統合
                    if 'federated_prediction' in fl_result:
                        # 重み付け係数（連合学習の方がより重視）
                        alpha = 0.6  # 連合学習の重み

                        # 重み付き平均
                        enhanced_predictions = alpha * np.array(fl_result['federated_prediction']) + (1 - alpha) * predictions

                        # 元の予測を保存
                        prediction_results['base_predicted'] = predictions
                        prediction_results['federated_predicted'] = fl_result['federated_prediction']
                        prediction_results['enhanced_predicted'] = enhanced_predictions
                        prediction_results['enhanced_difference'] = enhanced_predictions - y

                        # 予測に連合学習の結果を使用
                        predictions = enhanced_predictions

                    # 連合学習情報を保存
                    fl_prediction_data = {
                        'federated_used': True,
                        'federated_metrics': fl_result.get('metrics', {}),
                        'uncertainty': fl_result.get('uncertainty'),
                        'confidence_interval': fl_result.get('confidence_interval')
                    }

                    logger.info("Enhanced prediction with federated learning")
                except Exception as e:
                    logger.error(f"Failed to apply federated learning: {e}")
                    fl_prediction_data = {'federated_used': False, 'error': str(e)}
            else:
                fl_prediction_data = {'federated_used': False}

            # Firestoreに保存するデータの準備
            prediction_data = {
                'model_type': 'LinearRegression',
                'target_variable': target_variable,
                'feature_names': X.columns.tolist(),
                'metrics': {
                    'r2_score': float(self.models['regression'].score(X, y)),
                    'coefficients': self.models['regression'].coef_.tolist(),
                    'intercept': float(self.models['regression'].intercept_)
                },
                'predictions': prediction_results.to_dict('records'),
                'user_id': user_id,
                'created_at': datetime.now(),
                'metadata': metadata or {},
                'federated_learning': fl_prediction_data
            }

            # Firestoreに保存
            doc_ids = await self.firestore.save_results(
                results=[prediction_data],
                collection_name='performance_predictions'
            )

            # レスポンス準備
            response = {
                'document_id': doc_ids[0],
                'prediction_summary': {
                    'mean_prediction': float(predictions.mean()),
                    'min_prediction': float(predictions.min()),
                    'max_prediction': float(predictions.max())
                },
                'model_metrics': prediction_data['metrics'],
                'federated_learning': fl_prediction_data
            }

            # 不確実性情報を追加（連合学習から提供された場合）
            if fl_prediction_data.get('federated_used') and 'uncertainty' in fl_prediction_data:
                response['prediction_summary']['uncertainty'] = fl_prediction_data['uncertainty']
                response['prediction_summary']['confidence_interval'] = fl_prediction_data['confidence_interval']

            logger.info(f"Successfully completed prediction and saved results with ID: {doc_ids[0]}")
            return response

        except Exception as e:
            error_msg = f"Error in predict_and_save: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(error_msg) from e

    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        同期的に予測を実行する（連合学習連携用）

        Args:
            data (pd.DataFrame): 予測に使用するデータ

        Returns:
            Dict[str, Any]: 予測結果
        """
        try:
            # 入力データが単一行の場合は複数行に変換
            if len(data) == 1:
                # 複製して学習用データセットを作成
                train_data = pd.concat([data] * 3).reset_index(drop=True)

                # 数値型の列を抽出
                numeric_cols = data.select_dtypes(include=[np.number]).columns

                # 各行に少しノイズを加える（学習を可能にするため）
                for col in numeric_cols:
                    if col in train_data.columns:
                        # 元の値を取得
                        original_value = train_data.loc[0, col]

                        # 若干のノイズを加えたデータを生成
                        noise = np.random.normal(0, abs(original_value) * 0.1 + 0.001, len(train_data))
                        train_data[col] = original_value + noise

                # ターゲット変数を仮に作成（連合学習のために必要）
                target_col = 'target'
                train_data[target_col] = train_data[numeric_cols].mean(axis=1)

                # 学習
                X = train_data.drop(columns=[target_col])
                y = train_data[target_col]
                self.models['regression'].fit(X, y)

                # 元のデータで予測
                prediction = self.models['regression'].predict(data)[0]
            else:
                # 数値型の列を抽出
                numeric_cols = data.select_dtypes(include=[np.number]).columns

                # 最も相関の高い列をターゲットとして選択
                correlation_matrix = data[numeric_cols].corr().abs()
                mean_correlation = correlation_matrix.mean()
                target_col = mean_correlation.idxmax()

                # 学習と予測
                X = data.drop(columns=[target_col])
                y = data[target_col]
                self.models['regression'].fit(X, y)
                prediction = self.models['regression'].predict(X).mean()

            # 結果を返す
            return {
                'prediction': prediction,
                'model_type': 'LinearRegression'
            }

        except Exception as e:
            logger.error(f"Error in predict: {e}")
            # エラーの場合はダミー予測を返す
            return {'prediction': 0.5, 'error': str(e)}

    async def train_with_federated_data(
        self,
        data: pd.DataFrame,
        target_variable: str,
        submit_update: bool = True,
        industry_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        連合学習のために、ローカルデータを使用してモデルを訓練します。

        Args:
            data (pd.DataFrame): 訓練データ
            target_variable (str): 予測対象の変数名
            submit_update (bool): 更新を中央サーバーに送信するかどうか
            industry_type (Optional[str]): 業界タイプ

        Returns:
            Dict[str, Any]: 訓練結果
        """
        if not self.use_federated_learning or not self.fl_integration:
            raise PredictionError("連合学習が初期化されていません")

        try:
            # データのバリデーション
            if target_variable not in data.columns:
                raise PredictionError(f"Target variable '{target_variable}' not found in data")

            # 訓練データの準備
            X = data.drop(columns=[target_variable])
            y = data[[target_variable]]

            # 連合学習でモデルを訓練
            metrics = self.fl_integration.train_with_local_data(
                X, y, "financial_performance", submit_update
            )

            logger.info(f"Successfully trained with federated learning: {metrics}")
            return {
                'status': 'success',
                'metrics': metrics,
                'submitted': submit_update
            }

        except Exception as e:
            error_msg = f"Error in federated training: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(error_msg) from e

    async def get_prediction_history(
        self,
        user_id: str,
        limit: Optional[int] = 10
    ) -> List[Dict[str, Any]]:
        """
        ユーザーの予測履歴を取得します。

        Args:
            user_id (str): ユーザーID
            limit (Optional[int]): 取得する履歴の最大数

        Returns:
            List[Dict[str, Any]]: 予測履歴のリスト
        """
        try:
            conditions = [
                {'field': 'user_id', 'operator': '==', 'value': user_id}
            ]

            predictions = await self.firestore.fetch_documents(
                collection_name='performance_predictions',
                conditions=conditions,
                limit=limit
            )

            logger.info(f"Successfully retrieved prediction history for user: {user_id}")
            return predictions

        except Exception as e:
            error_msg = f"Error retrieving prediction history: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(error_msg) from e

    async def close(self) -> None:
        """
        リソースをクリーンアップします。
        """
        try:
            await self.firestore.close()
            logger.info("PerformancePredictor resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up PerformancePredictor resources: {str(e)}")
            raise

async def get_predictor() -> PerformancePredictor:
    """
    PerformancePredictorのインスタンスを取得します。
    """
    return PerformancePredictor()