# -*- coding: utf-8 -*-
"""
将来パフォーマンス予測
機械学習モデルを使用して将来パフォーマンス (売上高, 従業員満足度など) を予測し、
結果をFirestoreに保存します。
"""
from typing import Dict, Any, List, Optional
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
import logging
import asyncio
from backend.service.firestore.client import FirestoreService

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
    def __init__(self):
        """
        予測モデルとFirestoreサービスを初期化します。
        """
        try:
            self.model = LinearRegression()
            self.firestore_service = FirestoreService()
            logger.info("PerformancePredictor initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize PerformancePredictor: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(error_msg) from e

    async def predict_and_save(
        self,
        data: pd.DataFrame,
        target_variable: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        将来パフォーマンスを予測し、結果をFirestoreに保存します。

        Args:
            data (pd.DataFrame): 予測に使用するデータ
            target_variable (str): 予測対象の変数名
            user_id (str): 予測を実行したユーザーのID
            metadata (Optional[Dict[str, Any]]): 予測に関する追加メタデータ

        Returns:
            Dict[str, Any]: 予測結果とドキュメントID
        """
        try:
            logger.info(f"Starting prediction for target variable: {target_variable}")

            # データのバリデーション
            if target_variable not in data.columns:
                raise PredictionError(f"Target variable '{target_variable}' not found in data")

            # 予測の実行
            X = data.drop(columns=[target_variable])
            y = data[target_variable]

            self.model.fit(X, y)
            predictions = self.model.predict(X)

            # 予測結果の整形
            prediction_results = pd.DataFrame({
                'actual': y,
                'predicted': predictions,
                'difference': predictions - y
            })

            # Firestoreに保存するデータの準備
            prediction_data = {
                'model_type': 'LinearRegression',
                'target_variable': target_variable,
                'feature_names': X.columns.tolist(),
                'metrics': {
                    'r2_score': float(self.model.score(X, y)),
                    'coefficients': self.model.coef_.tolist(),
                    'intercept': float(self.model.intercept_)
                },
                'predictions': prediction_results.to_dict('records'),
                'user_id': user_id,
                'created_at': datetime.now(),
                'metadata': metadata or {}
            }

            # Firestoreに保存
            doc_ids = await self.firestore_service.save_results(
                results=[prediction_data],
                collection_name='performance_predictions'
            )

            response = {
                'document_id': doc_ids[0],
                'prediction_summary': {
                    'mean_prediction': float(predictions.mean()),
                    'min_prediction': float(predictions.min()),
                    'max_prediction': float(predictions.max())
                },
                'model_metrics': prediction_data['metrics']
            }

            logger.info(f"Successfully completed prediction and saved results with ID: {doc_ids[0]}")
            return response

        except Exception as e:
            error_msg = f"Error in predict_and_save: {str(e)}"
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

            predictions = await self.firestore_service.fetch_documents(
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
            await self.firestore_service.close()
            logger.info("PerformancePredictor resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up PerformancePredictor resources: {str(e)}")
            raise

async def get_predictor() -> PerformancePredictor:
    """
    PerformancePredictorのインスタンスを取得します。
    """
    return PerformancePredictor()