# -*- coding: utf-8 -*-
"""
モデル評価と改善のFirestoreサービス
予測モデルの精度を監視し、評価結果をFirestoreに保存します。
"""
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from google.cloud import firestore
from google.cloud.firestore_v1.document import DocumentReference
from google.cloud.firestore_v1.query import Query
import asyncio
from backend.service.firestore.client import get_firestore_client, StorageError

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class ModelEvaluationError(Exception):
    """モデル評価に関するエラー"""
    pass

class FirestoreModelEvaluator:
    """
    予測モデルの評価結果をFirestoreで管理するクラス
    """
    def __init__(self):
        """
        Firestoreクライアントを初期化
        """
        try:
            self.db: firestore.Client = get_firestore_client()
            self.collection_name: str = 'model_evaluations'
            logger.info("FirestoreModelEvaluator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FirestoreModelEvaluator: {str(e)}")
            raise ModelEvaluationError(f"Initialization error: {str(e)}") from e

    async def evaluate_and_save(
        self,
        data: pd.DataFrame,
        target_variable: str,
        model_id: str,
        model_version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        モデルを評価し、結果をFirestoreに保存します。

        Args:
            data (pd.DataFrame): 予測結果を含むデータフレーム
            target_variable (str): 予測対象の変数名
            model_id (str): モデルの識別子
            model_version (str): モデルのバージョン
            metadata (Optional[Dict[str, Any]]): 追加のメタデータ

        Returns:
            Dict[str, Any]: 評価結果と保存されたドキュメントID
        """
        try:
            # データバリデーション
            self._validate_input_data(data, target_variable)

            # 評価指標の計算
            metrics = self._calculate_metrics(
                actual=data[target_variable],
                predicted=data['predictions']
            )

            # 保存用のデータ構造作成
            evaluation_data = {
                'model_id': model_id,
                'model_version': model_version,
                'metrics': metrics,
                'timestamp': datetime.now(),
                'sample_size': len(data),
                'metadata': metadata or {}
            }

            # Firestoreに保存
            doc_ref = self.db.collection(self.collection_name).document()
            await self._save_evaluation(doc_ref, evaluation_data)

            logger.info(f"Model evaluation saved successfully: {doc_ref.id}")

            return {
                'document_id': doc_ref.id,
                'metrics': metrics,
                'timestamp': evaluation_data['timestamp']
            }

        except Exception as e:
            error_msg = f"Error in model evaluation: {str(e)}"
            logger.error(error_msg)
            raise ModelEvaluationError(error_msg) from e

    async def get_evaluation_history(
        self,
        model_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        特定モデルの評価履歴を取得します。

        Args:
            model_id (str): モデルの識別子
            limit (int): 取得する履歴の最大数

        Returns:
            List[Dict[str, Any]]: 評価履歴のリスト
        """
        try:
            collection_ref = self.db.collection(self.collection_name)
            query: Query = (collection_ref
                          .where('model_id', '==', model_id)
                          .order_by('timestamp', direction=firestore.Query.DESCENDING)
                          .limit(limit))

            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, query.get)

            history: List[Dict[str, Any]] = []
            for doc in docs:
                if doc is not None:  # Noneチェックの追加
                    data = doc.to_dict()
                    if data is not None:  # データの存在確認
                        data['id'] = doc.id
                        history.append(data)

            logger.info(f"Retrieved {len(history)} evaluation records for model {model_id}")
            return history

        except Exception as e:
            error_msg = f"Error retrieving evaluation history: {str(e)}"
            logger.error(error_msg)
            raise ModelEvaluationError(error_msg) from e

    def _calculate_metrics(
        self,
        actual: pd.Series,
        predicted: pd.Series
    ) -> Dict[str, float]:
        """
        評価指標を計算します。

        Args:
            actual (pd.Series): 実際の値
            predicted (pd.Series): 予測値

        Returns:
            Dict[str, float]: 各種評価指標
        """
        return {
            'mse': float(mean_squared_error(actual, predicted)),
            'mae': float(mean_absolute_error(actual, predicted)),
            'r2': float(r2_score(actual, predicted)),
            'rmse': float(mean_squared_error(actual, predicted, squared=False))
        }

    def _validate_input_data(
        self,
        data: pd.DataFrame,
        target_variable: str
    ) -> None:
        """
        入力データのバリデーションを行います。

        Args:
            data (pd.DataFrame): 検証対象のデータフレーム
            target_variable (str): 予測対象の変数名

        Raises:
            ModelEvaluationError: バリデーションエラー時
        """
        if not isinstance(data, pd.DataFrame):
            raise ModelEvaluationError("Input data must be a pandas DataFrame")

        required_columns = {target_variable, 'predictions'}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ModelEvaluationError(f"Missing required columns: {missing_columns}")

        if data.empty:
            raise ModelEvaluationError("Input DataFrame is empty")

        if data[target_variable].isnull().any() or data['predictions'].isnull().any():
            raise ModelEvaluationError("Input data contains null values")

    async def _save_evaluation(
        self,
        doc_ref: DocumentReference,
        evaluation_data: Dict[str, Any]
    ) -> None:
        """
        評価結果をFirestoreに保存します。

        Args:
            doc_ref (DocumentReference): ドキュメント参照
            evaluation_data (Dict[str, Any]): 保存するデータ
        """
        try:
            if doc_ref is None:
                raise ModelEvaluationError("Invalid document reference")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: doc_ref.set(evaluation_data))
        except Exception as e:
            raise ModelEvaluationError(f"Failed to save evaluation: {str(e)}") from e

    async def close(self) -> None:
        """
        リソースをクリーンアップします。
        """
        try:
            logger.info("FirestoreModelEvaluator closed successfully")
        except Exception as e:
            logger.error(f"Error closing FirestoreModelEvaluator: {str(e)}")
            raise ModelEvaluationError(f"Cleanup error: {str(e)}") from e