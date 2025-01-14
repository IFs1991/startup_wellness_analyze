# -*- coding: utf-8 -*-
"""
特徴量エンジニアリング
Firestoreと連携して分析に有効な特徴量を作成・保存します。
"""
from typing import Dict, List, Optional, Any, Union, Callable
import pandas as pd
import logging
from datetime import datetime
import asyncio
from firebase_admin import firestore
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 型エイリアスの定義
FirestoreClient = Any

class FeatureEngineeringError(Exception):
    """特徴量エンジニアリング処理に関するエラー"""
    pass

class FeatureConfig(BaseModel):
    """特徴量の設定を定義するモデル"""
    moving_average_window: int = 7
    calculate_change_rate: bool = True
    custom_features: Optional[List[str]] = None

class FeatureEngineer:
    """
    Firestoreと連携して特徴量エンジニアリングを行うクラスです。
    """
    def __init__(self, db: FirestoreClient):
        """
        初期化
        Args:
            db: Firestoreクライアント
        """
        self.db = db
        self._executor = ThreadPoolExecutor(max_workers=4)
        logger.info("FeatureEngineer initialized")

    async def process_and_save_features(
        self,
        collection_name: str,
        user_id: str,
        config: Optional[FeatureConfig] = None,
        conditions: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        データを取得し、特徴量エンジニアリングを行い、結果を保存します。
        """
        try:
            logger.info(f"Starting feature engineering process for user {user_id}")

            if config is None:
                config = FeatureConfig()

            data = await self._fetch_data(collection_name, conditions)
            if data.empty:
                raise FeatureEngineeringError("No data found for processing")

            processed_data = await self.engineer_features(data, config)
            doc_id = await self._save_results(processed_data, user_id)

            logger.info(f"Feature engineering completed successfully for user {user_id}")
            return doc_id

        except Exception as e:
            error_msg = f"Error in feature engineering process: {str(e)}"
            logger.error(error_msg)
            raise FeatureEngineeringError(error_msg) from e

    async def engineer_features(
        self,
        data: pd.DataFrame,
        config: FeatureConfig
    ) -> pd.DataFrame:
        """
        特徴量エンジニアリングを実行します。
        """
        try:
            logger.debug("Starting feature engineering calculations")
            result = data.copy()

            # 非同期タスクのリスト
            tasks = []
            loop = asyncio.get_event_loop()

            if config.calculate_change_rate:
                tasks.append(
                    loop.run_in_executor(
                        self._executor,
                        self._calculate_change_rates,
                        result
                    )
                )

            if config.moving_average_window > 0:
                tasks.append(
                    loop.run_in_executor(
                        self._executor,
                        self._calculate_moving_averages,
                        result,
                        config.moving_average_window
                    )
                )

            if config.custom_features:
                for feature in config.custom_features:
                    tasks.append(
                        loop.run_in_executor(
                            self._executor,
                            self._calculate_custom_feature,
                            result,
                            feature
                        )
                    )

            # すべてのタスクを実行して結果を待機
            await asyncio.gather(*tasks)

            logger.debug("Feature engineering calculations completed")
            return result

        except Exception as e:
            error_msg = f"Error during feature engineering: {str(e)}"
            logger.error(error_msg)
            raise FeatureEngineeringError(error_msg) from e

    async def _fetch_data(
        self,
        collection_name: str,
        conditions: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Firestoreからデータを取得します。
        """
        try:
            query = self.db.collection(collection_name)

            if conditions:
                for condition in conditions:
                    field = condition.get('field')
                    operator = condition.get('operator', '==')
                    value = condition.get('value')
                    if all([field, operator, value is not None]):
                        query = query.where(field, operator, value)

            docs = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: query.get()
            )

            data = [doc.to_dict() for doc in docs]
            return pd.DataFrame(data)

        except Exception as e:
            error_msg = f"Error fetching data: {str(e)}"
            logger.error(error_msg)
            raise FeatureEngineeringError(error_msg) from e

    async def _save_results(
        self,
        data: pd.DataFrame,
        user_id: str
    ) -> str:
        """
        特徴量エンジニアリング結果をFirestoreに保存します。
        """
        try:
            result_dict = {
                'user_id': user_id,
                'features': data.to_dict('records'),
                'created_at': datetime.now(),
                'metadata': {
                    'row_count': len(data),
                    'column_count': len(data.columns),
                    'columns': list(data.columns)
                }
            }

            doc_ref = self.db.collection('feature_engineering_results').document()

            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: doc_ref.set(result_dict)
            )

            return doc_ref.id

        except Exception as e:
            error_msg = f"Error saving results: {str(e)}"
            logger.error(error_msg)
            raise FeatureEngineeringError(error_msg) from e

    @staticmethod
    def _calculate_change_rates(data: pd.DataFrame) -> pd.DataFrame:
        """変化率を計算"""
        try:
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                data[f'{col}_change_rate'] = data[col].pct_change()
            return data
        except Exception as e:
            logger.error(f"Error calculating change rates: {str(e)}")
            raise

    @staticmethod
    def _calculate_moving_averages(
        data: pd.DataFrame,
        window: int
    ) -> pd.DataFrame:
        """移動平均を計算"""
        try:
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                data[f'{col}_ma_{window}'] = data[col].rolling(window=window).mean()
            return data
        except Exception as e:
            logger.error(f"Error calculating moving averages: {str(e)}")
            raise

    @staticmethod
    def _calculate_custom_feature(
        data: pd.DataFrame,
        feature_name: str
    ) -> pd.DataFrame:
        """カスタム特徴量を計算"""
        try:
            # カスタム特徴量の実装をここに追加
            return data
        except Exception as e:
            logger.error(f"Error calculating custom feature {feature_name}: {str(e)}")
            raise

    def __del__(self):
        """デストラクタ：Executorのシャットダウンを確実に行う"""
        self._executor.shutdown(wait=True)

def get_feature_engineer(db: FirestoreClient) -> FeatureEngineer:
    """
    FeatureEngineerのインスタンスを取得します。
    """
    return FeatureEngineer(db)