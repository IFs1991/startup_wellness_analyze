# -*- coding: utf-8 -*-
"""
データ品質管理モジュール
データの品質を監視・管理し、結果をFirestoreに保存します。

Features:
    - データの整合性チェック
    - 品質メトリクスの計算
    - チェック結果の永続化
"""
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from firebase_admin import firestore
from ..service.firestore.client import FirestoreService, StorageError  # パスを修正



# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class DataQualityError(Exception):
    """データ品質チェック関連のエラー"""
    pass

class DataQualityChecker:
    """
    データの品質をチェックし、結果をFirestoreに保存するクラス
    """
    def __init__(self, firestore_service: Optional[FirestoreService] = None):
        """
        初期化
        Args:
            firestore_service: FirestoreServiceのインスタンス
        """
        self.firestore_service = firestore_service or FirestoreService()
        self.collection_name = "data_quality_reports"

    async def check_data_quality(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        dataset_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        データ品質チェックを実行し、結果をFirestoreに保存

        Args:
            data: チェック対象のデータ
            dataset_id: データセットの識別子
            metadata: 追加のメタデータ

        Returns:
            Dict[str, Any]: チェック結果のレポート

        Raises:
            DataQualityError: データ品質チェック処理でエラーが発生した場合
        """
        try:
            # DataFrameへの変換
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

            # 基本的な品質メトリクスの計算
            quality_metrics = self._calculate_quality_metrics(df)

            # 詳細チェックの実行
            detailed_checks = self._perform_detailed_checks(df)

            # レポートの作成
            report = {
                'dataset_id': dataset_id,
                'timestamp': datetime.now(),
                'row_count': len(df),
                'column_count': len(df.columns),
                'quality_metrics': quality_metrics,
                'detailed_checks': detailed_checks,
                'metadata': metadata or {},
                'status': 'completed'
            }

            # Firestoreへの保存
            doc_ids = await self.firestore_service.save_results(
                results=[report],
                collection_name=self.collection_name
            )

            report['report_id'] = doc_ids[0]
            logger.info(f"Quality check completed for dataset {dataset_id}, report ID: {doc_ids[0]}")

            return report

        except Exception as e:
            error_msg = f"Error in data quality check: {str(e)}"
            logger.error(error_msg)
            raise DataQualityError(error_msg) from e

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        基本的な品質メトリクスを計算

        Args:
            df: 分析対象のDataFrame

        Returns:
            Dict[str, Any]: 品質メトリクス
        """
        try:
            return {
                'missing_values': {
                    'total': df.isna().sum().sum(),
                    'by_column': df.isna().sum().to_dict()
                },
                'duplicate_rows': df.duplicated().sum(),
                'column_types': df.dtypes.astype(str).to_dict(),
                'numeric_columns_stats': self._get_numeric_stats(df),
            }
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {str(e)}")
            raise DataQualityError("Failed to calculate quality metrics") from e

    def _perform_detailed_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        詳細なデータ品質チェックを実行

        Args:
            df: チェック対象のDataFrame

        Returns:
            Dict[str, Any]: 詳細チェック結果
        """
        try:
            return {
                'completeness': self._check_completeness(df),
                'consistency': self._check_consistency(df),
                'validity': self._check_validity(df)
            }
        except Exception as e:
            logger.error(f"Error performing detailed checks: {str(e)}")
            raise DataQualityError("Failed to perform detailed checks") from e

    def _get_numeric_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        数値列の統計情報を計算

        Args:
            df: 分析対象のDataFrame

        Returns:
            Dict[str, Dict[str, float]]: 数値列ごとの統計情報
        """
        numeric_stats = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            stats = df[col].describe()
            numeric_stats[col] = {
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max'],
                'quartile_25': stats['25%'],
                'median': stats['50%'],
                'quartile_75': stats['75%']
            }

        return numeric_stats

    def _check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        データの完全性をチェック
        """
        return {
            'missing_rate': (df.isna().sum() / len(df)).to_dict(),
            'complete_records': (len(df) - df.isna().any(axis=1).sum()) / len(df)
        }

    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        データの一貫性をチェック
        """
        return {
            'unique_counts': df.nunique().to_dict(),
            'constant_columns': [col for col in df.columns if df[col].nunique() == 1]
        }

    def _check_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        データの妥当性をチェック
        """
        validity_checks = {
            'zero_variance_columns': [
                col for col in df.select_dtypes(include=[np.number]).columns
                if df[col].std() == 0
            ]
        }

        return validity_checks

    async def get_quality_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        品質チェックレポートを取得

        Args:
            report_id: レポートID

        Returns:
            Optional[Dict[str, Any]]: 品質チェックレポート
        """
        try:
            results = await self.firestore_service.fetch_documents(
                collection_name=self.collection_name,
                conditions=[{'field': 'id', 'operator': '==', 'value': report_id}],
                limit=1
            )
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Error retrieving quality report: {str(e)}")
            raise DataQualityError("Failed to retrieve quality report") from e