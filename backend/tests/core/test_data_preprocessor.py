import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

from core.data_preprocessor import (
    DataPreprocessor,
    DataPreprocessingError
)

@pytest.fixture
def sample_raw_data():
    """生のデータサンプルを提供します"""
    return [
        {
            'id': 'doc1',
            'company_id': 'comp1',
            'revenue': 1000000,
            'expenses': 800000,
            'profit': 200000,
            'employees': 50,
            'timestamp': datetime(2022, 1, 1),
            'year': 2022,
            'quarter': 1,
            'growth_rate': 0.05,
            'customer_satisfaction': 4.2
        },
        {
            'id': 'doc2',
            'company_id': 'comp1',
            'revenue': 1200000,
            'expenses': 900000,
            'profit': 300000,
            'employees': 55,
            'timestamp': datetime(2022, 2, 1),
            'year': 2022,
            'quarter': 1,
            'growth_rate': 0.08,
            'customer_satisfaction': 4.3
        },
        {
            'id': 'doc3',
            'company_id': 'comp2',
            'revenue': 800000,
            'expenses': 700000,
            'profit': 100000,
            'employees': 30,
            'timestamp': datetime(2022, 1, 1),
            'year': 2022,
            'quarter': 1,
            'growth_rate': 0.03,
            'customer_satisfaction': 3.8
        }
    ]

@pytest.fixture
def sample_vas_data():
    """VASデータのサンプルを提供します"""
    return [
        {
            'id': 'vas1',
            'company_id': 'comp1',
            'timestamp': datetime(2022, 1, 1),
            'employee_satisfaction': 4.2,
            'work_life_balance': 3.8,
            'team_collaboration': 4.0,
            'leadership_quality': 3.5,
            'career_growth': 3.7
        },
        {
            'id': 'vas2',
            'company_id': 'comp1',
            'timestamp': datetime(2022, 2, 1),
            'employee_satisfaction': 4.3,
            'work_life_balance': 3.9,
            'team_collaboration': 4.1,
            'leadership_quality': 3.6,
            'career_growth': 3.8
        },
        {
            'id': 'vas3',
            'company_id': 'comp1',
            'timestamp': datetime(2022, 3, 1),
            'employee_satisfaction': 4.4,
            'work_life_balance': 4.0,
            'team_collaboration': 4.2,
            'leadership_quality': 3.8,
            'career_growth': 3.9
        }
    ]

@pytest.fixture
def mock_firestore_service():
    """FirestoreServiceのモックを提供します"""
    service = MagicMock()
    service.query_documents = AsyncMock()
    return service

def test_singleton_pattern():
    """シングルトンパターンが正しく機能することを確認するテスト"""
    # FirestoreServiceのモックを作成
    with patch('core.data_preprocessor.FirestoreService') as mock_service_class:
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        # 最初のインスタンス
        preprocessor1 = DataPreprocessor()

        # 2つ目のインスタンス（同じオブジェクトが返される）
        preprocessor2 = DataPreprocessor()

        # 同じインスタンスであることを確認
        assert preprocessor1 is preprocessor2

        # FirestoreServiceは一度だけ初期化されることを確認
        mock_service_class.assert_called_once()

def test_preprocess_firestore_data_financial(sample_raw_data):
    """財務データの前処理機能をテスト"""
    with patch('core.data_preprocessor.FirestoreService'):
        preprocessor = DataPreprocessor()

        # 前処理メソッドをパッチしてモックする
        with patch.object(preprocessor, '_process_timestamp') as mock_process_timestamp, \
             patch.object(preprocessor, '_convert_data_types') as mock_convert_types, \
             patch.object(preprocessor, '_handle_missing_values') as mock_handle_missing, \
             patch.object(preprocessor, '_handle_outliers') as mock_handle_outliers, \
             patch.object(preprocessor, '_validate_columns') as mock_validate:

            # モックの戻り値を設定
            sample_df = pd.DataFrame(sample_raw_data)
            mock_process_timestamp.return_value = sample_df
            mock_convert_types.return_value = sample_df
            mock_handle_missing.return_value = sample_df
            mock_handle_outliers.return_value = sample_df

            # 前処理を実行
            result = preprocessor.preprocess_firestore_data(
                sample_raw_data,
                data_type='financial_data'
            )

            # 結果がDataFrameであることを確認
            assert isinstance(result, pd.DataFrame)

            # 各メソッドが正しく呼び出されたことを確認
            mock_validate.assert_called_once()
            mock_process_timestamp.assert_called_once()
            mock_convert_types.assert_called_once()
            mock_handle_missing.assert_called_once()
            mock_handle_outliers.assert_called_once()

def test_preprocess_firestore_data_vas(sample_vas_data):
    """VASデータの前処理機能をテスト"""
    with patch('core.data_preprocessor.FirestoreService'):
        preprocessor = DataPreprocessor()

        # 前処理メソッドをパッチ
        with patch.object(preprocessor, '_process_timestamp') as mock_process_timestamp, \
             patch.object(preprocessor, '_convert_data_types') as mock_convert_types, \
             patch.object(preprocessor, '_handle_missing_values') as mock_handle_missing, \
             patch.object(preprocessor, '_handle_outliers') as mock_handle_outliers, \
             patch.object(preprocessor, '_validate_columns') as mock_validate:

            # モックの戻り値を設定
            sample_df = pd.DataFrame(sample_vas_data)
            mock_process_timestamp.return_value = sample_df
            mock_convert_types.return_value = sample_df
            mock_handle_missing.return_value = sample_df
            mock_handle_outliers.return_value = sample_df

            # 前処理を実行
            result = preprocessor.preprocess_firestore_data(
                sample_vas_data,
                data_type='vas_data'
            )

            # 結果がDataFrameであることを確認
            assert isinstance(result, pd.DataFrame)

            # 各メソッドが正しく呼び出されたことを確認
            mock_validate.assert_called_once()
            mock_process_timestamp.assert_called_once()
            mock_convert_types.assert_called_once()
            mock_handle_missing.assert_called_once()
            mock_handle_outliers.assert_called_once()

def test_process_timestamp():
    """タイムスタンプ処理機能をテスト"""
    with patch('core.data_preprocessor.FirestoreService'):
        preprocessor = DataPreprocessor()

        # テスト用のDataFrameを作成
        test_df = pd.DataFrame({
            'timestamp': [
                datetime(2022, 1, 1),
                datetime(2022, 2, 1),
                datetime(2022, 3, 1)
            ],
            'value': [1, 2, 3]
        })

        # タイムスタンプ処理を実行
        result = preprocessor._process_timestamp(test_df)

        # 結果を検証
        assert 'timestamp' in result.columns
        assert pd.api.types.is_datetime64_dtype(result['timestamp'])
        assert 'year' in result.columns
        assert 'month' in result.columns
        assert 'day' in result.columns

def test_handle_missing_values():
    """欠損値処理機能をテスト"""
    with patch('core.data_preprocessor.FirestoreService'):
        preprocessor = DataPreprocessor()

        # 欠損値を含むテスト用のDataFrameを作成
        test_df = pd.DataFrame({
            'numeric_col': [1.0, np.nan, 3.0, 4.0, 5.0],
            'category_col': ['A', 'B', np.nan, 'D', 'E'],
            'timestamp': [
                datetime(2022, 1, 1),
                datetime(2022, 2, 1),
                datetime(2022, 3, 1),
                datetime(2022, 4, 1),
                datetime(2022, 5, 1)
            ]
        })

        # 欠損値処理のオプションを設定
        options = {
            'numeric_fill_method': 'mean',
            'category_fill_method': 'most_frequent'
        }

        # 欠損値処理を実行
        result = preprocessor._handle_missing_values(test_df, options)

        # 結果を検証 - 欠損値がないことを確認
        assert result['numeric_col'].isna().sum() == 0
        assert result['category_col'].isna().sum() == 0

        # 数値カラムが平均値で埋められていることを確認
        assert result['numeric_col'].iloc[1] == test_df['numeric_col'].mean()

def test_handle_outliers():
    """異常値処理機能をテスト"""
    with patch('core.data_preprocessor.FirestoreService'):
        preprocessor = DataPreprocessor()

        # 異常値を含むテスト用のDataFrameを作成
        test_df = pd.DataFrame({
            'numeric_col': [1.0, 2.0, 3.0, 4.0, 100.0],  # 100は明らかな異常値
            'category_col': ['A', 'B', 'C', 'D', 'E'],
            'timestamp': [
                datetime(2022, 1, 1),
                datetime(2022, 2, 1),
                datetime(2022, 3, 1),
                datetime(2022, 4, 1),
                datetime(2022, 5, 1)
            ]
        })

        # 異常値処理のオプションを設定
        options = {
            'outlier_detection_method': 'zscore',
            'zscore_threshold': 2.0,
            'outlier_handling': 'cap'
        }

        # 異常値処理を実行
        result = preprocessor._handle_outliers(test_df, options)

        # 結果を検証 - 異常値が処理されていることを確認
        assert result['numeric_col'].max() < 100.0

@pytest.mark.asyncio
async def test_get_data(mock_firestore_service):
    """データ取得機能をテスト"""
    with patch('core.data_preprocessor.FirestoreService', return_value=mock_firestore_service):
        preprocessor = DataPreprocessor()

        # Firestoreからのレスポンスをモック
        mock_data = [{'id': 'doc1', 'data': {'field': 'value'}}]
        mock_firestore_service.query_documents.return_value = mock_data

        # 条件を設定
        conditions = [
            {'field': 'company_id', 'op': '==', 'value': 'comp1'}
        ]

        # データ取得を実行
        result = await preprocessor.get_data('collection_name', conditions)

        # 結果を検証
        mock_firestore_service.query_documents.assert_called_once_with(
            'collection_name',
            filters=conditions,
            order_by=None,
            limit=None
        )

        # DataFrameとして返されることを確認
        assert isinstance(result, pd.DataFrame)

def test_merge_datasets():
    """データセット結合機能をテスト"""
    with patch('core.data_preprocessor.FirestoreService'):
        preprocessor = DataPreprocessor()

        # テスト用のDataFrameを作成
        vas_df = pd.DataFrame({
            'company_id': ['comp1', 'comp1', 'comp2'],
            'timestamp': [
                datetime(2022, 1, 1),
                datetime(2022, 2, 1),
                datetime(2022, 1, 1)
            ],
            'employee_satisfaction': [4.2, 4.3, 3.8]
        })

        financial_df = pd.DataFrame({
            'company_id': ['comp1', 'comp1', 'comp2'],
            'timestamp': [
                datetime(2022, 1, 1),
                datetime(2022, 2, 1),
                datetime(2022, 1, 1)
            ],
            'revenue': [1000000, 1200000, 800000]
        })

        # データセットを結合
        result = preprocessor.merge_datasets(vas_df, financial_df, merge_on=['company_id', 'timestamp'])

        # 結果を検証
        assert 'company_id' in result.columns
        assert 'timestamp' in result.columns
        assert 'employee_satisfaction' in result.columns
        assert 'revenue' in result.columns
        assert len(result) == 3  # 3つの行があるはず

def test_forward_fill():
    """前方補間機能をテスト"""
    with patch('core.data_preprocessor.FirestoreService'):
        preprocessor = DataPreprocessor()

        # 欠損値を含むテスト用のDataFrameを作成
        test_df = pd.DataFrame({
            'numeric_col': [1.0, np.nan, 3.0, np.nan, 5.0],
            'category_col': ['A', np.nan, 'C', np.nan, 'E']
        })

        # 前方補間を実行
        result = preprocessor._forward_fill(test_df)

        # 結果を検証
        assert result['numeric_col'].iloc[1] == 1.0  # 2番目の欠損値が前の値で埋められている
        assert result['numeric_col'].iloc[3] == 3.0  # 4番目の欠損値が前の値で埋められている
        assert result['category_col'].iloc[1] == 'A'  # カテゴリ列も同様に処理されている

def test_backward_fill():
    """後方補間機能をテスト"""
    with patch('core.data_preprocessor.FirestoreService'):
        preprocessor = DataPreprocessor()

        # 欠損値を含むテスト用のDataFrameを作成
        test_df = pd.DataFrame({
            'numeric_col': [np.nan, 2.0, np.nan, 4.0, np.nan],
            'category_col': [np.nan, 'B', np.nan, 'D', np.nan]
        })

        # 後方補間を実行
        result = preprocessor._backward_fill(test_df)

        # 結果を検証
        assert result['numeric_col'].iloc[0] == 2.0  # 1番目の欠損値が後ろの値で埋められている
        assert result['numeric_col'].iloc[2] == 4.0  # 3番目の欠損値が後ろの値で埋められている
        assert result['category_col'].iloc[0] == 'B'  # カテゴリ列も同様に処理されている

def test_data_type_conversion():
    """データ型変換機能をテスト"""
    with patch('core.data_preprocessor.FirestoreService'):
        preprocessor = DataPreprocessor()

        # テスト用のDataFrameを作成
        test_df = pd.DataFrame({
            'numeric_col': ['1', '2', '3'],
            'float_col': ['1.1', '2.2', '3.3'],
            'category_col': ['A', 'B', 'C'],
            'bool_col': ['True', 'False', 'True']
        })

        # データ型変換を実行
        type_mapping = {
            'numeric_col': 'int',
            'float_col': 'float',
            'category_col': 'category',
            'bool_col': 'bool'
        }

        # _convert_data_typesメソッドをモックして直接テスト
        result = preprocessor._convert_data_types(test_df, type_mapping)

        # 結果を検証
        assert pd.api.types.is_integer_dtype(result['numeric_col'])
        assert pd.api.types.is_float_dtype(result['float_col'])
        assert pd.api.types.is_categorical_dtype(result['category_col'])
        assert pd.api.types.is_bool_dtype(result['bool_col'])