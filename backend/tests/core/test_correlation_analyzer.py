import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime
import json

from core.correlation_analyzer import CorrelationAnalyzer

@pytest.fixture
def sample_correlation_data():
    """テスト用のサンプルデータを提供します"""
    return pd.DataFrame({
        'revenue': [1000000, 1200000, 800000, 850000, 1500000],
        'expenses': [800000, 900000, 700000, 720000, 1200000],
        'profit': [200000, 300000, 100000, 130000, 300000],
        'employees': [50, 55, 30, 32, 70],
        'customer_satisfaction': [4.2, 4.3, 3.8, 3.9, 4.5]
    })

@pytest.mark.asyncio
@patch('core.correlation_analyzer.FirestoreService')
async def test_analyze(mock_firestore_service, sample_correlation_data):
    """相関分析の実行をテスト"""
    # モックの設定
    firestore_service_instance = AsyncMock()
    mock_firestore_service.return_value = firestore_service_instance
    firestore_service_instance.add_document.return_value = "test_analysis_id"

    # 相関分析ツールのインスタンス作成
    analyzer = CorrelationAnalyzer()

    # 分析対象の変数
    variables = ['revenue', 'expenses', 'profit', 'customer_satisfaction']

    # 分析の実行
    result = await analyzer.analyze(
        data=sample_correlation_data,
        variables=variables,
        user_id="test_user_id",
        analysis_name="テスト分析",
        metadata={"purpose": "テスト用"}
    )

    # 結果の検証
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (len(variables), len(variables))

    # 相関行列の値を検証
    # revenueとexpensesの相関は高いはず
    assert result.loc['revenue', 'expenses'] > 0.9

    # Firestoreへの保存が呼ばれたことを確認
    firestore_service_instance.add_document.assert_called_once()
    args, kwargs = firestore_service_instance.add_document.call_args

    # 保存されたデータの検証
    assert args[0] == 'correlation_analysis'
    assert 'variables' in args[1]
    assert 'correlation_matrix' in args[1]
    assert 'user_id' in args[1]
    assert args[1]['user_id'] == "test_user_id"
    assert args[1]['analysis_name'] == "テスト分析"

@pytest.mark.asyncio
@patch('core.correlation_analyzer.FirestoreService')
async def test_get_analysis_history(mock_firestore_service):
    """分析履歴の取得をテスト"""
    # モックの設定
    firestore_service_instance = AsyncMock()
    mock_firestore_service.return_value = firestore_service_instance

    # モックのレスポンス設定
    mock_analyses = [
        {
            'id': 'analysis1',
            'analysis_name': '分析1',
            'user_id': 'test_user_id',
            'created_at': datetime.now(),
            'variables': ['revenue', 'expenses', 'profit'],
            'correlation_matrix': json.dumps({
                'revenue': {'revenue': 1.0, 'expenses': 0.95, 'profit': 0.85},
                'expenses': {'revenue': 0.95, 'expenses': 1.0, 'profit': 0.65},
                'profit': {'revenue': 0.85, 'expenses': 0.65, 'profit': 1.0}
            })
        },
        {
            'id': 'analysis2',
            'analysis_name': '分析2',
            'user_id': 'test_user_id',
            'created_at': datetime.now(),
            'variables': ['revenue', 'customer_satisfaction'],
            'correlation_matrix': json.dumps({
                'revenue': {'revenue': 1.0, 'customer_satisfaction': 0.75},
                'customer_satisfaction': {'revenue': 0.75, 'customer_satisfaction': 1.0}
            })
        }
    ]
    firestore_service_instance.query_documents.return_value = mock_analyses

    # 相関分析ツールのインスタンス作成
    analyzer = CorrelationAnalyzer()

    # 分析履歴の取得
    result = await analyzer.get_analysis_history(user_id="test_user_id", limit=10)

    # 結果の検証
    assert result is not None
    assert len(result) == 2
    assert result[0]['id'] == 'analysis1'
    assert result[1]['id'] == 'analysis2'

    # Firestoreからのクエリが呼ばれたことを確認
    firestore_service_instance.query_documents.assert_called_once()
    args, kwargs = firestore_service_instance.query_documents.call_args

    # クエリ条件の検証
    assert args[0] == 'correlation_analysis'
    assert len(kwargs['filters']) == 1
    assert kwargs['filters'][0]['field'] == 'user_id'
    assert kwargs['filters'][0]['op'] == '=='
    assert kwargs['filters'][0]['value'] == 'test_user_id'
    assert kwargs['limit'] == 10

@pytest.mark.asyncio
@patch('core.correlation_analyzer.FirestoreService')
async def test_get_analysis_by_id(mock_firestore_service):
    """IDによる分析の取得をテスト"""
    # モックの設定
    firestore_service_instance = AsyncMock()
    mock_firestore_service.return_value = firestore_service_instance

    # モックのレスポンス設定
    mock_analysis = {
        'id': 'analysis1',
        'analysis_name': '分析1',
        'user_id': 'test_user_id',
        'created_at': datetime.now(),
        'variables': ['revenue', 'expenses', 'profit'],
        'correlation_matrix': json.dumps({
            'revenue': {'revenue': 1.0, 'expenses': 0.95, 'profit': 0.85},
            'expenses': {'revenue': 0.95, 'expenses': 1.0, 'profit': 0.65},
            'profit': {'revenue': 0.85, 'expenses': 0.65, 'profit': 1.0}
        })
    }
    firestore_service_instance.get_document.return_value = mock_analysis

    # 相関分析ツールのインスタンス作成
    analyzer = CorrelationAnalyzer()

    # IDによる分析の取得
    result = await analyzer.get_analysis_by_id("analysis1")

    # 結果の検証
    assert result is not None
    assert result['id'] == 'analysis1'
    assert result['analysis_name'] == '分析1'
    assert result['user_id'] == 'test_user_id'
    assert 'correlation_matrix' in result

    # Firestoreからのドキュメント取得が呼ばれたことを確認
    firestore_service_instance.get_document.assert_called_once_with('correlation_analysis', 'analysis1')

@pytest.mark.asyncio
@patch('core.correlation_analyzer.FirestoreService')
async def test_update_analysis_metadata(mock_firestore_service):
    """分析メタデータの更新をテスト"""
    # モックの設定
    firestore_service_instance = AsyncMock()
    mock_firestore_service.return_value = firestore_service_instance
    firestore_service_instance.update_document.return_value = None

    # 相関分析ツールのインスタンス作成
    analyzer = CorrelationAnalyzer()

    # 更新用メタデータ
    metadata = {
        'description': '更新された説明',
        'tags': ['重要', 'ビジネス', '財務']
    }

    # メタデータの更新
    await analyzer.update_analysis_metadata("analysis1", metadata)

    # Firestoreのドキュメント更新が呼ばれたことを確認
    firestore_service_instance.update_document.assert_called_once()
    args, kwargs = firestore_service_instance.update_document.call_args

    # 更新データの検証
    assert args[0] == 'correlation_analysis'
    assert args[1] == 'analysis1'
    assert args[2]['metadata'] == metadata