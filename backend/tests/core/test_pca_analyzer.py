import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from core.pca_analyzer import FirestorePCAAnalyzer, PCAAnalysisError

@pytest.fixture
def sample_pca_data():
    """テスト用のサンプルデータを提供します"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100),
        'feature4': np.random.normal(0, 1, 100),
        'feature5': np.random.normal(0, 1, 100)
    })

@pytest.mark.asyncio
@patch('core.pca_analyzer.FirestoreService')
@patch('core.pca_analyzer.PCA')
@patch('core.pca_analyzer.StandardScaler')
async def test_analyze_and_save(mock_standard_scaler, mock_pca, mock_firestore_service, sample_pca_data):
    """PCA分析の実行と保存をテスト"""
    # モックの設定
    firestore_service_instance = AsyncMock()
    mock_firestore_service.return_value = firestore_service_instance
    firestore_service_instance.add_document.return_value = "test_analysis_id"

    # StandardScalerのモック設定
    scaler_instance = MagicMock()
    mock_standard_scaler.return_value = scaler_instance
    scaled_data = sample_pca_data.copy()
    scaler_instance.fit_transform.return_value = scaled_data.values
    scaler_instance.inverse_transform.return_value = sample_pca_data.values

    # PCAのモック設定
    pca_instance = MagicMock()
    mock_pca.return_value = pca_instance
    pca_instance.fit_transform.return_value = np.random.random((100, 2))
    pca_instance.components_ = np.random.random((2, 5))
    pca_instance.explained_variance_ratio_ = np.array([0.6, 0.3])

    # PCA分析ツールのインスタンス作成
    analyzer = FirestorePCAAnalyzer()

    # 分析メタデータ
    analysis_metadata = {
        "analysis_name": "テストPCA分析",
        "description": "テスト用の説明",
        "tags": ["テスト", "PCA", "次元削減"]
    }

    # 分析の実行
    result, analysis_id = await analyzer.analyze_and_save(
        data=sample_pca_data,
        n_components=2,
        analysis_metadata=analysis_metadata,
        user_id="test_user_id"
    )

    # 結果の検証
    assert result is not None
    assert analysis_id == "test_analysis_id"

    # モック関数の呼び出しを検証
    scaler_instance.fit_transform.assert_called_once()
    pca_instance.fit_transform.assert_called_once()

    # Firestoreへの保存が呼ばれたことを確認
    firestore_service_instance.add_document.assert_called_once()
    args, kwargs = firestore_service_instance.add_document.call_args

    # 保存されたデータの検証
    assert args[0] == 'pca_analyses'
    assert 'components' in args[1]
    assert 'explained_variance_ratio' in args[1]
    assert 'transformed_data' in args[1]
    assert 'user_id' in args[1]
    assert args[1]['user_id'] == "test_user_id"
    assert args[1]['n_components'] == 2
    assert args[1]['metadata']['analysis_name'] == "テストPCA分析"

@pytest.mark.asyncio
@patch('core.pca_analyzer.FirestoreService')
async def test_get_analysis_history(mock_firestore_service):
    """分析履歴の取得をテスト"""
    # モックの設定
    firestore_service_instance = AsyncMock()
    mock_firestore_service.return_value = firestore_service_instance

    # モックのレスポンス設定
    mock_analyses = [
        {
            'id': 'pca1',
            'metadata': {
                'analysis_name': 'PCA分析1',
                'description': '説明1'
            },
            'user_id': 'test_user_id',
            'created_at': datetime.now(),
            'n_components': 2,
            'original_features': ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
            'explained_variance_ratio': [0.6, 0.3]
        },
        {
            'id': 'pca2',
            'metadata': {
                'analysis_name': 'PCA分析2',
                'description': '説明2'
            },
            'user_id': 'test_user_id',
            'created_at': datetime.now(),
            'n_components': 3,
            'original_features': ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
            'explained_variance_ratio': [0.5, 0.3, 0.15]
        }
    ]
    firestore_service_instance.query_documents.return_value = mock_analyses

    # PCA分析ツールのインスタンス作成
    analyzer = FirestorePCAAnalyzer()

    # 分析履歴の取得
    result = await analyzer.get_analysis_history(limit=10, user_id="test_user_id")

    # 結果の検証
    assert result is not None
    assert len(result) == 2
    assert result[0]['id'] == 'pca1'
    assert result[1]['id'] == 'pca2'
    assert result[0]['metadata']['analysis_name'] == 'PCA分析1'
    assert result[1]['metadata']['analysis_name'] == 'PCA分析2'

    # Firestoreからのクエリが呼ばれたことを確認
    firestore_service_instance.query_documents.assert_called_once()
    args, kwargs = firestore_service_instance.query_documents.call_args

    # クエリ条件の検証
    assert args[0] == 'pca_analyses'
    assert len(kwargs['filters']) == 1
    assert kwargs['filters'][0]['field'] == 'user_id'
    assert kwargs['filters'][0]['op'] == '=='
    assert kwargs['filters'][0]['value'] == 'test_user_id'
    assert kwargs['limit'] == 10

@pytest.mark.asyncio
@patch('core.pca_analyzer.FirestoreService')
async def test_error_handling(mock_firestore_service, sample_pca_data):
    """エラーハンドリングをテスト"""
    # モックの設定
    firestore_service_instance = AsyncMock()
    mock_firestore_service.return_value = firestore_service_instance

    # エラーを発生させる
    firestore_service_instance.add_document.side_effect = Exception("テストエラー")

    # PCA分析ツールのインスタンス作成
    analyzer = FirestorePCAAnalyzer()

    # エラーが発生することを確認
    with pytest.raises(PCAAnalysisError):
        await analyzer.analyze_and_save(
            data=sample_pca_data,
            n_components=2,
            user_id="test_user_id"
        )

@pytest.mark.asyncio
@patch('core.pca_analyzer.FirestoreService')
async def test_close(mock_firestore_service):
    """リソースのクローズをテスト"""
    # モックの設定
    firestore_service_instance = AsyncMock()
    mock_firestore_service.return_value = firestore_service_instance
    firestore_service_instance.close.return_value = None

    # PCA分析ツールのインスタンス作成
    analyzer = FirestorePCAAnalyzer()

    # クローズの実行
    await analyzer.close()

    # Firestoreクライアントがクローズされたことを確認
    firestore_service_instance.close.assert_called_once()