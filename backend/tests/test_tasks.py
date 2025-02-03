import pytest
from typing import Dict, Any, AsyncGenerator
from datetime import datetime
from pytest import FixtureRequest
from backend.src.tasks.tasks import TaskProcessor
from backend.src.database.firestore.client import FirestoreClient

@pytest.fixture
def task_processor() -> TaskProcessor:
    return TaskProcessor()

@pytest.fixture
def mock_config() -> Dict[str, Any]:
    return {
        'data_source': 'test_data_source',  # テスト用のコレクション名
        'filters': [
            {'field': 'status', 'operator': '==', 'value': 'active'},
            {'field': 'date', 'operator': '>=', 'value': '2024-01-01'}
        ]
    }

@pytest.fixture
async def cleanup_test_data() -> AsyncGenerator[None, None]:
    """テストデータのクリーンアップ"""
    yield
    client = FirestoreClient()
    # テストで使用したコレクションのデータを削除
    collections = ['test_data_source', 'visualizations']
    for collection in collections:
        docs = await client.query_documents(collection=collection)
        for doc in docs:
            if doc.get('id', '').startswith('test_'):
                await client.delete_document(collection=collection, doc_id=doc['id'])

@pytest.mark.asyncio
async def test_process_visualization_data_success(
    task_processor: TaskProcessor,
    mock_config: Dict[str, Any],
    cleanup_test_data: None
) -> None:
    """可視化データ処理の正常系テスト"""
    visualization_id = 'test_viz_001'
    user_id = 'test_user_001'

    # テストデータの作成
    client = FirestoreClient()
    test_data = {'id': 'test_1', 'status': 'active', 'date': '2024-01-01'}
    await client.create_document(
        collection='test_data_source',
        doc_id='test_1',
        data=test_data
    )

    # 処理実行
    await task_processor.process_visualization_data(
        visualization_id=visualization_id,
        config=mock_config,
        user_id=user_id
    )

    # 結果の検証
    result = await task_processor.firestore_client.get_document(
        collection='visualizations',
        doc_id=visualization_id
    )

    assert result is not None
    assert result['id'] == visualization_id
    assert result['status'] == 'completed'
    assert result['user_id'] == user_id
    assert 'data' in result
    assert 'source' in result['data']
    assert 'last_updated' in result['data']

@pytest.mark.asyncio
async def test_process_visualization_data_error(
    task_processor: TaskProcessor,
    cleanup_test_data: None
) -> None:
    """可視化データ処理のエラー系テスト"""
    visualization_id = 'test_viz_002'
    user_id = 'test_user_001'
    invalid_config: Dict[str, Any] = {
        'data_source': None  # 無効な設定
    }

    # 処理実行
    await task_processor.process_visualization_data(
        visualization_id=visualization_id,
        config=invalid_config,
        user_id=user_id
    )

    # 結果の検証
    result = await task_processor.firestore_client.get_document(
        collection='visualizations',
        doc_id=visualization_id
    )

    assert result is not None
    assert result['id'] == visualization_id
    assert result['status'] == 'error'
    assert result['user_id'] == user_id
    assert 'error' in result
    assert 'last_updated' in result