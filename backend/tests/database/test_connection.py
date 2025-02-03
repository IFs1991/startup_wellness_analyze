import pytest
import asyncio
from google.cloud import firestore
from google.cloud.firestore import AsyncClient
from backend.src.database.connection import FirestoreConnection
from backend.src.database.config import FirestoreConfig

@pytest.fixture
async def firestore_config():
    """テスト用のFirestore設定を作成"""
    return FirestoreConfig(
        project_id="test-project",
        emulator_host="localhost:8080"
    )

@pytest.fixture
async def db_connection(firestore_config):
    """テスト用のFirestore接続を作成"""
    connection = FirestoreConnection(firestore_config)
    await connection.initialize()
    yield connection
    await connection.cleanup()

@pytest.mark.asyncio
async def test_firestore_connection_initialization(db_connection):
    """Firestore接続の初期化をテスト"""
    # 接続を確認
    is_connected = await db_connection.check_connection()
    assert is_connected is True

    # クライアントを取得してコレクションにアクセス
    client = db_connection.get_client()
    test_ref = client.collection('test').document('test')
    await test_ref.set({'test': True})
    doc = await test_ref.get()
    assert doc.exists
    assert doc.to_dict()['test'] is True

@pytest.mark.asyncio
async def test_firestore_connection_from_config():
    """設定からのFirestore接続をテスト"""
    config = FirestoreConfig(
        project_id="test-project",
        emulator_host="localhost:8080"
    )

    connection = FirestoreConnection(config)
    try:
        await connection.initialize()
        is_connected = await connection.check_connection()
        assert is_connected is True

        # 基本的なドキュメント操作をテスト
        client = connection.get_client()
        test_ref = client.collection('test').document('test')
        await test_ref.set({'test': True})
        doc = await test_ref.get()
        assert doc.exists
    finally:
        await connection.cleanup()

@pytest.mark.asyncio
async def test_firestore_connection_cleanup(db_connection):
    """Firestore接続のクリーンアップをテスト"""
    # テストデータを作成
    client = db_connection.get_client()
    test_ref = client.collection('test').document('cleanup_test')
    await test_ref.set({'test': True})

    # クリーンアップを実行
    await db_connection.cleanup()

    # 接続が正しく切断されていることを確認
    with pytest.raises(Exception):
        await test_ref.get()

@pytest.mark.asyncio
async def test_firestore_connection_error_handling():
    """Firestore接続のエラーハンドリングをテスト"""
    # 無効な設定で接続を試みる
    invalid_config = FirestoreConfig(
        project_id="invalid-project",
        emulator_host="invalid:9999"
    )
    connection = FirestoreConnection(invalid_config)

    try:
        # 接続を確認
        is_connected = await connection.check_connection()
        assert is_connected is False

        # 初期化を試みる
        with pytest.raises(Exception):
            await connection.initialize()
    finally:
        await connection.cleanup()

@pytest.mark.asyncio
async def test_firestore_batch_operations(db_connection):
    """Firestoreのバッチ操作をテスト"""
    client = db_connection.get_client()
    batch = client.batch()

    # バッチで複数のドキュメントを作成
    docs = []
    for i in range(3):
        ref = client.collection('test').document(f'batch_test_{i}')
        batch.set(ref, {'index': i})
        docs.append(ref)

    await batch.commit()

    # 作成されたドキュメントを確認
    for i, ref in enumerate(docs):
        doc = await ref.get()
        assert doc.exists
        assert doc.to_dict()['index'] == i

@pytest.mark.asyncio
async def test_firestore_transaction(db_connection):
    """Firestoreのトランザクションをテスト"""
    client = db_connection.get_client()
    doc_ref = client.collection('test').document('transaction_test')

    # 初期データを設定
    await doc_ref.set({'counter': 0})

    @firestore.async_transactional
    async def increment_counter(transaction, doc_ref):
        snapshot = await transaction.get(doc_ref)
        transaction.update(doc_ref, {
            'counter': snapshot.get('counter') + 1
        })

    # トランザクションを実行
    transaction = client.transaction()
    await increment_counter(transaction, doc_ref)

    # 結果を確認
    doc = await doc_ref.get()
    assert doc.to_dict()['counter'] == 1