from firebase_admin import credentials, firestore, initialize_app
from google.cloud.firestore import Client
from typing import Generator
from contextlib import contextmanager
from backend.config import GOOGLE_CLOUD_PROJECT, CREDENTIALS_PATH

# Firebase初期化
cred = credentials.Certificate(CREDENTIALS_PATH)
firebase_app = initialize_app(cred, {
    'projectId': GOOGLE_CLOUD_PROJECT
})

# Firestoreクライアントの初期化
db = firestore.client()

@contextmanager
def get_db() -> Generator[Client, None, None]:
    """
    Firestoreデータベースクライアントを取得し、コンテキストマネージャーとして使用するための関数です。

    Yields:
        Client: Firestoreクライアントインスタンス

    Example:
        with get_db() as db:
            users_ref = db.collection('users')
            docs = users_ref.stream()
    """
    try:
        yield db
    except Exception as e:
        # エラーログ記録やエラーハンドリングをここで実装
        print(f"Database error: {str(e)}")
        raise
    finally:
        # Firestoreはステートレスなので、明示的なクローズは不要
        pass

# 使用例を含むヘルパー関数
async def get_collection_data(collection_name: str):
    """
    指定されたコレクションのデータを取得するヘルパー関数

    Args:
        collection_name (str): 取得するコレクション名

    Returns:
        list: ドキュメントのリスト
    """
    with get_db() as db:
        collection_ref = db.collection(collection_name)
        docs = collection_ref.stream()
        return [doc.to_dict() for doc in docs]

# バッチ処理用のヘルパー関数
def batch_write(operations: list):
    """
    バッチ処理を実行するヘルパー関数

    Args:
        operations (list): バッチ操作のリスト

    Example:
        operations = [
            {
                'type': 'set',
                'collection': 'users',
                'doc_id': 'user1',
                'data': {'name': 'John Doe'}
            }
        ]
    """
    batch = db.batch()

    try:
        for op in operations:
            ref = db.collection(op['collection']).document(op['doc_id'])
            if op['type'] == 'set':
                batch.set(ref, op['data'])
            elif op['type'] == 'update':
                batch.update(ref, op['data'])
            elif op['type'] == 'delete':
                batch.delete(ref)

        batch.commit()
    except Exception as e:
        print(f"Batch operation failed: {str(e)}")
        raise

# 設定ファイル (backend/config.py)