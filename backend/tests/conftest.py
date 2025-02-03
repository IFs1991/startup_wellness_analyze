"""テスト用の共通フィクスチャ"""
import os
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Dict, Generator
from unittest.mock import MagicMock, patch
from backend.src.database.firestore.client import FirestoreClient
import sys
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
sys.path.append(str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
def event_loop():
    """イベントループのフィクスチャ"""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def mock_firestore():
    """Firestoreクライアントのモック"""
    with patch('backend.src.database.firestore.client.get_firestore_client') as mock:
        mock_client = MagicMock()
        mock.return_value = mock_client
        yield mock_client

@pytest.fixture
def test_user() -> Dict[str, str]:
    """テスト用のユーザー情報"""
    return {
        "id": "test_user_id",
        "email": "test@example.com",
        "name": "Test User"
    }

@pytest.fixture
def test_template() -> Dict[str, str]:
    """テスト用のテンプレート情報"""
    return {
        "id": "test_template_id",
        "name": "Test Template",
        "description": "Test template description"
    }

@pytest.fixture
def auth_headers() -> Dict[str, str]:
    """認証用ヘッダー"""
    return {"Authorization": "Bearer test_token"}