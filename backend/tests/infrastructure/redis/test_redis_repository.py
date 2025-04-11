"""
Redisキャッシュリポジトリのテスト

このモジュールはRedisCacheRepositoryの機能をテストします。
モックRedisクライアントを使用してユニットテストを行います。
"""

import asyncio
import pytest
import json
from typing import Dict, Optional
from pydantic import BaseModel

from backend.infrastructure.redis.redis_client import RedisClientInterface
from backend.infrastructure.redis.redis_repository import RedisCacheRepository, create_redis_cache_repository


class MockRedisClient(RedisClientInterface):
    """
    テスト用のRedisクライアントモック
    """

    def __init__(self):
        self.data: Dict[str, str] = {}

    async def connect(self):
        """接続（モック）"""
        return True

    async def get(self, key: str) -> Optional[str]:
        """キーに対応する値を取得"""
        return self.data.get(key)

    async def get_json(self, key: str) -> Optional[dict]:
        """キーに対応するJSON値を取得して辞書に変換"""
        value = self.data.get(key)
        if value is None:
            return None
        return json.loads(value)

    async def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        """キーと値のペアを保存"""
        self.data[key] = value
        return True

    async def set_json(self, key: str, value: dict, expire: Optional[int] = None) -> bool:
        """辞書をJSON文字列に変換して保存"""
        self.data[key] = json.dumps(value)
        return True

    async def delete(self, key: str) -> bool:
        """キーの削除"""
        if key in self.data:
            del self.data[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        """キーの存在確認"""
        return key in self.data

    async def flush_all(self) -> bool:
        """すべてのキーを削除"""
        self.data.clear()
        return True


class TestUserModel(BaseModel):
    """テスト用のPydanticモデル"""
    id: int
    name: str
    email: str


class TestRedisCacheRepository:
    """RedisCacheRepositoryのテストクラス"""

    @pytest.fixture
    def redis_client(self):
        """モックRedisクライアントのフィクスチャ"""
        return MockRedisClient()

    @pytest.fixture
    def repository(self, redis_client):
        """テスト用リポジトリのフィクスチャ"""
        return RedisCacheRepository(redis_client, TestUserModel, "test")

    @pytest.fixture
    def test_user(self):
        """テスト用ユーザーデータのフィクスチャ"""
        return TestUserModel(id=1, name="テストユーザー", email="test@example.com")

    @pytest.mark.asyncio
    async def test_set_and_get(self, repository, test_user):
        """セットと取得のテスト"""
        # データを保存
        result = await repository.set("user:1", test_user)
        assert result is True

        # データを取得
        retrieved_user = await repository.get("user:1")
        assert retrieved_user is not None
        assert retrieved_user.id == test_user.id
        assert retrieved_user.name == test_user.name
        assert retrieved_user.email == test_user.email

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, repository):
        """存在しないキーの取得テスト"""
        result = await repository.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, repository, test_user):
        """削除テスト"""
        # データを保存
        await repository.set("user:1", test_user)

        # 削除する
        result = await repository.delete("user:1")
        assert result is True

        # 取得を試みる
        retrieved = await repository.get("user:1")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_exists(self, repository, test_user):
        """存在確認テスト"""
        # データを保存
        await repository.set("user:1", test_user)

        # 存在を確認
        exists = await repository.exists("user:1")
        assert exists is True

        # 存在しないキー
        exists = await repository.exists("nonexistent")
        assert exists is False

    @pytest.mark.asyncio
    async def test_clear_all(self, repository, test_user):
        """全削除テスト"""
        # 複数のデータを保存
        await repository.set("user:1", test_user)
        await repository.set("user:2", TestUserModel(id=2, name="ユーザー2", email="user2@example.com"))

        # すべて削除
        result = await repository.clear_all()
        assert result is True

        # 確認
        exists = await repository.exists("user:1")
        assert exists is False
        exists = await repository.exists("user:2")
        assert exists is False

    @pytest.mark.asyncio
    async def test_factory_function(self, redis_client):
        """ファクトリ関数のテスト"""
        repo = create_redis_cache_repository(redis_client, TestUserModel, "factory_test")
        assert isinstance(repo, RedisCacheRepository)

        # プレフィックスの確認
        test_user = TestUserModel(id=1, name="テストユーザー", email="test@example.com")
        await repo.set("user:1", test_user)

        # 内部的には正しいキーで保存されていることを確認
        internal_key = f"TestUserModel:factory_test:user:1"
        assert await redis_client.exists(internal_key) is True