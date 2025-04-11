# -*- coding: utf-8 -*-
"""
リポジトリパターンのテスト
新しいデータアクセス抽象化レイヤーのテストコードを提供します。
"""
import unittest
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys
import os

# テスト用のモックリポジトリ
class MockRepository:
    """テスト用モックリポジトリ"""

    def __init__(self):
        self.data = {}
        self.id_counter = 1

    def find_by_id(self, id):
        """IDによる検索"""
        return self.data.get(id)

    def find_all(self, skip=0, limit=100):
        """全件取得"""
        values = list(self.data.values())
        return values[skip:skip+limit]

    def find_by_criteria(self, criteria, skip=0, limit=100):
        """条件による検索"""
        result = []
        for item in self.data.values():
            match = True
            for key, value in criteria.items():
                if not hasattr(item, key) or getattr(item, key) != value:
                    match = False
                    break
            if match:
                result.append(item)
        return result[skip:skip+limit]

    def save(self, entity):
        """保存"""
        if not hasattr(entity, "entity_id") or not entity.entity_id:
            # 新しいIDを割り当て
            entity_id = str(self.id_counter)
            self.id_counter += 1
            # entity_idに直接アクセスできないので、dictに変換してからentity_idを設定
            data = entity.to_dict()
            # 新しいエンティティを作成
            cls = entity.__class__
            new_entity = cls(**data)
            # entity_idを設定（通常はエンティティのプロパティを通じて設定する）
            object.__setattr__(new_entity, "_entity_id", entity_id)
            self.data[entity_id] = new_entity
            return new_entity
        else:
            # 既存エンティティの更新
            self.data[entity.entity_id] = entity
            return entity

    def update(self, id, data):
        """更新"""
        entity = self.find_by_id(id)
        if not entity:
            raise ValueError(f"Entity with ID {id} not found")

        # データの更新
        for key, value in data.items():
            if hasattr(entity, key):
                setattr(entity, key, value)

        self.data[id] = entity
        return entity

    def delete(self, id):
        """削除"""
        if id not in self.data:
            raise ValueError(f"Entity with ID {id} not found")
        del self.data[id]
        return True

    def count(self, criteria=None):
        """件数取得"""
        if not criteria:
            return len(self.data)
        return len(self.find_by_criteria(criteria))

# テストケース
class RepositoryPatternTest(unittest.TestCase):
    """リポジトリパターンのテスト"""

    def setUp(self):
        """テスト前の準備"""
        # ルートディレクトリをパスに追加（インポート用）
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

        # モジュールをインポート
        from backend.database.models import StartupEntity, UserEntity, VASDataEntity, ModelType
        from backend.database.repository import Repository, DataCategory

        # テスト用のクラスと変数を保存
        self.StartupEntity = StartupEntity
        self.UserEntity = UserEntity
        self.VASDataEntity = VASDataEntity
        self.ModelType = ModelType
        self.DataCategory = DataCategory

        # モックリポジトリの準備
        self.repository = MockRepository()

    def test_entity_creation(self):
        """エンティティ作成のテスト"""
        # スタートアップエンティティを作成
        startup = self.StartupEntity(
            name="テストスタートアップ",
            industry="AI",
            founding_date=datetime(2023, 1, 1),
            owner_id="user1"
        )

        # 基本的なプロパティの検証
        self.assertEqual(startup.name, "テストスタートアップ")
        self.assertEqual(startup.industry, "AI")
        self.assertEqual(startup.owner_id, "user1")

        # エンティティIDと作成日時の検証
        self.assertIsNotNone(startup.created_at)
        self.assertIsNotNone(startup.updated_at)

    def test_repository_save_and_find(self):
        """リポジトリの保存と検索のテスト"""
        # スタートアップエンティティを作成
        startup = self.StartupEntity(
            name="テストスタートアップ2",
            industry="HealthTech",
            founding_date=datetime(2023, 2, 1),
            owner_id="user2"
        )

        # リポジトリに保存
        saved_startup = self.repository.save(startup)

        # エンティティIDが割り当てられていることを確認
        self.assertIsNotNone(saved_startup.entity_id)

        # IDで検索
        found_startup = self.repository.find_by_id(saved_startup.entity_id)

        # 検索結果の検証
        self.assertIsNotNone(found_startup)
        self.assertEqual(found_startup.name, "テストスタートアップ2")
        self.assertEqual(found_startup.industry, "HealthTech")

    def test_repository_update(self):
        """リポジトリの更新のテスト"""
        # スタートアップエンティティを作成して保存
        startup = self.StartupEntity(
            name="更新テスト",
            industry="IoT",
            founding_date=datetime(2023, 3, 1),
            owner_id="user3"
        )
        saved_startup = self.repository.save(startup)

        # エンティティを更新
        updated = self.repository.update(
            saved_startup.entity_id,
            {"name": "更新後", "employee_count": 10}
        )

        # 更新結果の検証
        self.assertEqual(updated.name, "更新後")
        self.assertEqual(updated.employee_count, 10)

        # 更新されていない項目の確認
        self.assertEqual(updated.industry, "IoT")

    def test_repository_delete(self):
        """リポジトリの削除のテスト"""
        # スタートアップエンティティを作成して保存
        startup = self.StartupEntity(
            name="削除テスト",
            industry="FinTech",
            founding_date=datetime(2023, 4, 1),
            owner_id="user4"
        )
        saved_startup = self.repository.save(startup)

        # 削除前の確認
        self.assertIsNotNone(self.repository.find_by_id(saved_startup.entity_id))

        # エンティティを削除
        result = self.repository.delete(saved_startup.entity_id)

        # 削除結果の検証
        self.assertTrue(result)
        self.assertIsNone(self.repository.find_by_id(saved_startup.entity_id))

    def test_repository_find_by_criteria(self):
        """条件による検索のテスト"""
        # 複数のスタートアップエンティティを作成して保存
        for i in range(5):
            startup = self.StartupEntity(
                name=f"検索テスト{i}",
                industry="AI" if i % 2 == 0 else "HealthTech",
                founding_date=datetime(2023, 5, i+1),
                owner_id=f"user{i+5}"
            )
            self.repository.save(startup)

        # 業界による検索
        ai_startups = self.repository.find_by_criteria({"industry": "AI"})

        # 検索結果の検証
        self.assertEqual(len(ai_startups), 3)  # i=0,2,4の3件
        for startup in ai_startups:
            self.assertEqual(startup.industry, "AI")

    def test_entity_to_dict(self):
        """エンティティの辞書変換のテスト"""
        # スタートアップエンティティを作成
        startup = self.StartupEntity(
            name="辞書変換テスト",
            industry="EdTech",
            founding_date=datetime(2023, 6, 1),
            owner_id="user10"
        )

        # 辞書に変換
        data = startup.to_dict()

        # 辞書の内容検証
        self.assertEqual(data["name"], "辞書変換テスト")
        self.assertEqual(data["industry"], "EdTech")
        self.assertIn("created_at", data)
        self.assertIn("updated_at", data)

# テスト実行
if __name__ == "__main__":
    unittest.main()