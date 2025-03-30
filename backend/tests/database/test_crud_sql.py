import pytest
from unittest.mock import patch, MagicMock
import uuid
from datetime import datetime

from backend.database import crud_sql
from backend.database.models_sql import User, Startup, VASData, FinancialData, Note

class TestUserCrudSQL:
    """ユーザー関連のSQL CRUD操作のテスト"""

    def test_get_user(self, test_db, sample_user_data):
        """ユーザー取得関数のテスト"""
        # テスト用のユーザーをデータベースに追加
        db_user = User(**sample_user_data)
        test_db.add(db_user)
        test_db.commit()

        # 関数を呼び出し
        user = crud_sql.get_user(test_db, sample_user_data["id"])

        # 結果を確認
        assert user is not None
        assert user.id == sample_user_data["id"]
        assert user.username == sample_user_data["username"]
        assert user.email == sample_user_data["email"]

    def test_get_user_not_found(self, test_db):
        """存在しないユーザーの取得テスト"""
        # 存在しないIDでユーザーを検索
        user = crud_sql.get_user(test_db, "nonexistent_id")

        # 結果を確認
        assert user is None

    def test_get_user_by_username(self, test_db, sample_user_data):
        """ユーザー名によるユーザー取得関数のテスト"""
        # テスト用のユーザーをデータベースに追加
        db_user = User(**sample_user_data)
        test_db.add(db_user)
        test_db.commit()

        # 関数を呼び出し
        user = crud_sql.get_user_by_username(test_db, sample_user_data["username"])

        # 結果を確認
        assert user is not None
        assert user.username == sample_user_data["username"]

    def test_get_user_by_email(self, test_db, sample_user_data):
        """メールアドレスによるユーザー取得関数のテスト"""
        # テスト用のユーザーをデータベースに追加
        db_user = User(**sample_user_data)
        test_db.add(db_user)
        test_db.commit()

        # 関数を呼び出し
        user = crud_sql.get_user_by_email(test_db, sample_user_data["email"])

        # 結果を確認
        assert user is not None
        assert user.email == sample_user_data["email"]

    def test_get_users(self, test_db, sample_user_data):
        """ユーザー一覧取得関数のテスト"""
        # テスト用のユーザーをデータベースに追加
        user1 = User(**sample_user_data)

        user2_data = sample_user_data.copy()
        user2_data["id"] = str(uuid.uuid4())
        user2_data["username"] = "testuser2"
        user2_data["email"] = "test2@example.com"
        user2 = User(**user2_data)

        test_db.add(user1)
        test_db.add(user2)
        test_db.commit()

        # 関数を呼び出し
        users = crud_sql.get_users(test_db, skip=0, limit=10)

        # 結果を確認
        assert len(users) >= 2
        assert any(u.id == sample_user_data["id"] for u in users)
        assert any(u.id == user2_data["id"] for u in users)

    def test_create_user(self, test_db):
        """ユーザー作成関数のテスト"""
        # テストデータ
        user_data = {
            "username": "newuser",
            "email": "new@example.com",
            "hashed_password": "hashedpwd",
            "is_active": True
        }

        # 関数を呼び出し
        user = crud_sql.create_user(test_db, user_data)

        # 結果を確認
        assert user is not None
        assert user.id is not None
        assert user.username == "newuser"
        assert user.email == "new@example.com"

        # データベースに実際に保存されたことを確認
        db_user = test_db.query(User).filter(User.id == user.id).first()
        assert db_user is not None
        assert db_user.username == "newuser"

    def test_update_user(self, test_db, sample_user_data):
        """ユーザー更新関数のテスト"""
        # テスト用のユーザーをデータベースに追加
        db_user = User(**sample_user_data)
        test_db.add(db_user)
        test_db.commit()

        # 更新データ
        update_data = {
            "username": "updateduser",
            "is_active": False
        }

        # 関数を呼び出し
        updated_user = crud_sql.update_user(test_db, sample_user_data["id"], update_data)

        # 結果を確認
        assert updated_user is not None
        assert updated_user.username == "updateduser"
        assert updated_user.is_active == False
        assert updated_user.email == sample_user_data["email"]  # 変更していない項目は維持される

        # データベースに実際に反映されたことを確認
        db_user = test_db.query(User).filter(User.id == sample_user_data["id"]).first()
        assert db_user.username == "updateduser"
        assert db_user.is_active == False

    def test_delete_user(self, test_db, sample_user_data):
        """ユーザー削除関数のテスト"""
        # テスト用のユーザーをデータベースに追加
        db_user = User(**sample_user_data)
        test_db.add(db_user)
        test_db.commit()

        # 関数を呼び出し
        result = crud_sql.delete_user(test_db, sample_user_data["id"])

        # 結果を確認
        assert result is True

        # データベースから実際に削除されたことを確認
        db_user = test_db.query(User).filter(User.id == sample_user_data["id"]).first()
        assert db_user is None


class TestStartupCrudSQL:
    """スタートアップ関連のSQL CRUD操作のテスト"""

    def test_get_startup(self, test_db, sample_startup_data, sample_user_data):
        """スタートアップ取得関数のテスト"""
        # テスト用のユーザーとスタートアップをデータベースに追加
        db_user = User(**sample_user_data)
        test_db.add(db_user)

        db_startup = Startup(**sample_startup_data)
        test_db.add(db_startup)
        test_db.commit()

        # 関数を呼び出し
        startup = crud_sql.get_startup(test_db, sample_startup_data["id"])

        # 結果を確認
        assert startup is not None
        assert startup.id == sample_startup_data["id"]
        assert startup.name == sample_startup_data["name"]
        assert startup.owner_id == sample_user_data["id"]

    def test_get_startups(self, test_db, sample_startup_data, sample_user_data):
        """スタートアップ一覧取得関数のテスト"""
        # テスト用のユーザーとスタートアップをデータベースに追加
        db_user = User(**sample_user_data)
        test_db.add(db_user)

        db_startup1 = Startup(**sample_startup_data)

        startup2_data = sample_startup_data.copy()
        startup2_data["id"] = str(uuid.uuid4())
        startup2_data["name"] = "Test Startup 2"
        db_startup2 = Startup(**startup2_data)

        test_db.add(db_startup1)
        test_db.add(db_startup2)
        test_db.commit()

        # 関数を呼び出し
        startups = crud_sql.get_startups(test_db, skip=0, limit=10)

        # 結果を確認
        assert len(startups) >= 2
        assert any(s.id == sample_startup_data["id"] for s in startups)
        assert any(s.id == startup2_data["id"] for s in startups)

    def test_get_startups_by_owner(self, test_db, sample_startup_data, sample_user_data):
        """オーナーごとのスタートアップ一覧取得関数のテスト"""
        # テスト用のユーザーとスタートアップをデータベースに追加
        db_user = User(**sample_user_data)
        test_db.add(db_user)

        db_startup = Startup(**sample_startup_data)
        test_db.add(db_startup)
        test_db.commit()

        # 関数を呼び出し
        startups = crud_sql.get_startups_by_owner(test_db, sample_user_data["id"])

        # 結果を確認
        assert len(startups) == 1
        assert startups[0].id == sample_startup_data["id"]
        assert startups[0].owner_id == sample_user_data["id"]

    def test_create_startup(self, test_db, sample_user_data):
        """スタートアップ作成関数のテスト"""
        # テスト用のユーザーをデータベースに追加
        db_user = User(**sample_user_data)
        test_db.add(db_user)
        test_db.commit()

        # テストデータ
        startup_data = {
            "name": "New Startup",
            "industry": "FinTech",
            "owner_id": sample_user_data["id"],
            "description": "Test description"
        }

        # 関数を呼び出し
        startup = crud_sql.create_startup(test_db, startup_data)

        # 結果を確認
        assert startup is not None
        assert startup.id is not None
        assert startup.name == "New Startup"
        assert startup.industry == "FinTech"
        assert startup.owner_id == sample_user_data["id"]

        # データベースに実際に保存されたことを確認
        db_startup = test_db.query(Startup).filter(Startup.id == startup.id).first()
        assert db_startup is not None
        assert db_startup.name == "New Startup"