import pytest
from sqlalchemy.orm import Session
import uuid
from datetime import datetime

from backend.database.models_sql import User, Startup, VASData, FinancialData, Note

class TestUserModel:
    """Userモデルのテスト"""

    def test_user_creation(self, test_db):
        """Userモデルの作成と取得をテスト"""
        # テスト用ユーザーデータ
        user_id = str(uuid.uuid4())
        username = "testuser"
        email = "test@example.com"

        # ユーザーインスタンス作成
        user = User(
            id=user_id,
            username=username,
            email=email,
            hashed_password="hashedpwd",
            is_active=True
        )

        # DBに保存
        test_db.add(user)
        test_db.commit()

        # DBから取得して検証
        db_user = test_db.query(User).filter(User.id == user_id).first()
        assert db_user is not None
        assert db_user.id == user_id
        assert db_user.username == username
        assert db_user.email == email
        assert db_user.hashed_password == "hashedpwd"
        assert db_user.is_active == True
        assert db_user.created_at is not None
        assert db_user.updated_at is not None

    def test_user_relationships(self, test_db):
        """Userモデルのリレーションシップをテスト"""
        # ユーザー作成
        user_id = str(uuid.uuid4())
        user = User(
            id=user_id,
            username="relationuser",
            email="relation@example.com",
            hashed_password="hashedpwd"
        )
        test_db.add(user)
        test_db.commit()

        # スタートアップ作成
        startup_id = str(uuid.uuid4())
        startup = Startup(
            id=startup_id,
            name="Test Startup",
            owner_id=user_id
        )
        test_db.add(startup)

        # ノート作成
        note_id = str(uuid.uuid4())
        note = Note(
            id=note_id,
            content="Test note",
            user_id=user_id,
            startup_id=startup_id
        )
        test_db.add(note)

        test_db.commit()

        # ユーザーを再取得して検証
        db_user = test_db.query(User).filter(User.id == user_id).first()

        # リレーションシップの確認
        assert len(db_user.startups) == 1
        assert db_user.startups[0].id == startup_id
        assert db_user.startups[0].name == "Test Startup"

        assert len(db_user.notes) == 1
        assert db_user.notes[0].id == note_id
        assert db_user.notes[0].content == "Test note"


class TestStartupModel:
    """Startupモデルのテスト"""

    def test_startup_creation(self, test_db, sample_user_data):
        """Startupモデルの作成と取得をテスト"""
        # テスト用のユーザーを作成
        user = User(**sample_user_data)
        test_db.add(user)
        test_db.commit()

        # テスト用スタートアップデータ
        startup_id = str(uuid.uuid4())
        name = "Test Startup"

        # スタートアップインスタンス作成
        startup = Startup(
            id=startup_id,
            name=name,
            description="A test startup company",
            industry="Technology",
            owner_id=sample_user_data["id"]
        )

        # DBに保存
        test_db.add(startup)
        test_db.commit()

        # DBから取得して検証
        db_startup = test_db.query(Startup).filter(Startup.id == startup_id).first()
        assert db_startup is not None
        assert db_startup.id == startup_id
        assert db_startup.name == name
        assert db_startup.description == "A test startup company"
        assert db_startup.industry == "Technology"
        assert db_startup.owner_id == sample_user_data["id"]
        assert db_startup.created_at is not None

    def test_startup_relationships(self, test_db, sample_user_data):
        """Startupモデルのリレーションシップをテスト"""
        # ユーザー作成
        user = User(**sample_user_data)
        test_db.add(user)
        test_db.commit()

        # スタートアップ作成
        startup_id = str(uuid.uuid4())
        startup = Startup(
            id=startup_id,
            name="Relation Startup",
            owner_id=sample_user_data["id"]
        )
        test_db.add(startup)
        test_db.commit()

        # VASデータ作成
        vas_id = str(uuid.uuid4())
        vas = VASData(
            id=vas_id,
            startup_id=startup_id,
            product_score=8.5,
            team_score=9.0,
            business_model_score=7.5,
            market_score=8.0,
            financial_score=7.0,
            total_score=8.0
        )
        test_db.add(vas)

        # 財務データ作成
        financial_id = str(uuid.uuid4())
        financial = FinancialData(
            id=financial_id,
            startup_id=startup_id,
            year=2023,
            quarter=1,
            revenue=1000000.0,
            expenses=800000.0,
            profit=200000.0
        )
        test_db.add(financial)

        # ノート作成
        note_id = str(uuid.uuid4())
        note = Note(
            id=note_id,
            content="Startup note",
            user_id=sample_user_data["id"],
            startup_id=startup_id
        )
        test_db.add(note)

        test_db.commit()

        # スタートアップを再取得して検証
        db_startup = test_db.query(Startup).filter(Startup.id == startup_id).first()

        # リレーションシップの確認
        assert db_startup.owner.id == sample_user_data["id"]

        assert len(db_startup.vas_data) == 1
        assert db_startup.vas_data[0].id == vas_id
        assert db_startup.vas_data[0].total_score == 8.0

        assert len(db_startup.financial_data) == 1
        assert db_startup.financial_data[0].id == financial_id
        assert db_startup.financial_data[0].revenue == 1000000.0

        assert len(db_startup.notes) == 1
        assert db_startup.notes[0].id == note_id
        assert db_startup.notes[0].content == "Startup note"


class TestVASDataModel:
    """VASDataモデルのテスト"""

    def test_vas_data_creation(self, test_db, sample_user_data, sample_startup_data):
        """VASDataモデルの作成と取得をテスト"""
        # テスト用のユーザーとスタートアップを作成
        user = User(**sample_user_data)
        test_db.add(user)

        startup = Startup(**sample_startup_data)
        test_db.add(startup)
        test_db.commit()

        # テスト用VASデータ
        vas_id = str(uuid.uuid4())

        # VASデータインスタンス作成
        vas = VASData(
            id=vas_id,
            startup_id=sample_startup_data["id"],
            product_score=9.0,
            team_score=8.5,
            business_model_score=7.5,
            market_score=8.0,
            financial_score=7.0,
            total_score=8.0,
            comments="Very promising"
        )

        # DBに保存
        test_db.add(vas)
        test_db.commit()

        # DBから取得して検証
        db_vas = test_db.query(VASData).filter(VASData.id == vas_id).first()
        assert db_vas is not None
        assert db_vas.id == vas_id
        assert db_vas.startup_id == sample_startup_data["id"]
        assert db_vas.product_score == 9.0
        assert db_vas.team_score == 8.5
        assert db_vas.total_score == 8.0
        assert db_vas.comments == "Very promising"

    def test_vas_data_relationships(self, test_db, sample_user_data, sample_startup_data):
        """VASDataモデルのリレーションシップをテスト"""
        # ユーザーとスタートアップを作成
        user = User(**sample_user_data)
        test_db.add(user)

        startup = Startup(**sample_startup_data)
        test_db.add(startup)
        test_db.commit()

        # VASデータ作成
        vas_id = str(uuid.uuid4())
        vas = VASData(
            id=vas_id,
            startup_id=sample_startup_data["id"],
            product_score=9.0,
            team_score=8.5,
            business_model_score=7.5,
            market_score=8.0,
            financial_score=7.0,
            total_score=8.0
        )
        test_db.add(vas)
        test_db.commit()

        # VASデータを再取得して検証
        db_vas = test_db.query(VASData).filter(VASData.id == vas_id).first()

        # リレーションシップの確認
        assert db_vas.startup.id == sample_startup_data["id"]
        assert db_vas.startup.name == sample_startup_data["name"]