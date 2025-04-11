"""
共通エンティティモデル定義
異なるデータベース間で共通して使用できるエンティティモデルを提供します。
"""
from datetime import datetime
from typing import Dict, List, Optional, Any, ClassVar
from pydantic import EmailStr, Field
from enum import Enum

from .base import BaseEntity, ModelType

class UserEntity(BaseEntity):
    """ユーザーエンティティ"""
    username: str = Field(..., description="ユーザー名")
    email: EmailStr = Field(..., description="メールアドレス")
    hashed_password: str = Field(..., description="ハッシュ化されたパスワード")
    is_active: bool = Field(default=True, description="アクティブ状態")
    is_vc: bool = Field(default=False, description="VCフラグ")
    hr_system_user_id: Optional[str] = Field(default=None, description="HR連携ユーザーID")

    # クラス変数
    model_type: ClassVar[ModelType] = ModelType.MEMORY

    @classmethod
    def get_collection_name(cls) -> str:
        return "users"

    @property
    def entity_id(self) -> str:
        return str(self.email)


class StartupEntity(BaseEntity):
    """スタートアップ企業エンティティ"""
    name: str = Field(..., description="企業名")
    description: Optional[str] = Field(default=None, description="説明")
    industry: Optional[str] = Field(default=None, description="業界")
    founding_date: Optional[datetime] = Field(default=None, description="設立日")
    location: Optional[str] = Field(default=None, description="所在地")
    website: Optional[str] = Field(default=None, description="ウェブサイト")
    logo_url: Optional[str] = Field(default=None, description="ロゴURL")
    employee_count: Optional[int] = Field(default=None, description="従業員数")
    funding_stage: Optional[str] = Field(default=None, description="資金調達ステージ")
    total_funding: Optional[float] = Field(default=None, description="調達総額")
    owner_id: str = Field(..., description="オーナーID")

    # クラス変数
    model_type: ClassVar[ModelType] = ModelType.MEMORY

    @classmethod
    def get_collection_name(cls) -> str:
        return "startups"

    @property
    def entity_id(self) -> str:
        return f"{self.name.lower().replace(' ', '-')}-{int(self.created_at.timestamp())}"


class VASDataEntity(BaseEntity):
    """VAS (Value Assessment Score) データエンティティ"""
    startup_id: str = Field(..., description="スタートアップID")
    timestamp: datetime = Field(default_factory=datetime.now, description="評価時点")
    product_score: float = Field(..., description="製品スコア")
    team_score: float = Field(..., description="チームスコア")
    business_model_score: float = Field(..., description="ビジネスモデルスコア")
    market_score: float = Field(..., description="市場スコア")
    financial_score: float = Field(..., description="財務スコア")
    total_score: float = Field(..., description="総合スコア")
    comments: Optional[str] = Field(default=None, description="コメント")

    # クラス変数
    model_type: ClassVar[ModelType] = ModelType.MEMORY

    @classmethod
    def get_collection_name(cls) -> str:
        return "vas_data"

    @property
    def entity_id(self) -> str:
        return f"{self.startup_id}-{int(self.timestamp.timestamp())}"


class FinancialDataEntity(BaseEntity):
    """財務データエンティティ"""
    startup_id: str = Field(..., description="スタートアップID")
    year: int = Field(..., description="年度")
    quarter: int = Field(..., description="四半期")
    revenue: Optional[float] = Field(default=None, description="売上")
    expenses: Optional[float] = Field(default=None, description="経費")
    profit: Optional[float] = Field(default=None, description="利益")
    burn_rate: Optional[float] = Field(default=None, description="資金消費率")
    runway: Optional[float] = Field(default=None, description="残存運転資金期間")
    cash_balance: Optional[float] = Field(default=None, description="現金残高")
    kpis: Optional[Dict[str, Any]] = Field(default=None, description="KPI")

    # クラス変数
    model_type: ClassVar[ModelType] = ModelType.MEMORY

    @classmethod
    def get_collection_name(cls) -> str:
        return "financial_data"

    @property
    def entity_id(self) -> str:
        return f"{self.startup_id}-{self.year}-{self.quarter}"


class NoteEntity(BaseEntity):
    """メモエンティティ"""
    startup_id: str = Field(..., description="スタートアップID")
    user_id: str = Field(..., description="ユーザーID")
    content: str = Field(..., description="内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="作成時間")

    # クラス変数
    model_type: ClassVar[ModelType] = ModelType.MEMORY

    @classmethod
    def get_collection_name(cls) -> str:
        return "notes"

    @property
    def entity_id(self) -> str:
        return f"{self.user_id}-{self.startup_id}-{int(self.timestamp.timestamp())}"