"""
データモデルパッケージ
異なるデータベースバックエンド間で共通のデータモデルを提供します。
"""
from .subscription import SubscriptionModel, SubscriptionPlan
from .base import BaseEntity, ModelType
from .entities import (
    UserEntity,
    StartupEntity,
    VASDataEntity,
    FinancialDataEntity,
    NoteEntity
)
from .adapters import (
    ModelAdapter,
    FirestoreAdapter,
    SQLAdapter,
    Neo4jAdapter,
    get_adapter_for_model_type
)
from pydantic import BaseModel, EmailStr, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class UserModel(BaseModel):
    """ユーザーモデル (レガシー - 非推奨)"""
    id: str = Field(..., description="ユーザーID")
    email: EmailStr = Field(..., description="メールアドレス")
    display_name: Optional[str] = Field(None, description="表示名")
    is_active: bool = Field(True, description="アクティブ状態")
    is_admin: bool = Field(False, description="管理者フラグ")
    created_at: datetime = Field(default_factory=datetime.now, description="作成日時")
    last_login: Optional[datetime] = Field(None, description="最終ログイン日時")
    company_id: Optional[str] = Field(None, description="企業ID")
    roles: List[str] = Field(default=["user"], description="ロール")
    permissions: Dict[str, Any] = Field(default_factory=dict, description="権限設定")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "user123",
                "email": "user@example.com",
                "display_name": "テストユーザー",
                "is_active": True,
                "is_admin": False,
                "created_at": "2023-01-01T00:00:00",
                "last_login": "2023-01-02T12:34:56",
                "company_id": "company123",
                "roles": ["user"],
                "permissions": {"read": True, "write": False}
            }
        }

__all__ = [
    # 基底クラス
    "BaseEntity",
    "ModelType",

    # エンティティモデル
    "UserEntity",
    "StartupEntity",
    "VASDataEntity",
    "FinancialDataEntity",
    "NoteEntity",

    # アダプタークラス
    "ModelAdapter",
    "FirestoreAdapter",
    "SQLAdapter",
    "Neo4jAdapter",
    "get_adapter_for_model_type",

    # レガシーモデル (非推奨)
    "SubscriptionModel",
    "SubscriptionPlan",
    "UserModel"
]