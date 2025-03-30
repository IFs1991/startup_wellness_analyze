"""
サブスクリプション関連のモデル
"""
from .subscription import SubscriptionModel, SubscriptionPlan
from pydantic import BaseModel, EmailStr, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class UserModel(BaseModel):
    """ユーザーモデル"""
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

__all__ = ["SubscriptionModel", "SubscriptionPlan", "UserModel"]