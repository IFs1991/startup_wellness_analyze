"""
サブスクリプションモデル
ユーザーのサブスクリプション情報を管理するモデルを定義します。
"""
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

class SubscriptionModel(BaseModel):
    """サブスクリプションモデル"""
    id: Optional[str] = Field(default=None)
    user_id: str = Field(...)
    stripe_customer_id: str = Field(...)
    stripe_subscription_id: str = Field(...)
    plan_type: str = Field(...)  # 'free', 'basic', 'premium' など
    status: str = Field(...)  # 'active', 'trialing', 'canceled', 'past_due'
    trial_start: Optional[datetime] = Field(default=None)
    trial_end: Optional[datetime] = Field(default=None)
    current_period_start: datetime = Field(...)
    current_period_end: datetime = Field(...)
    cancel_at: Optional[datetime] = Field(default=None)
    canceled_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def is_trialing(self) -> bool:
        """トライアル中かどうかを判定"""
        return self.status == 'trialing'

    @property
    def is_active(self) -> bool:
        """アクティブかどうかを判定"""
        return self.status == 'active'

    @property
    def is_canceled(self) -> bool:
        """キャンセル済みかどうかを判定"""
        return self.status == 'canceled' or self.canceled_at is not None

    @property
    def days_until_trial_end(self) -> Optional[int]:
        """トライアル終了までの日数を計算"""
        if not self.trial_end or not self.is_trialing:
            return None
        delta = self.trial_end - datetime.now()
        return max(0, delta.days)

    class Config:
        """Pydanticモデル設定"""
        collection_name = "subscriptions"
        schema_extra = {
            "example": {
                "user_id": "user123",
                "stripe_customer_id": "cus_123456789",
                "stripe_subscription_id": "sub_123456789",
                "plan_type": "premium",
                "status": "active",
                "trial_start": "2023-05-01T00:00:00Z",
                "trial_end": "2023-05-15T00:00:00Z",
                "current_period_start": "2023-05-01T00:00:00Z",
                "current_period_end": "2023-06-01T00:00:00Z"
            }
        }

class SubscriptionPlan(BaseModel):
    """サブスクリプションプランモデル"""
    id: str
    name: str
    price_id: str
    price: float
    currency: str
    interval: str  # 'month', 'year'
    description: str
    features: list[str]

    class Config:
        schema_extra = {
            "example": {
                "id": "basic",
                "name": "ベーシックプラン",
                "price_id": "price_basic123",
                "price": 9.99,
                "currency": "USD",
                "interval": "month",
                "description": "基本的な分析機能",
                "features": ["制限付きデータ分析", "基本ダッシュボード", "週次レポート"]
            }
        }