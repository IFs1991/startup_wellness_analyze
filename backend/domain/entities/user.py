"""
ユーザーエンティティ
ユーザーのドメインモデルを定義します。
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class UserRole(str, Enum):
    """ユーザーの役割を表す列挙型"""
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    GUEST = "guest"


class UserPosition(str, Enum):
    """ユーザーの職位を表す列挙型"""
    CEO = "ceo"
    CTO = "cto"
    CFO = "cfo"
    DIRECTOR = "director"
    MANAGER = "manager"
    TEAM_LEAD = "team_lead"
    SENIOR_STAFF = "senior_staff"
    STAFF = "staff"
    INTERN = "intern"
    OTHER = "other"


class SubscriptionStatus(str, Enum):
    """サブスクリプションの状態を表す列挙型"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRIAL = "trial"
    EXPIRED = "expired"
    CANCELED = "canceled"


@dataclass
class User:
    """
    ユーザーエンティティ
    アプリケーションのユーザーを表すドメインオブジェクト
    """
    id: str
    email: str
    display_name: Optional[str] = None
    roles: List[UserRole] = field(default_factory=lambda: [UserRole.USER])
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    company_id: Optional[str] = None
    department: Optional[str] = None
    position: Optional[UserPosition] = None
    hire_date: Optional[datetime] = None
    manager_id: Optional[str] = None
    is_company_admin: bool = False
    is_active: bool = True
    subscription_status: SubscriptionStatus = SubscriptionStatus.INACTIVE
    profile_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_admin(self) -> bool:
        """ユーザーが管理者かどうかを判定"""
        return UserRole.ADMIN in self.roles

    @property
    def has_active_subscription(self) -> bool:
        """アクティブなサブスクリプションを持っているかどうかを判定"""
        return self.subscription_status in [
            SubscriptionStatus.ACTIVE,
            SubscriptionStatus.TRIAL
        ]

    def update_profile(self, profile_data: Dict[str, Any]) -> None:
        """
        プロフィール情報を更新

        Args:
            profile_data: 更新するプロフィールデータ
        """
        self.profile_data.update(profile_data)
        self.updated_at = datetime.now()

    def add_role(self, role: UserRole) -> None:
        """
        ユーザーに役割を追加

        Args:
            role: 追加する役割
        """
        if role not in self.roles:
            self.roles.append(role)
            self.updated_at = datetime.now()

    def remove_role(self, role: UserRole) -> None:
        """
        ユーザーから役割を削除

        Args:
            role: 削除する役割
        """
        if role in self.roles and len(self.roles) > 1:
            self.roles.remove(role)
            self.updated_at = datetime.now()

    def update_company_info(self, company_id: str, department: Optional[str] = None,
                           position: Optional[UserPosition] = None,
                           is_company_admin: bool = False) -> None:
        """
        会社関連情報を更新

        Args:
            company_id: 会社ID
            department: 部署名
            position: 職位
            is_company_admin: 会社管理者かどうか
        """
        self.company_id = company_id
        self.department = department
        self.position = position
        self.is_company_admin = is_company_admin
        self.updated_at = datetime.now()

    def set_manager(self, manager_id: str) -> None:
        """
        マネージャーを設定

        Args:
            manager_id: マネージャーのユーザーID
        """
        self.manager_id = manager_id
        self.updated_at = datetime.now()

    def record_login(self) -> None:
        """ログイン時刻を記録"""
        self.last_login = datetime.now()

    def deactivate(self) -> None:
        """ユーザーを非アクティブ化"""
        self.is_active = False
        self.updated_at = datetime.now()

    def activate(self) -> None:
        """ユーザーをアクティブ化"""
        self.is_active = True
        self.updated_at = datetime.now()