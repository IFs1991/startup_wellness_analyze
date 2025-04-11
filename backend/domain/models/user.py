"""
ユーザードメインモデル
ユーザー関連のエンティティと値オブジェクトを定義
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class UserRole(str, Enum):
    """ユーザーロール"""
    ADMIN = "admin"
    USER = "user"
    ANALYST = "analyst"
    VC = "vc"


class MFAType(str, Enum):
    """多要素認証タイプ"""
    NONE = "none"
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"


@dataclass
class UserCredentials:
    """ユーザー認証情報"""
    email: str
    password_hash: str
    mfa_enabled: bool = False
    mfa_type: MFAType = MFAType.NONE
    mfa_secret: Optional[str] = None
    email_verified: bool = False


@dataclass
class User:
    """ユーザーエンティティ"""
    id: str
    email: str
    display_name: Optional[str] = None
    role: UserRole = UserRole.USER
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    company_id: Optional[str] = None
    profile_completed: bool = False
    login_count: int = 0
    data_access: List[str] = field(default_factory=list)
    mfa_enabled: bool = False
    mfa_type: MFAType = MFAType.NONE
    phone_number: Optional[str] = None

    @property
    def is_admin(self) -> bool:
        """管理者かどうか"""
        return self.role == UserRole.ADMIN

    @property
    def is_analyst(self) -> bool:
        """分析者かどうか"""
        return self.role == UserRole.ANALYST

    @property
    def full_name(self) -> str:
        """フルネーム（未設定の場合はメールアドレスを返す）"""
        return self.display_name if self.display_name else self.email

    def update_login(self) -> None:
        """ログイン情報を更新"""
        self.last_login = datetime.now()
        self.login_count += 1

    def can_access_data(self, data_id: str) -> bool:
        """特定のデータへのアクセス権があるかどうか"""
        # 管理者は全てのデータにアクセス可能
        if self.is_admin:
            return True

        # データアクセスリストにデータIDが含まれているか確認
        return data_id in self.data_access

    def requires_mfa(self) -> bool:
        """MFAが必要かどうか"""
        return self.mfa_enabled and self.mfa_type != MFAType.NONE


@dataclass
class UserProfile:
    """ユーザープロファイル（詳細情報）"""
    user_id: str
    full_name: Optional[str] = None
    company_name: Optional[str] = None
    job_title: Optional[str] = None
    bio: Optional[str] = None
    profile_image_url: Optional[str] = None
    preferences: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class TokenData:
    """トークンデータ"""
    user_id: str
    scopes: List[str]
    expires_at: datetime


@dataclass
class AuthToken:
    """認証トークン"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600  # 1時間（秒単位）
    refresh_token: Optional[str] = None