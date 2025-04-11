"""
企業エンティティ
企業のドメインモデルを定義します。
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum


class CompanySize(str, Enum):
    """企業の規模を表す列挙型"""
    SMALL = "small"  # 1-50人
    MEDIUM = "medium"  # 51-250人
    LARGE = "large"  # 251-1000人
    ENTERPRISE = "enterprise"  # 1000人以上


class CompanyStatus(str, Enum):
    """企業のステータスを表す列挙型"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRIAL = "trial"
    SUSPENDED = "suspended"


@dataclass
class CompanyAddress:
    """企業の住所情報"""
    postal_code: str = ""
    prefecture: str = ""
    city: str = ""
    street_address: str = ""
    building: Optional[str] = None
    country: str = "日本"


@dataclass
class Company:
    """
    企業エンティティ
    システム内の企業を表すドメインオブジェクト
    """
    id: str
    name: str
    size: CompanySize = CompanySize.SMALL
    status: CompanyStatus = CompanyStatus.ACTIVE
    address: Optional[CompanyAddress] = None
    industry: Optional[str] = None
    established_date: Optional[datetime] = None
    employee_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    website: Optional[str] = None
    departments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    admin_user_ids: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """初期化後の処理"""
        if self.address is None:
            self.address = CompanyAddress()

    @property
    def is_active(self) -> bool:
        """企業がアクティブかどうかを判定"""
        return self.status == CompanyStatus.ACTIVE or self.status == CompanyStatus.TRIAL

    def update_info(self, name: Optional[str] = None, industry: Optional[str] = None,
                   employee_count: Optional[int] = None) -> None:
        """
        企業の基本情報を更新

        Args:
            name: 企業名
            industry: 業種
            employee_count: 従業員数
        """
        if name:
            self.name = name
        if industry:
            self.industry = industry
        if employee_count is not None:
            self.employee_count = employee_count
            # 企業サイズの自動更新
            self._update_company_size()

        self.updated_at = datetime.now()

    def _update_company_size(self) -> None:
        """従業員数に基づいて企業サイズを更新"""
        if self.employee_count <= 50:
            self.size = CompanySize.SMALL
        elif self.employee_count <= 250:
            self.size = CompanySize.MEDIUM
        elif self.employee_count <= 1000:
            self.size = CompanySize.LARGE
        else:
            self.size = CompanySize.ENTERPRISE

    def update_status(self, status: CompanyStatus) -> None:
        """
        企業のステータスを更新

        Args:
            status: 新しいステータス
        """
        self.status = status
        self.updated_at = datetime.now()

    def update_address(self, address: CompanyAddress) -> None:
        """
        企業の住所を更新

        Args:
            address: 新しい住所
        """
        self.address = address
        self.updated_at = datetime.now()

    def add_department(self, department: str) -> None:
        """
        部署を追加

        Args:
            department: 追加する部署名
        """
        if department not in self.departments:
            self.departments.append(department)
            self.updated_at = datetime.now()

    def remove_department(self, department: str) -> None:
        """
        部署を削除

        Args:
            department: 削除する部署名
        """
        if department in self.departments:
            self.departments.remove(department)
            self.updated_at = datetime.now()

    def add_admin(self, user_id: str) -> None:
        """
        管理者ユーザーを追加

        Args:
            user_id: 管理者として追加するユーザーID
        """
        self.admin_user_ids.add(user_id)
        self.updated_at = datetime.now()

    def remove_admin(self, user_id: str) -> None:
        """
        管理者ユーザーを削除

        Args:
            user_id: 管理者から削除するユーザーID
        """
        if user_id in self.admin_user_ids:
            self.admin_user_ids.remove(user_id)
            self.updated_at = datetime.now()

    def is_admin(self, user_id: str) -> bool:
        """
        指定されたユーザーが管理者かどうかを判定

        Args:
            user_id: 確認するユーザーID

        Returns:
            管理者であればTrue、そうでなければFalse
        """
        return user_id in self.admin_user_ids