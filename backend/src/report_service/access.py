from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel
from enum import Enum

class AccessLevel(str, Enum):
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"

class AccessPolicy(BaseModel):
    id: str
    resource_type: str
    resource_id: str
    user_id: str
    access_level: AccessLevel
    expires_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

class AccessRequest(BaseModel):
    id: str
    user_id: str
    resource_type: str
    resource_id: str
    requested_level: AccessLevel
    reason: Optional[str] = None
    status: str  # pending, approved, rejected
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    created_at: datetime

class AccessController:
    def __init__(self, database):
        self.db = database

    async def create_policy(
        self,
        resource_type: str,
        resource_id: str,
        user_id: str,
        access_level: AccessLevel,
        expires_at: Optional[datetime] = None
    ) -> AccessPolicy:
        """アクセスポリシーを作成する"""
        policy_data = {
            "id": self._generate_id(),
            "resource_type": resource_type,
            "resource_id": resource_id,
            "user_id": user_id,
            "access_level": access_level,
            "expires_at": expires_at,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        policy = AccessPolicy(**policy_data)
        # TODO: データベースに保存する実装
        return policy

    async def update_policy(
        self,
        policy_id: str,
        access_level: AccessLevel,
        expires_at: Optional[datetime] = None
    ) -> Optional[AccessPolicy]:
        """アクセスポリシーを更新する"""
        policy = await self.get_policy(policy_id)
        if not policy:
            return None

        policy.access_level = access_level
        policy.expires_at = expires_at
        policy.updated_at = datetime.utcnow()

        # TODO: データベースを更新する実装
        return policy

    async def delete_policy(self, policy_id: str) -> bool:
        """アクセスポリシーを削除する"""
        policy = await self.get_policy(policy_id)
        if not policy:
            return False

        # TODO: データベースから��除する実装
        return True

    async def get_policy(
        self,
        policy_id: str
    ) -> Optional[AccessPolicy]:
        """アクセスポリシーを取得する"""
        # TODO: データベースからポリシーを取得する実装
        pass

    async def list_policies(
        self,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[AccessPolicy]:
        """アクセスポリシー一覧を取得する"""
        # TODO: データベースからポリシー一覧を取得する実装
        return []

    async def check_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        required_level: AccessLevel
    ) -> bool:
        """アクセス権限をチェックする"""
        policies = await self.list_policies(
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id
        )

        if not policies:
            return False

        # 有効なポリシーを確認
        now = datetime.utcnow()
        valid_policies = [
            p for p in policies
            if not p.expires_at or p.expires_at > now
        ]

        if not valid_policies:
            return False

        # アクセスレベルをチェック
        access_levels = {
            AccessLevel.NONE: 0,
            AccessLevel.READ: 1,
            AccessLevel.WRITE: 2,
            AccessLevel.ADMIN: 3
        }

        required_level_value = access_levels[required_level]
        max_granted_level = max(
            access_levels[p.access_level]
            for p in valid_policies
        )

        return max_granted_level >= required_level_value

    async def request_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        requested_level: AccessLevel,
        reason: Optional[str] = None
    ) -> AccessRequest:
        """アクセス権限をリクエストする"""
        request_data = {
            "id": self._generate_id(),
            "user_id": user_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "requested_level": requested_level,
            "reason": reason,
            "status": "pending",
            "created_at": datetime.utcnow()
        }
        request = AccessRequest(**request_data)
        # TODO: データベースに保存する実装
        return request

    async def review_access_request(
        self,
        request_id: str,
        reviewer_id: str,
        approved: bool,
        expires_at: Optional[datetime] = None
    ) -> Optional[AccessRequest]:
        """アクセスリクエストをレビューする"""
        request = await self.get_access_request(request_id)
        if not request or request.status != "pending":
            return None

        request.status = "approved" if approved else "rejected"
        request.reviewed_by = reviewer_id
        request.reviewed_at = datetime.utcnow()

        # TODO: データベースを更新する実装

        if approved:
            # 承認された場合、アクセスポリシーを作成
            await self.create_policy(
                resource_type=request.resource_type,
                resource_id=request.resource_id,
                user_id=request.user_id,
                access_level=request.requested_level,
                expires_at=expires_at
            )

        return request

    async def get_access_request(
        self,
        request_id: str
    ) -> Optional[AccessRequest]:
        """アクセスリクエストを取得する"""
        # TODO: データベースからリクエストを取得する実装
        pass

    async def list_access_requests(
        self,
        status: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[AccessRequest]:
        """アクセスリクエスト一覧を取得する"""
        # TODO: データベースからリクエスト一覧を取得する実装
        return []

    def _generate_id(self) -> str:
        """IDを生成する"""
        import uuid
        return str(uuid.uuid4())