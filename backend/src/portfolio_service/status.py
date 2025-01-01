from typing import Optional, List, Dict
from datetime import datetime
from pydantic import BaseModel

class Status(BaseModel):
    id: str
    company_id: str
    status_type: str
    value: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class StatusNotification(BaseModel):
    id: str
    status_id: str
    message: str
    recipients: List[str]
    created_at: datetime
    sent_at: Optional[datetime] = None

class StatusService:
    def __init__(self, database):
        self.db = database

    async def update_status(self, company_id: str, status_data: dict) -> Status:
        """ステータスを更新する"""
        status_data.update({
            "company_id": company_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
        status = Status(**status_data)
        # TODO: データベースに保存する実装
        await self._create_notification(status)
        return status

    async def get_status_history(self, company_id: str, status_type: Optional[str] = None) -> List[Status]:
        """��テータス履歴を取得する"""
        # TODO: データベースからステータス履歴を取得する実装
        return []

    async def _create_notification(self, status: Status) -> StatusNotification:
        """ステータス更新通知を作成する"""
        notification_data = {
            "status_id": status.id,
            "message": f"Status updated to {status.value} for company {status.company_id}",
            "recipients": await self._get_notification_recipients(status.company_id),
            "created_at": datetime.utcnow()
        }
        notification = StatusNotification(**notification_data)
        # TODO: 通知を送信する実装
        return notification

    async def _get_notification_recipients(self, company_id: str) -> List[str]:
        """通知受信者リストを取得する"""
        # TODO: 通知受信者リストを取得する実装
        return []