from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel
import aiohttp
import asyncio
import json

class IntegrationConfig(BaseModel):
    api_url: str
    api_key: Optional[str] = None
    method: str = "GET"
    headers: Dict[str, str] = {}
    params: Dict[str, Any] = {}
    retry_count: int = 3
    timeout: int = 30

class WebhookConfig(BaseModel):
    url: str
    secret: Optional[str] = None
    events: List[str] = []
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

class SyncStatus(BaseModel):
    id: str
    source: str
    target: str
    status: str
    records_processed: int
    records_failed: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class DataIntegration:
    def __init__(self, database):
        self.db = database
        self._session = None

    async def initialize(self):
        """HTTPセッションを初期化する"""
        if not self._session:
            self._session = aiohttp.ClientSession()

    async def cleanup(self):
        """HTTPセッションをクリーンアップする"""
        if self._session:
            await self._session.close()
            self._session = None

    async def integrate_external_api(
        self,
        config: IntegrationConfig,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """外部APIと統合する"""
        if not self._session:
            await self.initialize()

        headers = {
            'Content-Type': 'application/json',
            **config.headers
        }

        if config.api_key:
            headers['Authorization'] = f'Bearer {config.api_key}'

        for attempt in range(config.retry_count):
            try:
                async with self._session.request(
                    method=config.method,
                    url=config.api_url,
                    headers=headers,
                    params=config.params,
                    json=data if data else None,
                    timeout=config.timeout
                ) as response:
                    response.raise_for_status()
                    return await response.json()

            except aiohttp.ClientError as e:
                if attempt == config.retry_count - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def register_webhook(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None
    ) -> WebhookConfig:
        """Webhookを登録する"""
        webhook = WebhookConfig(
            url=url,
            secret=secret,
            events=events,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        # TODO: データベースに保存する実装
        return webhook

    async def trigger_webhook(
        self,
        webhook: WebhookConfig,
        event: str,
        payload: Dict[str, Any]
    ) -> bool:
        """Webhookをトリガーする"""
        if not webhook.is_active or event not in webhook.events:
            return False

        if not self._session:
            await self.initialize()

        headers = {'Content-Type': 'application/json'}
        if webhook.secret:
            headers['X-Webhook-Secret'] = webhook.secret

        try:
            async with self._session.post(
                webhook.url,
                headers=headers,
                json={
                    'event': event,
                    'timestamp': datetime.utcnow().isoformat(),
                    'payload': payload
                },
                timeout=30
            ) as response:
                return response.status == 200
        except:
            return False

    async def synchronize_data(
        self,
        source: str,
        target: str,
        data: List[Dict[str, Any]]
    ) -> SyncStatus:
        """データを同期する"""
        sync_id = self._generate_id()
        status = SyncStatus(
            id=sync_id,
            source=source,
            target=target,
            status="in_progress",
            records_processed=0,
            records_failed=0,
            started_at=datetime.utcnow()
        )

        try:
            for record in data:
                try:
                    # TODO: 実際の同期処理を実装
                    await self._sync_record(source, target, record)
                    status.records_processed += 1
                except Exception as e:
                    status.records_failed += 1
                    if not status.error_message:
                        status.error_message = str(e)

            status.status = "completed" if status.records_failed == 0 else "completed_with_errors"

        except Exception as e:
            status.status = "failed"
            status.error_message = str(e)

        finally:
            status.completed_at = datetime.utcnow()
            # TODO: ステータスをデータベースに保存する実装

        return status

    async def _sync_record(
        self,
        source: str,
        target: str,
        record: Dict[str, Any]
    ) -> None:
        """個別のレコードを同期する"""
        # TODO: レコードの同期ロジックを実装
        pass

    def _generate_id(self) -> str:
        """IDを生成する"""
        import uuid
        return str(uuid.uuid4())