"""
ハートビート管理システム
タイムアウト検知とストラグラー検出機能
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional
from structlog import get_logger

from .models import HealthMetrics


logger = get_logger(__name__)


class HeartbeatManager:
    """ハートビート管理システム"""

    def __init__(
        self,
        heartbeat_interval: float = 5.0,
        timeout_threshold: float = 30.0,
        straggler_response_threshold: float = 5.0,
        straggler_success_threshold: float = 0.7
    ):
        """
        Args:
            heartbeat_interval: ハートビート送信間隔（秒）
            timeout_threshold: タイムアウト判定時間（秒）
            straggler_response_threshold: ストラグラー判定レスポンス時間（秒）
            straggler_success_threshold: ストラグラー判定成功率閾値
        """
        self.heartbeat_interval = heartbeat_interval
        self.timeout_threshold = timeout_threshold
        self.straggler_response_threshold = straggler_response_threshold
        self.straggler_success_threshold = straggler_success_threshold

        # アクティブクライアント管理
        self.active_clients: Dict[str, datetime] = {}
        self.client_metrics: Dict[str, HealthMetrics] = {}

        # 実行制御
        self._tasks: Dict[str, asyncio.Task] = {}
        self._is_running = False

        logger.info("HeartbeatManager initialized",
                   heartbeat_interval=heartbeat_interval,
                   timeout_threshold=timeout_threshold)

    async def start_heartbeat(self, client_id: str) -> None:
        """クライアントのハートビート開始"""
        if client_id in self.active_clients:
            logger.warning("Heartbeat already started", client_id=client_id)
            return

        # クライアント登録
        self.active_clients[client_id] = datetime.now()
        self.client_metrics[client_id] = HealthMetrics()

        logger.info("Heartbeat started", client_id=client_id)

    async def stop_heartbeat(self, client_id: str) -> None:
        """クライアントのハートビート停止"""
        if client_id in self.active_clients:
            del self.active_clients[client_id]

        if client_id in self.client_metrics:
            del self.client_metrics[client_id]

        if client_id in self._tasks:
            task = self._tasks[client_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self._tasks[client_id]

        logger.info("Heartbeat stopped", client_id=client_id)

    async def update_heartbeat(self, client_id: str, response_time: Optional[float] = None, success: bool = True) -> None:
        """ハートビート更新"""
        if client_id not in self.active_clients:
            logger.warning("Client not in active list", client_id=client_id)
            return

        # 最後のハートビート時刻を更新
        self.active_clients[client_id] = datetime.now()

        # メトリクス更新
        if client_id in self.client_metrics:
            self.client_metrics[client_id].add_check_result(response_time, success)

        logger.debug("Heartbeat updated", client_id=client_id, success=success)

    def is_client_active(self, client_id: str) -> bool:
        """クライアントがアクティブかチェック"""
        if client_id not in self.active_clients:
            return False

        last_heartbeat = self.active_clients[client_id]
        current_time = datetime.now()
        time_diff = (current_time - last_heartbeat).total_seconds()

        return time_diff <= self.timeout_threshold

    async def detect_timeouts(self) -> List[str]:
        """タイムアウトしたクライアントを検知"""
        timed_out_clients = []
        current_time = datetime.now()

        for client_id, last_heartbeat in self.active_clients.items():
            time_diff = (current_time - last_heartbeat).total_seconds()

            if time_diff > self.timeout_threshold:
                timed_out_clients.append(client_id)
                logger.warning("Client timeout detected",
                             client_id=client_id,
                             time_since_last_heartbeat=time_diff)

        return timed_out_clients

    async def detect_stragglers(
        self,
        response_time_threshold: Optional[float] = None,
        success_rate_threshold: Optional[float] = None
    ) -> List[str]:
        """ストラグラー（遅延クライアント）を検知"""
        if response_time_threshold is None:
            response_time_threshold = self.straggler_response_threshold

        if success_rate_threshold is None:
            success_rate_threshold = self.straggler_success_threshold

        stragglers = []

        for client_id, metrics in self.client_metrics.items():
            if not self.is_client_active(client_id):
                continue

            # レスポンス時間チェック
            if metrics.avg_response_time > response_time_threshold:
                stragglers.append(client_id)
                logger.info("Straggler detected by response time",
                           client_id=client_id,
                           avg_response_time=metrics.avg_response_time)
                continue

            # 成功率チェック
            if metrics.success_rate < success_rate_threshold:
                stragglers.append(client_id)
                logger.info("Straggler detected by success rate",
                           client_id=client_id,
                           success_rate=metrics.success_rate)

        return stragglers

    async def get_client_metrics(self, client_id: str) -> Optional[HealthMetrics]:
        """クライアントのメトリクス取得"""
        return self.client_metrics.get(client_id)

    async def get_all_active_clients(self) -> List[str]:
        """アクティブなクライアント一覧取得"""
        active_clients = []
        for client_id in self.active_clients:
            if self.is_client_active(client_id):
                active_clients.append(client_id)

        return active_clients

    async def cleanup_inactive_clients(self) -> List[str]:
        """非アクティブクライアントのクリーンアップ"""
        timed_out_clients = await self.detect_timeouts()

        for client_id in timed_out_clients:
            await self.stop_heartbeat(client_id)

        if timed_out_clients:
            logger.info("Cleaned up inactive clients", count=len(timed_out_clients))

        return timed_out_clients

    async def get_health_summary(self) -> Dict[str, any]:
        """ヘルス状態の要約情報を取得"""
        active_clients = await self.get_all_active_clients()
        timed_out_clients = await self.detect_timeouts()
        stragglers = await self.detect_stragglers()

        summary = {
            "total_registered": len(self.active_clients),
            "active_clients": len(active_clients),
            "timed_out_clients": len(timed_out_clients),
            "straggler_clients": len(stragglers),
            "active_client_ids": active_clients,
            "timed_out_client_ids": timed_out_clients,
            "straggler_client_ids": stragglers,
            "timestamp": datetime.now().isoformat()
        }

        return summary