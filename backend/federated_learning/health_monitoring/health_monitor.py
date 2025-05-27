"""
メインヘルスモニタークラス
HTTP pollingベースの基本的なヘルスチェック機能
"""

import asyncio
import aiohttp
import time
from datetime import datetime
from typing import Dict, List, Optional
from structlog import get_logger

from .models import ClientHealthStatus, HealthCheckResult, HealthMetrics


logger = get_logger(__name__)


class HealthMonitor:
    """クライアントヘルス監視システム"""

    def __init__(
        self,
        check_interval: float = 5.0,
        timeout_threshold: float = 10.0,
        max_retries: int = 3,
        request_timeout: float = 3.0
    ):
        """
        Args:
            check_interval: ヘルスチェック間隔（秒）
            timeout_threshold: タイムアウト判定時間（秒）
            max_retries: 最大リトライ回数
            request_timeout: HTTPリクエストタイムアウト（秒）
        """
        self.check_interval = check_interval
        self.timeout_threshold = timeout_threshold
        self.max_retries = max_retries
        self.request_timeout = request_timeout

        # クライアント管理
        self.clients: Dict[str, ClientHealthStatus] = {}
        self.is_running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info("HealthMonitor initialized",
                   check_interval=check_interval,
                   timeout_threshold=timeout_threshold)

    async def start(self) -> None:
        """ヘルスモニタリング開始"""
        if self.is_running:
            logger.warning("HealthMonitor is already running")
            return

        try:
            # Windows互換性のためのコネクター設定
            connector = aiohttp.TCPConnector(use_dns_cache=False)
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.request_timeout),
                connector=connector
            )
        except RuntimeError as e:
            if "aiodns needs a SelectorEventLoop" in str(e):
                # Windows環境での回避策：単純なコネクター使用
                connector = aiohttp.TCPConnector(use_dns_cache=False, resolver=None)
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.request_timeout),
                    connector=connector
                )
            else:
                raise

        self.is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("HealthMonitor started")

    async def stop(self) -> None:
        """ヘルスモニタリング停止"""
        if not self.is_running:
            return

        self.is_running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("HealthMonitor stopped")

    async def register_client(self, client_id: str, endpoint: str) -> None:
        """クライアント登録"""
        if client_id in self.clients:
            logger.warning("Client already registered", client_id=client_id)
            return

        self.clients[client_id] = ClientHealthStatus(
            client_id=client_id,
            endpoint=endpoint
        )

        logger.info("Client registered", client_id=client_id, endpoint=endpoint)

    async def unregister_client(self, client_id: str) -> None:
        """クライアント登録解除"""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info("Client unregistered", client_id=client_id)
        else:
            logger.warning("Client not found for unregistration", client_id=client_id)

    async def check_client_health(self, client_id: str) -> Optional[HealthCheckResult]:
        """個別クライアントのヘルスチェック"""
        if client_id not in self.clients:
            logger.error("Client not found", client_id=client_id)
            return None

        client_status = self.clients[client_id]
        start_time = time.time()

        # セッションが初期化されていない場合は作成
        if self._session is None:
            try:
                # Windows互換性のためのコネクター設定
                connector = aiohttp.TCPConnector(use_dns_cache=False)
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.request_timeout),
                    connector=connector
                )
            except RuntimeError as e:
                if "aiodns needs a SelectorEventLoop" in str(e):
                    # Windows環境での回避策：単純なコネクター使用
                    connector = aiohttp.TCPConnector(use_dns_cache=False, resolver=None)
                    self._session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=self.request_timeout),
                        connector=connector
                    )
                else:
                    raise

        try:
            async with self._session.get(client_status.endpoint) as response:
                response_time = time.time() - start_time

                if response.status == 200:
                    # レスポンス内容を取得（オプション）
                    try:
                        response_data = await response.json()
                        details = response_data if isinstance(response_data, dict) else None
                    except:
                        details = None

                    result = HealthCheckResult(
                        client_id=client_id,
                        is_healthy=True,
                        response_time=response_time,
                        status_code=response.status,
                        timestamp=datetime.now(),
                        details=details
                    )

                    # クライアント状態更新
                    client_status.update_status(True, response_time)

                    logger.debug("Health check successful",
                               client_id=client_id,
                               response_time=response_time)

                    return result
                else:
                    # HTTPエラーステータス
                    result = HealthCheckResult(
                        client_id=client_id,
                        is_healthy=False,
                        response_time=response_time,
                        status_code=response.status,
                        timestamp=datetime.now(),
                        error_message=f"HTTP {response.status}"
                    )

                    client_status.update_status(False, response_time)
                    return result

        except asyncio.TimeoutError:
            result = HealthCheckResult(
                client_id=client_id,
                is_healthy=False,
                timestamp=datetime.now(),
                error_message="Request timeout"
            )
            client_status.update_status(False)
            logger.warning("Health check timeout", client_id=client_id)
            return result

        except Exception as e:
            result = HealthCheckResult(
                client_id=client_id,
                is_healthy=False,
                timestamp=datetime.now(),
                error_message=str(e)
            )
            client_status.update_status(False)
            logger.error("Health check failed", client_id=client_id, error=str(e))
            return result

    async def check_all_clients(self) -> List[HealthCheckResult]:
        """全クライアントのヘルスチェック"""
        if not self._session:
            logger.error("Session not initialized")
            return []

        tasks = []
        for client_id in self.clients:
            task = asyncio.create_task(self.check_client_health(client_id))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 例外処理
        valid_results = []
        for result in results:
            if isinstance(result, HealthCheckResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error("Health check task failed", error=str(result))

        return valid_results

    async def get_unhealthy_clients(self) -> List[str]:
        """不健全なクライアントのリストを取得"""
        unhealthy_clients = []
        for client_id, status in self.clients.items():
            if status.status in ["unhealthy", "timeout"]:
                unhealthy_clients.append(client_id)

        return unhealthy_clients

    async def get_client_status(self, client_id: str) -> Optional[ClientHealthStatus]:
        """クライアント状態取得"""
        return self.clients.get(client_id)

    async def _monitoring_loop(self) -> None:
        """バックグラウンド監視ループ"""
        logger.info("Health monitoring loop started")

        while self.is_running:
            try:
                results = await self.check_all_clients()

                # 統計情報をログ出力
                healthy_count = sum(1 for result in results if result.is_healthy)
                total_count = len(results)

                logger.info("Health check completed",
                           healthy_clients=healthy_count,
                           total_clients=total_count)

                # 次のチェックまで待機
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                logger.info("Health monitoring loop cancelled")
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(1.0)  # エラー時は短い間隔で再試行

        logger.info("Health monitoring loop stopped")