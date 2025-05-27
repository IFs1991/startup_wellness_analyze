"""
ヘルスモニタリングシステム用データモデル
基本的なデータクラスとヘルス状態管理
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class HealthStatus(Enum):
    """クライアントヘルス状態"""
    REGISTERED = "registered"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    TIMEOUT = "timeout"
    DISCONNECTED = "disconnected"


@dataclass
class ClientHealthStatus:
    """クライアントのヘルス状態管理"""
    client_id: str
    endpoint: str
    status: str = "registered"
    consecutive_failures: int = 0
    last_check: Optional[datetime] = None
    last_successful_check: Optional[datetime] = None
    total_checks: int = 0

    def update_status(self, is_healthy: bool, response_time: Optional[float] = None) -> None:
        """ヘルス状態を更新"""
        current_time = datetime.now()
        self.last_check = current_time
        self.total_checks += 1

        if is_healthy:
            self.status = "healthy"
            self.consecutive_failures = 0
            self.last_successful_check = current_time
        else:
            self.status = "unhealthy"
            self.consecutive_failures += 1


@dataclass
class HealthCheckResult:
    """ヘルスチェックの結果"""
    client_id: str
    is_healthy: bool
    timestamp: datetime
    response_time: Optional[float] = None
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class HealthMetrics:
    """ヘルスメトリクス統計情報"""
    response_times: List[float] = field(default_factory=list)
    check_results: List[bool] = field(default_factory=list)
    last_successful_check: Optional[datetime] = None

    def add_check_result(self, response_time: Optional[float], success: bool) -> None:
        """ヘルスチェック結果を追加"""
        if response_time is not None:
            self.response_times.append(response_time)

        self.check_results.append(success)

        if success:
            self.last_successful_check = datetime.now()

    @property
    def avg_response_time(self) -> float:
        """平均レスポンス時間"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    @property
    def success_rate(self) -> float:
        """成功率"""
        if not self.check_results:
            return 0.0
        return sum(self.check_results) / len(self.check_results)

    @property
    def total_checks(self) -> int:
        """総チェック回数"""
        return len(self.check_results)