"""
リトライ管理器
Task 4.3: 非同期ジョブキュー
"""

import time
import math
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from .models import JobResult
from .job_types import JobType


@dataclass
class CircuitBreakerState:
    """サーキットブレーカー状態"""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    is_open: bool = False
    reset_timeout: float = 60.0


class RetryManager:
    """ジョブリトライ管理器"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

        # サーキットブレーカー設定
        self.enable_circuit_breaker = False
        self.failure_threshold = 5
        self.reset_timeout = 60.0
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}

        # 一時的エラーパターン（リトライ対象）
        self.transient_error_patterns = [
            r'connection.*timeout',
            r'network.*unreachable',
            r'service.*temporarily.*unavailable',
            r'rate.*limit.*exceeded',
            r'resource.*temporarily.*unavailable',
            r'timeout',
            r'temporary.*failure',
            r'503.*service.*unavailable',
            r'502.*bad.*gateway',
            r'504.*gateway.*timeout',
            r'too.*many.*requests',
            r'connection.*refused',
            r'connection.*reset'
        ]

        # 永続的エラーパターン（リトライ非対象）
        self.permanent_error_patterns = [
            r'invalid.*credentials',
            r'malformed.*payload',
            r'authorization.*failed',
            r'resource.*not.*found',
            r'permission.*denied',
            r'access.*denied',
            r'404.*not.*found',
            r'401.*unauthorized',
            r'403.*forbidden',
            r'400.*bad.*request',
            r'invalid.*input',
            r'schema.*validation.*failed',
            r'syntax.*error'
        ]

    def should_retry(self, job_result: JobResult, worker_id: Optional[str] = None) -> bool:
        """
        ジョブをリトライすべきかを判定

        Args:
            job_result: ジョブ実行結果
            worker_id: ワーカーID

        Returns:
            bool: リトライすべきかどうか
        """
        # 最大リトライ回数チェック
        if job_result.retry_count >= self.max_retries:
            return False

        # サーキットブレーカーチェック
        if worker_id and self.enable_circuit_breaker:
            if self.is_circuit_open(worker_id):
                return False

        # エラータイプによる判定
        if job_result.error_message:
            if not self.is_retryable_error(job_result.error_message):
                return False

        return True

    def calculate_retry_delay(self, retry_count: int) -> float:
        """
        リトライ間隔を計算（エクスポネンシャルバックオフ）

        Args:
            retry_count: リトライ回数

        Returns:
            float: 待機時間（秒）
        """
        # エクスポネンシャルバックオフ計算
        delay = self.base_delay * (self.exponential_base ** (retry_count - 1))

        # 最大遅延時間の制限
        delay = min(delay, self.max_delay)

        # ジッター追加（雪崩現象防止）
        if self.jitter:
            import random
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)

        return max(delay, 0.0)

    def is_retryable_error(self, error_message: str) -> bool:
        """
        エラーがリトライ可能かを判定

        Args:
            error_message: エラーメッセージ

        Returns:
            bool: リトライ可能かどうか
        """
        error_lower = error_message.lower()

        # 永続的エラーチェック（リトライ不可）
        for pattern in self.permanent_error_patterns:
            if re.search(pattern, error_lower):
                return False

        # 一時的エラーチェック（リトライ可能）
        for pattern in self.transient_error_patterns:
            if re.search(pattern, error_lower):
                return True

        # パターンにマッチしない場合はデフォルトでリトライ可能
        return True

    def should_move_to_dead_letter(self, job_result: JobResult) -> bool:
        """
        ジョブをデッドレターキューに移動すべきかを判定

        Args:
            job_result: ジョブ実行結果

        Returns:
            bool: デッドレターキューに移動すべきかどうか
        """
        # 最大リトライ回数に達している
        if job_result.retry_count >= self.max_retries:
            return True

        # 永続的エラーの場合
        if job_result.error_message:
            if not self.is_retryable_error(job_result.error_message):
                return True

        return False

    def record_failure(self, worker_id: str):
        """
        ワーカーの失敗を記録（サーキットブレーカー用）

        Args:
            worker_id: ワーカーID
        """
        if not self.enable_circuit_breaker:
            return

        if worker_id not in self.circuit_breakers:
            self.circuit_breakers[worker_id] = CircuitBreakerState()

        cb_state = self.circuit_breakers[worker_id]
        cb_state.failure_count += 1
        cb_state.last_failure_time = datetime.now()

        # 失敗閾値を超えた場合はサーキットブレーカーを開く
        if cb_state.failure_count >= self.failure_threshold:
            cb_state.is_open = True

    def record_success(self, worker_id: str):
        """
        ワーカーの成功を記録（サーキットブレーカーリセット）

        Args:
            worker_id: ワーカーID
        """
        if not self.enable_circuit_breaker:
            return

        if worker_id in self.circuit_breakers:
            # 成功したらサーキットブレーカーをリセット
            self.circuit_breakers[worker_id] = CircuitBreakerState()

    def is_circuit_open(self, worker_id: str) -> bool:
        """
        サーキットブレーカーが開いているかチェック

        Args:
            worker_id: ワーカーID

        Returns:
            bool: サーキットブレーカーが開いているかどうか
        """
        if not self.enable_circuit_breaker:
            return False

        if worker_id not in self.circuit_breakers:
            return False

        cb_state = self.circuit_breakers[worker_id]

        if not cb_state.is_open:
            return False

        # リセットタイムアウト時間が経過した場合はサーキットブレーカーを閉じる
        if cb_state.last_failure_time:
            time_since_failure = (datetime.now() - cb_state.last_failure_time).total_seconds()
            if time_since_failure >= self.reset_timeout:
                cb_state.is_open = False
                cb_state.failure_count = 0
                return False

        return True

    def get_retry_statistics(self) -> Dict[str, any]:
        """
        リトライ統計情報を取得

        Returns:
            Dict[str, any]: 統計情報
        """
        active_circuits = sum(1 for cb in self.circuit_breakers.values() if cb.is_open)
        total_failures = sum(cb.failure_count for cb in self.circuit_breakers.values())

        return {
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "exponential_base": self.exponential_base,
            "circuit_breaker_enabled": self.enable_circuit_breaker,
            "active_circuit_breakers": active_circuits,
            "total_workers_monitored": len(self.circuit_breakers),
            "total_recorded_failures": total_failures,
            "transient_error_patterns": len(self.transient_error_patterns),
            "permanent_error_patterns": len(self.permanent_error_patterns)
        }

    def reset_circuit_breaker(self, worker_id: str):
        """
        特定ワーカーのサーキットブレーカーを手動でリセット

        Args:
            worker_id: ワーカーID
        """
        if worker_id in self.circuit_breakers:
            self.circuit_breakers[worker_id] = CircuitBreakerState()

    def reset_all_circuit_breakers(self):
        """すべてのサーキットブレーカーをリセット"""
        self.circuit_breakers.clear()

    def add_transient_error_pattern(self, pattern: str):
        """
        一時的エラーパターンを追加

        Args:
            pattern: 正規表現パターン
        """
        if pattern not in self.transient_error_patterns:
            self.transient_error_patterns.append(pattern)

    def add_permanent_error_pattern(self, pattern: str):
        """
        永続的エラーパターンを追加

        Args:
            pattern: 正規表現パターン
        """
        if pattern not in self.permanent_error_patterns:
            self.permanent_error_patterns.append(pattern)

    def get_next_retry_time(self, retry_count: int) -> datetime:
        """
        次のリトライ時刻を取得

        Args:
            retry_count: リトライ回数

        Returns:
            datetime: 次のリトライ時刻
        """
        delay = self.calculate_retry_delay(retry_count)
        return datetime.now() + timedelta(seconds=delay)