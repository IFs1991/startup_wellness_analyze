"""
優先度ジョブキュー
Task 4.3: 非同期ジョブキュー
"""

import asyncio
import heapq
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from .models import JobRequest, QueueStatistics
from .job_types import JobPriority, JobType, get_priority_queue_name


@dataclass
class PriorityJobItem:
    """優先度付きジョブアイテム"""
    priority: int
    timestamp: float
    sequence: int
    job_request: JobRequest

    def __lt__(self, other):
        """優先度比較（数値が小さいほど高優先度）"""
        if self.priority != other.priority:
            return self.priority < other.priority
        # 同一優先度の場合はFIFO
        return self.timestamp < other.timestamp


class PriorityJobQueue:
    """優先度ジョブキュー"""

    def __init__(self, max_queue_size: Optional[int] = None):
        self.max_queue_size = max_queue_size or 1000
        self._sequence_counter = 0
        self._lock = asyncio.Lock()

        # 優先度別のヒープキュー
        self._priority_heaps: Dict[JobPriority, List[PriorityJobItem]] = {
            priority: [] for priority in JobPriority
        }

        # 統計情報
        self._statistics = QueueStatistics(queue_name="priority_queue")
        self._job_counts_by_priority: Dict[JobPriority, int] = defaultdict(int)
        self._enqueue_times: Dict[str, float] = {}

        # 公平性制御
        self._fairness_counters: Dict[JobPriority, int] = defaultdict(int)
        self._last_served_priority = None

        # パフォーマンス追跡
        self._total_enqueued = 0
        self._total_dequeued = 0

    async def enqueue(self, job_request: JobRequest, allow_overflow: bool = False) -> bool:
        """
        ジョブをキューに追加

        Args:
            job_request: ジョブリクエスト
            allow_overflow: 容量超過を許可するか

        Returns:
            bool: 追加に成功したかどうか
        """
        async with self._lock:
            # 容量チェック
            current_size = self.get_total_size()
            if current_size >= self.max_queue_size and not allow_overflow:
                # 高優先度ジョブの場合は例外的に追加を許可
                if job_request.priority not in [JobPriority.CRITICAL, JobPriority.HIGH]:
                    self._statistics.failed_jobs += 1
                    return False

            # 優先度設定
            if job_request.priority is None:
                from .job_types import get_job_config
                config = get_job_config(job_request.job_type)
                job_request.priority = config.get("default_priority", JobPriority.NORMAL)

            # 優先度アイテム作成
            priority_item = PriorityJobItem(
                priority=job_request.priority.value,
                timestamp=time.time(),
                sequence=self._get_next_sequence(),
                job_request=job_request
            )

            # 優先度別ヒープに追加
            heap = self._priority_heaps[job_request.priority]
            heapq.heappush(heap, priority_item)

            # 統計更新
            self._job_counts_by_priority[job_request.priority] += 1
            self._enqueue_times[job_request.job_id] = time.time()
            self._total_enqueued += 1
            self._statistics.pending_jobs += 1

            return True

    async def dequeue(self) -> Optional[JobRequest]:
        """
        最高優先度のジョブを取得

        Returns:
            Optional[JobRequest]: ジョブリクエスト（キューが空の場合はNone）
        """
        async with self._lock:
            # 最高優先度のジョブを検索
            for priority in JobPriority:
                heap = self._priority_heaps[priority]
                if heap:
                    # 公平性制御の適用
                    if self._should_apply_fairness(priority):
                        continue

                    # ジョブを取得
                    priority_item = heapq.heappop(heap)
                    job_request = priority_item.job_request

                    # キューイング時間の計算
                    if job_request.job_id in self._enqueue_times:
                        queue_time = time.time() - self._enqueue_times[job_request.job_id]
                        del self._enqueue_times[job_request.job_id]

                    # 統計更新
                    self._job_counts_by_priority[priority] -= 1
                    self._total_dequeued += 1
                    self._statistics.pending_jobs -= 1
                    self._statistics.active_jobs += 1

                    # 公平性カウンター更新
                    self._fairness_counters[priority] += 1
                    self._last_served_priority = priority

                    return job_request

            return None

    async def peek(self) -> Optional[JobRequest]:
        """
        次に処理されるジョブを取得（キューから削除しない）

        Returns:
            Optional[JobRequest]: ジョブリクエスト
        """
        async with self._lock:
            for priority in JobPriority:
                heap = self._priority_heaps[priority]
                if heap:
                    return heap[0].job_request
            return None

    async def remove_job(self, job_id: str) -> bool:
        """
        特定のジョブをキューから削除

        Args:
            job_id: ジョブID

        Returns:
            bool: 削除に成功したかどうか
        """
        async with self._lock:
            for priority, heap in self._priority_heaps.items():
                for i, item in enumerate(heap):
                    if item.job_request.job_id == job_id:
                        heap.pop(i)
                        heapq.heapify(heap)

                        # 統計更新
                        self._job_counts_by_priority[priority] -= 1
                        self._statistics.pending_jobs -= 1

                        if job_id in self._enqueue_times:
                            del self._enqueue_times[job_id]

                        return True
            return False

    def get_total_size(self) -> int:
        """キューの総サイズを取得"""
        return sum(len(heap) for heap in self._priority_heaps.values())

    def get_size_by_priority(self) -> Dict[JobPriority, int]:
        """優先度別のキューサイズを取得"""
        return dict(self._job_counts_by_priority)

    async def get_statistics(self) -> QueueStatistics:
        """キュー統計情報を取得"""
        async with self._lock:
            self._statistics.pending_jobs = self.get_total_size()
            self._statistics.last_updated = datetime.now()

            # レート計算
            if self._total_dequeued > 0:
                elapsed_time = (datetime.now() - datetime.fromtimestamp(0)).total_seconds()
                self._statistics.jobs_per_minute = (self._total_dequeued / elapsed_time) * 60

            return self._statistics

    async def clear_queue(self, priority: Optional[JobPriority] = None):
        """
        キューをクリア

        Args:
            priority: 特定優先度のみクリアする場合
        """
        async with self._lock:
            if priority:
                self._priority_heaps[priority].clear()
                self._job_counts_by_priority[priority] = 0
            else:
                for heap in self._priority_heaps.values():
                    heap.clear()
                self._job_counts_by_priority.clear()
                self._enqueue_times.clear()

    async def get_jobs_by_priority(self, priority: JobPriority) -> List[JobRequest]:
        """
        特定優先度のジョブ一覧を取得

        Args:
            priority: 優先度

        Returns:
            List[JobRequest]: ジョブリクエスト一覧
        """
        async with self._lock:
            heap = self._priority_heaps[priority]
            return [item.job_request for item in heap]

    async def get_oldest_job_age(self) -> Optional[float]:
        """
        最も古いジョブの経過時間を取得

        Returns:
            Optional[float]: 経過時間（秒）、キューが空の場合はNone
        """
        async with self._lock:
            oldest_time = None
            current_time = time.time()

            for heap in self._priority_heaps.values():
                for item in heap:
                    if oldest_time is None or item.timestamp < oldest_time:
                        oldest_time = item.timestamp

            if oldest_time:
                return current_time - oldest_time
            return None

    def _get_next_sequence(self) -> int:
        """次のシーケンス番号を取得"""
        self._sequence_counter += 1
        return self._sequence_counter

    def _should_apply_fairness(self, priority: JobPriority) -> bool:
        """
        公平性制御を適用すべきかを判定

        Args:
            priority: 優先度

        Returns:
            bool: 公平性制御を適用するかどうか
        """
        # 最高優先度（CRITICAL）は公平性制御の対象外
        if priority == JobPriority.CRITICAL:
            return False

        # 前回と同じ優先度が連続して処理された場合の制御
        if self._last_served_priority == priority:
            served_count = self._fairness_counters[priority]
            # 同一優先度で3回連続は避ける
            if served_count >= 3:
                return True

        return False

    async def rebalance_queues(self):
        """キューの再バランス（公平性向上）"""
        async with self._lock:
            # 公平性カウンターをリセット
            self._fairness_counters.clear()
            self._last_served_priority = None

    async def get_queue_health(self) -> Dict[str, Any]:
        """
        キューの健全性情報を取得

        Returns:
            Dict[str, Any]: 健全性情報
        """
        async with self._lock:
            total_jobs = self.get_total_size()
            oldest_age = await self.get_oldest_job_age()

            health_status = "healthy"
            if total_jobs > self.max_queue_size * 0.8:
                health_status = "warning"
            if total_jobs >= self.max_queue_size:
                health_status = "critical"

            return {
                "status": health_status,
                "total_jobs": total_jobs,
                "max_capacity": self.max_queue_size,
                "utilization_percent": (total_jobs / self.max_queue_size) * 100,
                "oldest_job_age_seconds": oldest_age,
                "total_enqueued": self._total_enqueued,
                "total_dequeued": self._total_dequeued,
                "jobs_by_priority": dict(self._job_counts_by_priority),
                "fairness_counters": dict(self._fairness_counters)
            }