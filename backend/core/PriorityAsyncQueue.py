"""
優先度付き非同期キュー

優先度ベースでタスクを処理するための非同期キューを提供します。
"""

import asyncio
import heapq
from typing import Any, List, TypeVar, Set, Generic, Optional

from .common_logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

class PriorityAsyncQueue(Generic[T]):
    """
    優先度付き非同期キュークラス

    優先度ベースでタスクを処理するための非同期キューを提供します。
    """

    def __init__(self, maxsize: int = 0):
        """
        初期化

        Args:
            maxsize: キューの最大サイズ（0は無制限）
        """
        self._queue = []  # [(優先度, カウンタ, アイテム)]
        self._counter = 0  # 同じ優先度のタスクの挿入順を保持
        self._maxsize = maxsize
        self._unfinished_tasks = 0
        self._finished = asyncio.Event()
        self._finished.set()
        self._mutex = asyncio.Lock()
        self._waiting_puts: List[asyncio.Event] = []
        self._waiting_gets: List[asyncio.Event] = []

    async def put(self, item: T, priority: int = 0) -> None:
        """
        キューにアイテムを追加します

        Args:
            item: 追加するアイテム
            priority: アイテムの優先度（低いほど優先）
        """
        while self._maxsize > 0 and len(self._queue) >= self._maxsize:
            # キューに空きができるまで待機
            put_event = asyncio.Event()
            async with self._mutex:
                self._waiting_puts.append(put_event)
            await put_event.wait()

        async with self._mutex:
            # (優先度, カウンタ, アイテム)の形式でヒープに追加
            heapq.heappush(self._queue, (priority, self._counter, item))
            self._counter += 1
            self._unfinished_tasks += 1
            self._finished.clear()

            # 待機中のgetがあれば通知
            if self._waiting_gets:
                self._waiting_gets.pop(0).set()

    async def get(self) -> T:
        """
        キューから最高優先度のアイテムを取得します

        Returns:
            取得したアイテム
        """
        async with self._mutex:
            while not self._queue:
                get_event = asyncio.Event()
                self._waiting_gets.append(get_event)
                async with self._mutex:
                    pass  # ロックを解放して待機
                await get_event.wait()

            priority, counter, item = heapq.heappop(self._queue)

            # 待機中のputがあれば通知
            if self._waiting_puts and len(self._queue) < self._maxsize:
                self._waiting_puts.pop(0).set()

            return item

    async def task_done(self) -> None:
        """タスク完了を通知します"""
        async with self._mutex:
            self._unfinished_tasks -= 1
            if self._unfinished_tasks <= 0:
                self._finished.set()

    async def join(self) -> None:
        """すべてのタスクが完了するまで待機します"""
        await self._finished.wait()

    async def empty(self) -> bool:
        """キューが空かどうかを確認します"""
        async with self._mutex:
            return len(self._queue) == 0

    async def qsize(self) -> int:
        """キューのサイズを取得します"""
        async with self._mutex:
            return len(self._queue)

    async def clear(self) -> None:
        """キューをクリアします"""
        async with self._mutex:
            self._queue.clear()
            # 待機中のすべてのgetイベントをトリガーして例外を発生させる
            for event in self._waiting_gets:
                event.set()
            self._waiting_gets.clear()