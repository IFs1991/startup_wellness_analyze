"""
非同期処理ユーティリティ

効率的な非同期処理のためのヘルパー関数とユーティリティを提供します。
"""

import asyncio
import logging
import time
import functools
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar, Set, Union, cast, Tuple

from .common_logger import get_logger
from .PriorityAsyncQueue import PriorityAsyncQueue

logger = get_logger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class TaskLimiter:
    """
    非同期タスクの同時実行数を制限するためのセマフォライクなクラス

    最大同時実行数を制限しながら非同期タスクを実行します。
    """

    def __init__(self, max_concurrency: int = 10):
        """
        初期化

        Args:
            max_concurrency: 最大同時実行数（デフォルト: 10）
        """
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._active_tasks: Set[asyncio.Task] = set()
        self._total_tasks_created = 0
        self._completed_tasks = 0

    async def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """
        制限付きで非同期タスクを実行します

        Args:
            coro: 実行するコルーチン

        Returns:
            コルーチンの実行結果
        """
        async with self._semaphore:
            self._total_tasks_created += 1
            task = asyncio.create_task(coro)
            self._active_tasks.add(task)

            try:
                result = await task
                self._completed_tasks += 1
                return result
            finally:
                self._active_tasks.remove(task)

    async def gather(self, coros: List[Coroutine[Any, Any, T]]) -> List[T]:
        """
        複数のコルーチンを制限付きで並列実行し、結果をまとめて返します

        Args:
            coros: 実行するコルーチンのリスト

        Returns:
            各コルーチンの結果のリスト
        """
        results = []
        for coro in coros:
            result = await self.run(coro)
            results.append(result)
        return results

    @property
    def active_tasks(self) -> int:
        """現在アクティブなタスク数"""
        return len(self._active_tasks)

    @property
    def total_tasks(self) -> int:
        """これまでに作成されたタスク数"""
        return self._total_tasks_created

    @property
    def completed_tasks(self) -> int:
        """完了したタスク数"""
        return self._completed_tasks


async def gather_with_concurrency(n: int, *coros: Coroutine) -> List[Any]:
    """
    最大同時実行数を制限しながら複数のコルーチンを実行します

    Args:
        n: 最大同時実行数
        *coros: 実行するコルーチン

    Returns:
        各コルーチンの結果のリスト
    """
    semaphore = asyncio.Semaphore(n)

    async def wrapped_coro(coro: Coroutine) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(wrapped_coro(c) for c in coros))


async def process_in_batches(
    items: List[T],
    processor: Callable[[T], Coroutine[Any, Any, R]],
    batch_size: int = 10,
    max_concurrency: int = 5
) -> List[R]:
    """
    アイテムのリストをバッチ処理します

    Args:
        items: 処理するアイテムのリスト
        processor: 各アイテムを処理する非同期関数
        batch_size: 一度に処理するバッチのサイズ
        max_concurrency: バッチ内の最大同時実行数

    Returns:
        処理結果のリスト
    """
    results: List[R] = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch_results = await gather_with_concurrency(
            max_concurrency,
            *(processor(item) for item in batch)
        )
        results.extend(batch_results)

    return results


def async_timed(func: Callable) -> Callable:
    """
    非同期関数の実行時間を測定するデコレータ

    Args:
        func: 測定対象の非同期関数

    Returns:
        ラップされた関数
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        try:
            return await func(*args, **kwargs)
        finally:
            end = time.time()
            total = end - start
            logger.info(f"関数 {func.__name__} の実行時間: {total:.4f} 秒")
    return wrapper


async def retry_async(
    func: Callable[..., Coroutine],
    *args: Any,
    retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Any:
    """
    非同期関数の実行を指定回数リトライします

    Args:
        func: リトライする非同期関数
        *args: 関数に渡す引数
        retries: 最大リトライ回数
        delay: 初期遅延時間（秒）
        backoff_factor: バックオフ係数
        exceptions: キャッチする例外のタプル
        logger: ロガーインスタンス

    Returns:
        関数の実行結果

    Raises:
        最後の例外が再送出されます
    """
    _logger = logger or get_logger(__name__)

    for attempt in range(retries + 1):
        try:
            return await func(*args)
        except exceptions as e:
            if attempt == retries:
                _logger.error(f"最大リトライ回数 ({retries}) に達しました: {str(e)}")
                raise

            wait_time = delay * (backoff_factor ** attempt)
            _logger.warning(f"試行 {attempt + 1}/{retries + 1} 失敗: {str(e)}. {wait_time:.2f}秒後にリトライします")
            await asyncio.sleep(wait_time)


class AsyncBatchProcessor:
    """
    非同期バッチ処理のためのユーティリティクラス

    大量のアイテムを効率的に処理するためのバッチ処理機能を提供します。
    """

    def __init__(
        self,
        batch_size: int = 100,
        max_concurrency: int = 10,
        batch_interval: float = 0.0,
        use_prioritization: bool = False
    ):
        """
        初期化

        Args:
            batch_size: バッチあたりの最大アイテム数
            max_concurrency: 最大同時実行数
            batch_interval: バッチ間の待機時間（秒）
            use_prioritization: 優先度ベースの処理を使用するかどうか
        """
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.batch_interval = batch_interval
        self.task_limiter = TaskLimiter(max_concurrency)
        self.use_prioritization = use_prioritization
        self.stats = {
            "total_processed": 0,
            "batches_processed": 0,
            "errors": 0,
            "processing_time": 0.0
        }

    async def process(
        self,
        items: List[T],
        processor: Callable[[T], Coroutine[Any, Any, R]],
        error_handler: Optional[Callable[[Exception, T], Coroutine[Any, Any, None]]] = None
    ) -> List[Optional[R]]:
        """
        アイテムのリストをバッチで処理します

        Args:
            items: 処理するアイテムのリスト
            processor: 各アイテムを処理する非同期関数
            error_handler: エラー発生時のハンドラ関数

        Returns:
            処理結果のリスト。エラーが発生した場合はNoneが含まれる場合があります
        """
        if not items:
            return []

        start_time = time.time()
        results: List[Optional[R]] = []
        batches = [items[i:i+self.batch_size] for i in range(0, len(items), self.batch_size)]

        self.stats["batches_processed"] = 0

        try:
            for i, batch in enumerate(batches):
                batch_results = await self._process_batch(batch, processor, error_handler)
                results.extend(batch_results)

                self.stats["batches_processed"] += 1
                self.stats["total_processed"] += len(batch)

                if self.batch_interval > 0 and i < len(batches) - 1:
                    await asyncio.sleep(self.batch_interval)

            self.stats["processing_time"] = time.time() - start_time
            return results
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"バッチ処理中にエラーが発生しました: {str(e)}")
            raise

    async def _process_batch(
        self,
        batch: List[T],
        processor: Callable[[T], Coroutine[Any, Any, R]],
        error_handler: Optional[Callable[[Exception, T], Coroutine[Any, Any, None]]] = None
    ) -> List[Optional[R]]:
        """
        単一バッチを処理します

        Args:
            batch: 処理するアイテムのバッチ
            processor: 各アイテムを処理する非同期関数
            error_handler: エラー発生時のハンドラ関数

        Returns:
            バッチ処理結果のリスト
        """
        tasks = []
        for item in batch:
            # 各アイテムの処理をラップして、エラーをキャッチする
            async def process_with_error_handling(item: T) -> Optional[R]:
                try:
                    return await processor(item)
                except Exception as e:
                    self.stats["errors"] += 1
                    logger.error(f"アイテム処理中にエラーが発生しました: {str(e)}")
                    if error_handler:
                        try:
                            await error_handler(e, item)
                        except Exception as handler_error:
                            logger.error(f"エラーハンドラでエラーが発生しました: {str(handler_error)}")
                    return None

            tasks.append(self.task_limiter.run(process_with_error_handling(item)))

        return await asyncio.gather(*tasks)

    async def process_with_priority(
        self,
        items: List[Tuple[T, int]],
        processor: Callable[[T], Coroutine[Any, Any, R]],
        error_handler: Optional[Callable[[Exception, T], Coroutine[Any, Any, None]]] = None
    ) -> List[Optional[R]]:
        """
        優先度付きでアイテムを処理します

        Args:
            items: 処理するアイテムと優先度のタプルのリスト (アイテム, 優先度)
                  優先度は低いほど優先（0が最高優先度）
            processor: 各アイテムを処理する非同期関数
            error_handler: エラー発生時のハンドラ関数

        Returns:
            処理結果のリスト（入力順）
        """
        if not items:
            return []

        # 入力順を保持
        original_order = {i: idx for idx, (i, _) in enumerate(items)}

        # 優先度でソート
        sorted_items = sorted(items, key=lambda x: x[1])
        sorted_data = [item for item, _ in sorted_items]

        # 処理実行
        results = await self.process(sorted_data, processor, error_handler)

        # 元の順序に戻す
        ordered_results = [None] * len(results)
        for idx, (item, _) in enumerate(sorted_items):
            original_idx = original_order[item]
            ordered_results[original_idx] = results[idx]

        return ordered_results

    async def process_with_backpressure(
        self,
        items: List[T],
        processor: Callable[[T], Coroutine[Any, Any, R]],
        max_pending: int = 20,
        error_handler: Optional[Callable[[Exception, T], Coroutine[Any, Any, None]]] = None
    ) -> List[Optional[R]]:
        """
        バックプレッシャー制御付きでアイテムを処理します

        システムの負荷に応じて処理速度を調整します

        Args:
            items: 処理するアイテムのリスト
            processor: 各アイテムを処理する非同期関数
            max_pending: 最大保留タスク数
            error_handler: エラー発生時のハンドラ関数

        Returns:
            処理結果のリスト
        """
        if not items:
            return []

        results: List[Optional[R]] = [None] * len(items)
        pending_tasks = set()
        next_item_index = 0

        while pending_tasks or next_item_index < len(items):
            # 保留タスクが上限未満で、まだ処理していないアイテムがある場合
            while len(pending_tasks) < max_pending and next_item_index < len(items):
                item = items[next_item_index]

                # プロセッサをラップして、エラーハンドリングを追加
                async def process_with_error_handling(item: T, idx: int) -> None:
                    try:
                        result = await processor(item)
                        results[idx] = result
                    except Exception as e:
                        self.stats["errors"] += 1
                        logger.error(f"アイテム処理中にエラーが発生しました: {str(e)}")
                        if error_handler:
                            try:
                                await error_handler(e, item)
                            except Exception as handler_error:
                                logger.error(f"エラーハンドラでエラーが発生しました: {str(handler_error)}")

                task = asyncio.create_task(process_with_error_handling(item, next_item_index))
                pending_tasks.add(task)
                next_item_index += 1

            if not pending_tasks:
                break

            # 完了したタスクを待機
            done, pending_tasks = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED
            )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        バッチ処理の統計情報を取得します

        Returns:
            統計情報を含む辞書
        """
        return self.stats.copy()

# 優先度付き非同期キューの実装
class PriorityAsyncQueue:
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
        import heapq
        self._queue = []
        self._counter = 0  # 同じ優先度のタスクの挿入順を保持
        self._maxsize = maxsize
        self._unfinished_tasks = 0
        self._finished = asyncio.Event()
        self._finished.set()
        self._mutex = asyncio.Lock()

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
            import heapq
            # (優先度, カウンタ, アイテム)の形式でヒープに追加
            heapq.heappush(self._queue, (priority, self._counter, item))
            self._counter += 1
            self._unfinished_tasks += 1
            self._finished.clear()

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
                await get_event.wait()

            import heapq
            _, _, item = heapq.heappop(self._queue)
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


async def as_completed_with_timeout(
    coros: List[Coroutine],
    timeout: float,
    return_exceptions: bool = False
) -> List[Any]:
    """
    コルーチンを実行し、完了したものから順に結果を返します

    指定されたタイムアウト内に完了しないコルーチンは中断されます。

    Args:
        coros: 実行するコルーチンのリスト
        timeout: 各コルーチンのタイムアウト時間（秒）
        return_exceptions: Trueの場合、例外も結果として返します

    Returns:
        完了したコルーチンの結果のリスト
    """
    tasks = [asyncio.create_task(coro) for coro in coros]
    results = []

    for task in asyncio.as_completed(tasks):
        try:
            result = await asyncio.wait_for(task, timeout=timeout)
            results.append(result)
        except asyncio.TimeoutError:
            task.cancel()
            if return_exceptions:
                results.append(asyncio.TimeoutError())
            logger.warning(f"タスクがタイムアウトしました: {timeout}秒")
        except Exception as e:
            if return_exceptions:
                results.append(e)
            else:
                for t in tasks:
                    if not t.done():
                        t.cancel()
                raise

    return results