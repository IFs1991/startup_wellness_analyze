"""
デッドレターキュー
Task 4.3: 非同期ジョブキュー
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter
from .models import JobResult, JobRequest, DeadLetterRecord
from .job_types import JobType, JobPriority


class DeadLetterQueue:
    """デッドレターキュー"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._lock = asyncio.Lock()

        # デッドレターレコード保存
        self._records: Dict[str, DeadLetterRecord] = {}

        # インデックス（高速検索用）
        self._records_by_type: Dict[JobType, List[str]] = defaultdict(list)
        self._records_by_date: Dict[str, List[str]] = defaultdict(list)  # YYYY-MM-DD

        # 統計情報
        self._total_added = 0
        self._total_resubmitted = 0
        self._failure_patterns: Counter = Counter()

    async def add_failed_job(
        self,
        job_result: JobResult,
        original_payload: Dict[str, Any]
    ) -> DeadLetterRecord:
        """
        失敗ジョブをデッドレターキューに追加

        Args:
            job_result: ジョブ実行結果
            original_payload: 元のペイロード

        Returns:
            DeadLetterRecord: デッドレターレコード
        """
        async with self._lock:
            # 容量チェック
            if len(self._records) >= self.max_size:
                await self._cleanup_old_records()

            # デッドレターレコード作成
            now = datetime.now()
            record = DeadLetterRecord(
                job_id=job_result.job_id,
                job_type=job_result.job_type,
                original_payload=original_payload,
                failure_reason=job_result.error_message or "Unknown failure",
                retry_count=job_result.retry_count,
                first_failed_at=now,
                last_failed_at=now,
                worker_errors=[],
                metadata={
                    "worker_id": job_result.worker_id,
                    "execution_time": job_result.execution_time,
                    "started_at": job_result.started_at.isoformat() if job_result.started_at else None,
                    "completed_at": job_result.completed_at.isoformat() if job_result.completed_at else None
                }
            )

            # 既存レコードの更新または新規追加
            if job_result.job_id in self._records:
                existing_record = self._records[job_result.job_id]
                existing_record.last_failed_at = now
                existing_record.retry_count = job_result.retry_count
                existing_record.add_error(
                    job_result.error_message or "Unknown failure",
                    job_result.worker_id
                )
                record = existing_record
            else:
                self._records[job_result.job_id] = record

                # インデックス更新
                self._records_by_type[job_result.job_type].append(job_result.job_id)
                date_key = now.strftime("%Y-%m-%d")
                self._records_by_date[date_key].append(job_result.job_id)

                self._total_added += 1

            # 失敗パターン分析
            if job_result.error_message:
                self._failure_patterns[job_result.error_message] += 1

            return record

    async def get_failed_jobs(
        self,
        job_type: Optional[JobType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[DeadLetterRecord]:
        """
        失敗ジョブ一覧を取得

        Args:
            job_type: ジョブタイプフィルター
            start_date: 開始日時フィルター
            end_date: 終了日時フィルター
            limit: 取得数制限

        Returns:
            List[DeadLetterRecord]: デッドレターレコード一覧
        """
        async with self._lock:
            records = []

            if job_type:
                # ジョブタイプでフィルター
                job_ids = self._records_by_type.get(job_type, [])
                for job_id in job_ids:
                    if job_id in self._records:
                        records.append(self._records[job_id])
            else:
                # 全レコード取得
                records = list(self._records.values())

            # 日時フィルター
            if start_date or end_date:
                filtered_records = []
                for record in records:
                    record_date = record.first_failed_at
                    if start_date and record_date < start_date:
                        continue
                    if end_date and record_date > end_date:
                        continue
                    filtered_records.append(record)
                records = filtered_records

            # ソート（最新の失敗順）
            records.sort(key=lambda r: r.last_failed_at, reverse=True)

            # 制限適用
            if limit:
                records = records[:limit]

            return records

    async def create_resubmission_request(self, job_id: str) -> Optional[JobRequest]:
        """
        再実行用のジョブリクエストを作成

        Args:
            job_id: ジョブID

        Returns:
            Optional[JobRequest]: ジョブリクエスト（見つからない場合はNone）
        """
        async with self._lock:
            if job_id not in self._records:
                return None

            record = self._records[job_id]

            # 再実行用リクエスト作成
            resubmit_request = JobRequest(
                job_type=record.job_type,
                payload=record.original_payload,
                priority=JobPriority.HIGH,  # 再実行は高優先度
                max_retries=3,  # リトライ回数をリセット
                metadata={
                    "resubmitted_from_dlq": True,
                    "original_job_id": job_id,
                    "original_failure_reason": record.failure_reason,
                    "original_retry_count": record.retry_count,
                    "resubmitted_at": datetime.now().isoformat()
                }
            )

            return resubmit_request

    async def mark_resubmitted(self, job_id: str) -> bool:
        """
        ジョブを再実行済みとしてマーク（デッドレターキューから削除）

        Args:
            job_id: ジョブID

        Returns:
            bool: 削除に成功したかどうか
        """
        async with self._lock:
            if job_id not in self._records:
                return False

            record = self._records[job_id]

            # インデックスから削除
            self._records_by_type[record.job_type].remove(job_id)

            # 日付インデックスからも削除
            date_key = record.first_failed_at.strftime("%Y-%m-%d")
            if date_key in self._records_by_date:
                self._records_by_date[date_key].remove(job_id)

            # レコード削除
            del self._records[job_id]
            self._total_resubmitted += 1

            return True

    async def analyze_failures(self) -> Dict[str, Any]:
        """
        失敗分析を実行

        Returns:
            Dict[str, Any]: 分析結果
        """
        async with self._lock:
            # 基本統計
            total_failed_jobs = len(self._records)

            # ジョブタイプ別失敗数
            failure_by_type = defaultdict(int)
            for record in self._records.values():
                failure_by_type[record.job_type] += 1

            # 失敗パターン分析
            common_errors = self._failure_patterns.most_common(10)

            # 時系列分析
            failure_trends = defaultdict(int)
            for record in self._records.values():
                date_key = record.first_failed_at.strftime("%Y-%m-%d")
                failure_trends[date_key] += 1

            # リトライ回数分析
            retry_distribution = defaultdict(int)
            for record in self._records.values():
                retry_distribution[record.retry_count] += 1

            # 平均失敗間隔
            if len(self._records) > 1:
                sorted_records = sorted(self._records.values(), key=lambda r: r.first_failed_at)
                intervals = []
                for i in range(1, len(sorted_records)):
                    interval = (sorted_records[i].first_failed_at - sorted_records[i-1].first_failed_at).total_seconds()
                    intervals.append(interval)
                avg_failure_interval = sum(intervals) / len(intervals) if intervals else 0
            else:
                avg_failure_interval = 0

            return {
                "total_failed_jobs": total_failed_jobs,
                "total_resubmitted": self._total_resubmitted,
                "failure_by_type": dict(failure_by_type),
                "common_errors": [{"error": error, "count": count} for error, count in common_errors],
                "failure_trends": dict(failure_trends),
                "retry_distribution": dict(retry_distribution),
                "average_failure_interval_seconds": avg_failure_interval,
                "oldest_failure": min(r.first_failed_at for r in self._records.values()) if self._records else None,
                "newest_failure": max(r.last_failed_at for r in self._records.values()) if self._records else None
            }

    async def get_statistics(self) -> Dict[str, Any]:
        """
        デッドレターキュー統計情報を取得

        Returns:
            Dict[str, Any]: 統計情報
        """
        async with self._lock:
            return {
                "total_records": len(self._records),
                "max_capacity": self.max_size,
                "utilization_percent": (len(self._records) / self.max_size) * 100,
                "total_added": self._total_added,
                "total_resubmitted": self._total_resubmitted,
                "resubmission_rate": (self._total_resubmitted / self._total_added) * 100 if self._total_added > 0 else 0,
                "unique_failure_patterns": len(self._failure_patterns),
                "job_types_in_dlq": len(self._records_by_type),
                "last_updated": datetime.now().isoformat()
            }

    async def cleanup_old_records(self, days_to_keep: int = 30):
        """
        古いレコードをクリーンアップ

        Args:
            days_to_keep: 保持日数
        """
        async with self._lock:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            records_to_remove = []
            for job_id, record in self._records.items():
                if record.first_failed_at < cutoff_date:
                    records_to_remove.append(job_id)

            for job_id in records_to_remove:
                await self._remove_record(job_id)

    async def export_records(
        self,
        format: str = "json",
        job_type: Optional[JobType] = None
    ) -> str:
        """
        デッドレターレコードをエクスポート

        Args:
            format: エクスポート形式 (json/csv)
            job_type: ジョブタイプフィルター

        Returns:
            str: エクスポートデータ
        """
        records = await self.get_failed_jobs(job_type=job_type)

        if format == "json":
            export_data = []
            for record in records:
                export_data.append({
                    "job_id": record.job_id,
                    "job_type": record.job_type.value,
                    "failure_reason": record.failure_reason,
                    "retry_count": record.retry_count,
                    "first_failed_at": record.first_failed_at.isoformat(),
                    "last_failed_at": record.last_failed_at.isoformat(),
                    "worker_errors": record.worker_errors,
                    "original_payload": record.original_payload,
                    "metadata": record.metadata
                })
            return json.dumps(export_data, indent=2, ensure_ascii=False)

        elif format == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # ヘッダー
            writer.writerow([
                "job_id", "job_type", "failure_reason", "retry_count",
                "first_failed_at", "last_failed_at", "error_count"
            ])

            # データ
            for record in records:
                writer.writerow([
                    record.job_id,
                    record.job_type.value,
                    record.failure_reason,
                    record.retry_count,
                    record.first_failed_at.isoformat(),
                    record.last_failed_at.isoformat(),
                    len(record.worker_errors)
                ])

            return output.getvalue()

        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def _cleanup_old_records(self):
        """内部: 容量制限のための古いレコード削除"""
        # 最も古いレコードを削除
        if not self._records:
            return

        oldest_job_id = min(
            self._records.keys(),
            key=lambda job_id: self._records[job_id].first_failed_at
        )
        await self._remove_record(oldest_job_id)

    async def _remove_record(self, job_id: str):
        """内部: レコード削除"""
        if job_id not in self._records:
            return

        record = self._records[job_id]

        # インデックスから削除
        if job_id in self._records_by_type[record.job_type]:
            self._records_by_type[record.job_type].remove(job_id)

        date_key = record.first_failed_at.strftime("%Y-%m-%d")
        if job_id in self._records_by_date[date_key]:
            self._records_by_date[date_key].remove(job_id)

        # レコード削除
        del self._records[job_id]