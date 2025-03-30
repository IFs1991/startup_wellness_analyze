"""
連合学習最適スケジューラ

このモジュールは、GCPの東京リージョン(asia-northeast1)の割引時間帯を
活用して連合学習ワークフローを最適にスケジュールします。
"""

import os
import logging
from datetime import datetime, timedelta, time
import pytz
from typing import Dict, List, Tuple, Optional
from google.cloud import scheduler_v1
from google.cloud.scheduler_v1.types import Job, HttpTarget

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 東京リージョンのタイムゾーン
TOKYO_TZ = pytz.timezone('Asia/Tokyo')

# GCP割引時間帯の定義（日本時間）
# 参考: https://cloud.google.com/compute/docs/regions-zones/
DISCOUNT_HOURS = {
    # 平日の割引時間帯
    'weekday': {
        'start': time(22, 0),  # 22:00
        'end': time(8, 0)      # 翌日8:00
    },
    # 週末の割引時間帯（終日）
    'weekend': {
        'start': time(0, 0),   # 0:00
        'end': time(23, 59)    # 23:59
    }
}

class FederatedLearningScheduler:
    """
    連合学習ワークフローを最適なコストで実行するためのスケジューラ
    """

    def __init__(self, project_id: str, location: str = 'asia-northeast1'):
        """
        スケジューラを初期化します。

        Args:
            project_id: GCPプロジェクトID
            location: GCPロケーション（デフォルトは東京リージョン）
        """
        self.project_id = project_id
        self.location = location
        self.client = scheduler_v1.CloudSchedulerClient()
        self.parent = f"projects/{project_id}/locations/{location}"

    def _is_weekend(self, dt: datetime) -> bool:
        """
        指定された日付が週末かどうかを判定します。

        Args:
            dt: 確認する日時

        Returns:
            週末の場合はTrue、そうでない場合はFalse
        """
        # 5=土曜日, 6=日曜日
        return dt.weekday() >= 5

    def _get_next_discount_window(
        self, from_time: Optional[datetime] = None
    ) -> Tuple[datetime, datetime, bool]:
        """
        次の割引時間帯の開始時刻と終了時刻を取得します。

        Args:
            from_time: この時刻以降で検索する基準時刻（デフォルトは現在時刻）

        Returns:
            (開始時刻, 終了時刻, 週末フラグ)のタプル
        """
        if from_time is None:
            # デフォルトは現在時刻（東京時間）
            from_time = datetime.now(TOKYO_TZ)

        current_date = from_time.date()
        is_weekend = self._is_weekend(from_time)

        # 今日の割引時間帯を計算
        if is_weekend:
            window = DISCOUNT_HOURS['weekend']
            start_time = datetime.combine(current_date, window['start']).replace(tzinfo=TOKYO_TZ)
            end_time = datetime.combine(current_date, window['end']).replace(tzinfo=TOKYO_TZ)
        else:
            window = DISCOUNT_HOURS['weekday']
            start_time = datetime.combine(current_date, window['start']).replace(tzinfo=TOKYO_TZ)
            # 翌日の朝まで
            end_time = datetime.combine(current_date + timedelta(days=1), window['end']).replace(tzinfo=TOKYO_TZ)

        # 現在時刻が今日の割引時間帯を過ぎている場合、次の割引時間帯を検索
        if from_time > end_time:
            next_date = current_date + timedelta(days=1)
            return self._get_next_discount_window(
                datetime.combine(next_date, time(0, 0)).replace(tzinfo=TOKYO_TZ)
            )

        # 現在時刻が今日の割引時間帯の開始前の場合、開始時刻を調整
        if from_time < start_time:
            start_time = from_time

        return start_time, end_time, is_weekend

    def _calculate_optimal_run_time(
        self,
        estimated_duration_minutes: int,
        from_time: Optional[datetime] = None
    ) -> datetime:
        """
        与えられた実行時間の見積もりに基づいて、最適な実行開始時刻を計算します。

        Args:
            estimated_duration_minutes: 予想される実行時間（分）
            from_time: この時刻以降で検索する基準時刻（デフォルトは現在時刻）

        Returns:
            最適な実行開始時刻
        """
        start_time, end_time, is_weekend = self._get_next_discount_window(from_time)

        # 割引時間帯内で実行が完了するかチェック
        estimated_end_time = start_time + timedelta(minutes=estimated_duration_minutes)

        # 割引時間帯内に収まる場合
        if estimated_end_time <= end_time:
            return start_time

        # 割引時間帯内に収まらない場合、次の割引時間帯を検討
        # 平日→週末の移行で最大の割引時間帯を得られるため、その境界をチェック
        tomorrow = start_time.date() + timedelta(days=1)
        tomorrow_is_weekend = self._is_weekend(datetime.combine(tomorrow, time(0, 0)).replace(tzinfo=TOKYO_TZ))

        # 明日が週末で、現在は平日の場合
        if tomorrow_is_weekend and not is_weekend:
            # 週末開始直前に調整（金曜日の夜に開始）
            weekend_start = datetime.combine(tomorrow, time(0, 0)).replace(tzinfo=TOKYO_TZ)
            return weekend_start - timedelta(minutes=estimated_duration_minutes)

        # それ以外の場合は、現在の割引時間帯を最大限活用
        return max(start_time, end_time - timedelta(minutes=estimated_duration_minutes))

    def schedule_federated_learning(
        self,
        job_name: str,
        endpoint_url: str,
        body: Dict,
        estimated_duration_minutes: int = 120,
        headers: Optional[Dict[str, str]] = None
    ) -> Job:
        """
        連合学習ジョブを最適な時間帯にスケジュールします。

        Args:
            job_name: スケジュールするジョブの名前
            endpoint_url: 呼び出すエンドポイントURL
            body: リクエストボディ
            estimated_duration_minutes: 予想される実行時間（分）
            headers: リクエストヘッダー

        Returns:
            作成されたスケジュールジョブ
        """
        # 最適な実行時間を計算
        optimal_time = self._calculate_optimal_run_time(estimated_duration_minutes)

        # スケジュール時刻をcronフォーマットに変換
        # GCPスケジューラはUTC時間を使用するため変換
        utc_time = optimal_time.astimezone(pytz.UTC)

        # cronフォーマット: "分 時 日 月 曜日"
        cron_schedule = f"{utc_time.minute} {utc_time.hour} {utc_time.day} {utc_time.month} *"

        # HTTPターゲットの設定
        if headers is None:
            headers = {}

        http_target = HttpTarget(
            uri=endpoint_url,
            http_method=scheduler_v1.HttpMethod.POST,
            headers=headers,
            body=str(body).encode("utf-8")
        )

        # ジョブの設定
        job = Job(
            name=f"{self.parent}/jobs/{job_name}",
            description=f"Federated Learning Job scheduled to run at {optimal_time.isoformat()}",
            schedule=cron_schedule,
            time_zone="Asia/Tokyo",
            http_target=http_target
        )

        # 既存のジョブをチェック
        try:
            existing_job = self.client.get_job(name=job.name)
            # 既存のジョブを更新
            updated_job = self.client.update_job(job=job)
            logger.info(f"Updated federated learning job: {job_name} to run at {optimal_time.isoformat()}")
            return updated_job
        except Exception:
            # 新しいジョブを作成
            created_job = self.client.create_job(parent=self.parent, job=job)
            logger.info(f"Created federated learning job: {job_name} to run at {optimal_time.isoformat()}")
            return created_job

    def list_scheduled_jobs(self) -> List[Job]:
        """
        すべてのスケジュールされたジョブを取得します。

        Returns:
            スケジュールされたジョブのリスト
        """
        jobs = self.client.list_jobs(parent=self.parent)
        return list(jobs)

    def delete_scheduled_job(self, job_name: str) -> None:
        """
        指定されたジョブを削除します。

        Args:
            job_name: 削除するジョブの名前
        """
        job_path = f"{self.parent}/jobs/{job_name}"
        self.client.delete_job(name=job_path)
        logger.info(f"Deleted job: {job_name}")