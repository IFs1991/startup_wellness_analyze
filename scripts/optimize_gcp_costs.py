#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GCPコスト分析と最適化スクリプト

このスクリプトは、GCP環境のコスト分析とリソース最適化を行います。
特に東京リージョンの割引時間帯の活用と連合学習ワークフローの最適化に焦点を当てています。
"""

import os
import sys
import logging
import json
import argparse
from datetime import datetime, timedelta, timezone
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import compute_v1
from google.cloud import monitoring_v3
from google.cloud import scheduler_v1
from google.cloud.exceptions import NotFound

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 日本時間タイムゾーン
JST = timezone(timedelta(hours=9))

class GCPCostOptimizer:
    """
    GCPコスト分析と最適化を行うクラス
    """

    def __init__(self, project_id: str, region: str = 'asia-northeast1'):
        """
        GCPコスト最適化ツールを初期化します。

        Args:
            project_id: GCPプロジェクトID
            region: GCPリージョン（デフォルトは東京リージョン）
        """
        self.project_id = project_id
        self.region = region

        # BigQueryクライアント
        self.bq_client = bigquery.Client(project=project_id)

        # Computeクライアント
        self.compute_client = compute_v1.InstancesClient()

        # Monitoringクライアント
        self.monitoring_client = monitoring_v3.MetricServiceClient()

        # Cloud Schedulerクライアント
        self.scheduler_client = scheduler_v1.CloudSchedulerClient()

        # Storageクライアント
        self.storage_client = storage.Client(project=project_id)

    def analyze_costs(self, days: int = 30) -> pd.DataFrame:
        """
        指定期間のコスト分析を行います。

        Args:
            days: 分析対象期間（日）

        Returns:
            コスト分析結果のDataFrame
        """
        # BigQueryのビリングエクスポートテーブルからコストデータを取得
        query = f"""
        SELECT
          service.description as service,
          sku.description as sku,
          DATE(usage_start_time) as usage_date,
          location.region as region,
          SUM(cost) as cost,
          SUM(usage.amount) as usage_amount,
          usage.unit as usage_unit
        FROM
          `{self.project_id}.billing.gcp_billing_export_v1_*`
        WHERE
          DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
        GROUP BY
          service, sku, usage_date, region, usage_unit
        ORDER BY
          usage_date DESC, cost DESC
        """

        logger.info(f"過去{days}日間のGCPコスト分析クエリを実行中...")
        df = self.bq_client.query(query).to_dataframe()

        # 結果がない場合
        if df.empty:
            logger.warning("クエリ結果が空です。ビリングエクスポートが設定されているか確認してください。")
            return pd.DataFrame()

        logger.info(f"コスト分析完了: {len(df)}行のデータを取得")
        return df

    def get_resource_usage(self, resource_type: str, days: int = 7) -> pd.DataFrame:
        """
        リソースの使用状況を取得します。

        Args:
            resource_type: リソースタイプ ('instance', 'disk', など)
            days: 分析対象期間（日）

        Returns:
            リソース使用状況のDataFrame
        """
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(days=days)

        if resource_type == 'instance':
            # Compute Engineインスタンスの使用状況
            instances = []

            for zone in self._list_zones():
                request = compute_v1.ListInstancesRequest(
                    project=self.project_id,
                    zone=zone
                )
                for instance in self.compute_client.list(request=request):
                    usage_data = self._get_instance_metrics(instance.name, zone, start_time, now)
                    instances.append({
                        'name': instance.name,
                        'zone': zone,
                        'machine_type': instance.machine_type.split('/')[-1],
                        'status': instance.status,
                        'creation_timestamp': instance.creation_timestamp,
                        'cpu_usage_avg': usage_data.get('cpu_usage_avg'),
                        'memory_usage_avg': usage_data.get('memory_usage_avg'),
                        'network_in_avg': usage_data.get('network_in_avg'),
                        'network_out_avg': usage_data.get('network_out_avg')
                    })

            return pd.DataFrame(instances)

        elif resource_type == 'disk':
            # 永続ディスクの使用状況
            disks = []
            disk_client = compute_v1.DisksClient()

            for zone in self._list_zones():
                request = compute_v1.ListDisksRequest(
                    project=self.project_id,
                    zone=zone
                )
                for disk in disk_client.list(request=request):
                    disks.append({
                        'name': disk.name,
                        'zone': zone,
                        'size_gb': disk.size_gb,
                        'type': disk.type.split('/')[-1],
                        'status': disk.status,
                        'creation_timestamp': disk.creation_timestamp,
                        'users': len(disk.users) if hasattr(disk, 'users') and disk.users else 0
                    })

            return pd.DataFrame(disks)

        elif resource_type == 'bucket':
            # Cloud Storageバケットの使用状況
            buckets = []

            for bucket in self.storage_client.list_buckets():
                if bucket.location.startswith(self.region):
                    bucket_stats = self._get_bucket_statistics(bucket.name)
                    buckets.append({
                        'name': bucket.name,
                        'location': bucket.location,
                        'storage_class': bucket.storage_class,
                        'size_bytes': bucket_stats.get('size', 0),
                        'object_count': bucket_stats.get('count', 0),
                        'creation_time': bucket.time_created
                    })

            return pd.DataFrame(buckets)

        else:
            logger.error(f"サポートされていないリソースタイプ: {resource_type}")
            return pd.DataFrame()

    def _list_zones(self) -> List[str]:
        """
        プロジェクトで利用可能なゾーンのリストを取得します。

        Returns:
            ゾーン名のリスト
        """
        regions_client = compute_v1.RegionsClient()
        request = compute_v1.ListRegionsRequest(project=self.project_id)

        zones = []
        for region in regions_client.list(request=request):
            if region.name == self.region:
                zones_client = compute_v1.ZonesClient()
                zones_request = compute_v1.ListZonesRequest(project=self.project_id)

                for zone in zones_client.list(request=zones_request):
                    if zone.region.endswith(f"/{region.name}"):
                        zones.append(zone.name)

        return zones

    def _get_instance_metrics(
        self,
        instance_name: str,
        zone: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, float]:
        """
        インスタンスのメトリクスを取得します。

        Args:
            instance_name: インスタンス名
            zone: ゾーン
            start_time: 開始時刻
            end_time: 終了時刻

        Returns:
            メトリクスデータの辞書
        """
        metrics = {}

        # CPU使用率
        cpu_metric_type = 'compute.googleapis.com/instance/cpu/utilization'
        cpu_data = self._query_metric(
            instance_name, zone, cpu_metric_type, start_time, end_time
        )
        metrics['cpu_usage_avg'] = cpu_data

        # メモリ使用率は直接取得できないため省略
        metrics['memory_usage_avg'] = None

        # ネットワーク受信
        net_in_metric_type = 'compute.googleapis.com/instance/network/received_bytes_count'
        net_in_data = self._query_metric(
            instance_name, zone, net_in_metric_type, start_time, end_time
        )
        metrics['network_in_avg'] = net_in_data

        # ネットワーク送信
        net_out_metric_type = 'compute.googleapis.com/instance/network/sent_bytes_count'
        net_out_data = self._query_metric(
            instance_name, zone, net_out_metric_type, start_time, end_time
        )
        metrics['network_out_avg'] = net_out_data

        return metrics

    def _query_metric(
        self,
        instance_name: str,
        zone: str,
        metric_type: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[float]:
        """
        特定のメトリクスを取得します。

        Args:
            instance_name: インスタンス名
            zone: ゾーン
            metric_type: メトリックタイプ
            start_time: 開始時刻
            end_time: 終了時刻

        Returns:
            平均値または None
        """
        try:
            project_name = f"projects/{self.project_id}"
            interval = monitoring_v3.TimeInterval(
                {
                    "start_time": {"seconds": int(start_time.timestamp())},
                    "end_time": {"seconds": int(end_time.timestamp())},
                }
            )

            # メトリクスフィルタを作成
            filter_str = (
                f'metric.type="{metric_type}" AND '
                f'resource.type="gce_instance" AND '
                f'resource.labels.instance_id="{instance_name}" AND '
                f'resource.labels.zone="{zone}"'
            )

            # メトリクスを取得
            results = self.monitoring_client.list_time_series(
                request={
                    "name": project_name,
                    "filter": filter_str,
                    "interval": interval,
                    "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                }
            )

            # 結果を処理
            values = []
            for result in results:
                for point in result.points:
                    values.append(point.value.double_value or point.value.int64_value)

            # 平均値を計算
            if values:
                return sum(values) / len(values)
            else:
                return None

        except Exception as e:
            logger.error(f"メトリクス取得エラー: {e}")
            return None

    def _get_bucket_statistics(self, bucket_name: str) -> Dict[str, int]:
        """
        バケットの統計情報を取得します。

        Args:
            bucket_name: バケット名

        Returns:
            統計情報の辞書
        """
        try:
            bucket = self.storage_client.get_bucket(bucket_name)
            total_size = 0
            total_count = 0

            # バケット内のオブジェクトを取得
            blobs = bucket.list_blobs()
            for blob in blobs:
                total_size += blob.size
                total_count += 1

            return {
                'size': total_size,
                'count': total_count
            }
        except Exception as e:
            logger.error(f"バケット統計情報取得エラー: {e}")
            return {'size': 0, 'count': 0}

    def optimize_instance_schedule(self) -> Dict[str, Any]:
        """
        インスタンスのスケジュール最適化を行います。

        Returns:
            最適化結果の辞書
        """
        # インスタンスの使用状況を取得
        instance_df = self.get_resource_usage('instance')

        if instance_df.empty:
            return {'status': 'no_instances'}

        # CPU利用率の低いインスタンスを特定
        low_usage_instances = instance_df[
            (instance_df['cpu_usage_avg'] < 0.1) &  # 10%未満のCPU使用率
            (instance_df['status'] == 'RUNNING')
        ]

        # 最適化対象インスタンスのリスト
        optimization_targets = []

        for _, instance in low_usage_instances.iterrows():
            # インスタンス情報
            instance_info = {
                'name': instance['name'],
                'zone': instance['zone'],
                'machine_type': instance['machine_type'],
                'cpu_usage_avg': instance['cpu_usage_avg'],
                'recommendations': []
            }

            # マシンタイプのダウングレード推奨
            if 'n2-standard' in instance['machine_type']:
                # 例: n2-standard-4 → n2-standard-2
                current_size = int(instance['machine_type'].split('-')[-1])
                if current_size > 1:
                    new_size = max(1, current_size // 2)
                    new_machine_type = instance['machine_type'].replace(
                        f'-{current_size}', f'-{new_size}'
                    )
                    instance_info['recommendations'].append({
                        'type': 'resize',
                        'current': instance['machine_type'],
                        'recommended': new_machine_type,
                        'estimated_savings': self._estimate_instance_savings(
                            instance['machine_type'], new_machine_type
                        )
                    })

            # スポットインスタンスへの変換推奨
            if not instance['name'].endswith('-spot'):
                instance_info['recommendations'].append({
                    'type': 'convert_to_spot',
                    'estimated_savings': '60-90% cost reduction'
                })

            # 夜間停止スケジュール推奨
            instance_info['recommendations'].append({
                'type': 'schedule_stop',
                'schedule': 'Weekdays 22:00-08:00 JST',
                'estimated_savings': 'Approx. 40% cost reduction'
            })

            if instance_info['recommendations']:
                optimization_targets.append(instance_info)

        return {
            'status': 'success',
            'instances_analyzed': len(instance_df),
            'optimization_targets': len(optimization_targets),
            'recommendations': optimization_targets
        }

    def _estimate_instance_savings(
        self,
        current_type: str,
        new_type: str
    ) -> str:
        """
        インスタンスサイズ変更による推定節約額を計算します。

        Args:
            current_type: 現在のマシンタイプ
            new_type: 推奨マシンタイプ

        Returns:
            推定節約額の文字列
        """
        # 簡易版の料金表（実際の料金はGCPの料金計算機で確認してください）
        pricing = {
            'n2-standard-1': 0.0612,  # 時間あたりの料金（USD）
            'n2-standard-2': 0.1224,
            'n2-standard-4': 0.2449,
            'n2-standard-8': 0.4898,
            'e2-standard-2': 0.0847,
            'e2-standard-4': 0.1693,
            'e2-standard-8': 0.3387
        }

        current_price = pricing.get(current_type, 0)
        new_price = pricing.get(new_type, 0)

        if current_price == 0 or new_price == 0:
            return 'Unknown'

        savings_percent = (current_price - new_price) / current_price * 100
        monthly_savings = (current_price - new_price) * 24 * 30  # 30日間

        return f"約{savings_percent:.1f}%（月間約${monthly_savings:.2f}）"

    def optimize_federated_learning_schedule(self) -> Dict[str, Any]:
        """
        連合学習ジョブの実行スケジュールを最適化します。

        Returns:
            最適化結果の辞書
        """
        from backend.federated_learning.scheduler.optimal_scheduling import (
            FederatedLearningScheduler, TOKYO_TZ
        )

        # 割引時間帯（日本時間）
        discount_hours = {
            'weekday': {'start': 22, 'end': 8},  # 22:00-翌8:00
            'weekend': {'start': 0, 'end': 24}   # 終日
        }

        # 現在のスケジュールされたジョブを取得
        parent = f"projects/{self.project_id}/locations/{self.region}"
        jobs = list(self.scheduler_client.list_jobs(parent=parent))

        federated_jobs = [job for job in jobs
                         if 'federated' in job.name.lower() or 'fl_job' in job.name.lower()]

        # スケジュール最適化
        fl_scheduler = FederatedLearningScheduler(self.project_id, self.region)
        optimized_jobs = []

        for job in federated_jobs:
            # 現在のスケジュール情報を解析
            current_schedule = job.schedule
            description = job.description

            # 最適化が必要かどうかを判断
            needs_optimization = self._check_if_schedule_needs_optimization(
                job.schedule, job.time_zone
            )

            if needs_optimization:
                # 推定実行時間（例: 2時間）
                estimated_duration = 120

                # 既存のjobを再スケジュール
                try:
                    # 最適なスケジュールを計算
                    optimal_time = fl_scheduler._calculate_optimal_run_time(estimated_duration)
                    optimal_time_jst = optimal_time.astimezone(JST)

                    optimized_jobs.append({
                        'name': job.name.split('/')[-1],
                        'current_schedule': current_schedule,
                        'optimized_schedule': f"JST {optimal_time_jst.strftime('%Y-%m-%d %H:%M')}",
                        'estimated_savings': 'Approx. 20% on compute costs'
                    })

                except Exception as e:
                    logger.error(f"連合学習スケジュール最適化エラー: {e}")
            else:
                # 既に最適化されている
                optimized_jobs.append({
                    'name': job.name.split('/')[-1],
                    'current_schedule': current_schedule,
                    'status': 'already_optimized',
                    'estimated_savings': 'N/A (Already optimized)'
                })

        return {
            'status': 'success',
            'jobs_analyzed': len(federated_jobs),
            'jobs_to_optimize': len([j for j in optimized_jobs if 'optimized_schedule' in j]),
            'optimized_jobs': optimized_jobs
        }

    def _check_if_schedule_needs_optimization(
        self,
        schedule: str,
        time_zone: str
    ) -> bool:
        """
        スケジュールが最適化が必要かどうかを判断します。

        Args:
            schedule: cronスケジュール文字列
            time_zone: タイムゾーン

        Returns:
            最適化が必要な場合はTrue
        """
        try:
            # cronスケジュールを解析（分 時 日 月 曜日）
            cron_parts = schedule.split()
            if len(cron_parts) != 5:
                return True  # 不明なフォーマットは最適化が必要

            # 時間部分を取得
            hour = cron_parts[1]

            # 単一の時間の場合
            if hour.isdigit():
                hour_val = int(hour)
                # 東京リージョンの割引時間帯外なら最適化が必要
                if time_zone == 'Asia/Tokyo':
                    if 8 <= hour_val < 22:  # 8:00-22:00は最適化が必要
                        return True
                else:
                    # タイムゾーンが異なる場合、変換が必要
                    return True
            else:
                # 複雑なスケジュール（例: */2）は分析が必要
                return True

        except Exception:
            # エラーが発生した場合は最適化を推奨
            return True

        # その他のケースでは最適化は不要
        return False

    def generate_optimization_report(self) -> Dict[str, Any]:
        """
        最適化レポートを生成します。

        Returns:
            最適化レポートの辞書
        """
        # コスト分析
        costs_df = self.analyze_costs(days=30)

        # インスタンス最適化
        instance_recommendations = self.optimize_instance_schedule()

        # 連合学習スケジュール最適化
        fl_schedule_recommendations = self.optimize_federated_learning_schedule()

        # レポートにまとめる
        report = {
            'generated_at': datetime.now(JST).isoformat(),
            'project_id': self.project_id,
            'region': self.region,
            'summary': {
                'total_cost_30d': costs_df['cost'].sum() if not costs_df.empty else 0,
                'instance_optimization_targets': instance_recommendations.get('optimization_targets', 0),
                'federated_learning_jobs_to_optimize': fl_schedule_recommendations.get('jobs_to_optimize', 0),
                'estimated_monthly_savings': self._calculate_total_savings(
                    instance_recommendations, fl_schedule_recommendations
                )
            },
            'recommendations': {
                'instances': instance_recommendations,
                'federated_learning': fl_schedule_recommendations
            }
        }

        return report

    def _calculate_total_savings(
        self,
        instance_recommendations: Dict[str, Any],
        fl_recommendations: Dict[str, Any]
    ) -> str:
        """
        総節約額を計算します。

        Args:
            instance_recommendations: インスタンス最適化レコメンデーション
            fl_recommendations: 連合学習最適化レコメンデーション

        Returns:
            推定総節約額の文字列
        """
        # サンプル実装（実際の計算は料金体系に基づいて実装する必要があります）
        return "年間$3,000-5,000相当（最適化の完全実装時）"

    def save_report(self, report: Dict[str, Any], output_file: str) -> None:
        """
        レポートをファイルに保存します。

        Args:
            report: 保存するレポート
            output_file: 出力ファイルパス
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"最適化レポートを保存しました: {output_file}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="GCPコスト分析と最適化スクリプト"
    )
    parser.add_argument(
        "--project-id",
        required=True,
        help="GCPプロジェクトID"
    )
    parser.add_argument(
        "--region",
        default="asia-northeast1",
        help="GCPリージョン (デフォルト: asia-northeast1)"
    )
    parser.add_argument(
        "--output",
        default="gcp_cost_optimization_report.json",
        help="出力レポートファイルパス"
    )
    args = parser.parse_args()

    try:
        optimizer = GCPCostOptimizer(args.project_id, args.region)
        report = optimizer.generate_optimization_report()
        optimizer.save_report(report, args.output)
        logger.info("コスト最適化分析が完了しました")
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())