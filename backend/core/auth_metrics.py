# -*- coding: utf-8 -*-
"""
認証メトリクスコレクター
ユーザー認証に関するメトリクス収集と監視を行います。
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import threading
import redis
import json
import os
from enum import Enum

# ロガーの設定
logger = logging.getLogger(__name__)

# Redis接続設定
REDIS_HOST = os.environ.get("REDIS_HOST", "startup-wellness-redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 1))  # メトリクス用に別DBを使用
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)

class MetricType(str, Enum):
    """メトリクスタイプの列挙型"""
    COUNTER = "counter"  # 単調増加するカウンター
    GAUGE = "gauge"      # 任意の値をとるゲージ
    HISTOGRAM = "histogram"  # 値の分布を記録するヒストグラム
    SUMMARY = "summary"  # サマリー統計

class AuthMetricsCollector:
    """
    認証メトリクスの収集と監視を行うクラス
    シングルトンパターンで実装
    """
    _instance = None
    _redis_client = None
    _lock = threading.Lock()
    _metrics = {}
    _last_alert_sent = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AuthMetricsCollector, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """メトリクスコレクターの初期化"""
        try:
            self._initialize_redis()
            self._initialize_metrics()
            logger.info("AuthMetricsCollectorが正常に初期化されました")
        except Exception as e:
            logger.error(f"AuthMetricsCollectorの初期化中にエラーが発生しました: {str(e)}")

    def _initialize_redis(self):
        """Redisクライアントの初期化"""
        try:
            self._redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True
            )
            self._redis_client.ping()  # 接続テスト
            logger.info("メトリクス用Redisクライアントが初期化されました")
        except Exception as e:
            logger.error(f"メトリクス用Redisクライアントの初期化中にエラーが発生しました: {str(e)}")
            self._redis_client = None

    def _initialize_metrics(self):
        """メトリクスの初期化"""
        # メトリクス定義
        self._metrics = {
            # ログイン関連メトリクス
            "login_attempts": {
                "type": MetricType.COUNTER,
                "description": "ログイン試行回数",
                "labels": ["success", "user_role"]
            },
            "login_failures": {
                "type": MetricType.COUNTER,
                "description": "ログイン失敗回数",
                "labels": ["reason", "ip_address"]
            },
            "active_sessions": {
                "type": MetricType.GAUGE,
                "description": "アクティブなセッション数",
                "labels": ["user_role"]
            },
            "session_duration": {
                "type": MetricType.HISTOGRAM,
                "description": "セッション持続時間（秒）",
                "labels": ["user_role"],
                "buckets": [60, 300, 900, 1800, 3600, 7200, 14400, 28800, 86400]
            },

            # MFA関連メトリクス
            "mfa_attempts": {
                "type": MetricType.COUNTER,
                "description": "MFA認証試行回数",
                "labels": ["success", "mfa_type"]
            },
            "mfa_setup_attempts": {
                "type": MetricType.COUNTER,
                "description": "MFA設定試行回数",
                "labels": ["success", "mfa_type"]
            },
            "enabled_mfa_count": {
                "type": MetricType.GAUGE,
                "description": "MFAを有効にしているユーザー数",
                "labels": ["mfa_type"]
            },

            # パスワード関連メトリクス
            "password_reset_requests": {
                "type": MetricType.COUNTER,
                "description": "パスワードリセット要求回数",
                "labels": []
            },
            "password_changes": {
                "type": MetricType.COUNTER,
                "description": "パスワード変更回数",
                "labels": ["self_initiated"]
            },

            # ユーザー関連メトリクス
            "user_registrations": {
                "type": MetricType.COUNTER,
                "description": "ユーザー登録回数",
                "labels": ["user_role"]
            },
            "active_users": {
                "type": MetricType.GAUGE,
                "description": "アクティブなユーザー数",
                "labels": ["user_role"]
            },

            # 認証パフォーマンス関連メトリクス
            "auth_latency": {
                "type": MetricType.HISTOGRAM,
                "description": "認証処理のレイテンシ（ミリ秒）",
                "labels": ["operation"],
                "buckets": [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
            },

            # セキュリティ関連メトリクス
            "suspicious_activities": {
                "type": MetricType.COUNTER,
                "description": "不審なアクティビティの検出回数",
                "labels": ["activity_type", "severity"]
            },
            "rate_limit_hits": {
                "type": MetricType.COUNTER,
                "description": "レート制限ヒット回数",
                "labels": ["endpoint", "ip_address"]
            }
        }

        # Redisが利用可能な場合、初期値を設定
        if self._redis_client:
            try:
                # 各メトリクスの定義をRedisに保存
                for metric_name, metric_def in self._metrics.items():
                    self._redis_client.hset(
                        f"metric_def:{metric_name}",
                        mapping={
                            "type": metric_def["type"].value,
                            "description": metric_def["description"],
                            "labels": json.dumps(metric_def.get("labels", [])),
                            "buckets": json.dumps(metric_def.get("buckets", []))
                        }
                    )

                # ゲージタイプのメトリクスの初期値を0に設定
                for metric_name, metric_def in self._metrics.items():
                    if metric_def["type"] == MetricType.GAUGE:
                        # ラベルがある場合は基本的なラベルの組み合わせで初期化
                        if metric_def.get("labels"):
                            for label in metric_def.get("labels", []):
                                self._redis_client.hset(
                                    f"metric:{metric_name}",
                                    f"default_{label}",
                                    0
                                )
                        else:
                            self._redis_client.hset(f"metric:{metric_name}", "default", 0)

            except Exception as e:
                logger.error(f"メトリクスの初期化中にエラーが発生しました: {str(e)}")

    def increment(self, metric_name: str, value: int = 1, labels: Dict[str, str] = None) -> bool:
        """
        カウンターメトリクスをインクリメント

        Args:
            metric_name: メトリクス名
            value: インクリメントする値（デフォルト: 1）
            labels: メトリクスのラベル

        Returns:
            bool: 成功した場合はTrue
        """
        if metric_name not in self._metrics:
            logger.warning(f"未定義のメトリクスです: {metric_name}")
            return False

        if self._metrics[metric_name]["type"] != MetricType.COUNTER:
            logger.warning(f"メトリクス {metric_name} はカウンタータイプではありません")
            return False

        try:
            # Redisが利用可能な場合
            if self._redis_client:
                # ラベルが指定されている場合はラベル付きでカウント
                if labels:
                    label_str = self._format_labels(labels)
                    self._redis_client.hincrby(f"metric:{metric_name}", label_str, value)
                else:
                    self._redis_client.hincrby(f"metric:{metric_name}", "default", value)

                # 監視のために最新の値と時刻を記録
                self._redis_client.hset(
                    f"metric_meta:{metric_name}",
                    mapping={
                        "last_updated": datetime.now().isoformat(),
                        "last_value": value
                    }
                )

                return True
            else:
                # Redisが利用できない場合はインメモリで記録
                label_str = self._format_labels(labels) if labels else "default"
                key = f"{metric_name}:{label_str}"

                if key not in self._metrics:
                    self._metrics[key] = 0

                self._metrics[key] += value
                return True

        except Exception as e:
            logger.error(f"メトリクスの更新中にエラーが発生しました: {metric_name}, {str(e)}")
            return False

    def set_gauge(self, metric_name: str, value: float, labels: Dict[str, str] = None) -> bool:
        """
        ゲージメトリクスを設定

        Args:
            metric_name: メトリクス名
            value: 設定する値
            labels: メトリクスのラベル

        Returns:
            bool: 成功した場合はTrue
        """
        if metric_name not in self._metrics:
            logger.warning(f"未定義のメトリクスです: {metric_name}")
            return False

        if self._metrics[metric_name]["type"] != MetricType.GAUGE:
            logger.warning(f"メトリクス {metric_name} はゲージタイプではありません")
            return False

        try:
            # Redisが利用可能な場合
            if self._redis_client:
                # ラベルが指定されている場合はラベル付きで設定
                if labels:
                    label_str = self._format_labels(labels)
                    self._redis_client.hset(f"metric:{metric_name}", label_str, value)
                else:
                    self._redis_client.hset(f"metric:{metric_name}", "default", value)

                # 監視のために最新の値と時刻を記録
                self._redis_client.hset(
                    f"metric_meta:{metric_name}",
                    mapping={
                        "last_updated": datetime.now().isoformat(),
                        "last_value": value
                    }
                )

                return True
            else:
                # Redisが利用できない場合はインメモリで記録
                label_str = self._format_labels(labels) if labels else "default"
                key = f"{metric_name}:{label_str}"
                self._metrics[key] = value
                return True

        except Exception as e:
            logger.error(f"メトリクスの更新中にエラーが発生しました: {metric_name}, {str(e)}")
            return False

    def observe(self, metric_name: str, value: float, labels: Dict[str, str] = None) -> bool:
        """
        ヒストグラムまたはサマリーメトリクスに値を観測

        Args:
            metric_name: メトリクス名
            value: 観測値
            labels: メトリクスのラベル

        Returns:
            bool: 成功した場合はTrue
        """
        if metric_name not in self._metrics:
            logger.warning(f"未定義のメトリクスです: {metric_name}")
            return False

        if self._metrics[metric_name]["type"] not in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
            logger.warning(f"メトリクス {metric_name} はヒストグラムまたはサマリータイプではありません")
            return False

        try:
            # Redisが利用可能な場合
            if self._redis_client:
                # ラベルが指定されている場合はラベル付きで記録
                label_str = self._format_labels(labels) if labels else "default"

                # 値をタイムスタンプ付きで記録
                timestamp = datetime.now().timestamp()
                self._redis_client.zadd(
                    f"metric_values:{metric_name}:{label_str}",
                    {f"{value}:{timestamp}": timestamp}
                )

                # バケットの更新（ヒストグラムの場合）
                if self._metrics[metric_name]["type"] == MetricType.HISTOGRAM:
                    buckets = self._metrics[metric_name].get("buckets", [])
                    for bucket in buckets:
                        if value <= bucket:
                            self._redis_client.hincrby(
                                f"metric_buckets:{metric_name}:{label_str}",
                                f"le:{bucket}",
                                1
                            )

                # 古い値の削除（24時間以上前のデータ）
                old_timestamp = (datetime.now() - timedelta(days=1)).timestamp()
                self._redis_client.zremrangebyscore(
                    f"metric_values:{metric_name}:{label_str}",
                    0, old_timestamp
                )

                # 監視のために最新の値と時刻を記録
                self._redis_client.hset(
                    f"metric_meta:{metric_name}",
                    mapping={
                        "last_updated": datetime.now().isoformat(),
                        "last_value": value
                    }
                )

                return True
            else:
                # Redisが利用できない場合は簡易的に記録
                label_str = self._format_labels(labels) if labels else "default"
                key = f"{metric_name}:{label_str}"

                if key not in self._metrics:
                    self._metrics[key] = []

                self._metrics[key].append((datetime.now(), value))

                # 古いデータの削除
                old_time = datetime.now() - timedelta(days=1)
                self._metrics[key] = [item for item in self._metrics[key] if item[0] > old_time]

                return True

        except Exception as e:
            logger.error(f"メトリクスの観測中にエラーが発生しました: {metric_name}, {str(e)}")
            return False

    def time_operation(self, metric_name: str, labels: Dict[str, str] = None):
        """
        操作の実行時間を計測するコンテキストマネージャ

        Args:
            metric_name: メトリクス名
            labels: メトリクスのラベル

        Returns:
            コンテキストマネージャ
        """
        class Timer:
            def __init__(self, collector, metric_name, labels):
                self.collector = collector
                self.metric_name = metric_name
                self.labels = labels
                self.start_time = None

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.start_time is not None:
                    elapsed_ms = (time.time() - self.start_time) * 1000
                    self.collector.observe(self.metric_name, elapsed_ms, self.labels)

        return Timer(self, metric_name, labels)

    def get_metrics(self) -> Dict[str, Any]:
        """
        全メトリクスを取得

        Returns:
            Dict: メトリクスデータ
        """
        metrics_data = {}

        try:
            # Redisが利用可能な場合
            if self._redis_client:
                for metric_name in self._metrics:
                    metric_data = {
                        "type": self._metrics[metric_name]["type"].value,
                        "description": self._metrics[metric_name]["description"],
                        "values": {}
                    }

                    # メトリクスタイプに応じたデータ取得
                    if self._metrics[metric_name]["type"] in [MetricType.COUNTER, MetricType.GAUGE]:
                        # カウンターとゲージは直接値を取得
                        values = self._redis_client.hgetall(f"metric:{metric_name}")
                        metric_data["values"] = {k: float(v) for k, v in values.items()}

                    elif self._metrics[metric_name]["type"] == MetricType.HISTOGRAM:
                        # ヒストグラムはバケットと最近の値を取得
                        # ラベルごとのキーを取得
                        label_keys = self._redis_client.keys(f"metric_values:{metric_name}:*")

                        for label_key in label_keys:
                            label = label_key.split(":")[-1]

                            # 最近の値（最大100個）
                            recent_values = self._redis_client.zrevrange(
                                label_key, 0, 99, withscores=True
                            )
                            values = [float(v.split(":")[0]) for v, _ in recent_values]

                            # バケット情報
                            buckets = self._redis_client.hgetall(f"metric_buckets:{metric_name}:{label}")

                            metric_data["values"][label] = {
                                "recent_values": values,
                                "buckets": {k: int(v) for k, v in buckets.items()}
                            }

                    metrics_data[metric_name] = metric_data
            else:
                # Redisが利用できない場合はインメモリデータを返す
                for key, value in self._metrics.items():
                    if ":" in key:
                        # キーが "メトリクス名:ラベル" の形式
                        metric_name, label = key.split(":", 1)

                        if metric_name not in metrics_data:
                            metrics_data[metric_name] = {
                                "type": self._metrics[metric_name]["type"].value,
                                "description": self._metrics[metric_name]["description"],
                                "values": {}
                            }

                        metrics_data[metric_name]["values"][label] = value
                    elif isinstance(value, dict) and "type" in value:
                        # メトリクス定義
                        continue
                    else:
                        # 単純なメトリクス値
                        metrics_data[key] = {
                            "type": "unknown",
                            "description": "In-memory metric",
                            "values": {"default": value}
                        }

            return metrics_data

        except Exception as e:
            logger.error(f"メトリクスの取得中にエラーが発生しました: {str(e)}")
            return {"error": str(e)}

    def get_metric(self, metric_name: str) -> Dict[str, Any]:
        """
        特定のメトリクスを取得

        Args:
            metric_name: メトリクス名

        Returns:
            Dict: メトリクスデータ
        """
        if metric_name not in self._metrics:
            logger.warning(f"未定義のメトリクスです: {metric_name}")
            return {}

        try:
            # Redisが利用可能な場合
            if self._redis_client:
                metric_data = {
                    "type": self._metrics[metric_name]["type"].value,
                    "description": self._metrics[metric_name]["description"],
                    "values": {}
                }

                # メトリクスタイプに応じたデータ取得
                if self._metrics[metric_name]["type"] in [MetricType.COUNTER, MetricType.GAUGE]:
                    # カウンターとゲージは直接値を取得
                    values = self._redis_client.hgetall(f"metric:{metric_name}")
                    metric_data["values"] = {k: float(v) for k, v in values.items()}

                elif self._metrics[metric_name]["type"] == MetricType.HISTOGRAM:
                    # ヒストグラムはバケットと最近の値を取得
                    # ラベルごとのキーを取得
                    label_keys = self._redis_client.keys(f"metric_values:{metric_name}:*")

                    for label_key in label_keys:
                        label = label_key.split(":")[-1]

                        # 最近の値（最大100個）
                        recent_values = self._redis_client.zrevrange(
                            label_key, 0, 99, withscores=True
                        )
                        values = [float(v.split(":")[0]) for v, _ in recent_values]

                        # バケット情報
                        buckets = self._redis_client.hgetall(f"metric_buckets:{metric_name}:{label}")

                        metric_data["values"][label] = {
                            "recent_values": values,
                            "buckets": {k: int(v) for k, v in buckets.items()}
                        }

                return metric_data
            else:
                # Redisが利用できない場合はインメモリデータを返す
                metric_data = {
                    "type": self._metrics[metric_name]["type"].value,
                    "description": self._metrics[metric_name]["description"],
                    "values": {}
                }

                for key, value in self._metrics.items():
                    if key.startswith(f"{metric_name}:"):
                        # キーが "メトリクス名:ラベル" の形式
                        label = key.split(":", 1)[1]
                        metric_data["values"][label] = value

                return metric_data

        except Exception as e:
            logger.error(f"メトリクスの取得中にエラーが発生しました: {metric_name}, {str(e)}")
            return {"error": str(e)}

    def check_alerts(self) -> List[Dict[str, Any]]:
        """
        アラート条件をチェックし、必要に応じてアラートを発行

        Returns:
            List[Dict]: 発行されたアラート
        """
        alerts = []

        try:
            # アラートルール定義
            alert_rules = [
                {
                    "name": "high_login_failures",
                    "description": "ログイン失敗回数が多すぎます",
                    "metric": "login_failures",
                    "condition": lambda x: sum(float(v) for v in x.values()) > 10,
                    "severity": "warning",
                    "cooldown": 3600  # 1時間のクールダウン
                },
                {
                    "name": "high_suspicious_activities",
                    "description": "不審なアクティビティが多数検出されました",
                    "metric": "suspicious_activities",
                    "condition": lambda x: sum(float(v) for v in x.values()) > 5,
                    "severity": "critical",
                    "cooldown": 1800  # 30分のクールダウン
                },
                {
                    "name": "high_auth_latency",
                    "description": "認証レイテンシが高すぎます",
                    "metric": "auth_latency",
                    "condition": lambda x: any(
                        v.get("recent_values") and max(v["recent_values"]) > 5000
                        for v in x.values() if isinstance(v, dict)
                    ),
                    "severity": "warning",
                    "cooldown": 1800  # 30分のクールダウン
                }
            ]

            # 各アラートルールをチェック
            for rule in alert_rules:
                metric_data = self.get_metric(rule["metric"])

                if "values" in metric_data and metric_data["values"]:
                    # 条件をチェック
                    if rule["condition"](metric_data["values"]):
                        # クールダウン期間をチェック
                        last_alert_time = self._last_alert_sent.get(rule["name"])
                        now = datetime.now()

                        if (not last_alert_time or
                            (now - last_alert_time).total_seconds() > rule["cooldown"]):
                            # アラートを発行
                            alert = {
                                "name": rule["name"],
                                "description": rule["description"],
                                "severity": rule["severity"],
                                "metric": rule["metric"],
                                "values": metric_data["values"],
                                "timestamp": now.isoformat()
                            }

                            alerts.append(alert)
                            self._last_alert_sent[rule["name"]] = now

                            # ログに記録
                            logger.warning(
                                f"アラート発行: {rule['name']}, "
                                f"重要度: {rule['severity']}, "
                                f"メトリクス: {rule['metric']}"
                            )

            return alerts

        except Exception as e:
            logger.error(f"アラートチェック中にエラーが発生しました: {str(e)}")
            return [{"error": str(e)}]

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """
        ラベル辞書を文字列にフォーマット

        Args:
            labels: ラベル辞書

        Returns:
            str: フォーマットされたラベル文字列
        """
        if not labels:
            return "default"

        # ラベルをアルファベット順にソートして文字列化
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def track_login_attempt(self, success: bool, user_role: str = "unknown",
                           failure_reason: str = None, ip_address: str = None) -> None:
        """
        ログイン試行を記録

        Args:
            success: 成功したかどうか
            user_role: ユーザーロール
            failure_reason: 失敗理由（失敗時のみ）
            ip_address: IPアドレス（失敗時のみ）
        """
        # ログイン試行をカウント
        self.increment("login_attempts", 1, {"success": str(success), "user_role": user_role})

        # 失敗の場合は失敗カウントも増やす
        if not success and failure_reason:
            labels = {"reason": failure_reason}
            if ip_address:
                labels["ip_address"] = ip_address

            self.increment("login_failures", 1, labels)

    def track_mfa_attempt(self, success: bool, mfa_type: str) -> None:
        """
        MFA試行を記録

        Args:
            success: 成功したかどうか
            mfa_type: MFAタイプ
        """
        self.increment("mfa_attempts", 1, {"success": str(success), "mfa_type": mfa_type})

    def track_mfa_setup(self, success: bool, mfa_type: str) -> None:
        """
        MFA設定を記録

        Args:
            success: 成功したかどうか
            mfa_type: MFAタイプ
        """
        self.increment("mfa_setup_attempts", 1, {"success": str(success), "mfa_type": mfa_type})

        # 成功した場合はMFA有効ユーザー数を更新
        if success:
            # 現在の値を取得して増やす
            current = self.get_metric("enabled_mfa_count")
            current_value = 0
            if current and "values" in current:
                values = current.get("values", {})
                label_key = f"mfa_type={mfa_type}"
                if label_key in values:
                    current_value = float(values[label_key])

            self.set_gauge("enabled_mfa_count", current_value + 1, {"mfa_type": mfa_type})

    def track_suspicious_activity(self, activity_type: str, severity: str = "low") -> None:
        """
        不審なアクティビティを記録

        Args:
            activity_type: アクティビティタイプ
            severity: 重要度（low, medium, high, critical）
        """
        self.increment(
            "suspicious_activities",
            1,
            {"activity_type": activity_type, "severity": severity}
        )

    def track_rate_limit_hit(self, endpoint: str, ip_address: str) -> None:
        """
        レート制限ヒットを記録

        Args:
            endpoint: エンドポイント
            ip_address: IPアドレス
        """
        self.increment(
            "rate_limit_hits",
            1,
            {"endpoint": endpoint, "ip_address": ip_address}
        )

    def update_active_users(self, user_role: str, count: int) -> None:
        """
        アクティブユーザー数を更新

        Args:
            user_role: ユーザーロール
            count: ユーザー数
        """
        self.set_gauge("active_users", count, {"user_role": user_role})

    def update_active_sessions(self, user_role: str, count: int) -> None:
        """
        アクティブセッション数を更新

        Args:
            user_role: ユーザーロール
            count: セッション数
        """
        self.set_gauge("active_sessions", count, {"user_role": user_role})

    def track_session_duration(self, duration_seconds: float, user_role: str) -> None:
        """
        セッション持続時間を記録

        Args:
            duration_seconds: 持続時間（秒）
            user_role: ユーザーロール
        """
        self.observe("session_duration", duration_seconds, {"user_role": user_role})

    def track_user_registration(self, user_role: str) -> None:
        """
        ユーザー登録を記録

        Args:
            user_role: ユーザーロール
        """
        self.increment("user_registrations", 1, {"user_role": user_role})

    def track_password_reset(self) -> None:
        """パスワードリセット要求を記録"""
        self.increment("password_reset_requests", 1)

    def track_password_change(self, self_initiated: bool) -> None:
        """
        パスワード変更を記録

        Args:
            self_initiated: ユーザー自身による変更かどうか
        """
        self.increment("password_changes", 1, {"self_initiated": str(self_initiated)})

# シングルトンインスタンスの取得関数
def get_auth_metrics() -> AuthMetricsCollector:
    """AuthMetricsCollectorのシングルトンインスタンスを返す"""
    return AuthMetricsCollector()