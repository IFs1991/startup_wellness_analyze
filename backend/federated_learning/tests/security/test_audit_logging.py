"""
監査ログシステムのテスト

TDD Phase 1, Task 1.4: 監査ログシステム
RED段階: 失敗するテストを最初に書く

TDD.yamlに基づく要件:
- test_audit_log_completeness: 監査ログの完全性
- test_log_tamper_detection: ログ改ざん検出
- test_gdpr_compliance_logs: GDPR準拠ログ
"""

import pytest
import asyncio
import json
import hashlib
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Optional
import tempfile
from pathlib import Path


class TestAuditLoggingSystem:
    """監査ログシステムのテストクラス

    TDD.yamlに基づく要件実装:
    - 監査ログの完全性テスト
    - ログ改ざん検出テスト
    - GDPR準拠ログテスト
    - ログストレージ最適化テスト
    """

    def test_audit_log_completeness(self):
        """監査ログの完全性テスト

        要件:
        - すべての重要な操作がログに記録されること
        - ユーザー認証イベントの記録
        - データアクセスイベントの記録
        - システム設定変更の記録
        - エラーイベントの記録
        - メタデータの完全性
        """
        # まだAuditLoggerクラスが実装されていないため、このテストは失敗する（RED段階）
        from backend.federated_learning.security.audit_logging import AuditLogger

        logger = AuditLogger(
            log_level="INFO",
            storage_backend="file",
            enable_encryption=True,
            retention_days=365
        )

        # ユーザー認証イベントのログ
        auth_event = {
            "event_type": "user_authentication",
            "user_id": "user_123",
            "client_ip": "192.168.1.100",
            "user_agent": "FlowerClient/1.0",
            "auth_method": "mTLS",
            "certificate_fingerprint": "sha256:abc123...",
            "success": True
        }

        log_entry = logger.log_event(auth_event)

        # ログエントリの検証
        assert log_entry.id is not None, "ログエントリにIDが設定されること"
        assert log_entry.timestamp is not None, "タイムスタンプが設定されること"
        assert log_entry.event_type == "user_authentication", "イベントタイプが正しく記録されること"
        assert log_entry.user_id == "user_123", "ユーザーIDが記録されること"
        assert log_entry.integrity_hash is not None, "完全性ハッシュが生成されること"

        # データアクセスイベントのログ
        data_access_event = {
            "event_type": "data_access",
            "user_id": "user_123",
            "resource_type": "model_weights",
            "resource_id": "model_v1.2.3",
            "operation": "read",
            "data_classification": "confidential",
            "data_size_bytes": 1048576
        }

        data_log = logger.log_event(data_access_event)
        assert data_log.event_type == "data_access", "データアクセスが記録されること"
        assert data_log.resource_type == "model_weights", "リソースタイプが記録されること"

        # システム設定変更のログ
        config_change_event = {
            "event_type": "configuration_change",
            "user_id": "admin_456",
            "component": "differential_privacy",
            "setting": "epsilon",
            "old_value": "3.0",
            "new_value": "2.5",
            "change_reason": "privacy_enhancement"
        }

        config_log = logger.log_event(config_change_event)
        assert config_log.event_type == "configuration_change", "設定変更が記録されること"

    def test_log_tamper_detection(self):
        """ログ改ざん検出テスト

        要件:
        - ハッシュチェーンによる改ざん検出
        - デジタル署名による検証
        - 改ざんされたログの特定
        - チェーン整合性の検証
        """
        from backend.federated_learning.security.audit_logging import AuditLogger

        logger = AuditLogger(
            log_level="INFO",
            storage_backend="file",
            enable_integrity_chain=True,
            digital_signature=True
        )

        # 複数のログエントリを作成
        events = [
            {"event_type": "login", "user_id": "user_1"},
            {"event_type": "data_access", "user_id": "user_1", "resource": "model_1"},
            {"event_type": "logout", "user_id": "user_1"}
        ]

        log_entries = []
        for event in events:
            entry = logger.log_event(event)
            log_entries.append(entry)

        # ハッシュチェーンの整合性確認
        chain_valid = logger.verify_integrity_chain(log_entries)
        assert chain_valid, "ハッシュチェーンが有効であること"

        # ログエントリの改ざんをシミュレート
        corrupted_entry = log_entries[1]
        original_data = corrupted_entry.data.copy()
        corrupted_entry.data["user_id"] = "malicious_user"

        # 改ざん検出
        tamper_detected = logger.detect_tampering(corrupted_entry)
        assert tamper_detected, "ログの改ざんが検出されること"

        # チェーン検証で改ざんを検出
        chain_valid_after_tamper = logger.verify_integrity_chain(log_entries)
        assert not chain_valid_after_tamper, "改ざん後にチェーンが無効になること"

        # 改ざんされたエントリの特定
        corrupted_entries = logger.find_corrupted_entries(log_entries)
        assert len(corrupted_entries) == 1, "改ざんされたエントリが1つ特定されること"
        assert corrupted_entries[0].id == corrupted_entry.id, "正しいエントリが特定されること"

    def test_gdpr_compliance_logs(self):
        """GDPR準拠ログテスト

        要件:
        - 個人データの匿名化
        - データ保持期間の管理
        - データ削除の記録
        - プライバシー影響評価
        - 同意管理の記録
        """
        from backend.federated_learning.security.audit_logging import AuditLogger

        logger = AuditLogger(
            log_level="INFO",
            storage_backend="file",
            gdpr_compliance=True,
            data_retention_days=365,
            anonymization_level="strict"
        )

        # 個人データを含むイベント
        personal_data_event = {
            "event_type": "personal_data_processing",
            "user_id": "user_789",
            "email": "user@example.com",
            "ip_address": "192.168.1.50",
            "processing_purpose": "federated_learning",
            "legal_basis": "consent",
            "consent_id": "consent_123"
        }

        log_entry = logger.log_event(personal_data_event)

        # 個人データの匿名化確認
        assert "user@example.com" not in str(log_entry.data), "メールアドレスが匿名化されること"
        assert log_entry.data.get("email_hash") is not None, "メールアドレスのハッシュが記録されること"
        assert log_entry.data.get("ip_masked") is not None, "IPアドレスがマスクされること"

        # データ保持期間の確認
        retention_info = logger.get_retention_info(log_entry.id)
        assert retention_info.retention_days == 365, "データ保持期間が設定されること"
        assert retention_info.deletion_date is not None, "削除予定日が設定されること"

        # データ削除リクエストの処理
        deletion_request = {
            "event_type": "data_deletion_request",
            "user_id": "user_789",
            "request_type": "gdpr_erasure",
            "data_categories": ["personal_data", "processing_logs"],
            "verification_method": "email_confirmation"
        }

        deletion_log = logger.log_event(deletion_request)
        assert deletion_log.event_type == "data_deletion_request", "削除リクエストが記録されること"

        # データ削除の実行
        deletion_result = logger.execute_data_deletion("user_789", ["personal_data"])
        assert deletion_result.success, "データ削除が成功すること"
        assert deletion_result.deleted_entries > 0, "削除されたエントリ数が記録されること"

        # 削除後の確認
        remaining_logs = logger.search_logs(user_id="user_789")
        personal_logs = [log for log in remaining_logs if "personal_data" in log.data_categories]
        assert len(personal_logs) == 0, "個人データログが削除されること"

    def test_log_storage_optimization(self):
        """ログストレージ最適化テスト

        要件:
        - 圧縮による容量削減
        - インデックス作成による検索最適化
        - アーカイブ機能
        - 分散ストレージ対応
        """
        from backend.federated_learning.security.audit_logging import AuditLogger

        logger = AuditLogger(
            log_level="INFO",
            storage_backend="optimized_file",
            compression_enabled=True,
            indexing_enabled=True,
            archive_threshold_days=30
        )

        # 大量のログエントリを作成
        events = []
        for i in range(1000):
            event = {
                "event_type": "test_event",
                "user_id": f"user_{i % 100}",
                "timestamp": datetime.utcnow() - timedelta(days=i % 60),
                "data": f"test_data_{i}" * 100  # 大きなデータ
            }
            events.append(event)

        # ログの記録
        start_time = time.time()
        for event in events:
            logger.log_event(event)
        write_time = time.time() - start_time

        # パフォーマンステスト
        assert write_time < 5.0, f"1000エントリの書き込みが5秒以内に完了すること: {write_time}秒"

        # 圧縮効果の確認
        storage_stats = logger.get_storage_stats()
        assert storage_stats.compression_ratio > 0.5, "50%以上の圧縮率を達成すること"

        # インデックス検索のテスト
        start_time = time.time()
        search_results = logger.search_logs(
            user_id="user_42",
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow()
        )
        search_time = time.time() - start_time

        assert search_time < 0.1, f"検索が0.1秒以内に完了すること: {search_time}秒"
        assert len(search_results) > 0, "検索結果が返されること"

        # アーカイブ機能のテスト
        archive_result = logger.archive_old_logs(days_threshold=30)
        assert archive_result.archived_count > 0, "古いログがアーカイブされること"
        assert archive_result.active_count < 1000, "アクティブログ数が減少すること"

    def test_audit_log_security_events(self):
        """セキュリティイベントの監査ログテスト

        要件:
        - 不正アクセス試行の記録
        - セキュリティ違反の検出
        - アラート生成
        - インシデント追跡
        """
        from backend.federated_learning.security.audit_logging import AuditLogger

        logger = AuditLogger(
            log_level="INFO",
            storage_backend="file",
            security_monitoring=True,
            alert_thresholds={
                "failed_login_attempts": 5,
                "suspicious_data_access": 10
            }
        )

        # 不正アクセス試行
        for i in range(6):  # 閾値を超える
            failed_login = {
                "event_type": "authentication_failure",
                "user_id": "attacker",
                "client_ip": "10.0.0.100",
                "failure_reason": "invalid_certificate",
                "attempt_number": i + 1
            }
            logger.log_event(failed_login)

        # セキュリティアラートの確認
        alerts = logger.get_security_alerts()
        assert len(alerts) > 0, "セキュリティアラートが生成されること"

        failed_login_alert = next(
            (alert for alert in alerts if alert.event_type == "excessive_login_failures"),
            None
        )
        assert failed_login_alert is not None, "ログイン失敗アラートが生成されること"
        assert failed_login_alert.severity == "high", "高い重要度が設定されること"

        # インシデント作成
        incident = logger.create_security_incident(
            alert_id=failed_login_alert.id,
            incident_type="brute_force_attack",
            severity="critical"
        )

        assert incident.id is not None, "インシデントが作成されること"
        assert incident.status == "open", "インシデントがオープン状態であること"

    @pytest.mark.asyncio
    async def test_async_audit_logging(self):
        """非同期監査ログテスト

        要件:
        - 非ブロッキングログ記録
        - 並行ログ処理
        - バッチ処理
        - 高スループット対応
        """
        from backend.federated_learning.security.audit_logging import AsyncAuditLogger

        logger = AsyncAuditLogger(
            log_level="INFO",
            storage_backend="async_file",
            batch_size=100,
            flush_interval=1.0
        )

        # 並行ログ記録のテスト
        events = [
            {"event_type": "async_test", "user_id": f"user_{i}", "data": f"data_{i}"}
            for i in range(500)
        ]

        # 非同期でログ記録
        start_time = time.time()
        tasks = [logger.log_event_async(event) for event in events]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # パフォーマンス確認
        throughput = len(events) / (end_time - start_time)
        assert throughput > 100, f"100 events/sec以上のスループットを達成すること: {throughput:.2f} events/sec"

        # 全ログが記録されていることを確認
        assert len(results) == 500, "全てのログが記録されること"
        assert all(result.success for result in results), "全てのログ記録が成功すること"

        # バッファのフラッシュ
        await logger.flush()

        # 永続化の確認
        stored_logs = await logger.search_logs_async(event_type="async_test")
        assert len(stored_logs) == 500, "全てのログが永続化されること"

    def test_audit_log_compliance_reporting(self):
        """監査ログのコンプライアンスレポートテスト

        要件:
        - 定期レポート生成
        - コンプライアンス指標
        - 監査証跡
        - レポート出力フォーマット
        """
        from backend.federated_learning.security.audit_logging import AuditLogger

        logger = AuditLogger(
            log_level="INFO",
            storage_backend="file",
            compliance_reporting=True
        )

        # 様々なイベントを記録
        events = [
            {"event_type": "user_authentication", "user_id": "user_1", "success": True},
            {"event_type": "data_access", "user_id": "user_1", "resource": "model_weights"},
            {"event_type": "configuration_change", "user_id": "admin", "component": "privacy"},
            {"event_type": "data_deletion", "user_id": "user_2", "gdpr_request": True}
        ]

        for event in events:
            logger.log_event(event)

        # コンプライアンスレポート生成
        report_period = {
            "start_date": datetime.utcnow() - timedelta(days=30),
            "end_date": datetime.utcnow()
        }

        compliance_report = logger.generate_compliance_report(**report_period)

        # レポート内容の確認
        assert compliance_report.total_events > 0, "イベント総数が記録されること"
        assert compliance_report.authentication_events > 0, "認証イベント数が記録されること"
        assert compliance_report.data_access_events > 0, "データアクセス数が記録されること"
        assert compliance_report.gdpr_compliance_score > 0.8, "GDPR準拠スコアが80%以上であること"

        # レポート出力フォーマットテスト
        report_json = logger.export_compliance_report(compliance_report, format="json")
        assert "total_events" in report_json, "JSON形式でエクスポートできること"

        report_pdf = logger.export_compliance_report(compliance_report, format="pdf")
        assert report_pdf is not None, "PDF形式でエクスポートできること"

    def test_log_performance_monitoring(self):
        """ログパフォーマンス監視テスト

        要件:
        - ログ記録レイテンシ監視
        - ストレージ使用量監視
        - エラー率監視
        - パフォーマンスアラート
        """
        from backend.federated_learning.security.audit_logging import AuditLogger

        logger = AuditLogger(
            log_level="INFO",
            storage_backend="file",
            performance_monitoring=True,
            performance_thresholds={
                "max_latency_ms": 100,
                "max_storage_gb": 10,
                "max_error_rate": 0.01
            }
        )

        # パフォーマンス指標の収集
        for i in range(100):
            event = {"event_type": "performance_test", "data": f"test_{i}"}
            logger.log_event(event)

        # パフォーマンス統計の取得
        perf_stats = logger.get_performance_stats()

        assert perf_stats.avg_latency_ms < 100, "平均レイテンシが100ms未満であること"
        assert perf_stats.p99_latency_ms < 200, "99パーセンタイルレイテンシが200ms未満であること"
        assert perf_stats.error_rate < 0.01, "エラー率が1%未満であること"
        assert perf_stats.throughput_eps > 10, "10 events/sec以上のスループットを達成すること"

        # ストレージ使用量の確認
        storage_usage = logger.get_storage_usage()
        assert storage_usage.used_gb < 10, "ストレージ使用量が10GB未満であること"
        assert storage_usage.compression_ratio > 0.3, "30%以上の圧縮率を達成すること"


@pytest.fixture
def temp_log_directory():
    """テスト用一時ログディレクトリ"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_security_context():
    """モックセキュリティコンテキスト"""
    with patch('backend.federated_learning.security.audit_logging.get_security_context') as mock:
        mock.return_value = {
            "user_id": "test_user",
            "session_id": "test_session",
            "client_ip": "127.0.0.1",
            "timestamp": datetime.utcnow()
        }
        yield mock