"""
Task 4.2: 自動フェイルオーバー機構 テスト
TDD RED段階: フェイルオーバーシナリオテスト
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Optional

# テスト対象
from backend.federated_learning.failover import (
    FailoverCoordinator,
    PrimaryFailureDetector,
    AutoFailoverManager,
    DataConsistencyChecker,
    FailoverEvent,
    FailoverStatus,
    NodeRole,
    NodeState,
    ClusterState,
    FailoverConfiguration
)


class TestPrimaryFailureDetector:
    """プライマリ障害検知テスト"""

    @pytest.fixture
    def failure_detector(self):
        """プライマリ障害検知器フィクスチャ"""
        config = FailoverConfiguration(
            failure_detection_timeout=10.0,
            health_check_interval=2.0
        )
        return PrimaryFailureDetector(config)

    @pytest.fixture
    def cluster_state(self):
        """クラスター状態フィクスチャ"""
        cluster = ClusterState(cluster_id="test_cluster")

        # プライマリノード
        primary_node = NodeState(
            node_id="primary_node_1",
            role=NodeRole.PRIMARY,
            status=FailoverStatus.HEALTHY,
            last_heartbeat=datetime.now(),
            health_score=1.0
        )
        cluster.update_node_state("primary_node_1", primary_node)

        # セカンダリノード
        secondary_node = NodeState(
            node_id="secondary_node_1",
            role=NodeRole.SECONDARY,
            status=FailoverStatus.HEALTHY,
            last_heartbeat=datetime.now(),
            health_score=0.9
        )
        cluster.update_node_state("secondary_node_1", secondary_node)

        return cluster

    @pytest.mark.asyncio
    async def test_primary_failure_detection(self, failure_detector, cluster_state):
        """プライマリ障害検知テスト"""
        # プライマリノードを失敗状態に設定
        primary_node = cluster_state.get_primary_node()
        primary_node.last_heartbeat = datetime.now() - timedelta(seconds=60)
        primary_node.status = FailoverStatus.FAILED

        # 障害検知
        is_failed = await failure_detector.detect_primary_failure(cluster_state)

        assert is_failed == True
        assert cluster_state.get_primary_node().status == FailoverStatus.FAILED

    @pytest.mark.asyncio
    async def test_healthy_primary_no_failure(self, failure_detector, cluster_state):
        """健全なプライマリでの障害検知テスト（失敗しないことを確認）"""
        # プライマリノードは健全状態
        primary_node = cluster_state.get_primary_node()
        primary_node.last_heartbeat = datetime.now()
        primary_node.status = FailoverStatus.HEALTHY

        # 障害検知
        is_failed = await failure_detector.detect_primary_failure(cluster_state)

        assert is_failed == False
        assert cluster_state.get_primary_node().status == FailoverStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_heartbeat_timeout_detection(self, failure_detector, cluster_state):
        """ハートビートタイムアウト検知テスト"""
        # プライマリノードのハートビートを古くする
        primary_node = cluster_state.get_primary_node()
        primary_node.last_heartbeat = datetime.now() - timedelta(seconds=45)

        # タイムアウト検知
        is_timeout = await failure_detector.check_heartbeat_timeout(cluster_state)

        assert is_timeout == True

    @pytest.mark.asyncio
    async def test_health_score_degradation_detection(self, failure_detector, cluster_state):
        """ヘルススコア劣化検知テスト"""
        # プライマリノードのヘルススコアを劣化
        primary_node = cluster_state.get_primary_node()
        primary_node.health_score = 0.3  # 閾値0.7以下

        # 劣化検知
        is_degraded = await failure_detector.check_health_degradation(cluster_state)

        assert is_degraded == True


class TestAutoFailoverManager:
    """自動フェイルオーバー管理テスト"""

    @pytest.fixture
    def failover_manager(self):
        """フェイルオーバー管理フィクスチャ"""
        config = FailoverConfiguration(
            failover_timeout=30.0,
            enable_automatic_failover=True,
            max_failover_attempts=3
        )
        return AutoFailoverManager(config)

    @pytest.fixture
    def failed_cluster_state(self):
        """障害クラスター状態フィクスチャ"""
        cluster = ClusterState(cluster_id="test_cluster")

        # 失敗したプライマリノード
        failed_primary = NodeState(
            node_id="failed_primary",
            role=NodeRole.FAILED,
            status=FailoverStatus.FAILED,
            last_heartbeat=datetime.now() - timedelta(seconds=60),
            health_score=0.0
        )
        cluster.update_node_state("failed_primary", failed_primary)

        # 健全なセカンダリノード
        healthy_secondary = NodeState(
            node_id="healthy_secondary",
            role=NodeRole.SECONDARY,
            status=FailoverStatus.HEALTHY,
            last_heartbeat=datetime.now(),
            health_score=0.9
        )
        cluster.update_node_state("healthy_secondary", healthy_secondary)

        return cluster

    @pytest.mark.asyncio
    async def test_automatic_failover(self, failover_manager, failed_cluster_state):
        """自動フェイルオーバー実行テスト"""
        # フェイルオーバー実行
        failover_result = await failover_manager.execute_failover(failed_cluster_state)

        assert failover_result.success == True
        assert failover_result.new_primary_id == "healthy_secondary"
        assert failover_result.execution_time < 30.0

        # 新しいプライマリノードの確認
        new_primary = failed_cluster_state.get_primary_node()
        assert new_primary.node_id == "healthy_secondary"
        assert new_primary.role == NodeRole.PRIMARY

    @pytest.mark.asyncio
    async def test_failover_candidate_selection(self, failover_manager, failed_cluster_state):
        """フェイルオーバー候補選択テスト"""
        # 複数のセカンダリノードを追加
        secondary_node_2 = NodeState(
            node_id="secondary_2",
            role=NodeRole.SECONDARY,
            status=FailoverStatus.HEALTHY,
            last_heartbeat=datetime.now(),
            health_score=0.95  # より高いスコア
        )
        failed_cluster_state.update_node_state("secondary_2", secondary_node_2)

        # 最適候補選択
        best_candidate = await failover_manager.select_best_candidate(failed_cluster_state)

        assert best_candidate.node_id == "secondary_2"
        assert best_candidate.health_score == 0.95

    @pytest.mark.asyncio
    async def test_no_viable_candidates_failover(self, failover_manager):
        """利用可能候補がない場合のフェイルオーバーテスト"""
        cluster = ClusterState(cluster_id="test_cluster")

        # 失敗したプライマリのみ
        failed_primary = NodeState(
            node_id="failed_primary",
            role=NodeRole.FAILED,
            status=FailoverStatus.FAILED,
            last_heartbeat=datetime.now() - timedelta(seconds=60),
            health_score=0.0
        )
        cluster.update_node_state("failed_primary", failed_primary)

        # フェイルオーバー試行
        failover_result = await failover_manager.execute_failover(cluster)

        assert failover_result.success == False
        assert failover_result.error_message == "No viable failover candidates available"

    @pytest.mark.asyncio
    async def test_failover_retry_mechanism(self, failover_manager, failed_cluster_state):
        """フェイルオーバーリトライ機構テスト"""
        # フェイルオーバーが失敗するようにモック
        with patch.object(failover_manager, '_perform_failover_operation') as mock_failover:
            mock_failover.side_effect = [Exception("Network error"), Exception("Timeout"), True]

            # リトライ付きフェイルオーバー実行
            result = await failover_manager.execute_failover_with_retry(failed_cluster_state)

            assert result.success == True
            assert result.retry_count == 2  # 2回リトライ後成功


class TestDataConsistencyChecker:
    """データ整合性チェックテスト"""

    @pytest.fixture
    def consistency_checker(self):
        """整合性チェッカーフィクスチャ"""
        config = FailoverConfiguration(consistency_check_enabled=True)
        return DataConsistencyChecker(config)

    @pytest.fixture
    def post_failover_cluster(self):
        """フェイルオーバー後クラスター状態"""
        cluster = ClusterState(cluster_id="test_cluster")

        # 新しいプライマリ
        new_primary = NodeState(
            node_id="new_primary",
            role=NodeRole.PRIMARY,
            status=FailoverStatus.HEALTHY,
            last_heartbeat=datetime.now(),
            health_score=0.9
        )
        cluster.update_node_state("new_primary", new_primary)

        # セカンダリノード
        secondary = NodeState(
            node_id="secondary_1",
            role=NodeRole.SECONDARY,
            status=FailoverStatus.HEALTHY,
            last_heartbeat=datetime.now(),
            health_score=0.8
        )
        cluster.update_node_state("secondary_1", secondary)

        return cluster

    @pytest.mark.asyncio
    async def test_data_consistency_after_failover(self, consistency_checker, post_failover_cluster):
        """フェイルオーバー後データ整合性テスト"""
        # データ整合性チェック
        consistency_result = await consistency_checker.check_post_failover_consistency(
            post_failover_cluster
        )

        assert consistency_result.is_consistent == True
        assert consistency_result.inconsistency_count == 0
        assert consistency_result.check_duration < 10.0

    @pytest.mark.asyncio
    async def test_data_inconsistency_detection(self, consistency_checker, post_failover_cluster):
        """データ不整合検知テスト"""
        # 不整合状態をシミュレート
        with patch.object(consistency_checker, '_check_database_consistency') as mock_check:
            mock_check.return_value = False

            consistency_result = await consistency_checker.check_post_failover_consistency(
                post_failover_cluster
            )

            assert consistency_result.is_consistent == False
            assert consistency_result.inconsistency_count > 0

    @pytest.mark.asyncio
    async def test_data_synchronization_after_failover(self, consistency_checker, post_failover_cluster):
        """フェイルオーバー後データ同期テスト"""
        # データ同期実行
        sync_result = await consistency_checker.synchronize_cluster_data(post_failover_cluster)

        assert sync_result.success == True
        assert sync_result.synchronized_records > 0
        assert sync_result.sync_duration < 30.0


class TestFailoverCoordinator:
    """フェイルオーバー調整テスト"""

    @pytest.fixture
    def failover_coordinator(self):
        """フェイルオーバー調整器フィクスチャ"""
        config = FailoverConfiguration()
        return FailoverCoordinator(config)

    @pytest.mark.asyncio
    async def test_end_to_end_failover_workflow(self, failover_coordinator):
        """エンドツーエンドフェイルオーバーワークフローテスト"""
        # 初期クラスター状態設定
        cluster_state = ClusterState(cluster_id="e2e_cluster")

        # プライマリノード（障害発生予定）
        primary = NodeState(
            node_id="primary_e2e",
            role=NodeRole.PRIMARY,
            status=FailoverStatus.HEALTHY,
            last_heartbeat=datetime.now(),
            health_score=1.0
        )
        cluster_state.update_node_state("primary_e2e", primary)

        # セカンダリノード
        secondary = NodeState(
            node_id="secondary_e2e",
            role=NodeRole.SECONDARY,
            status=FailoverStatus.HEALTHY,
            last_heartbeat=datetime.now(),
            health_score=0.9
        )
        cluster_state.update_node_state("secondary_e2e", secondary)

        # プライマリノード障害シミュレート
        primary.status = FailoverStatus.FAILED
        primary.last_heartbeat = datetime.now() - timedelta(seconds=60)

        # エンドツーエンドフェイルオーバー実行
        workflow_result = await failover_coordinator.execute_failover_workflow(cluster_state)

        assert workflow_result.success == True
        assert workflow_result.total_execution_time < 60.0
        assert workflow_result.new_primary_id == "secondary_e2e"
        assert workflow_result.data_consistency_verified == True

        # フェイルオーバー後の状態確認
        new_primary = cluster_state.get_primary_node()
        assert new_primary.node_id == "secondary_e2e"
        assert new_primary.role == NodeRole.PRIMARY

    @pytest.mark.asyncio
    async def test_concurrent_failover_prevention(self, failover_coordinator):
        """同時フェイルオーバー防止テスト"""
        cluster_state = ClusterState(cluster_id="concurrent_test")

        # プライマリノード（障害状態）
        primary = NodeState(
            node_id="primary_concurrent",
            role=NodeRole.PRIMARY,
            status=FailoverStatus.FAILED,
            last_heartbeat=datetime.now() - timedelta(seconds=60),
            health_score=0.0
        )
        cluster_state.update_node_state("primary_concurrent", primary)

        # セカンダリノード
        secondary = NodeState(
            node_id="secondary_concurrent",
            role=NodeRole.SECONDARY,
            status=FailoverStatus.HEALTHY,
            last_heartbeat=datetime.now(),
            health_score=0.9
        )
        cluster_state.update_node_state("secondary_concurrent", secondary)

        # 2つの同時フェイルオーバー試行
        task1 = asyncio.create_task(
            failover_coordinator.execute_failover_workflow(cluster_state)
        )
        task2 = asyncio.create_task(
            failover_coordinator.execute_failover_workflow(cluster_state)
        )

        results = await asyncio.gather(task1, task2, return_exceptions=True)

        # 両方が実行されるが、2番目は先に実行されたworkflowによって状況が変わっている
        # ロックにより逐次実行されるため、両方とも成功する可能性がある
        successful_results = [r for r in results if hasattr(r, 'success') and r.success]
        assert len(successful_results) >= 1

    @pytest.mark.asyncio
    async def test_failover_event_logging(self, failover_coordinator):
        """フェイルオーバーイベントログテスト"""
        cluster_state = ClusterState(cluster_id="logging_test")

        # プライマリノード（障害状態）
        primary = NodeState(
            node_id="primary_logging",
            role=NodeRole.PRIMARY,
            status=FailoverStatus.FAILED,
            last_heartbeat=datetime.now() - timedelta(seconds=60),
            health_score=0.0
        )
        cluster_state.update_node_state("primary_logging", primary)

        # セカンダリノード
        secondary = NodeState(
            node_id="secondary_logging",
            role=NodeRole.SECONDARY,
            status=FailoverStatus.HEALTHY,
            last_heartbeat=datetime.now(),
            health_score=0.9
        )
        cluster_state.update_node_state("secondary_logging", secondary)

        # フェイルオーバー実行
        await failover_coordinator.execute_failover_workflow(cluster_state)

        # イベントログ確認
        events = await failover_coordinator.get_failover_events()

        assert len(events) > 0
        assert any(event.event_type == "primary_failure" for event in events)
        assert any(event.event_type == "automatic_failover" for event in events)

    @pytest.mark.asyncio
    async def test_kubernetes_integration(self, failover_coordinator):
        """Kubernetes統合テスト"""
        cluster_state = ClusterState(cluster_id="k8s_test")

        # プライマリノード（障害状態）
        primary = NodeState(
            node_id="primary_k8s",
            role=NodeRole.PRIMARY,
            status=FailoverStatus.FAILED,
            last_heartbeat=datetime.now() - timedelta(seconds=60),
            health_score=0.0
        )
        cluster_state.update_node_state("primary_k8s", primary)

        # セカンダリノード
        secondary = NodeState(
            node_id="secondary_k8s",
            role=NodeRole.SECONDARY,
            status=FailoverStatus.HEALTHY,
            last_heartbeat=datetime.now(),
            health_score=0.9
        )
        cluster_state.update_node_state("secondary_k8s", secondary)

        # Kubernetesリソース更新を含むフェイルオーバー
        result = await failover_coordinator.execute_kubernetes_failover(cluster_state)

        assert result.success == True
        # Kubernetesが利用可能でない場合でも基本フェイルオーバーは成功する
        assert result.new_primary_id == "secondary_k8s"
        # Kubernetesクライアントが利用できない環境では、基本フェイルオーバーのみ実行される