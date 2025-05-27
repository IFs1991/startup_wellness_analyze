# Phase 3 Task 3.3: モデルバージョニングシステムの実装
# TDD RED段階: 失敗するテストから開始

import pytest
import pytest_asyncio
import asyncio
import json
import hashlib
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Optional, List, Dict, Any, Union
import uuid

# テスト対象となるクラス（まだ存在しない）
from ..versioning.model_version_manager import ModelVersionManager
from ..versioning.version_storage import VersionStorage
from ..versioning.model_registry import ModelRegistry
from ..versioning.version_comparator import VersionComparator
from ..versioning.rollback_manager import RollbackManager
from ..versioning.models import ModelVersion, VersionMetadata, ModelArtifact


# 共有フィクスチャ
@pytest_asyncio.fixture
async def mock_version_storage():
    """モック化されたVersionStorageのフィクスチャ"""
    storage = Mock(spec=VersionStorage)

    # 非同期メソッドの設定
    storage.store_model_artifact = AsyncMock(return_value="artifact_id_123")
    storage.load_model_artifact = AsyncMock(return_value={"model_data": "test"})
    storage.delete_model_artifact = AsyncMock(return_value=True)
    storage.list_artifacts = AsyncMock(return_value=["artifact_1", "artifact_2"])
    storage.get_artifact_info = AsyncMock(return_value={
        "size": 1024,
        "checksum": "abc123",
        "created_at": datetime.now(timezone.utc).isoformat()
    })
    storage.artifact_exists = AsyncMock(return_value=True)

    return storage


@pytest_asyncio.fixture
async def mock_model_registry():
    """モック化されたModelRegistryのフィクスチャ"""
    registry = Mock(spec=ModelRegistry)

    # 非同期メソッドの設定
    registry.register_version = AsyncMock(return_value="version_id_123")
    registry.get_version = AsyncMock(return_value=None)
    registry.list_versions = AsyncMock(return_value=[])
    registry.get_latest_version = AsyncMock(return_value=None)
    registry.update_version_status = AsyncMock(return_value=True)
    registry.delete_version = AsyncMock(return_value=True)
    registry.search_versions = AsyncMock(return_value=[])

    return registry


@pytest_asyncio.fixture
async def model_version_manager(mock_version_storage, mock_model_registry):
    """ModelVersionManagerのフィクスチャ"""
    manager = ModelVersionManager(
        storage=mock_version_storage,
        registry=mock_model_registry
    )
    return manager


@pytest_asyncio.fixture
async def version_comparator():
    """VersionComparatorのフィクスチャ"""
    return VersionComparator()


@pytest_asyncio.fixture
async def rollback_manager(mock_version_storage, mock_model_registry):
    """RollbackManagerのフィクスチャ"""
    return RollbackManager(
        storage=mock_version_storage,
        registry=mock_model_registry
    )


@pytest_asyncio.fixture
async def sample_model_data():
    """サンプルモデルデータ"""
    return {
        "model_architecture": {
            "layers": [
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dense", "units": 10, "activation": "softmax"}
            ]
        },
        "weights": [0.1, 0.2, 0.3, 0.4, 0.5],  # 簡略化された重み
        "optimizer_config": {
            "type": "adam",
            "learning_rate": 0.001,
            "beta_1": 0.9,
            "beta_2": 0.999
        }
    }


@pytest_asyncio.fixture
async def sample_version_metadata():
    """サンプルバージョンメタデータ"""
    return VersionMetadata(
        model_name="federated_mnist_classifier",
        version="1.0.0",
        created_at=datetime.now(timezone.utc),
        creator="test_user",
        description="Initial federated learning model",
        tags=["federated", "mnist", "baseline"],
        metrics={
            "accuracy": 0.92,
            "loss": 0.15,
            "num_clients": 10,
            "num_rounds": 100
        },
        environment_info={
            "python_version": "3.9.0",
            "tensorflow_version": "2.8.0",
            "federated_learning_version": "1.0.0"
        }
    )


# ModelVersionManagerのテスト
class TestModelVersionManager:
    """ModelVersionManagerクラスのテスト"""

    async def test_create_new_version(self, model_version_manager, sample_model_data, sample_version_metadata):
        """新しいモデルバージョンの作成テスト"""
        # モックの設定
        model_version_manager.storage.store_model_artifact.return_value = "artifact_123"
        model_version_manager.registry.register_version.return_value = "version_123"

        # バージョン作成
        version_id = await model_version_manager.create_version(
            model_data=sample_model_data,
            metadata=sample_version_metadata
        )

        assert version_id == "version_123"
        model_version_manager.storage.store_model_artifact.assert_called_once()
        model_version_manager.registry.register_version.assert_called_once()

    async def test_get_version_details(self, model_version_manager):
        """バージョン詳細取得テスト"""
        # モックレスポンスの設定
        mock_version = ModelVersion(
            id="version_123",
            model_name="test_model",
            version="1.0.0",
            artifact_id="artifact_123",
            metadata=VersionMetadata(
                model_name="test_model",
                version="1.0.0",
                created_at=datetime.now(timezone.utc),
                creator="test_user"
            ),
            status="active"
        )
        model_version_manager.registry.get_version.return_value = mock_version

        # バージョン取得
        version = await model_version_manager.get_version("version_123")

        assert version is not None
        assert version.id == "version_123"
        assert version.model_name == "test_model"
        assert version.version == "1.0.0"

    async def test_list_model_versions(self, model_version_manager):
        """モデルバージョン一覧取得テスト"""
        # モックバージョンリスト
        mock_versions = [
            ModelVersion(id="v1", model_name="test_model", version="1.0.0", artifact_id="a1"),
            ModelVersion(id="v2", model_name="test_model", version="1.1.0", artifact_id="a2"),
            ModelVersion(id="v3", model_name="test_model", version="2.0.0", artifact_id="a3")
        ]
        model_version_manager.registry.list_versions.return_value = mock_versions

        # バージョンリスト取得
        versions = await model_version_manager.list_versions("test_model")

        assert len(versions) == 3
        assert versions[0].version == "1.0.0"
        assert versions[1].version == "1.1.0"
        assert versions[2].version == "2.0.0"

    async def test_get_latest_version(self, model_version_manager):
        """最新バージョン取得テスト"""
        mock_latest = ModelVersion(
            id="latest_version",
            model_name="test_model",
            version="2.1.0",
            artifact_id="latest_artifact"
        )
        model_version_manager.registry.get_latest_version.return_value = mock_latest

        latest = await model_version_manager.get_latest_version("test_model")

        assert latest is not None
        assert latest.version == "2.1.0"
        assert latest.id == "latest_version"

    async def test_update_version_status(self, model_version_manager):
        """バージョンステータス更新テスト"""
        model_version_manager.registry.update_version_status.return_value = True

        success = await model_version_manager.update_version_status(
            "version_123",
            "deprecated",
            reason="Replaced by newer version"
        )

        assert success is True
        model_version_manager.registry.update_version_status.assert_called_once_with(
            "version_123", "deprecated", "Replaced by newer version"
        )

    async def test_delete_version(self, model_version_manager):
        """バージョン削除テスト"""
        # モック設定
        model_version_manager.registry.get_version.return_value = ModelVersion(
            id="version_123",
            artifact_id="artifact_123",
            model_name="test_model",
            version="1.0.0"
        )
        model_version_manager.storage.delete_model_artifact.return_value = True
        model_version_manager.registry.delete_version.return_value = True

        success = await model_version_manager.delete_version("version_123")

        assert success is True
        model_version_manager.storage.delete_model_artifact.assert_called_once_with("artifact_123")
        model_version_manager.registry.delete_version.assert_called_once_with("version_123")


# バージョンワークフローのテスト
class TestModelVersioningWorkflow:
    """モデルバージョニングワークフローのテスト"""

    async def test_complete_versioning_workflow(self, model_version_manager, sample_model_data, sample_version_metadata):
        """完全なバージョニングワークフローテスト"""
        # 1. 新しいバージョンを作成
        model_version_manager.storage.store_model_artifact.return_value = "artifact_v1"
        model_version_manager.registry.register_version.return_value = "version_v1"

        version_id = await model_version_manager.create_version(
            model_data=sample_model_data,
            metadata=sample_version_metadata
        )
        assert version_id == "version_v1"

        # 2. バージョンを取得
        mock_version = ModelVersion(
            id="version_v1",
            model_name="federated_mnist_classifier",
            version="1.0.0",
            artifact_id="artifact_v1",
            metadata=sample_version_metadata
        )
        model_version_manager.registry.get_version.return_value = mock_version

        retrieved_version = await model_version_manager.get_version("version_v1")
        assert retrieved_version.id == "version_v1"

        # 3. ステータスを更新
        model_version_manager.registry.update_version_status.return_value = True
        success = await model_version_manager.update_version_status("version_v1", "active")
        assert success is True

    async def test_version_promotion_workflow(self, model_version_manager):
        """バージョン昇格ワークフローテスト"""
        # ステージング環境でのテスト
        model_version_manager.registry.promote_version.return_value = True

        # ステージングに昇格
        success = await model_version_manager.promote_version(
            "version_123",
            from_env="development",
            to_env="staging"
        )
        assert success is True

        # プロダクションに昇格
        success = await model_version_manager.promote_version(
            "version_123",
            from_env="staging",
            to_env="production"
        )
        assert success is True

    async def test_version_tagging_workflow(self, model_version_manager):
        """バージョンタギングワークフローテスト"""
        model_version_manager.registry.add_tags.return_value = True
        model_version_manager.registry.remove_tags.return_value = True

        # タグ追加
        success = await model_version_manager.add_tags("version_123", ["stable", "tested"])
        assert success is True

        # タグ削除
        success = await model_version_manager.remove_tags("version_123", ["experimental"])
        assert success is True


# VersionComparatorのテスト
class TestVersionComparator:
    """VersionComparatorクラスのテスト"""

    async def test_compare_model_architectures(self, version_comparator):
        """モデルアーキテクチャ比較テスト"""
        arch1 = {
            "layers": [
                {"type": "dense", "units": 128},
                {"type": "dense", "units": 64}
            ]
        }
        arch2 = {
            "layers": [
                {"type": "dense", "units": 128},
                {"type": "dense", "units": 32}  # 異なるユニット数
            ]
        }

        comparison = await version_comparator.compare_architectures(arch1, arch2)

        assert comparison["is_identical"] is False
        assert "layer_differences" in comparison
        assert len(comparison["layer_differences"]) > 0

    async def test_compare_model_metrics(self, version_comparator):
        """モデルメトリクス比較テスト"""
        metrics1 = {"accuracy": 0.92, "loss": 0.15, "f1_score": 0.89}
        metrics2 = {"accuracy": 0.94, "loss": 0.12, "f1_score": 0.91}

        comparison = await version_comparator.compare_metrics(metrics1, metrics2)

        assert comparison["accuracy"]["improvement"] > 0
        assert comparison["loss"]["improvement"] > 0  # 損失は低い方が良い
        assert comparison["f1_score"]["improvement"] > 0

    async def test_compare_version_metadata(self, version_comparator):
        """バージョンメタデータ比較テスト"""
        metadata1 = VersionMetadata(
            model_name="test_model",
            version="1.0.0",
            created_at=datetime.now(timezone.utc),
            metrics={"accuracy": 0.92}
        )
        metadata2 = VersionMetadata(
            model_name="test_model",
            version="1.1.0",
            created_at=datetime.now(timezone.utc),
            metrics={"accuracy": 0.94}
        )

        comparison = await version_comparator.compare_versions(metadata1, metadata2)

        assert comparison["version_change"]["is_upgrade"] is True
        assert comparison["metrics_comparison"]["accuracy"]["improvement"] > 0

    async def test_semantic_version_comparison(self, version_comparator):
        """セマンティックバージョン比較テスト"""
        # メジャーバージョンアップ
        comparison = await version_comparator.compare_semantic_versions("1.0.0", "2.0.0")
        assert comparison["change_type"] == "major"
        assert comparison["is_upgrade"] is True

        # マイナーバージョンアップ
        comparison = await version_comparator.compare_semantic_versions("1.0.0", "1.1.0")
        assert comparison["change_type"] == "minor"

        # パッチバージョンアップ
        comparison = await version_comparator.compare_semantic_versions("1.0.0", "1.0.1")
        assert comparison["change_type"] == "patch"


# RollbackManagerのテスト
class TestRollbackManager:
    """RollbackManagerクラスのテスト"""

    async def test_create_rollback_point(self, rollback_manager):
        """ロールバックポイント作成テスト"""
        # モックバージョンを設定
        mock_version = ModelVersion(
            id="version_123",
            model_name="test_model",
            version="2.0.0",
            artifact_id="artifact_123"
        )
        rollback_manager.registry.list_versions.return_value = [mock_version]
        rollback_manager.registry.create_rollback_point.return_value = "rollback_123"

        rollback_id = await rollback_manager.create_rollback_point(
            model_name="test_model",
            current_version="2.0.0",
            reason="Before major update"
        )

        assert rollback_id == "rollback_123"
        rollback_manager.registry.create_rollback_point.assert_called_once()

    async def test_rollback_to_version(self, rollback_manager):
        """指定バージョンへのロールバックテスト"""
        # モック設定
        mock_version = ModelVersion(
            id="version_123",
            model_name="test_model",
            version="1.0.0",
            artifact_id="artifact_123"
        )
        rollback_manager.registry.list_versions.return_value = [mock_version]
        rollback_manager.storage.load_model_artifact.return_value = {"model": "data"}
        rollback_manager.registry.update_version_status.return_value = True

        success = await rollback_manager.rollback_to_version(
            model_name="test_model",
            target_version="1.0.0",
            reason="Performance regression in v2.0.0"
        )

        assert success is True
        rollback_manager.storage.load_model_artifact.assert_called_once_with("artifact_123")

    async def test_rollback_validation(self, rollback_manager):
        """ロールバック妥当性検証テスト"""
        mock_version = ModelVersion(
            id="version_123",
            model_name="test_model",
            version="1.0.0",
            artifact_id="artifact_123",
            status="active"
        )
        rollback_manager.registry.list_versions.return_value = [mock_version]

        is_valid = await rollback_manager.validate_rollback(
            model_name="test_model",
            target_version="1.0.0"
        )

        assert is_valid is True

        # 存在しないバージョンへのロールバック
        rollback_manager.registry.list_versions.return_value = []
        is_valid = await rollback_manager.validate_rollback(
            model_name="test_model",
            target_version="999.0.0"
        )

        assert is_valid is False

    async def test_rollback_history_tracking(self, rollback_manager):
        """ロールバック履歴追跡テスト"""
        mock_history = [
            {
                "rollback_id": "rb_1",
                "from_version": "2.0.0",
                "to_version": "1.0.0",
                "timestamp": datetime.now(timezone.utc),
                "reason": "Performance issue"
            }
        ]
        rollback_manager.registry.get_rollback_history.return_value = mock_history

        history = await rollback_manager.get_rollback_history("test_model")

        assert len(history) == 1
        assert history[0]["from_version"] == "2.0.0"
        assert history[0]["to_version"] == "1.0.0"

    async def test_automatic_rollback_trigger(self, rollback_manager):
        """自動ロールバックトリガーテスト"""
        # パフォーマンス劣化による自動ロールバック
        latest_metadata = VersionMetadata(
            model_name="test_model",
            version="2.0.0",
            metrics={"accuracy": 0.85}
        )
        stable_metadata = VersionMetadata(
            model_name="test_model",
            version="1.0.0",
            metrics={"accuracy": 0.92}
        )

        latest_version = ModelVersion(
            id="latest_version",
            model_name="test_model",
            version="2.0.0",
            artifact_id="latest_artifact",
            metadata=latest_metadata
        )
        stable_version = ModelVersion(
            id="stable_version",
            model_name="test_model",
            version="1.0.0",
            artifact_id="stable_artifact",
            metadata=stable_metadata
        )

        rollback_manager.registry.get_latest_version.return_value = latest_version
        rollback_manager.registry.list_versions.return_value = [stable_version, latest_version]

        should_rollback = await rollback_manager.should_auto_rollback(
            model_name="test_model",
            current_metrics={"accuracy": 0.85},
            threshold_degradation=0.05
        )

        assert should_rollback is True


# 統合テスト
class TestVersioningIntegration:
    """モデルバージョニング統合テスト"""

    async def test_end_to_end_versioning_workflow(self, model_version_manager, sample_model_data, sample_version_metadata):
        """エンドツーエンドバージョニングワークフローテスト"""
        # 完全なライフサイクルをテスト

        # 1. 初期バージョン作成
        model_version_manager.storage.store_model_artifact.return_value = "artifact_v1"
        model_version_manager.registry.register_version.return_value = "version_v1"

        v1_id = await model_version_manager.create_version(
            model_data=sample_model_data,
            metadata=sample_version_metadata
        )

        # 2. 改良版モデル作成
        improved_data = sample_model_data.copy()
        improved_data["optimizer_config"]["learning_rate"] = 0.0005  # 改良

        improved_metadata = sample_version_metadata
        improved_metadata.version = "1.1.0"
        improved_metadata.metrics["accuracy"] = 0.95  # 改善

        model_version_manager.storage.store_model_artifact.return_value = "artifact_v1_1"
        model_version_manager.registry.register_version.return_value = "version_v1_1"

        v1_1_id = await model_version_manager.create_version(
            model_data=improved_data,
            metadata=improved_metadata
        )

        # 3. バージョン比較
        comparator = VersionComparator()
        # 異なるメタデータオブジェクトを作成
        original_metadata = VersionMetadata(
            model_name="federated_mnist_classifier",
            version="1.0.0",
            created_at=datetime.now(timezone.utc),
            creator="test_user",
            description="Initial federated learning model",
            tags=["federated", "mnist", "baseline"],
            metrics={"accuracy": 0.92}
        )
        improved_metadata_obj = VersionMetadata(
            model_name="federated_mnist_classifier",
            version="1.1.0",
            created_at=datetime.now(timezone.utc),
            creator="test_user",
            description="Improved federated learning model",
            tags=["federated", "mnist", "improved"],
            metrics={"accuracy": 0.95}
        )
        comparison = await comparator.compare_versions(original_metadata, improved_metadata_obj)

        assert comparison["metrics_comparison"]["accuracy"]["improvement"] > 0

        # 4. 問題があった場合のロールバック
        rollback_manager = RollbackManager(
            storage=model_version_manager.storage,
            registry=model_version_manager.registry
        )

        rollback_version = ModelVersion(
            id=v1_id,
            version="1.0.0",
            model_name="federated_mnist_classifier",
            artifact_id="artifact_v1"
        )
        rollback_manager.registry.list_versions.return_value = [rollback_version]
        rollback_manager.storage.load_model_artifact.return_value = sample_model_data

        success = await rollback_manager.rollback_to_version(
            model_name="federated_mnist_classifier",
            target_version="1.0.0",
            reason="Issue found in v1.1.0"
        )

        assert success is True

    async def test_multi_model_versioning(self, model_version_manager):
        """複数モデルバージョニングテスト"""
        # 複数のモデルを同時に管理
        models = ["mnist_classifier", "cifar_classifier", "nlp_model"]

        for model_name in models:
            model_version_manager.registry.list_versions.return_value = [
                ModelVersion(id=f"{model_name}_v1", model_name=model_name, version="1.0.0", artifact_id=f"{model_name}_artifact_1"),
                ModelVersion(id=f"{model_name}_v2", model_name=model_name, version="1.1.0", artifact_id=f"{model_name}_artifact_2")
            ]

            versions = await model_version_manager.list_versions(model_name)
            assert len(versions) == 2
            assert all(v.model_name == model_name for v in versions)

    async def test_concurrent_version_operations(self, model_version_manager):
        """並行バージョン操作テスト"""
        # 複数の操作を並行実行
        model_version_manager.registry.get_version.return_value = ModelVersion(
            id="test_version",
            model_name="test_model",
            version="1.0.0",
            artifact_id="test_artifact"
        )

        operations = [
            model_version_manager.get_version("version_1"),
            model_version_manager.get_version("version_2"),
            model_version_manager.get_version("version_3")
        ]

        results = await asyncio.gather(*operations)
        assert len(results) == 3
        assert all(result is not None for result in results)