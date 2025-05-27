# Phase 3 Task 3.3: モデルバージョニングシステムの実装
# TDD GREEN段階: RollbackManager実装

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

import structlog
from .version_storage import VersionStorage
from .model_registry import ModelRegistry
from .models import ModelVersion, VersionMetadata

logger = structlog.get_logger(__name__)


class RollbackManager:
    """
    ロールバック管理クラス

    モデルバージョンのロールバック操作を管理
    """

    def __init__(self, storage: VersionStorage, registry: ModelRegistry):
        """
        RollbackManagerの初期化

        Args:
            storage: バージョンストレージ
            registry: モデルレジストリ
        """
        self.storage = storage
        self.registry = registry

    async def create_rollback_point(
        self,
        model_name: str,
        current_version: str,
        reason: str = "",
        creator: str = "system"
    ) -> str:
        """
        ロールバックポイントを作成

        Args:
            model_name: モデル名
            current_version: 現在のバージョン
            reason: ロールバックポイント作成理由
            creator: 作成者

        Returns:
            ロールバックポイントID
        """
        try:
            # 現在のバージョンを取得
            versions = await self.registry.list_versions(model_name)
            current_version_obj = None

            for version in versions:
                if version.version == current_version:
                    current_version_obj = version
                    break

            if not current_version_obj:
                raise ValueError(f"Version {current_version} not found for model {model_name}")

            # ロールバックポイント作成
            rollback_id = await self.registry.create_rollback_point(
                model_name=model_name,
                version_id=current_version_obj.id,
                reason=reason,
                creator=creator
            )

            logger.info(f"Rollback point created: {rollback_id} for {model_name} v{current_version}")
            return rollback_id

        except Exception as e:
            logger.error(f"Failed to create rollback point: {e}")
            raise

    async def rollback_to_version(
        self,
        model_name: str,
        target_version: str,
        reason: str = ""
    ) -> bool:
        """
        指定バージョンにロールバック

        Args:
            model_name: モデル名
            target_version: ターゲットバージョン
            reason: ロールバック理由

        Returns:
            ロールバック成功フラグ
        """
        try:
            # 妥当性検証
            is_valid = await self.validate_rollback(model_name, target_version)
            if not is_valid:
                raise ValueError(f"Invalid rollback to version {target_version}")

            # ターゲットバージョンを取得
            versions = await self.registry.list_versions(model_name)
            target_version_obj = None

            for version in versions:
                if version.version == target_version:
                    target_version_obj = version
                    break

            if not target_version_obj:
                raise ValueError(f"Target version {target_version} not found")

            # アーティファクトを読み込み
            model_data = await self.storage.load_model_artifact(target_version_obj.artifact_id)

            # ステータス更新
            await self.registry.update_version_status(
                target_version_obj.id,
                "active",
                f"Rolled back: {reason}"
            )

            logger.info(f"Rollback completed: {model_name} to v{target_version}")
            return True

        except Exception as e:
            logger.error(f"Failed to rollback to version {target_version}: {e}")
            return False

    async def validate_rollback(
        self,
        model_name: str,
        target_version: str
    ) -> bool:
        """
        ロールバック妥当性を検証

        Args:
            model_name: モデル名
            target_version: ターゲットバージョン

        Returns:
            妥当性フラグ
        """
        try:
            # バージョン存在確認
            versions = await self.registry.list_versions(model_name)
            target_exists = any(v.version == target_version for v in versions)

            if not target_exists:
                logger.warning(f"Target version {target_version} does not exist")
                return False

            # ターゲットバージョンのステータス確認
            target_version_obj = None
            for version in versions:
                if version.version == target_version:
                    target_version_obj = version
                    break

            if target_version_obj and target_version_obj.status.value in ["archived", "failed"]:
                logger.warning(f"Cannot rollback to {target_version_obj.status.value} version")
                return False

            return True

        except Exception as e:
            logger.error(f"Rollback validation failed: {e}")
            return False

    async def get_rollback_history(self, model_name: str) -> List[Dict[str, Any]]:
        """
        ロールバック履歴を取得

        Args:
            model_name: モデル名

        Returns:
            ロールバック履歴リスト
        """
        try:
            return await self.registry.get_rollback_history(model_name)
        except Exception as e:
            logger.error(f"Failed to get rollback history: {e}")
            return []

    async def should_auto_rollback(
        self,
        model_name: str,
        current_metrics: Dict[str, Any],
        threshold_degradation: float = 0.05
    ) -> bool:
        """
        自動ロールバックが必要かを判定

        Args:
            model_name: モデル名
            current_metrics: 現在のメトリクス
            threshold_degradation: 劣化閾値

        Returns:
            自動ロールバック要否
        """
        try:
            # 最新バージョンと前の安定バージョンを取得（モック実装）
            latest_version = await self.registry.get_latest_version(model_name)
            if not latest_version or not latest_version.metadata:
                return False

            # 簡略実装：前の安定バージョンを想定
            versions = await self.registry.list_versions(model_name)
            if len(versions) < 2:
                return False

            previous_stable = versions[-2]  # 簡略実装
            if not previous_stable.metadata:
                return False

            # メトリクス比較
            if "accuracy" in current_metrics and "accuracy" in previous_stable.metadata.metrics:
                current_accuracy = current_metrics["accuracy"]
                stable_accuracy = previous_stable.metadata.metrics["accuracy"]

                degradation = stable_accuracy - current_accuracy
                if degradation > threshold_degradation:
                    logger.warning(f"Performance degradation detected: {degradation:.3f}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Auto-rollback check failed: {e}")
            return False