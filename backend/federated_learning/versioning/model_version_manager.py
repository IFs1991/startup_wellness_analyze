# Phase 3 Task 3.3: モデルバージョニングシステムの実装
# TDD REFACTOR段階: ModelVersionManager機能拡張

import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timezone

import structlog
from .version_storage import VersionStorage
from .model_registry import ModelRegistry
from .version_comparator import VersionComparator
from .rollback_manager import RollbackManager
from .models import ModelVersion, VersionMetadata, generate_version_id

logger = structlog.get_logger(__name__)


class ModelVersionManager:
    """
    モデルバージョンマネージャークラス

    モデルバージョニングシステムの中心的なコンポーネント
    """

    def __init__(
        self,
        storage: VersionStorage,
        registry: ModelRegistry,
        comparator: Optional[VersionComparator] = None,
        rollback_manager: Optional[RollbackManager] = None
    ):
        """
        ModelVersionManagerの初期化

        Args:
            storage: バージョンストレージ
            registry: モデルレジストリ
            comparator: バージョン比較器
            rollback_manager: ロールバックマネージャー
        """
        self.storage = storage
        self.registry = registry
        self.comparator = comparator or VersionComparator()
        self.rollback_manager = rollback_manager or RollbackManager(storage, registry)

    async def create_version(
        self,
        model_data: Any,
        metadata: VersionMetadata,
        parent_version_id: Optional[str] = None
    ) -> str:
        """
        新しいモデルバージョンを作成

        Args:
            model_data: モデルデータ
            metadata: バージョンメタデータ
            parent_version_id: 親バージョンID

        Returns:
            作成されたバージョンID
        """
        try:
            # アーティファクトを保存
            artifact_id = await self.storage.store_model_artifact(
                model_data=model_data,
                metadata=metadata.to_dict()
            )

            # レジストリに登録
            version_id = await self.registry.register_version(
                model_name=metadata.model_name,
                version=metadata.version,
                artifact_id=artifact_id,
                metadata=metadata,
                parent_version_id=parent_version_id
            )

            logger.info(f"Model version created: {metadata.model_name} v{metadata.version} ({version_id})")
            return version_id

        except Exception as e:
            logger.error(f"Failed to create version: {e}")
            raise

    async def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """
        バージョンを取得

        Args:
            version_id: バージョンID

        Returns:
            モデルバージョン
        """
        try:
            return await self.registry.get_version(version_id)
        except Exception as e:
            logger.error(f"Failed to get version {version_id}: {e}")
            return None

    async def list_versions(self, model_name: str) -> List[ModelVersion]:
        """
        モデルのバージョンリストを取得

        Args:
            model_name: モデル名

        Returns:
            バージョンリスト
        """
        try:
            return await self.registry.list_versions(model_name)
        except Exception as e:
            logger.error(f"Failed to list versions for {model_name}: {e}")
            return []

    async def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """
        最新バージョンを取得

        Args:
            model_name: モデル名

        Returns:
            最新バージョン
        """
        try:
            return await self.registry.get_latest_version(model_name)
        except Exception as e:
            logger.error(f"Failed to get latest version for {model_name}: {e}")
            return None

    async def update_version_status(
        self,
        version_id: str,
        status: str,
        reason: str = ""
    ) -> bool:
        """
        バージョンステータスを更新

        Args:
            version_id: バージョンID
            status: 新しいステータス
            reason: 更新理由

        Returns:
            更新成功フラグ
        """
        try:
            return await self.registry.update_version_status(version_id, status, reason)
        except Exception as e:
            logger.error(f"Failed to update version status: {e}")
            return False

    async def delete_version(self, version_id: str) -> bool:
        """
        バージョンを削除

        Args:
            version_id: バージョンID

        Returns:
            削除成功フラグ
        """
        try:
            # バージョン情報取得
            version = await self.registry.get_version(version_id)
            if not version:
                return False

            # アーティファクト削除
            artifact_deleted = await self.storage.delete_model_artifact(version.artifact_id)

            # レジストリから削除
            registry_deleted = await self.registry.delete_version(version_id)

            success = artifact_deleted and registry_deleted
            if success:
                logger.info(f"Version deleted: {version_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to delete version {version_id}: {e}")
            return False

    async def load_model_data(self, version_id: str) -> Optional[Any]:
        """
        モデルデータを読み込み

        Args:
            version_id: バージョンID

        Returns:
            モデルデータ
        """
        try:
            version = await self.registry.get_version(version_id)
            if not version:
                return None

            return await self.storage.load_model_artifact(version.artifact_id)

        except Exception as e:
            logger.error(f"Failed to load model data for {version_id}: {e}")
            return None

    async def compare_versions(
        self,
        version_id_a: str,
        version_id_b: str
    ) -> Optional[Dict[str, Any]]:
        """
        バージョンを比較

        Args:
            version_id_a: バージョンA ID
            version_id_b: バージョンB ID

        Returns:
            比較結果
        """
        try:
            version_a = await self.registry.get_version(version_id_a)
            version_b = await self.registry.get_version(version_id_b)

            if not version_a or not version_b:
                return None

            if not version_a.metadata or not version_b.metadata:
                return None

            return await self.comparator.compare_versions(
                version_a.metadata,
                version_b.metadata
            )

        except Exception as e:
            logger.error(f"Failed to compare versions: {e}")
            return None

    # プロモーション・ワークフロー関連
    async def promote_version(
        self,
        version_id: str,
        from_env: str,
        to_env: str
    ) -> bool:
        """
        バージョンを昇格

        Args:
            version_id: バージョンID
            from_env: 元環境
            to_env: 昇格先環境

        Returns:
            昇格成功フラグ
        """
        try:
            return await self.registry.promote_version(version_id, from_env, to_env)
        except Exception as e:
            logger.error(f"Failed to promote version: {e}")
            return False

    async def add_tags(self, version_id: str, tags: List[str]) -> bool:
        """タグを追加"""
        try:
            return await self.registry.add_tags(version_id, tags)
        except Exception as e:
            logger.error(f"Failed to add tags: {e}")
            return False

    async def remove_tags(self, version_id: str, tags: List[str]) -> bool:
        """タグを削除"""
        try:
            return await self.registry.remove_tags(version_id, tags)
        except Exception as e:
            logger.error(f"Failed to remove tags: {e}")
            return False

    # ロールバック機能
    async def create_rollback_point(
        self,
        model_name: str,
        current_version: str,
        reason: str = ""
    ) -> str:
        """ロールバックポイントを作成"""
        try:
            return await self.rollback_manager.create_rollback_point(
                model_name, current_version, reason
            )
        except Exception as e:
            logger.error(f"Failed to create rollback point: {e}")
            raise

    async def rollback_to_version(
        self,
        model_name: str,
        target_version: str,
        reason: str = ""
    ) -> bool:
        """指定バージョンにロールバック"""
        try:
            return await self.rollback_manager.rollback_to_version(
                model_name, target_version, reason
            )
        except Exception as e:
            logger.error(f"Failed to rollback: {e}")
            return False

    async def validate_rollback(
        self,
        model_name: str,
        target_version: str
    ) -> bool:
        """ロールバック妥当性を検証"""
        try:
            return await self.rollback_manager.validate_rollback(model_name, target_version)
        except Exception as e:
            logger.error(f"Failed to validate rollback: {e}")
            return False

    async def get_rollback_history(self, model_name: str) -> List[Dict[str, Any]]:
        """ロールバック履歴を取得"""
        try:
            return await self.rollback_manager.get_rollback_history(model_name)
        except Exception as e:
            logger.error(f"Failed to get rollback history: {e}")
            return []

    # 統計・管理機能
    async def get_storage_stats(self) -> Dict[str, Any]:
        """ストレージ統計を取得"""
        try:
            return await self.storage.get_storage_stats()
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}

    async def cleanup_temp_files(self) -> int:
        """一時ファイルをクリーンアップ"""
        try:
            return await self.storage.cleanup_temp_files()
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
            return 0

    async def search_versions(self, query: Dict[str, Any]) -> List[ModelVersion]:
        """バージョンを検索"""
        try:
            return await self.registry.search_versions(query)
        except Exception as e:
            logger.error(f"Failed to search versions: {e}")
            return []

    # REFACTOR段階で追加: 高度な機能

    async def bulk_create_versions(
        self,
        version_specs: List[Tuple[Any, VersionMetadata]]
    ) -> List[str]:
        """
        複数バージョンを一括作成

        Args:
            version_specs: (model_data, metadata)のタプルリスト

        Returns:
            作成されたバージョンIDのリスト
        """
        try:
            version_ids = []
            tasks = []

            for model_data, metadata in version_specs:
                task = self.create_version(model_data, metadata)
                tasks.append(task)

            version_ids = await asyncio.gather(*tasks, return_exceptions=True)

            # 例外をフィルタして成功したIDのみ返す
            successful_ids = [vid for vid in version_ids if isinstance(vid, str)]

            logger.info(f"Bulk created {len(successful_ids)}/{len(version_specs)} versions")
            return successful_ids

        except Exception as e:
            logger.error(f"Failed to bulk create versions: {e}")
            return []

    async def get_version_lineage(self, version_id: str) -> Dict[str, Any]:
        """
        バージョンの系譜を取得

        Args:
            version_id: バージョンID

        Returns:
            系譜情報
        """
        try:
            version = await self.registry.get_version(version_id)
            if not version:
                return {}

            lineage = {
                "current": version.to_dict(),
                "ancestors": [],
                "descendants": []
            }

            # 祖先を辿る
            current_parent_id = version.parent_version_id
            while current_parent_id:
                parent = await self.registry.get_version(current_parent_id)
                if parent:
                    lineage["ancestors"].append(parent.to_dict())
                    current_parent_id = parent.parent_version_id
                else:
                    break

            # 子孫を探す（簡略実装）
            all_versions = await self.registry.list_versions(version.model_name)
            for v in all_versions:
                if v.parent_version_id == version_id:
                    lineage["descendants"].append(v.to_dict())

            return lineage

        except Exception as e:
            logger.error(f"Failed to get version lineage: {e}")
            return {}

    async def auto_version_management(self, model_name: str) -> Dict[str, Any]:
        """
        自動バージョン管理

        Args:
            model_name: モデル名

        Returns:
            管理アクション結果
        """
        try:
            results = {
                "cleanup_count": 0,
                "archive_count": 0,
                "rollback_recommendations": []
            }

            # 一時ファイルのクリーンアップ
            results["cleanup_count"] = await self.cleanup_temp_files()

            # 古いバージョンのアーカイブ（簡略実装）
            versions = await self.registry.list_versions(model_name)
            old_versions = [v for v in versions if
                          (datetime.now(timezone.utc) - v.created_at).days > 90]

            for old_version in old_versions:
                if old_version.status.value == "active":
                    await self.registry.update_version_status(
                        old_version.id, "archived", "Auto-archived due to age"
                    )
                    results["archive_count"] += 1

            logger.info(f"Auto management completed for {model_name}: {results}")
            return results

        except Exception as e:
            logger.error(f"Failed auto version management: {e}")
            return {}

    async def generate_version_report(self, model_name: str) -> Dict[str, Any]:
        """
        バージョンレポートを生成

        Args:
            model_name: モデル名

        Returns:
            詳細レポート
        """
        try:
            versions = await self.registry.list_versions(model_name)
            storage_stats = await self.storage.get_storage_stats()

            report = {
                "model_name": model_name,
                "total_versions": len(versions),
                "status_breakdown": {},
                "metrics_evolution": [],
                "storage_usage": storage_stats,
                "latest_version": None,
                "recommendations": []
            }

            # ステータス別統計
            for version in versions:
                status = version.status.value
                report["status_breakdown"][status] = report["status_breakdown"].get(status, 0) + 1

            # 最新バージョン
            if versions:
                latest = max(versions, key=lambda v: v.created_at)
                report["latest_version"] = latest.to_dict()

            # メトリクス進化の追跡
            for version in sorted(versions, key=lambda v: v.created_at):
                if version.metadata and version.metadata.metrics:
                    report["metrics_evolution"].append({
                        "version": version.version,
                        "created_at": version.created_at.isoformat(),
                        "metrics": version.metadata.metrics
                    })

            # 推奨事項
            if len(versions) == 0:
                report["recommendations"].append("No versions found. Consider creating initial version.")
            elif len([v for v in versions if v.status.value == "active"]) == 0:
                report["recommendations"].append("No active versions. Consider activating a stable version.")

            return report

        except Exception as e:
            logger.error(f"Failed to generate version report: {e}")
            return {}

    async def export_version_metadata(
        self,
        model_name: str,
        format: str = "json"
    ) -> Optional[str]:
        """
        バージョンメタデータをエクスポート

        Args:
            model_name: モデル名
            format: エクスポート形式

        Returns:
            エクスポートされたデータ（文字列）
        """
        try:
            versions = await self.registry.list_versions(model_name)

            export_data = {
                "model_name": model_name,
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "versions": [v.to_dict() for v in versions]
            }

            if format == "json":
                import json
                return json.dumps(export_data, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            logger.error(f"Failed to export version metadata: {e}")
            return None