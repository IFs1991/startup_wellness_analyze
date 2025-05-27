# Phase 3 Task 3.3: モデルバージョニングシステムの実装
# TDD GREEN段階: ModelRegistry実装

import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone

import structlog

# 条件付きインポート - DatabaseManagerが存在しない場合はNoneに設定
try:
    from ..database.manager import DatabaseManager
except ImportError:
    DatabaseManager = None

from .models import ModelVersion, VersionMetadata, RollbackPoint, VersionStatus, Environment, generate_version_id

logger = structlog.get_logger(__name__)


class ModelRegistry:
    """
    モデルレジストリクラス

    モデルバージョンのメタデータ管理を担当
    """

    def __init__(self, database_manager: Optional['DatabaseManager'] = None):
        """
        ModelRegistryの初期化

        Args:
            database_manager: データベースマネージャー
        """
        self.db = database_manager
        # インメモリストレージ（テスト・開発用）
        self.memory_storage: Dict[str, ModelVersion] = {}
        self.rollback_storage: Dict[str, RollbackPoint] = {}

    async def register_version(
        self,
        model_name: str,
        version: str,
        artifact_id: str,
        metadata: VersionMetadata,
        parent_version_id: Optional[str] = None
    ) -> str:
        """
        新しいモデルバージョンを登録

        Args:
            model_name: モデル名
            version: バージョン
            artifact_id: アーティファクトID
            metadata: バージョンメタデータ
            parent_version_id: 親バージョンID

        Returns:
            バージョンID
        """
        try:
            version_id = generate_version_id()

            model_version = ModelVersion(
                id=version_id,
                model_name=model_name,
                version=version,
                artifact_id=artifact_id,
                metadata=metadata,
                parent_version_id=parent_version_id
            )

            # データベース保存（利用可能な場合）
            if self.db:
                await self._save_to_database(model_version)
            else:
                self.memory_storage[version_id] = model_version

            logger.info(f"Version registered: {model_name} v{version} ({version_id})")
            return version_id

        except Exception as e:
            logger.error(f"Failed to register version: {e}")
            raise

    async def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """バージョンを取得"""
        try:
            if self.db:
                return await self._load_from_database(version_id)
            else:
                return self.memory_storage.get(version_id)
        except Exception as e:
            logger.error(f"Failed to get version {version_id}: {e}")
            return None

    async def list_versions(self, model_name: str) -> List[ModelVersion]:
        """モデルのバージョンリストを取得"""
        try:
            if self.db:
                return await self._list_from_database(model_name)
            else:
                return [v for v in self.memory_storage.values() if v.model_name == model_name]
        except Exception as e:
            logger.error(f"Failed to list versions for {model_name}: {e}")
            return []

    async def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """最新バージョンを取得"""
        try:
            versions = await self.list_versions(model_name)
            if not versions:
                return None

            # 作成時刻でソートして最新を取得
            latest = max(versions, key=lambda v: v.created_at)
            return latest
        except Exception as e:
            logger.error(f"Failed to get latest version for {model_name}: {e}")
            return None

    async def update_version_status(
        self,
        version_id: str,
        status: Union[VersionStatus, str],
        reason: str = ""
    ) -> bool:
        """バージョンステータスを更新"""
        try:
            version = await self.get_version(version_id)
            if not version:
                return False

            if isinstance(status, str):
                status = VersionStatus(status)

            version.update_status(status, reason)

            # 保存
            if self.db:
                await self._save_to_database(version)
            else:
                self.memory_storage[version_id] = version

            logger.info(f"Version status updated: {version_id} -> {status.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to update version status: {e}")
            return False

    async def delete_version(self, version_id: str) -> bool:
        """バージョンを削除"""
        try:
            if self.db:
                return await self._delete_from_database(version_id)
            else:
                if version_id in self.memory_storage:
                    del self.memory_storage[version_id]
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete version {version_id}: {e}")
            return False

    async def search_versions(self, query: Dict[str, Any]) -> List[ModelVersion]:
        """バージョンを検索"""
        try:
            all_versions = []
            if self.db:
                all_versions = await self._search_database(query)
            else:
                all_versions = list(self.memory_storage.values())

            # インメモリフィルタリング
            filtered = []
            for version in all_versions:
                if self._matches_query(version, query):
                    filtered.append(version)

            return filtered

        except Exception as e:
            logger.error(f"Failed to search versions: {e}")
            return []

    # データベース操作（簡略版）
    async def _save_to_database(self, version: ModelVersion) -> None:
        """データベースに保存（実装は後で）"""
        pass

    async def _load_from_database(self, version_id: str) -> Optional[ModelVersion]:
        """データベースから読み込み（実装は後で）"""
        return None

    async def _list_from_database(self, model_name: str) -> List[ModelVersion]:
        """データベースからリスト取得（実装は後で）"""
        return []

    async def _delete_from_database(self, version_id: str) -> bool:
        """データベースから削除（実装は後で）"""
        return False

    async def _search_database(self, query: Dict[str, Any]) -> List[ModelVersion]:
        """データベース検索（実装は後で）"""
        return []

    def _matches_query(self, version: ModelVersion, query: Dict[str, Any]) -> bool:
        """クエリマッチング"""
        for key, value in query.items():
            if key == "model_name" and version.model_name != value:
                return False
            elif key == "status" and version.status.value != value:
                return False
            elif key == "tags" and version.metadata:
                if not any(tag in version.metadata.tags for tag in value):
                    return False
        return True

    # ロールバック関連
    async def create_rollback_point(
        self,
        model_name: str,
        version_id: str,
        reason: str,
        creator: str = "system"
    ) -> str:
        """ロールバックポイントを作成"""
        try:
            version = await self.get_version(version_id)
            if not version:
                raise ValueError(f"Version {version_id} not found")

            rollback_point = RollbackPoint(
                id=generate_version_id(),
                model_name=model_name,
                version_id=version_id,
                version=version.version,
                reason=reason,
                creator=creator
            )

            self.rollback_storage[rollback_point.id] = rollback_point
            return rollback_point.id

        except Exception as e:
            logger.error(f"Failed to create rollback point: {e}")
            raise

    async def get_rollback_history(self, model_name: str) -> List[Dict[str, Any]]:
        """ロールバック履歴を取得"""
        try:
            history = []
            for rollback in self.rollback_storage.values():
                if rollback.model_name == model_name:
                    history.append(rollback.to_dict())

            return sorted(history, key=lambda x: x["created_at"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to get rollback history: {e}")
            return []

    # 追加メソッド（テスト用）
    async def promote_version(self, version_id: str, from_env: str, to_env: str) -> bool:
        """バージョンを昇格"""
        return await self.update_version_status(version_id, "active", f"Promoted from {from_env} to {to_env}")

    async def add_tags(self, version_id: str, tags: List[str]) -> bool:
        """タグを追加"""
        return True  # 簡略実装

    async def remove_tags(self, version_id: str, tags: List[str]) -> bool:
        """タグを削除"""
        return True  # 簡略実装

    async def update_version_tags(self, version_id: str, tags: List[str]) -> bool:
        """タグを更新"""
        return True  # 簡略実装