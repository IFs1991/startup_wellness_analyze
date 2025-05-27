# Phase 3 Task 3.3: モデルバージョニングシステムの実装
# TDD GREEN段階: VersionStorage実装

import os
import json
import pickle
import hashlib
import asyncio
import aiofiles
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, BinaryIO
from datetime import datetime, timezone
import gzip
import lz4.frame

import structlog
from .models import ModelArtifact, generate_artifact_id

logger = structlog.get_logger(__name__)


class VersionStorage:
    """
    バージョンストレージクラス

    モデルアーティファクトの保存・管理を担当
    """

    def __init__(
        self,
        storage_path: str = "./model_storage",
        enable_compression: bool = True,
        default_compression: str = "gzip",
        max_file_size_mb: int = 100
    ):
        """
        VersionStorageの初期化

        Args:
            storage_path: ストレージパス
            enable_compression: 圧縮を有効にするか
            default_compression: デフォルト圧縮形式
            max_file_size_mb: 最大ファイルサイズ（MB）
        """
        self.storage_path = Path(storage_path)
        self.enable_compression = enable_compression
        self.default_compression = default_compression
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

        # ストレージディレクトリ作成
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # サブディレクトリ作成
        (self.storage_path / "artifacts").mkdir(exist_ok=True)
        (self.storage_path / "metadata").mkdir(exist_ok=True)
        (self.storage_path / "temp").mkdir(exist_ok=True)

    async def store_model_artifact(
        self,
        model_data: Any,
        artifact_id: Optional[str] = None,
        format: str = "pickle",
        compression: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        モデルアーティファクトを保存

        Args:
            model_data: 保存するモデルデータ
            artifact_id: アーティファクトID（指定しない場合は自動生成）
            format: 保存形式
            compression: 圧縮形式
            metadata: 追加メタデータ

        Returns:
            アーティファクトID
        """
        try:
            if artifact_id is None:
                artifact_id = generate_artifact_id()

            if compression is None and self.enable_compression:
                compression = self.default_compression

            # 一時ファイルでシリアライゼーション
            temp_path = self.storage_path / "temp" / f"{artifact_id}_temp"

            # データをシリアライズ
            serialized_data = await self._serialize_data(model_data, format)

            # 圧縮
            if compression and compression != "none":
                serialized_data = await self._compress_data(serialized_data, compression)

            # ファイルサイズチェック
            if len(serialized_data) > self.max_file_size_bytes:
                raise ValueError(f"Model size ({len(serialized_data)} bytes) exceeds maximum allowed size")

            # ファイル保存
            artifact_path = self.storage_path / "artifacts" / f"{artifact_id}.bin"
            async with aiofiles.open(artifact_path, 'wb') as f:
                await f.write(serialized_data)

            # チェックサム計算
            checksum = hashlib.sha256(serialized_data).hexdigest()

            # アーティファクト情報を作成
            artifact = ModelArtifact(
                id=artifact_id,
                path=str(artifact_path),
                size=len(serialized_data),
                checksum=checksum,
                format=format,
                compression=compression or "none",
                metadata=metadata or {}
            )

            # メタデータ保存
            metadata_path = self.storage_path / "metadata" / f"{artifact_id}.json"
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(artifact.to_dict(), indent=2))

            logger.info(f"Artifact stored: {artifact_id}, size: {len(serialized_data)} bytes")
            return artifact_id

        except Exception as e:
            logger.error(f"Failed to store artifact {artifact_id}: {e}")
            # クリーンアップ
            await self._cleanup_failed_storage(artifact_id)
            raise

    async def load_model_artifact(self, artifact_id: str) -> Any:
        """
        モデルアーティファクトを読み込み

        Args:
            artifact_id: アーティファクトID

        Returns:
            読み込まれたモデルデータ
        """
        try:
            # アーティファクト情報取得
            artifact_info = await self.get_artifact_info(artifact_id)
            if not artifact_info:
                raise FileNotFoundError(f"Artifact {artifact_id} not found")

            artifact_path = Path(artifact_info["path"])
            if not artifact_path.exists():
                raise FileNotFoundError(f"Artifact file not found: {artifact_path}")

            # ファイル読み込み
            async with aiofiles.open(artifact_path, 'rb') as f:
                data = await f.read()

            # チェックサム検証
            actual_checksum = hashlib.sha256(data).hexdigest()
            expected_checksum = artifact_info["checksum"]
            if actual_checksum != expected_checksum:
                raise ValueError(f"Checksum mismatch for {artifact_id}")

            # 解凍
            compression = artifact_info.get("compression", "none")
            if compression != "none":
                data = await self._decompress_data(data, compression)

            # デシリアライゼーション
            format = artifact_info.get("format", "pickle")
            model_data = await self._deserialize_data(data, format)

            logger.debug(f"Artifact loaded: {artifact_id}")
            return model_data

        except Exception as e:
            logger.error(f"Failed to load artifact {artifact_id}: {e}")
            raise

    async def delete_model_artifact(self, artifact_id: str) -> bool:
        """
        モデルアーティファクトを削除

        Args:
            artifact_id: アーティファクトID

        Returns:
            削除成功フラグ
        """
        try:
            artifact_path = self.storage_path / "artifacts" / f"{artifact_id}.bin"
            metadata_path = self.storage_path / "metadata" / f"{artifact_id}.json"

            deleted_files = 0

            # アーティファクトファイル削除
            if artifact_path.exists():
                artifact_path.unlink()
                deleted_files += 1

            # メタデータファイル削除
            if metadata_path.exists():
                metadata_path.unlink()
                deleted_files += 1

            success = deleted_files > 0
            if success:
                logger.info(f"Artifact deleted: {artifact_id}")
            else:
                logger.warning(f"Artifact not found for deletion: {artifact_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to delete artifact {artifact_id}: {e}")
            return False

    async def list_artifacts(self) -> List[str]:
        """
        保存されているアーティファクトのリストを取得

        Returns:
            アーティファクトIDのリスト
        """
        try:
            artifacts = []
            artifacts_dir = self.storage_path / "artifacts"

            if artifacts_dir.exists():
                for file_path in artifacts_dir.glob("*.bin"):
                    artifact_id = file_path.stem
                    artifacts.append(artifact_id)

            logger.debug(f"Found {len(artifacts)} artifacts")
            return artifacts

        except Exception as e:
            logger.error(f"Failed to list artifacts: {e}")
            return []

    async def get_artifact_info(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """
        アーティファクト情報を取得

        Args:
            artifact_id: アーティファクトID

        Returns:
            アーティファクト情報辞書
        """
        try:
            metadata_path = self.storage_path / "metadata" / f"{artifact_id}.json"

            if not metadata_path.exists():
                return None

            async with aiofiles.open(metadata_path, 'r') as f:
                content = await f.read()
                return json.loads(content)

        except Exception as e:
            logger.error(f"Failed to get artifact info for {artifact_id}: {e}")
            return None

    async def artifact_exists(self, artifact_id: str) -> bool:
        """
        アーティファクトの存在確認

        Args:
            artifact_id: アーティファクトID

        Returns:
            存在フラグ
        """
        try:
            artifact_path = self.storage_path / "artifacts" / f"{artifact_id}.bin"
            metadata_path = self.storage_path / "metadata" / f"{artifact_id}.json"

            return artifact_path.exists() and metadata_path.exists()

        except Exception:
            return False

    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        ストレージ統計情報を取得

        Returns:
            ストレージ統計辞書
        """
        try:
            stats = {
                "total_artifacts": 0,
                "total_size_bytes": 0,
                "artifacts_by_format": {},
                "artifacts_by_compression": {},
                "average_size_bytes": 0
            }

            artifacts = await self.list_artifacts()
            stats["total_artifacts"] = len(artifacts)

            for artifact_id in artifacts:
                info = await self.get_artifact_info(artifact_id)
                if info:
                    stats["total_size_bytes"] += info.get("size", 0)

                    # 形式別統計
                    format_type = info.get("format", "unknown")
                    stats["artifacts_by_format"][format_type] = stats["artifacts_by_format"].get(format_type, 0) + 1

                    # 圧縮形式別統計
                    compression = info.get("compression", "none")
                    stats["artifacts_by_compression"][compression] = stats["artifacts_by_compression"].get(compression, 0) + 1

            if stats["total_artifacts"] > 0:
                stats["average_size_bytes"] = stats["total_size_bytes"] / stats["total_artifacts"]

            return stats

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}

    async def cleanup_temp_files(self) -> int:
        """
        一時ファイルをクリーンアップ

        Returns:
            削除されたファイル数
        """
        try:
            temp_dir = self.storage_path / "temp"
            deleted_count = 0

            if temp_dir.exists():
                for temp_file in temp_dir.glob("*_temp"):
                    try:
                        temp_file.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {temp_file}: {e}")

            logger.info(f"Cleaned up {deleted_count} temporary files")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
            return 0

    # REFACTOR段階で追加: 高度なストレージ機能

    async def backup_artifacts(self, backup_path: str) -> Dict[str, Any]:
        """
        アーティファクトをバックアップ

        Args:
            backup_path: バックアップ先パス

        Returns:
            バックアップ結果
        """
        try:
            import shutil
            from pathlib import Path

            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)

            result = {
                "backed_up_artifacts": 0,
                "backed_up_metadata": 0,
                "total_size_bytes": 0,
                "backup_path": str(backup_dir)
            }

            # アーティファクトファイルをバックアップ
            artifacts_dir = self.storage_path / "artifacts"
            backup_artifacts_dir = backup_dir / "artifacts"
            backup_artifacts_dir.mkdir(exist_ok=True)

            if artifacts_dir.exists():
                for artifact_file in artifacts_dir.glob("*.bin"):
                    destination = backup_artifacts_dir / artifact_file.name
                    shutil.copy2(artifact_file, destination)
                    result["backed_up_artifacts"] += 1
                    result["total_size_bytes"] += artifact_file.stat().st_size

            # メタデータファイルをバックアップ
            metadata_dir = self.storage_path / "metadata"
            backup_metadata_dir = backup_dir / "metadata"
            backup_metadata_dir.mkdir(exist_ok=True)

            if metadata_dir.exists():
                for metadata_file in metadata_dir.glob("*.json"):
                    destination = backup_metadata_dir / metadata_file.name
                    shutil.copy2(metadata_file, destination)
                    result["backed_up_metadata"] += 1

            logger.info(f"Backup completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to backup artifacts: {e}")
            return {"error": str(e)}

    async def restore_from_backup(self, backup_path: str) -> Dict[str, Any]:
        """
        バックアップからリストア

        Args:
            backup_path: バックアップ元パス

        Returns:
            リストア結果
        """
        try:
            import shutil
            from pathlib import Path

            backup_dir = Path(backup_path)
            if not backup_dir.exists():
                raise FileNotFoundError(f"Backup directory not found: {backup_path}")

            result = {
                "restored_artifacts": 0,
                "restored_metadata": 0,
                "total_size_bytes": 0
            }

            # アーティファクトをリストア
            backup_artifacts_dir = backup_dir / "artifacts"
            if backup_artifacts_dir.exists():
                artifacts_dir = self.storage_path / "artifacts"
                artifacts_dir.mkdir(exist_ok=True)

                for backup_file in backup_artifacts_dir.glob("*.bin"):
                    destination = artifacts_dir / backup_file.name
                    shutil.copy2(backup_file, destination)
                    result["restored_artifacts"] += 1
                    result["total_size_bytes"] += backup_file.stat().st_size

            # メタデータをリストア
            backup_metadata_dir = backup_dir / "metadata"
            if backup_metadata_dir.exists():
                metadata_dir = self.storage_path / "metadata"
                metadata_dir.mkdir(exist_ok=True)

                for backup_file in backup_metadata_dir.glob("*.json"):
                    destination = metadata_dir / backup_file.name
                    shutil.copy2(backup_file, destination)
                    result["restored_metadata"] += 1

            logger.info(f"Restore completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return {"error": str(e)}

    async def optimize_storage(self) -> Dict[str, Any]:
        """
        ストレージを最適化

        Returns:
            最適化結果
        """
        try:
            result = {
                "compressed_artifacts": 0,
                "space_saved_bytes": 0,
                "orphaned_cleaned": 0
            }

            # 孤立ファイルのクリーンアップ
            artifacts = await self.list_artifacts()

            # アーティファクトファイルの存在確認
            artifacts_dir = self.storage_path / "artifacts"
            if artifacts_dir.exists():
                for artifact_file in artifacts_dir.glob("*.bin"):
                    artifact_id = artifact_file.stem
                    if artifact_id not in artifacts:
                        # 孤立したアーティファクト
                        artifact_file.unlink()
                        result["orphaned_cleaned"] += 1

            # メタデータファイルの存在確認
            metadata_dir = self.storage_path / "metadata"
            if metadata_dir.exists():
                for metadata_file in metadata_dir.glob("*.json"):
                    artifact_id = metadata_file.stem
                    if artifact_id not in artifacts:
                        # 孤立したメタデータ
                        metadata_file.unlink()
                        result["orphaned_cleaned"] += 1

            logger.info(f"Storage optimization completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to optimize storage: {e}")
            return {"error": str(e)}

    async def get_storage_health(self) -> Dict[str, Any]:
        """
        ストレージヘルス情報を取得

        Returns:
            ヘルス情報
        """
        try:
            import os

            health = {
                "status": "healthy",
                "issues": [],
                "warnings": [],
                "disk_usage": {},
                "file_integrity": {}
            }

            # ディスク使用量チェック
            stats = await self.get_storage_stats()
            if stats.get("total_size_bytes", 0) > self.max_file_size_bytes * 100:  # 100ファイル分
                health["warnings"].append("High storage usage detected")

            # ディスク空き容量チェック
            disk_usage = shutil.disk_usage(self.storage_path)
            health["disk_usage"] = {
                "total_bytes": disk_usage.total,
                "used_bytes": disk_usage.used,
                "free_bytes": disk_usage.free,
                "free_percentage": (disk_usage.free / disk_usage.total) * 100
            }

            if health["disk_usage"]["free_percentage"] < 10:
                health["issues"].append("Low disk space (< 10%)")
                health["status"] = "critical"
            elif health["disk_usage"]["free_percentage"] < 20:
                health["warnings"].append("Low disk space (< 20%)")
                health["status"] = "warning"

            # ファイル整合性チェック
            artifacts = await self.list_artifacts()
            for artifact_id in artifacts[:10]:  # サンプルチェック
                try:
                    info = await self.get_artifact_info(artifact_id)
                    if info:
                        artifact_path = Path(info["path"])
                        if artifact_path.exists():
                            # チェックサム検証（簡略版）
                            health["file_integrity"][artifact_id] = "ok"
                        else:
                            health["file_integrity"][artifact_id] = "missing"
                            health["issues"].append(f"Missing artifact file: {artifact_id}")
                except Exception as e:
                    health["file_integrity"][artifact_id] = f"error: {e}"

            return health

        except Exception as e:
            logger.error(f"Failed to get storage health: {e}")
            return {"status": "error", "error": str(e)}

    # プライベートメソッド

    async def _serialize_data(self, data: Any, format: str) -> bytes:
        """データをシリアライズ"""
        if format == "pickle":
            return pickle.dumps(data)
        elif format == "json":
            return json.dumps(data, ensure_ascii=False).encode('utf-8')
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def _deserialize_data(self, data: bytes, format: str) -> Any:
        """データをデシリアライズ"""
        if format == "pickle":
            return pickle.loads(data)
        elif format == "json":
            return json.loads(data.decode('utf-8'))
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def _compress_data(self, data: bytes, compression: str) -> bytes:
        """データを圧縮"""
        if compression == "gzip":
            return gzip.compress(data)
        elif compression == "lz4":
            return lz4.frame.compress(data)
        else:
            raise ValueError(f"Unsupported compression: {compression}")

    async def _decompress_data(self, data: bytes, compression: str) -> bytes:
        """データを解凍"""
        if compression == "gzip":
            return gzip.decompress(data)
        elif compression == "lz4":
            return lz4.frame.decompress(data)
        else:
            raise ValueError(f"Unsupported compression: {compression}")

    async def _cleanup_failed_storage(self, artifact_id: str) -> None:
        """失敗したストレージ操作のクリーンアップ"""
        try:
            # 一時ファイル削除
            temp_path = self.storage_path / "temp" / f"{artifact_id}_temp"
            if temp_path.exists():
                temp_path.unlink()

            # 部分的に作成されたファイル削除
            artifact_path = self.storage_path / "artifacts" / f"{artifact_id}.bin"
            if artifact_path.exists():
                artifact_path.unlink()

            metadata_path = self.storage_path / "metadata" / f"{artifact_id}.json"
            if metadata_path.exists():
                metadata_path.unlink()

        except Exception as e:
            logger.warning(f"Failed to cleanup after failed storage: {e}")