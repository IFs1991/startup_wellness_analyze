# Phase 3 Task 3.3: モデルバージョニングシステムの実装
# TDD GREEN段階: データモデル実装

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import json
import uuid


class VersionStatus(Enum):
    """バージョンステータス列挙型"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FAILED = "failed"


class Environment(Enum):
    """デプロイメント環境列挙型"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class VersionMetadata:
    """
    バージョンメタデータクラス

    モデルバージョンに関連するメタ情報を格納
    """
    model_name: str
    version: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    creator: str = "system"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    environment_info: Dict[str, Any] = field(default_factory=dict)
    training_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionMetadata':
        """辞書からVersionMetadataを復元"""
        if isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
        return cls(**data)


@dataclass
class ModelArtifact:
    """
    モデルアーティファクトクラス

    保存されたモデルファイルの情報を管理
    """
    id: str
    path: str
    size: int
    checksum: str
    format: str = "pickle"  # pickle, onnx, savedmodel, etc.
    compression: str = "none"  # none, gzip, lz4, etc.
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelArtifact':
        """辞書からModelArtifactを復元"""
        if isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
        return cls(**data)


@dataclass
class ModelVersion:
    """
    モデルバージョンクラス

    モデルの特定バージョンを表現する中心的なクラス
    """
    id: str
    model_name: str
    version: str
    artifact_id: str
    metadata: Optional[VersionMetadata] = None
    status: VersionStatus = VersionStatus.DRAFT
    environment: Environment = Environment.DEVELOPMENT
    parent_version_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """初期化後処理"""
        if isinstance(self.status, str):
            self.status = VersionStatus(self.status)
        if isinstance(self.environment, str):
            self.environment = Environment(self.environment)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = {
            "id": self.id,
            "model_name": self.model_name,
            "version": self.version,
            "artifact_id": self.artifact_id,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "status": self.status.value,
            "environment": self.environment.value,
            "parent_version_id": self.parent_version_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """辞書からModelVersionを復元"""
        # 日時文字列を変換
        if isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
        if isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"].replace('Z', '+00:00'))

        # メタデータを復元
        if data.get("metadata"):
            data["metadata"] = VersionMetadata.from_dict(data["metadata"])

        return cls(**data)

    def update_status(self, new_status: VersionStatus, reason: str = "") -> None:
        """ステータスを更新"""
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc)
        if reason and self.metadata:
            if "status_history" not in self.metadata.training_info:
                self.metadata.training_info["status_history"] = []
            self.metadata.training_info["status_history"].append({
                "status": new_status.value,
                "reason": reason,
                "timestamp": self.updated_at.isoformat()
            })


@dataclass
class RollbackPoint:
    """
    ロールバックポイントクラス

    特定時点でのモデル状態を記録
    """
    id: str
    model_name: str
    version_id: str
    version: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""
    creator: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RollbackPoint':
        """辞書からRollbackPointを復元"""
        if isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
        return cls(**data)


@dataclass
class VersionComparison:
    """
    バージョン比較結果クラス

    2つのモデルバージョンの比較結果を保存
    """
    version_a: str
    version_b: str
    comparison_type: str  # "metrics", "architecture", "full"
    result: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionComparison':
        """辞書からVersionComparisonを復元"""
        if isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
        return cls(**data)


# ユーティリティ関数

def generate_version_id() -> str:
    """一意のバージョンIDを生成"""
    return str(uuid.uuid4())


def generate_artifact_id() -> str:
    """一意のアーティファクトIDを生成"""
    return f"artifact_{uuid.uuid4().hex[:12]}"


def parse_semantic_version(version: str) -> tuple:
    """セマンティックバージョンを解析"""
    try:
        parts = version.split('.')
        if len(parts) != 3:
            raise ValueError("Invalid semantic version format")
        return int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid semantic version '{version}': {e}")


def compare_semantic_versions(version_a: str, version_b: str) -> Dict[str, Any]:
    """セマンティックバージョンを比較"""
    try:
        major_a, minor_a, patch_a = parse_semantic_version(version_a)
        major_b, minor_b, patch_b = parse_semantic_version(version_b)

        result = {
            "version_a": version_a,
            "version_b": version_b,
            "is_upgrade": False,
            "change_type": "none",
            "major_diff": major_b - major_a,
            "minor_diff": minor_b - minor_a,
            "patch_diff": patch_b - patch_a
        }

        if major_b > major_a:
            result["is_upgrade"] = True
            result["change_type"] = "major"
        elif major_b == major_a and minor_b > minor_a:
            result["is_upgrade"] = True
            result["change_type"] = "minor"
        elif major_b == major_a and minor_b == minor_a and patch_b > patch_a:
            result["is_upgrade"] = True
            result["change_type"] = "patch"
        elif major_b < major_a or (major_b == major_a and minor_b < minor_a) or \
             (major_b == major_a and minor_b == minor_a and patch_b < patch_a):
            result["change_type"] = "downgrade"

        return result

    except ValueError as e:
        return {
            "version_a": version_a,
            "version_b": version_b,
            "error": str(e),
            "is_upgrade": False,
            "change_type": "invalid"
        }