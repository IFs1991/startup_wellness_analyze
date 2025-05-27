# Phase 3 Task 3.3: モデルバージョニングシステムの実装
# TDD GREEN段階: VersionComparator実装

import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from statistics import mean
from datetime import datetime, timezone

import structlog
from .models import VersionMetadata, VersionComparison, compare_semantic_versions

logger = structlog.get_logger(__name__)


class VersionComparator:
    """
    バージョン比較クラス

    異なるモデルバージョン間の比較機能を提供
    """

    def __init__(self):
        """VersionComparatorの初期化"""
        self.comparison_cache: Dict[str, VersionComparison] = {}

    async def compare_architectures(
        self,
        architecture_a: Dict[str, Any],
        architecture_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        モデルアーキテクチャを比較

        Args:
            architecture_a: 比較対象アーキテクチャA
            architecture_b: 比較対象アーキテクチャB

        Returns:
            比較結果辞書
        """
        try:
            comparison_result = {
                "is_identical": False,
                "layer_differences": [],
                "parameter_count_diff": 0,
                "structural_changes": [],
                "compatibility_score": 0.0
            }

            # レイヤー構造比較
            layers_a = architecture_a.get("layers", [])
            layers_b = architecture_b.get("layers", [])

            # レイヤー数比較
            if len(layers_a) != len(layers_b):
                comparison_result["structural_changes"].append(
                    f"Layer count changed: {len(layers_a)} -> {len(layers_b)}"
                )

            # 各レイヤーの詳細比較
            max_layers = max(len(layers_a), len(layers_b))
            identical_layers = 0

            for i in range(max_layers):
                layer_a = layers_a[i] if i < len(layers_a) else None
                layer_b = layers_b[i] if i < len(layers_b) else None

                if layer_a is None:
                    comparison_result["layer_differences"].append({
                        "layer_index": i,
                        "change": "added",
                        "details": layer_b
                    })
                elif layer_b is None:
                    comparison_result["layer_differences"].append({
                        "layer_index": i,
                        "change": "removed",
                        "details": layer_a
                    })
                elif layer_a != layer_b:
                    comparison_result["layer_differences"].append({
                        "layer_index": i,
                        "change": "modified",
                        "before": layer_a,
                        "after": layer_b
                    })
                else:
                    identical_layers += 1

            # 互換性スコア計算（0.0-1.0）
            if max_layers > 0:
                comparison_result["compatibility_score"] = identical_layers / max_layers

            # アーキテクチャが完全に同一か判定
            comparison_result["is_identical"] = (
                len(comparison_result["layer_differences"]) == 0 and
                len(comparison_result["structural_changes"]) == 0
            )

            # パラメータ数の違いを概算
            param_count_a = self._estimate_parameter_count(architecture_a)
            param_count_b = self._estimate_parameter_count(architecture_b)
            comparison_result["parameter_count_diff"] = param_count_b - param_count_a

            logger.debug(f"Architecture comparison completed: {comparison_result['compatibility_score']:.2f} compatibility")
            return comparison_result

        except Exception as e:
            logger.error(f"Architecture comparison failed: {e}")
            return {
                "error": str(e),
                "is_identical": False,
                "layer_differences": [],
                "structural_changes": [],
                "compatibility_score": 0.0
            }

    async def compare_metrics(
        self,
        metrics_a: Dict[str, Any],
        metrics_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        モデルメトリクスを比較

        Args:
            metrics_a: 比較対象メトリクスA
            metrics_b: 比較対象メトリクスB

        Returns:
            メトリクス比較結果
        """
        try:
            comparison_result = {}

            # 共通メトリクスの特定
            common_metrics = set(metrics_a.keys()) & set(metrics_b.keys())

            for metric_name in common_metrics:
                value_a = metrics_a[metric_name]
                value_b = metrics_b[metric_name]

                if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
                    # 数値メトリクスの比較
                    improvement = self._calculate_improvement(metric_name, value_a, value_b)

                    comparison_result[metric_name] = {
                        "value_a": value_a,
                        "value_b": value_b,
                        "improvement": improvement,
                        "improvement_percentage": (improvement / value_a * 100) if value_a != 0 else 0,
                        "is_better": improvement > 0
                    }
                else:
                    # 非数値メトリクスの比較
                    comparison_result[metric_name] = {
                        "value_a": value_a,
                        "value_b": value_b,
                        "is_identical": value_a == value_b
                    }

            # 新規追加メトリクス
            new_metrics = set(metrics_b.keys()) - set(metrics_a.keys())
            if new_metrics:
                comparison_result["new_metrics"] = {
                    metric: metrics_b[metric] for metric in new_metrics
                }

            # 削除されたメトリクス
            removed_metrics = set(metrics_a.keys()) - set(metrics_b.keys())
            if removed_metrics:
                comparison_result["removed_metrics"] = {
                    metric: metrics_a[metric] for metric in removed_metrics
                }

            # 全体的な改善スコア計算
            improvement_scores = []
            for metric_name, result in comparison_result.items():
                if isinstance(result, dict) and "improvement" in result:
                    # 正規化された改善スコア（-1.0 to 1.0）
                    normalized_score = max(-1.0, min(1.0, result["improvement"] / abs(result["value_a"]) if result["value_a"] != 0 else 0))
                    improvement_scores.append(normalized_score)

            overall_improvement = mean(improvement_scores) if improvement_scores else 0.0
            comparison_result["overall_improvement"] = overall_improvement

            logger.debug(f"Metrics comparison completed: {overall_improvement:.3f} overall improvement")
            return comparison_result

        except Exception as e:
            logger.error(f"Metrics comparison failed: {e}")
            return {"error": str(e), "overall_improvement": 0.0}

    async def compare_versions(
        self,
        metadata_a: VersionMetadata,
        metadata_b: VersionMetadata
    ) -> Dict[str, Any]:
        """
        バージョンメタデータを包括的に比較

        Args:
            metadata_a: 比較対象メタデータA
            metadata_b: 比較対象メタデータB

        Returns:
            包括的比較結果
        """
        try:
            comparison_result = {
                "version_a": metadata_a.version,
                "version_b": metadata_b.version,
                "comparison_timestamp": datetime.now(timezone.utc).isoformat()
            }

            # セマンティックバージョン比較
            comparison_result["version_change"] = compare_semantic_versions(
                metadata_a.version,
                metadata_b.version
            )

            # メトリクス比較
            if metadata_a.metrics and metadata_b.metrics:
                comparison_result["metrics_comparison"] = await self.compare_metrics(
                    metadata_a.metrics,
                    metadata_b.metrics
                )

            # 環境情報比較
            if metadata_a.environment_info and metadata_b.environment_info:
                comparison_result["environment_changes"] = self._compare_environments(
                    metadata_a.environment_info,
                    metadata_b.environment_info
                )

            # タグ比較
            comparison_result["tag_changes"] = self._compare_tags(
                metadata_a.tags,
                metadata_b.tags
            )

            # 作成者・説明比較
            comparison_result["metadata_changes"] = {
                "creator_changed": metadata_a.creator != metadata_b.creator,
                "description_changed": metadata_a.description != metadata_b.description,
                "time_diff_days": (metadata_b.created_at - metadata_a.created_at).days
            }

            logger.info(f"Version comparison completed: {metadata_a.version} vs {metadata_b.version}")
            return comparison_result

        except Exception as e:
            logger.error(f"Version comparison failed: {e}")
            return {
                "error": str(e),
                "version_a": metadata_a.version,
                "version_b": metadata_b.version
            }

    async def compare_semantic_versions(self, version_a: str, version_b: str) -> Dict[str, Any]:
        """
        セマンティックバージョンを比較

        Args:
            version_a: バージョンA
            version_b: バージョンB

        Returns:
            セマンティックバージョン比較結果
        """
        try:
            return compare_semantic_versions(version_a, version_b)
        except Exception as e:
            logger.error(f"Semantic version comparison failed: {e}")
            return {
                "error": str(e),
                "version_a": version_a,
                "version_b": version_b
            }

    def _calculate_improvement(self, metric_name: str, value_a: float, value_b: float) -> float:
        """
        メトリクスの改善度を計算

        Args:
            metric_name: メトリクス名
            value_a: 元の値
            value_b: 新しい値

        Returns:
            改善度（正の値は改善、負の値は悪化）
        """
        # 損失系メトリクス（小さい方が良い）
        loss_metrics = {"loss", "error", "mse", "mae", "rmse", "cross_entropy"}

        if any(loss_term in metric_name.lower() for loss_term in loss_metrics):
            # 損失の場合：減少が改善
            return value_a - value_b
        else:
            # 精度系メトリクス（大きい方が良い）
            return value_b - value_a

    def _estimate_parameter_count(self, architecture: Dict[str, Any]) -> int:
        """
        アーキテクチャからパラメータ数を概算

        Args:
            architecture: モデルアーキテクチャ

        Returns:
            概算パラメータ数
        """
        try:
            total_params = 0
            layers = architecture.get("layers", [])
            prev_units = None

            for layer in layers:
                layer_type = layer.get("type", "").lower()
                units = layer.get("units", 0)

                if layer_type == "dense" and prev_units is not None:
                    # Dense層：(入力ユニット数 + 1) * 出力ユニット数
                    total_params += (prev_units + 1) * units

                prev_units = units

            return total_params

        except Exception as e:
            logger.warning(f"Parameter count estimation failed: {e}")
            return 0

    def _compare_environments(
        self,
        env_a: Dict[str, Any],
        env_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """環境情報を比較"""
        changes = {}

        all_keys = set(env_a.keys()) | set(env_b.keys())

        for key in all_keys:
            val_a = env_a.get(key)
            val_b = env_b.get(key)

            if val_a != val_b:
                changes[key] = {
                    "before": val_a,
                    "after": val_b
                }

        return changes

    def _compare_tags(self, tags_a: List[str], tags_b: List[str]) -> Dict[str, Any]:
        """タグを比較"""
        set_a = set(tags_a)
        set_b = set(tags_b)

        return {
            "added_tags": list(set_b - set_a),
            "removed_tags": list(set_a - set_b),
            "common_tags": list(set_a & set_b)
        }

    async def bulk_compare_versions(
        self,
        version_list: List[VersionMetadata]
    ) -> Dict[str, Any]:
        """
        複数バージョンの一括比較

        Args:
            version_list: 比較対象バージョンリスト

        Returns:
            一括比較結果
        """
        try:
            if len(version_list) < 2:
                return {"error": "At least 2 versions required for comparison"}

            # バージョンでソート
            sorted_versions = sorted(version_list, key=lambda v: v.created_at)

            comparison_matrix = {}
            trend_analysis = {
                "version_progression": [],
                "metric_trends": {},
                "improvement_timeline": []
            }

            # 順次比較
            for i in range(len(sorted_versions) - 1):
                current = sorted_versions[i]
                next_version = sorted_versions[i + 1]

                comparison = await self.compare_versions(current, next_version)
                comparison_key = f"{current.version}_vs_{next_version.version}"
                comparison_matrix[comparison_key] = comparison

                # トレンド分析
                if "metrics_comparison" in comparison:
                    for metric_name, metric_data in comparison["metrics_comparison"].items():
                        if isinstance(metric_data, dict) and "improvement" in metric_data:
                            if metric_name not in trend_analysis["metric_trends"]:
                                trend_analysis["metric_trends"][metric_name] = []

                            trend_analysis["metric_trends"][metric_name].append({
                                "version": next_version.version,
                                "value": metric_data["value_b"],
                                "improvement": metric_data["improvement"]
                            })

            return {
                "comparison_matrix": comparison_matrix,
                "trend_analysis": trend_analysis,
                "total_versions": len(version_list),
                "comparison_timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Bulk version comparison failed: {e}")
            return {"error": str(e)}