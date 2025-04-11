"""
ウェルネスエンティティ
ウェルネススコアと関連メトリクスのドメインモデルを定義します。
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class WellnessDimension(str, Enum):
    """ウェルネスの次元を表す列挙型"""
    PHYSICAL = "physical"
    MENTAL = "mental"
    FINANCIAL = "financial"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"
    OCCUPATIONAL = "occupational"


class WellnessScoreCategory(str, Enum):
    """ウェルネススコアのカテゴリを表す列挙型"""
    CRITICAL = "critical"  # 0-20
    POOR = "poor"          # 21-40
    FAIR = "fair"          # 41-60
    GOOD = "good"          # 61-80
    EXCELLENT = "excellent"  # 81-100


@dataclass
class WellnessMetric:
    """
    ウェルネスメトリック
    特定の指標に対するウェルネスの測定値
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    value: float = 0.0
    dimension: WellnessDimension = WellnessDimension.PHYSICAL
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_recent(self) -> bool:
        """過去30日以内のデータかどうか"""
        return (datetime.now() - self.timestamp).days <= 30


@dataclass
class WellnessScore:
    """
    ウェルネススコア
    ユーザーや企業のウェルネス状態を表す総合スコア
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    company_id: Optional[str] = None
    score: float = 0.0
    dimension_scores: Dict[WellnessDimension, float] = field(default_factory=dict)
    metrics: List[WellnessMetric] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """初期化後の処理"""
        # dimension_scoresのデフォルト値設定
        for dim in WellnessDimension:
            if dim not in self.dimension_scores:
                self.dimension_scores[dim] = 0.0

    @property
    def category(self) -> WellnessScoreCategory:
        """スコアに基づくカテゴリを返す"""
        if self.score <= 20:
            return WellnessScoreCategory.CRITICAL
        elif self.score <= 40:
            return WellnessScoreCategory.POOR
        elif self.score <= 60:
            return WellnessScoreCategory.FAIR
        elif self.score <= 80:
            return WellnessScoreCategory.GOOD
        else:
            return WellnessScoreCategory.EXCELLENT

    def add_metric(self, metric: WellnessMetric) -> None:
        """
        メトリックを追加し、スコアを再計算

        Args:
            metric: 追加するメトリック
        """
        self.metrics.append(metric)
        self._recalculate_scores()

    def _recalculate_scores(self) -> None:
        """ディメンションスコアと総合スコアを再計算"""
        # ディメンションごとにメトリックを集計
        dimension_metrics: Dict[WellnessDimension, List[WellnessMetric]] = {}
        for dim in WellnessDimension:
            dimension_metrics[dim] = []

        for metric in self.metrics:
            dimension_metrics[metric.dimension].append(metric)

        # 各ディメンションのスコアを計算
        for dim, metrics in dimension_metrics.items():
            if metrics:
                # 単純な平均を使用（より洗練された計算は実装によって異なる）
                self.dimension_scores[dim] = sum(m.value for m in metrics) / len(metrics)
            else:
                self.dimension_scores[dim] = 0.0

        # 総合スコアを計算（すべてのディメンションの平均）
        if self.dimension_scores:
            self.score = sum(self.dimension_scores.values()) / len(self.dimension_scores)
        else:
            self.score = 0.0


@dataclass
class WellnessRecommendation:
    """
    ウェルネス改善のための推奨事項
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    description: str = ""
    priority: int = 0  # 優先度（1-10）
    target_dimensions: List[WellnessDimension] = field(default_factory=list)
    impact_estimate: float = 0.0  # 推定される影響度（0-1）
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_high_priority(self) -> bool:
        """高優先度の推奨事項かどうか"""
        return self.priority >= 8