"""
ウェルネスドメインモデル
ウェルネススコアとその関連データを表現するエンティティと値オブジェクト
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class ScoreCategory(str, Enum):
    """スコアカテゴリ"""
    FINANCIAL = "financial"
    HEALTH = "health"
    WORK_LIFE_BALANCE = "work_life_balance"
    TEAM_CULTURE = "team_culture"
    GROWTH = "growth"
    INNOVATION = "innovation"
    LEADERSHIP = "leadership"
    RESILIENCE = "resilience"


class ScoreSeverity(str, Enum):
    """スコア重大度"""
    CRITICAL = "critical"  # 0-20
    POOR = "poor"          # 21-40
    FAIR = "fair"          # 41-60
    GOOD = "good"          # 61-80
    EXCELLENT = "excellent"  # 81-100


@dataclass
class ScoreMetric:
    """スコアメトリック（測定値）"""
    name: str
    value: float
    weight: float = 1.0
    category: ScoreCategory = ScoreCategory.FINANCIAL
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def get_weighted_value(self) -> float:
        """重み付けされた値を取得"""
        return self.value * self.weight


@dataclass
class WellnessScore:
    """ウェルネススコア（集計されたスコア）"""
    id: str
    company_id: str
    total_score: float
    category_scores: Dict[ScoreCategory, float]
    metrics: List[ScoreMetric] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None

    @property
    def severity(self) -> ScoreSeverity:
        """スコアの重大度を取得"""
        if self.total_score >= 81:
            return ScoreSeverity.EXCELLENT
        elif self.total_score >= 61:
            return ScoreSeverity.GOOD
        elif self.total_score >= 41:
            return ScoreSeverity.FAIR
        elif self.total_score >= 21:
            return ScoreSeverity.POOR
        else:
            return ScoreSeverity.CRITICAL

    def get_category_score(self, category: ScoreCategory) -> float:
        """特定カテゴリのスコアを取得"""
        return self.category_scores.get(category, 0.0)

    def is_healthy(self) -> bool:
        """健全なスコアかどうか（GOOD以上）"""
        return self.total_score >= 61

    def needs_attention(self) -> bool:
        """注意が必要なスコアかどうか（FAIR以下）"""
        return self.total_score <= 60

    def needs_immediate_action(self) -> bool:
        """即時対応が必要なスコアかどうか（POOR以下）"""
        return self.total_score <= 40


@dataclass
class ScoreHistory:
    """スコア履歴"""
    company_id: str
    scores: List[WellnessScore] = field(default_factory=list)
    time_period: str = "monthly"  # monthly, quarterly, yearly

    def get_trend(self) -> float:
        """スコアの傾向（直近のスコア変化率）を取得"""
        if len(self.scores) < 2:
            return 0.0

        # 最新と1つ前のスコアを取得
        latest = self.scores[0].total_score
        previous = self.scores[1].total_score

        # 変化がない場合は0を返す
        if previous == 0:
            return 0.0

        # 変化率を計算して返す
        return (latest - previous) / previous * 100


@dataclass
class RecommendationAction:
    """改善推奨アクション"""
    id: str
    title: str
    description: str
    category: ScoreCategory
    impact_level: int  # 1-5 (5が最高)
    effort_level: int  # 1-5 (5が最高)
    time_frame: str  # short, medium, long
    resources: List[str] = field(default_factory=list)

    @property
    def roi_score(self) -> float:
        """投資対効果スコア"""
        if self.effort_level == 0:
            return 0.0
        return self.impact_level / self.effort_level


@dataclass
class RecommendationPlan:
    """改善推奨プラン（複数のアクションをまとめたもの）"""
    company_id: str
    score_id: str
    actions: List[RecommendationAction] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    generated_by: Optional[str] = None

    def get_priority_actions(self, limit: int = 3) -> List[RecommendationAction]:
        """優先度の高いアクションを取得"""
        # ROIスコアでソートして上位を返す
        sorted_actions = sorted(
            self.actions,
            key=lambda x: x.roi_score,
            reverse=True
        )
        return sorted_actions[:limit]