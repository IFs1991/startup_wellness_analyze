"""
ドメインモデルパッケージ
エンティティと値オブジェクトを提供します。
"""

from domain.models.user import (
    User,
    UserCredentials,
    UserProfile,
    UserRole,
    MFAType,
    TokenData,
    AuthToken
)

from domain.models.wellness import (
    ScoreCategory,
    ScoreSeverity,
    ScoreMetric,
    WellnessScore,
    ScoreHistory,
    RecommendationAction,
    RecommendationPlan
)

__all__ = [
    # ユーザー関連
    'User',
    'UserCredentials',
    'UserProfile',
    'UserRole',
    'MFAType',
    'TokenData',
    'AuthToken',

    # ウェルネス関連
    'ScoreCategory',
    'ScoreSeverity',
    'ScoreMetric',
    'WellnessScore',
    'ScoreHistory',
    'RecommendationAction',
    'RecommendationPlan'
]