"""
連合学習アダプターモジュール

このモジュールは、連合学習システムと他のシステムコンポーネント（コアモジュールや分析モジュール）
との統合を容易にするためのアダプターを提供します。
"""

from .health_impact_adapter import HealthImpactAdapter
from .core_integration import CoreModelIntegration

__all__ = [
    'HealthImpactAdapter',
    'CoreModelIntegration'
]