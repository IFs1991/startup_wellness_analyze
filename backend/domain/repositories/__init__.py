"""
リポジトリパッケージ
データアクセスの抽象化を提供するリポジトリインターフェースを定義します。
"""

from domain.repositories.user_repository import UserRepositoryInterface
from domain.repositories.wellness_repository import WellnessRepositoryInterface

__all__ = [
    'UserRepositoryInterface',
    'WellnessRepositoryInterface'
]