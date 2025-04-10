"""
インフラストラクチャパッケージ
外部システムとの連携や永続化を担当するコンポーネントを提供します。
"""

from infrastructure.firebase import FirebaseWellnessRepository

__all__ = [
    'FirebaseWellnessRepository'
]