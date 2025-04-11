"""
リポジトリ関連の例外
データアクセスに関連するエラーを表す例外クラスを定義します。
"""
from typing import Optional


class RepositoryError(Exception):
    """リポジトリ操作の基本例外クラス"""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause


class UserRepositoryError(RepositoryError):
    """ユーザーリポジトリの操作に関するエラー"""
    pass


class WellnessRepositoryError(RepositoryError):
    """ウェルネスリポジトリの操作に関するエラー"""
    pass


class EntityNotFoundError(RepositoryError):
    """エンティティが見つからないエラー"""

    def __init__(self, entity_type: str, entity_id: str):
        message = f"{entity_type} ID: {entity_id} が見つかりません"
        super().__init__(message)
        self.entity_type = entity_type
        self.entity_id = entity_id


class EntityAlreadyExistsError(RepositoryError):
    """エンティティが既に存在するエラー"""

    def __init__(self, entity_type: str, entity_id: str):
        message = f"{entity_type} ID: {entity_id} は既に存在します"
        super().__init__(message)
        self.entity_type = entity_type
        self.entity_id = entity_id


class DatabaseConnectionError(RepositoryError):
    """データベース接続エラー"""
    pass


class InvalidQueryError(RepositoryError):
    """無効なクエリエラー"""
    pass


class TransactionError(RepositoryError):
    """トランザクションエラー"""
    pass


class ConstraintViolationError(RepositoryError):
    """制約違反エラー"""
    pass


class OptimisticLockError(RepositoryError):
    """楽観的ロックエラー（データが既に変更されている）"""
    pass