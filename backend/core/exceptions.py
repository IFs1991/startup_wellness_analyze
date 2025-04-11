"""
共通例外モジュール
アプリケーション全体で一貫した例外階層を提供します。
"""
from typing import Optional, Dict, Any, List, Union


class AppError(Exception):
    """
    アプリケーション基底例外クラス。
    すべてのカスタム例外はこのクラスを継承します。
    """
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        detail: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            message: エラーメッセージ
            status_code: HTTPステータスコード
            detail: 追加のエラー詳細
        """
        self.message = message
        self.status_code = status_code
        self.detail = detail or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """例外情報を辞書形式で返却"""
        result = {
            "error": self.__class__.__name__,
            "message": self.message,
            "status_code": self.status_code
        }
        if self.detail:
            result["detail"] = self.detail
        return result


# データ関連例外
class DataError(AppError):
    """データ処理に関する基底例外クラス"""
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        detail: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code, detail)


class DataValidationError(DataError):
    """データ検証エラー"""
    def __init__(
        self,
        message: str,
        fields: Optional[Dict[str, List[str]]] = None,
        status_code: int = 400,
        detail: Optional[Dict[str, Any]] = None
    ):
        detail = detail or {}
        if fields:
            detail["fields"] = fields
        super().__init__(message, status_code, detail)


class DataProcessingError(DataError):
    """データ処理エラー"""
    pass


class DataPreprocessingError(DataError):
    """データ前処理エラー"""
    pass


class DataNotFoundError(DataError):
    """データ未検出エラー"""
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        status_code: int = 404,
        detail: Optional[Dict[str, Any]] = None
    ):
        detail = detail or {}
        if resource_type:
            detail["resource_type"] = resource_type
        if resource_id:
            detail["resource_id"] = resource_id
        super().__init__(message, status_code, detail)


# 分析関連例外
class AnalysisError(AppError):
    """分析処理に関する基底例外クラス"""
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        detail: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code, detail)


class CalculationError(AnalysisError):
    """計算処理エラー"""
    pass


class ModelError(AnalysisError):
    """モデル関連エラー"""
    pass


class WellnessScoreError(AnalysisError):
    """ウェルネススコア計算エラー"""
    pass


class VisualizationError(AnalysisError):
    """可視化処理エラー"""
    pass


# 認証/認可関連例外
class AuthError(AppError):
    """認証・認可に関する基底例外クラス"""
    def __init__(
        self,
        message: str,
        status_code: int = 401,
        detail: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code, detail)


class AuthenticationError(AuthError):
    """認証エラー"""
    pass


class AuthorizationError(AuthError):
    """認可エラー"""
    def __init__(
        self,
        message: str,
        status_code: int = 403,
        detail: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code, detail)


class UserNotFoundError(AuthError):
    """ユーザーが見つからないエラー"""
    def __init__(
        self,
        message: str,
        status_code: int = 404,
        detail: Optional[Dict[str, Any]] = None
    ):
        detail = detail or {}
        super().__init__(message, status_code, detail)


class CompanyNotFoundError(DataNotFoundError):
    """企業が見つからないエラー"""
    def __init__(
        self,
        message: str = "指定された企業が見つかりません",
        company_id: Optional[str] = None,
        status_code: int = 404,
        detail: Optional[Dict[str, Any]] = None
    ):
        detail = detail or {}
        if company_id:
            detail["company_id"] = company_id
        super().__init__(message, "company", company_id, status_code, detail)


class CompanyAlreadyExistsError(DataError):
    """企業が既に存在するエラー"""
    def __init__(
        self,
        message: str = "指定された企業は既に存在しています",
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
        status_code: int = 409,
        detail: Optional[Dict[str, Any]] = None
    ):
        detail = detail or {}
        if company_id:
            detail["company_id"] = company_id
        if company_name:
            detail["company_name"] = company_name
        super().__init__(message, status_code, detail)


class TokenError(AuthError):
    """トークン関連エラー"""
    pass


# API関連例外
class APIError(AppError):
    """API処理に関する基底例外クラス"""
    pass


class RateLimitError(APIError):
    """レート制限エラー"""
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        status_code: int = 429,
        detail: Optional[Dict[str, Any]] = None
    ):
        detail = detail or {}
        if retry_after:
            detail["retry_after"] = retry_after
        super().__init__(message, status_code, detail)


class ExternalAPIError(APIError):
    """外部API連携エラー"""
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        status_code: int = 502,
        detail: Optional[Dict[str, Any]] = None
    ):
        detail = detail or {}
        if service_name:
            detail["service_name"] = service_name
        super().__init__(message, status_code, detail)


# データベース関連例外
class DatabaseError(AppError):
    """データベース操作に関する基底例外クラス"""
    pass


class FirestoreError(DatabaseError):
    """Firestore操作エラー"""
    pass


class RedisError(DatabaseError):
    """Redis操作エラー"""
    pass


# ビジネスロジック関連例外
class BusinessError(AppError):
    """ビジネスロジックに関する基底例外クラス"""
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        detail: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code, detail)


class SubscriptionError(BusinessError):
    """サブスクリプション関連エラー"""
    pass


class ComplianceError(BusinessError):
    """コンプライアンス関連エラー"""
    pass


# ファイル操作関連例外
class FileError(AppError):
    """ファイル操作に関する基底例外クラス"""
    pass


class FileNotFoundError(FileError):
    """ファイル未検出エラー"""
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        status_code: int = 404,
        detail: Optional[Dict[str, Any]] = None
    ):
        detail = detail or {}
        if file_path:
            detail["file_path"] = file_path
        super().__init__(message, status_code, detail)


class FileUploadError(FileError):
    """ファイルアップロードエラー"""
    pass


class FileProcessingError(FileError):
    """ファイル処理エラー"""
    pass


# 非同期処理関連例外
class AsyncError(AppError):
    """非同期処理に関する基底例外クラス"""
    pass


class TaskExecutionError(AsyncError):
    """タスク実行エラー"""
    pass


class TimeoutError(AsyncError):
    """タイムアウトエラー"""
    def __init__(
        self,
        message: str,
        timeout: Optional[float] = None,
        status_code: int = 408,
        detail: Optional[Dict[str, Any]] = None
    ):
        detail = detail or {}
        if timeout:
            detail["timeout"] = timeout
        super().__init__(message, status_code, detail)


# キャッシュ関連例外
class CacheError(AppError):
    """キャッシュ操作に関する基底例外クラス"""
    pass


class CacheConnectionError(CacheError):
    """キャッシュ接続エラー"""
    pass


class CacheOperationError(CacheError):
    """キャッシュ操作エラー"""
    pass


# 設定関連例外
class ConfigError(AppError):
    """設定に関する基底例外クラス"""
    pass


class ConfigValidationError(ConfigError):
    """設定検証エラー"""
    pass


class ConfigLoadError(ConfigError):
    """設定読み込みエラー"""
    pass