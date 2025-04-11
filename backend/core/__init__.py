"""
Core モジュール
ウェルネス分析プラットフォームのコア機能を提供します。
"""

from .config import get_settings
from .common_logger import get_logger, setup_application_logging
from .exceptions import (
    AppError,
    DataError, DataValidationError, DataProcessingError, DataPreprocessingError, DataNotFoundError,
    AnalysisError, CalculationError, ModelError, WellnessScoreError, VisualizationError,
    AuthError, AuthenticationError, AuthorizationError, TokenError, UserNotFoundError,
    APIError, RateLimitError, ExternalAPIError,
    DatabaseError, FirestoreError, RedisError,
    BusinessError, SubscriptionError, ComplianceError,
    FileError, FileNotFoundError, FileUploadError, FileProcessingError,
    AsyncError, TaskExecutionError, TimeoutError,
    CacheError, CacheConnectionError, CacheOperationError,
    ConfigError, ConfigValidationError, ConfigLoadError
)
from .firebase_client import (
    FirebaseClientInterface,
    FirebaseClient,
    MockFirebaseClient,
    get_firebase_client,
    get_firestore_client
)

# 新しい依存性注入システムのインポート
from .di import get_container, inject, DIContainer
from .di_config import setup_di_container, reset_di_container, get_wellness_repository

# デザインパターンのインポート
from .patterns import (
    Singleton,
    LazyImport,
    lazy_property,
    reset_singleton,
    reset_all_singletons
)

__all__ = [
    # 設定関連
    'get_settings',

    # ロギング関連
    'get_logger',
    'setup_application_logging',

    # 例外関連
    'AppError',
    'DataError', 'DataValidationError', 'DataProcessingError', 'DataPreprocessingError', 'DataNotFoundError',
    'AnalysisError', 'CalculationError', 'ModelError', 'WellnessScoreError', 'VisualizationError',
    'AuthError', 'AuthenticationError', 'AuthorizationError', 'TokenError', 'UserNotFoundError',
    'APIError', 'RateLimitError', 'ExternalAPIError',
    'DatabaseError', 'FirestoreError', 'RedisError',
    'BusinessError', 'SubscriptionError', 'ComplianceError',
    'FileError', 'FileNotFoundError', 'FileUploadError', 'FileProcessingError',
    'AsyncError', 'TaskExecutionError', 'TimeoutError',
    'CacheError', 'CacheConnectionError', 'CacheOperationError',
    'ConfigError', 'ConfigValidationError', 'ConfigLoadError',

    # Firebase関連
    'FirebaseClientInterface',
    'FirebaseClient',
    'MockFirebaseClient',
    'get_firebase_client',
    'get_firestore_client',

    # 依存性注入関連
    'DIContainer',
    'get_container',
    'inject',
    'setup_di_container',
    'reset_di_container',
    'get_wellness_repository',

    # デザインパターン関連
    'Singleton',
    'LazyImport',
    'lazy_property',
    'reset_singleton',
    'reset_all_singletons',
]
