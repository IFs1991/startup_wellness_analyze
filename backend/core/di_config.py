"""
DIコンテナ設定
依存性注入コンテナの初期設定と登録を行います。
"""
from core.common_logger import get_logger
from core.di import get_container
from core.firebase_client import FirebaseClientInterface, get_firebase_client
from domain.repositories.user_repository import UserRepositoryInterface
from domain.repositories.wellness_repository import WellnessRepositoryInterface
from infrastructure.firebase.firebase_wellness_repository import FirebaseWellnessRepository
from typing import Any, Optional

# 循環参照を避けるための型導入
from typing import Protocol

# 循環インポート対策のためのインターフェース定義
class AuthManagerInterface(Protocol):
    """AuthManager用プロトコル"""
    def get_user_data(self, user_id: str) -> Any: ...
    def validate_token(self, token: str) -> Any: ...

class ComplianceManagerInterface(Protocol):
    """ComplianceManager用プロトコル"""
    def log_event(self, event: Any) -> Any: ...
    def check_compliance(self, user_id: str, action: str) -> bool: ...

class SubscriptionManagerInterface(Protocol):
    """SubscriptionManager用プロトコル"""
    def get_user_subscription(self, user_id: str) -> Any: ...
    def check_subscription_status(self, user_id: str) -> bool: ...

# ウェルネススコアユースケースインターフェース定義
class WellnessScoreUseCaseInterface(Protocol):
    """WellnessScoreUseCase用プロトコル"""
    async def calculate_wellness_score(self, company_id: str, industry: str, stage: str, calculation_date: Optional[Any] = None) -> Any: ...
    async def get_score_history(self, company_id: str, time_period: str, limit: int) -> Any: ...

logger = get_logger(__name__)


def setup_di_container(use_mock: bool = False) -> None:
    """
    DIコンテナのセットアップ
    アプリケーション起動時に呼び出されます。

    Args:
        use_mock: モックを使用するかどうか（テスト用）
    """
    container = get_container()

    # 環境変数に基づいて設定
    import os
    environment = os.environ.get("ENVIRONMENT", "development")

    logger.info(f"DIコンテナをセットアップします（環境: {environment}）")

    # Firebaseクライアントの登録
    def firebase_client_factory():
        return get_firebase_client(use_mock)

    container.register_factory(
        FirebaseClientInterface,
        firebase_client_factory,
        singleton=True
    )

    # ウェルネスリポジトリの登録
    def wellness_repository_factory():
        firebase_client = container.get(FirebaseClientInterface)
        return FirebaseWellnessRepository(firebase_client)

    container.register_factory(
        WellnessRepositoryInterface,
        wellness_repository_factory,
        singleton=True
    )

    # ウェルネススコアユースケースの登録
    def wellness_score_usecase_factory():
        from usecases.wellness_score_usecase import WellnessScoreUseCase
        wellness_repository = container.get(WellnessRepositoryInterface)
        firebase_client = container.get(FirebaseClientInterface)
        return WellnessScoreUseCase(wellness_repository, firebase_client)

    container.register_factory(
        WellnessScoreUseCaseInterface,
        wellness_score_usecase_factory,
        singleton=True
    )

    # AuthManagerの登録
    def auth_manager_factory():
        from core.auth_manager import AuthManager
        return AuthManager()

    container.register_factory(
        AuthManagerInterface,
        auth_manager_factory,
        singleton=True
    )

    # ComplianceManagerの登録
    def compliance_manager_factory():
        from core.compliance_manager import ComplianceManager
        # AuthManagerを直接DIコンテナから取得せず、必要時に取得する設計に
        return ComplianceManager()

    container.register_factory(
        ComplianceManagerInterface,
        compliance_manager_factory,
        singleton=True
    )

    # SubscriptionManagerの登録
    def subscription_manager_factory():
        from core.subscription_manager import SubscriptionManager
        from service.firestore.client import FirestoreService
        # 必要なサービスを用意
        firestore_service = FirestoreService()
        # AuthManagerを直接DIコンテナから取得せず、必要時に取得する設計に
        return SubscriptionManager(firestore_service, None)

    container.register_factory(
        SubscriptionManagerInterface,
        subscription_manager_factory,
        singleton=True
    )

    # ユーザーリポジトリの登録（将来的に実装）
    # 当面は auth_manager.py が既存の実装を提供

    logger.info("DIコンテナの設定が完了しました")


def reset_di_container() -> None:
    """
    DIコンテナのリセット
    主にテスト時に使用されます。
    """
    container = get_container()
    container.clear()
    logger.info("DIコンテナをリセットしました")


def get_wellness_repository() -> WellnessRepositoryInterface:
    """
    ウェルネスリポジトリの取得
    レガシーコードとの互換性のためのヘルパー関数

    Returns:
        ウェルネスリポジトリのインスタンス
    """
    container = get_container()
    return container.get(WellnessRepositoryInterface)


def get_wellness_score_usecase_from_di() -> WellnessScoreUseCaseInterface:
    """
    ウェルネススコアユースケースの取得
    レガシーコードとの互換性のためのヘルパー関数

    Returns:
        ウェルネススコアユースケースのインスタンス
    """
    container = get_container()
    return container.get(WellnessScoreUseCaseInterface)


def get_auth_manager_from_di() -> AuthManagerInterface:
    """
    AuthManagerの取得
    レガシーコードとの互換性のためのヘルパー関数

    Returns:
        AuthManagerのインスタンス
    """
    container = get_container()
    return container.get(AuthManagerInterface)


def get_compliance_manager_from_di() -> ComplianceManagerInterface:
    """
    ComplianceManagerの取得
    レガシーコードとの互換性のためのヘルパー関数

    Returns:
        ComplianceManagerのインスタンス
    """
    container = get_container()
    return container.get(ComplianceManagerInterface)


def get_subscription_manager_from_di() -> SubscriptionManagerInterface:
    """
    SubscriptionManagerの取得
    レガシーコードとの互換性のためのヘルパー関数

    Returns:
        SubscriptionManagerのインスタンス
    """
    container = get_container()
    return container.get(SubscriptionManagerInterface)