"""
依存性注入のRedis関連モジュール

このモジュールはRedisリポジトリとサービスを依存性注入コンテナに登録します。
クリーンアーキテクチャのインフラストラクチャ層のコンポーネントを提供します。
"""

from functools import lru_cache
from typing import Optional, Dict, Any, Callable

from injector import Module, provider, singleton

from backend.config.settings import get_settings
from backend.infrastructure.redis.redis_service import RedisService, create_redis_service
from backend.infrastructure.redis.redis_user_repository import RedisUserRepository, create_redis_user_repository
from backend.core.di import DIContainer
from backend.domain.repositories.user_repository import UserRepositoryInterface
from backend.domain.repositories.wellness_repository import WellnessRepositoryInterface
from backend.domain.repositories.company_repository import CompanyRepositoryInterface

from backend.infrastructure.redis.redis_wellness_repository import create_redis_wellness_repository
from backend.infrastructure.redis.redis_company_repository import create_redis_company_repository


class RedisModule(Module):
    """
    RedisサービスをDIコンテナに登録するためのモジュール

    アプリケーション設定からRedisの接続情報を取得し、
    RedisServiceのシングルトンインスタンスを提供します。
    """

    @provider
    @singleton
    def provide_redis_service(self) -> RedisService:
        """
        Redisサービスのインスタンスを提供します

        設定からRedisの接続情報を取得し、RedisServiceの
        シングルトンインスタンスを作成して返します。

        Returns:
            RedisService: 設定されたRedisサービスのインスタンス
        """
        settings = get_settings()

        # 設定からRedis接続情報を取得
        redis_config = self._get_redis_config(settings.redis)

        # Redisサービスを作成して返す
        return create_redis_service(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            password=redis_config.get("password"),
            use_ssl=redis_config.get("use_ssl", False),
            default_ttl=redis_config.get("default_ttl", 3600)
        )

    @lru_cache(maxsize=1)
    def _get_redis_config(self, redis_settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        設定からRedis構成を取得します

        Args:
            redis_settings: 設定から取得したRedis設定

        Returns:
            Dict[str, Any]: Redis接続パラメータの辞書
        """
        if redis_settings is None:
            # デフォルト設定を返す
            return {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "password": None,
                "use_ssl": False,
                "default_ttl": 3600
            }

        return redis_settings


def register_redis_module(
    container: DIContainer,
    redis_url: Optional[str] = None,
    ttl: int = 3600,
) -> None:
    """
    Redisモジュールをコンテナに登録

    Args:
        container: DIコンテナ
        redis_url: Redis接続URL（デフォルト: 設定から取得）
        ttl: キャッシュのデフォルトTTL（秒）
    """
    settings = get_settings()

    # Redis URLが指定されていない場合は設定から取得
    if redis_url is None:
        redis_url = settings.REDIS_URL

    # Redisクライアントの登録
    container.register(
        RedisClient,
        lambda: RedisClient(redis_url)
    )

    # Redisサービスの登録
    container.register(
        RedisService,
        lambda c: create_redis_service(c.resolve(RedisClient))
    )

    # Redisユーザーリポジトリの登録 (メインリポジトリがある場合にのみ)
    def _create_redis_user_repository(c: DIContainer) -> Optional[RedisUserRepository]:
        try:
            # メインのユーザーリポジトリを解決
            main_repo = c.resolve(UserRepositoryInterface)
            redis_service = c.resolve(RedisService)

            # Redisユーザーリポジトリの作成
            return create_redis_user_repository(
                redis_service=redis_service,
                main_repository=main_repo,
                ttl=ttl
            )
        except Exception as e:
            # 依存関係が解決できない場合はNoneを返す
            return None

    # コンテナに登録
    container.register_factory(
        "redis_user_repository",
        _create_redis_user_repository
    )


def register_redis_user_repository_as_main(container: DIContainer) -> None:
    """
    RedisユーザーリポジトリをメインのUserRepositoryInterfaceとして登録します。
    これによりアプリケーションはキャッシュ層を通じてユーザーデータにアクセスします。

    Args:
        container: DIコンテナ
    """
    # 既存のRedisユーザーリポジトリがある場合はそれを使用
    if container.has("redis_user_repository"):
        redis_repo = container.resolve("redis_user_repository")
        if redis_repo:
            container.register(UserRepositoryInterface, lambda: redis_repo)
            return

    # 警告: この関数を実行する前に必ずregister_redis_moduleを呼び出してください
    raise ValueError("Redisモジュールが登録されていないか、RedisUserRepositoryの作成に失敗しました")


def setup_redis_repositories(container: DIContainer) -> None:
    """
    RedisリポジトリをDIコンテナに登録します

    Args:
        container: DIコンテナインスタンス
    """
    # Redisサービスの設定
    setup_redis_service(container)

    # Redisユーザーリポジトリの設定
    setup_redis_user_repository(container)

    # Redisウェルネスリポジトリの設定
    setup_redis_wellness_repository(container)

    # Redis企業リポジトリの設定
    setup_redis_company_repository(container)


def setup_redis_company_repository(container: DIContainer) -> None:
    """
    Redis企業リポジトリをDIコンテナに登録します

    Args:
        container: DIコンテナインスタンス
    """
    try:
        # RedisServiceの取得
        redis_service = container.get(RedisService)

        # メインの企業リポジトリを取得
        main_repo = container.get(CompanyRepositoryInterface)

        # Redis企業リポジトリの作成
        redis_repo = create_redis_company_repository(
            redis_service=redis_service,
            main_repository=main_repo
        )

        # 一時的に元のRepositoryをバックアップ
        container.register_alias(CompanyRepositoryInterface, "original_company_repository")

        # Redis企業リポジトリをメインのCompanyRepositoryInterfaceとして登録
        container.register(CompanyRepositoryInterface, lambda: redis_repo)

        logger.info("Redis企業リポジトリを登録しました")
    except Exception as e:
        logger.error(f"Redis企業リポジトリの設定中にエラーが発生しました: {e}")
        # 既存のリポジトリをそのまま使用
        pass