import redis
from .database.user_repository import SQLUserRepository
from .redis.redis_user_repository import RedisUserRepository
from core.di import DIContainer

def configure_infrastructure(container: DIContainer) -> None:
    """
    インフラストラクチャコンポーネントをDIコンテナに登録します。

    Args:
        container: 設定対象のDIコンテナ
    """
    # データベースリポジトリの登録
    container.register("user_repository_sql", SQLUserRepository)

    # Redisクライアントの設定と登録
    redis_client = redis.Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True
    )
    container.register_instance("redis_client", redis_client)

    # キャッシュリポジトリの登録（SQLリポジトリにデコレートする）
    container.register_factory(
        "user_repository",
        lambda c: RedisUserRepository(
            redis_client=c.resolve("redis_client"),
            main_repository=c.resolve("user_repository_sql")
        )
    )