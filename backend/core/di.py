"""
依存性注入システム
アプリケーションコンポーネント間の疎結合化とテスト容易性向上のためのDIコンテナ
"""
from typing import Dict, Any, Type, TypeVar, Callable, Optional, cast
import inspect
from functools import wraps
from .common_logger import get_logger

logger = get_logger(__name__)

# インターフェースと実装の型定義
T = TypeVar('T')
ServiceFactory = Callable[[], T]


class DIContainer:
    """
    依存性注入コンテナ
    コンポーネント間の依存関係を管理し、疎結合なアーキテクチャをサポート
    """
    _instance = None
    _services: Dict[str, Any] = {}
    _factories: Dict[str, ServiceFactory] = {}
    _singletons: Dict[str, bool] = {}

    def __new__(cls):
        """シングルトンパターンによるインスタンス生成"""
        if cls._instance is None:
            cls._instance = super(DIContainer, cls).__new__(cls)
        return cls._instance

    def register(self, interface: Type[T], implementation: Type[T], singleton: bool = True) -> None:
        """
        インターフェースと実装の登録

        Args:
            interface: 依存性の抽象インターフェース
            implementation: インターフェースの具体的な実装
            singleton: Trueの場合、シングルトンとして管理される
        """
        interface_name = self._get_type_name(interface)

        def factory() -> T:
            try:
                return implementation()
            except Exception as e:
                logger.error(f"Failed to create implementation for {interface_name}: {str(e)}")
                raise

        self._factories[interface_name] = factory
        self._singletons[interface_name] = singleton
        logger.debug(f"Registered {interface_name} with implementation {implementation.__name__}")

    def register_instance(self, interface: Type[T], instance: T) -> None:
        """
        既存インスタンスの登録

        Args:
            interface: 依存性の抽象インターフェース
            instance: インターフェースの実装インスタンス
        """
        interface_name = self._get_type_name(interface)
        self._services[interface_name] = instance
        logger.debug(f"Registered instance for {interface_name}")

    def register_factory(self, interface: Type[T], factory: ServiceFactory, singleton: bool = True) -> None:
        """
        ファクトリ関数の登録

        Args:
            interface: 依存性の抽象インターフェース
            factory: インスタンスを生成するファクトリ関数
            singleton: Trueの場合、シングルトンとして管理される
        """
        interface_name = self._get_type_name(interface)
        self._factories[interface_name] = factory
        self._singletons[interface_name] = singleton
        logger.debug(f"Registered factory for {interface_name}")

    def get(self, interface: Type[T]) -> T:
        """
        依存性の解決

        Args:
            interface: 取得するインターフェース

        Returns:
            インターフェースの実装

        Raises:
            KeyError: インターフェースが登録されていない場合
        """
        interface_name = self._get_type_name(interface)

        # 既存のインスタンスを確認
        if interface_name in self._services:
            return cast(T, self._services[interface_name])

        # ファクトリが登録されているか確認
        if interface_name not in self._factories:
            raise KeyError(f"No implementation registered for {interface_name}")

        # インスタンスの生成
        instance = self._factories[interface_name]()

        # シングルトンの場合はキャッシュ
        if self._singletons.get(interface_name, False):
            self._services[interface_name] = instance

        return cast(T, instance)

    def inject(self, interface: Type[T]) -> T:
        """
        依存性の注入用デコレータファクトリ

        使用例:
            @inject(IUserRepository)
            def function(user_repo):
                # user_repoにはIUserRepositoryの実装が注入される

        Args:
            interface: 注入するインターフェース

        Returns:
            デコレータ関数
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = self.get(interface)
                return func(instance, *args, **kwargs)
            return wrapper
        return decorator

    def clear(self) -> None:
        """
        登録された全てのサービスとファクトリをクリア（主にテスト用）
        """
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        logger.debug("Cleared all registrations from DI container")

    def reset_instance(self, interface: Type[T]) -> None:
        """
        特定のインターフェースのインスタンスをリセット（主にテスト用）

        Args:
            interface: リセットするインターフェース
        """
        interface_name = self._get_type_name(interface)
        if interface_name in self._services:
            del self._services[interface_name]
            logger.debug(f"Reset instance for {interface_name}")

    def _get_type_name(self, cls: Type) -> str:
        """型の完全修飾名を取得"""
        if hasattr(cls, "__name__"):
            module = cls.__module__
            name = cls.__name__
            return f"{module}.{name}"
        return str(cls)


# シングルトンインスタンス
_container = DIContainer()


def get_container() -> DIContainer:
    """
    DIコンテナのシングルトンインスタンスを取得

    Returns:
        DIコンテナのインスタンス
    """
    return _container


def inject(interface: Type[T]) -> Callable:
    """
    依存性の注入用デコレータ（グローバルショートカット）

    使用例:
        @inject(IUserRepository)
        def function(user_repo):
            # user_repoにはIUserRepositoryの実装が注入される

    Args:
        interface: 注入するインターフェース

    Returns:
        デコレータ関数
    """
    container = get_container()
    return container.inject(interface)