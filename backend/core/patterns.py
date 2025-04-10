"""
デザインパターン実装モジュール

アプリケーション全体で一貫して使用できる標準的なデザインパターンの実装を提供します。
"""
from functools import wraps
import inspect
from typing import Any, Dict, Type, TypeVar, Generic, Optional, Callable
from .common_logger import get_logger

logger = get_logger(__name__)

# 型変数
T = TypeVar('T')


class Singleton:
    """
    シングルトンパターンをクラスデコレータとして実装します。

    使用例:
        @Singleton
        class MyService:
            def __init__(self, config=None):
                self.config = config or {}
    """
    def __init__(self, cls: Type[T]):
        self._cls = cls
        self._instance = None
        self.__name__ = cls.__name__
        self.__doc__ = cls.__doc__
        self.__module__ = cls.__module__
        # オリジナルのクラスのメソッドシグネチャを保持
        self.__signature__ = inspect.signature(cls.__init__)

    def __call__(self, *args, **kwargs) -> T:
        """
        クラスの単一インスタンスを返します。
        最初の呼び出し時にのみインスタンスが作成されます。

        Returns:
            クラスの単一インスタンス
        """
        if self._instance is None:
            self._instance = self._cls(*args, **kwargs)
            logger.debug(f"シングルトンクラス {self.__name__} のインスタンスを生成しました")
        return self._instance

    def reset(self) -> None:
        """
        シングルトンインスタンスをリセットします。
        主にテスト目的で使用されます。
        """
        self._instance = None
        logger.debug(f"シングルトンクラス {self.__name__} のインスタンスをリセットしました")


class LazyImport:
    """
    循環インポートを回避するための遅延インポートユーティリティ。

    使用例:
        # モジュールレベルで定義
        DataPreprocessor = LazyImport('core.data_preprocessor', 'DataPreprocessor')

        # 使用時
        preprocessor = DataPreprocessor()  # この時点で実際にインポートされる
    """
    def __init__(self, module_path: str, class_name: str):
        """
        初期化メソッド

        Args:
            module_path: インポートするモジュールのパス
            class_name: モジュールからインポートするクラスの名前
        """
        self.module_path = module_path
        self.class_name = class_name
        self._cls = None

    def __call__(self, *args, **kwargs) -> Any:
        """
        遅延インポートされたクラスのインスタンスを生成

        Returns:
            インポートされたクラスのインスタンス
        """
        if self._cls is None:
            self._import()
        return self._cls(*args, **kwargs)

    def _import(self) -> None:
        """実際のインポート処理を行う"""
        try:
            module = __import__(self.module_path, fromlist=[self.class_name])
            self._cls = getattr(module, self.class_name)
            logger.debug(f"{self.module_path}.{self.class_name} を遅延インポートしました")
        except (ImportError, AttributeError) as e:
            logger.error(f"{self.module_path}.{self.class_name} の遅延インポートに失敗しました: {e}")
            raise

    def __getattr__(self, name: str) -> Any:
        """
        クラスのスタティックメソッドやプロパティへのアクセスをサポート

        Args:
            name: アクセスするメンバー名

        Returns:
            クラスメンバー
        """
        if self._cls is None:
            self._import()
        return getattr(self._cls, name)


def lazy_property(fn: Callable) -> property:
    """
    初回アクセス時にのみ値を計算するプロパティデコレータ。

    使用例:
        class Service:
            @lazy_property
            def expensive_resource(self):
                # 初回アクセス時にのみ実行される重い処理
                return ExpensiveResource()

    Args:
        fn: 値を計算する関数

    Returns:
        遅延評価するプロパティ
    """
    attr_name = '_lazy_' + fn.__name__

    @property
    @wraps(fn)
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


# テスト用ユーティリティ
singleton_registry: Dict[str, Singleton] = {}

def register_singleton(singleton_instance: Singleton) -> None:
    """シングルトンインスタンスをグローバルレジストリに登録"""
    singleton_registry[singleton_instance.__name__] = singleton_instance

def reset_singleton(cls_name: str) -> None:
    """
    名前によりシングルトンインスタンスをリセット

    Args:
        cls_name: リセットするシングルトンクラスの名前
    """
    if cls_name in singleton_registry:
        singleton_registry[cls_name].reset()
        logger.debug(f"シングルトン {cls_name} をリセットしました")
    else:
        logger.warning(f"シングルトン {cls_name} がレジストリに見つかりません")

def reset_all_singletons() -> None:
    """すべてのシングルトンインスタンスをリセット（主にテスト用）"""
    for singleton in singleton_registry.values():
        singleton.reset()
    logger.debug("すべてのシングルトンをリセットしました")