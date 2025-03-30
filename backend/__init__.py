"""
Startup Wellness データ分析システム バックエンド
"""

# バージョン情報
__version__ = '0.1.0'

# アプリケーションの起動時にimportされるようにsubmoduleをimportしておく
try:
    from . import service
    from . import database
    from . import api
    from . import analysis
    from . import core
    # 連合学習モジュールをインポート
    from . import federated_learning
except ImportError:
    # モジュールインポートにエラーがあった場合のフォールバック
    import warnings
    warnings.warn("一部のモジュールのインポートに失敗しました。機能が制限される可能性があります。")

import warnings

# データベース設定と接続のインポート
try:
    # 新しいモジュール構造
    from backend.config import database_config
    from backend.database import connection
except ImportError:
    warnings.warn(
        "新しいデータベースモジュール構造のインポートに失敗しました。古い構造を使用します。",
        ImportWarning
    )

# 後方互換性のために古いインポートを維持
# 警告を表示して非推奨を示す
try:
    from . import database
    warnings.warn(
        "backend.database モジュールは非推奨です。新しいコードでは backend.config.database_config と "
        "backend.database.connection を使用してください。",
        DeprecationWarning,
        stacklevel=2
    )
except ImportError:
    pass
