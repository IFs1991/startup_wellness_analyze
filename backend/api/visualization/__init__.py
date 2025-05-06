"""
可視化共通コンポーネントパッケージ

このパッケージは、様々な分析タイプにわたる可視化機能を共通化し、
コード重複を削減するための共通コンポーネントを提供します。
"""

__version__ = "1.0.0"

# プロセッサを初期化時にインポート
from . import models
from . import errors
from . import factory
from . import association_processor
from . import correlation_processor