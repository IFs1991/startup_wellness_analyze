"""
Startup Wellness データ分析システム バックエンド

このモジュールは、アプリケーションのエントリーポイントとして機能し、
主要なコンポーネントへのアクセスを提供します。
"""

# アプリケーション設定をインポート
from backend.app.core.config import settings

# バージョン情報とアプリケーション識別子の定義
__version__ = settings.VERSION
__app_name__ = settings.APP_NAME
__project_name__ = settings.PROJECT_NAME

# このモジュールから外部に公開するシンボルを定義
__all__ = [
    'settings',
    '__version__',
    '__app_name__',
    '__project_name__'
]