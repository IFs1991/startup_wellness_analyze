"""
Firestore サービスモジュール
Firestoreを利用した高度なビジネスロジックを提供します。

このモジュールはデータベース層（backend.database）の基本的なCRUD操作を活用し、
アプリケーション要件に特化した複合的なデータ操作、バッチ処理、ファイル管理など
高レベルなサービスを実装します。

データベース層との関係:
- データベース層: 基本的なCRUD、データモデル定義、DB接続管理
- サービス層: 高度な業務ロジック、外部API連携、ファイル管理
"""

# モジュールをインポート
from . import client

# エクスポートする関数を明示的に指定
__all__ = ['client']
