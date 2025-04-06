"""
PostgreSQL database connection utilities for Startup Wellness Analyze
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
from contextlib import contextmanager
from typing import Generator
import os
from dotenv import load_dotenv

# backendディレクトリへのパスを取得
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# backend/.envファイルのパスを設定
ENV_PATH = os.path.join(BACKEND_DIR, '.env')

# 開発環境の場合のみ .env を読み込み
if os.getenv("ENVIRONMENT") != "production":
    # backend/.envファイルを読み込む
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)
    else:
        # ENVファイルが見つからない場合はログ出力
        print(f"Warning: .env file not found at {ENV_PATH}")

# 環境変数からデータベースURLを取得するか、デフォルト値を使用
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:1spel!2stack3win@localhost:5432/startup_wellness_analyze"
)

# エンジンの作成
engine = create_engine(DATABASE_URL)

# セッションの作成
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# モデルのベースクラス
Base = declarative_base()

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    PostgreSQLデータベースセッションを取得し、コンテキストマネージャーとして使用するための関数です。

    Yields:
        Session: SQLAlchemyセッションインスタンス

    Example:
        with get_db() as db:
            users = db.query(models.User).all()
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        print(f"Database error: {str(e)}")
        raise
    finally:
        db.close()

def init_db():
    """
    データベーススキーマを初期化します。
    アプリケーション起動時に呼び出されます。
    """
    # モデルをインポート（モデルがBaseを継承している必要あり）
    from backend.database.models_sql import Base as SQLModels
    SQLModels.metadata.create_all(bind=engine)

# バッチ処理用のヘルパー関数
@contextmanager
def db_transaction() -> Generator[Session, None, None]:
    """
    トランザクション内で複数の操作を実行するためのコンテキストマネージャー

    Example:
        with db_transaction() as db:
            db.add(model1)
            db.add(model2)
            # トランザクションは自動的にコミットまたはロールバックされます
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Transaction failed: {str(e)}")
        raise
    finally:
        db.close()