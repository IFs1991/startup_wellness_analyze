import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

# モデルのメタデータをインポート
from ..models import Base
from ..config import PostgresConfig

# Alembic設定を取得
config = context.config

# ログ設定を適用（alembic.iniで定義）
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 対象のメタデータを設定
target_metadata = Base.metadata

def get_url():
    """データベースURLを取得する"""
    db_config = PostgresConfig()
    return db_config.get_database_url()

def run_migrations_offline() -> None:
    """
    オフラインでマイグレーションを実行する。
    このシナリオでは、スクリプトを直接データベースに送信する代わりに、
    SQL文を書き出してファイルに保存したり、後で実行したりすることができる。
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection: Connection) -> None:
    """マイグレーションを実行する"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True
    )

    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations() -> None:
    """非同期でマイグレーションを実行する"""
    configuration = config.get_section(config.config_ini_section)
    if configuration is None:
        configuration = {}
    configuration["sqlalchemy.url"] = get_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

def run_migrations_online() -> None:
    """
    オンラインでマイグレーションを実行する。
    これは通常の実行モードで、データベースに直接接続してマイグレーションを実行する。
    """
    asyncio.run(run_async_migrations())