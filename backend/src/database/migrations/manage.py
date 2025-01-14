import os
import sys
import argparse
import logging
from alembic import command
from alembic.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_alembic_config() -> Config:
    """Alembicの設定を取得する"""
    # プロジェクトのルートディレクトリを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

    # alembic.iniのパスを設定
    alembic_ini = os.path.join(project_root, "alembic.ini")
    if not os.path.exists(alembic_ini):
        raise FileNotFoundError(f"alembic.iniが見つかりません: {alembic_ini}")

    # Alembicの設定を作成
    config = Config(alembic_ini)
    return config

def create_migration(message: str) -> None:
    """マイグレーションを作成する"""
    try:
        config = get_alembic_config()
        command.revision(config, message=message, autogenerate=True)
        logger.info(f"マイグレーションを作成しました: {message}")
    except Exception as e:
        logger.error(f"マイグレーションの作成中にエラーが発生しました: {e}")
        sys.exit(1)

def upgrade_database(revision: str = "head") -> None:
    """データベースをアップグレードする"""
    try:
        config = get_alembic_config()
        command.upgrade(config, revision)
        logger.info(f"データベースをアップグレードしました: {revision}")
    except Exception as e:
        logger.error(f"データベースのアップグレード中にエラーが発生しました: {e}")
        sys.exit(1)

def downgrade_database(revision: str) -> None:
    """データベースをダウングレードする"""
    try:
        config = get_alembic_config()
        command.downgrade(config, revision)
        logger.info(f"データベースをダウングレードしました: {revision}")
    except Exception as e:
        logger.error(f"データベースのダウングレード中にエラーが発生しました: {e}")
        sys.exit(1)

def show_history() -> None:
    """マイグレーション履歴を表示する"""
    try:
        config = get_alembic_config()
        command.history(config)
    except Exception as e:
        logger.error(f"マイグレーション履歴の表示中にエラーが発生しました: {e}")
        sys.exit(1)

def show_current() -> None:
    """現在のリビジョンを表示する"""
    try:
        config = get_alembic_config()
        command.current(config)
    except Exception as e:
        logger.error(f"現在のリビジョンの表示中にエラーが発生しました: {e}")
        sys.exit(1)

def main() -> None:
    """メイン関数"""
    parser = argparse.ArgumentParser(description="データベースマイグレーション管理スクリプト")
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンド")

    # createコマンド
    create_parser = subparsers.add_parser("create", help="新しいマイグレーションを作成する")
    create_parser.add_argument("message", help="マイグレーションの説明")

    # upgradeコマンド
    upgrade_parser = subparsers.add_parser("upgrade", help="データベースをアップグレードする")
    upgrade_parser.add_argument(
        "--revision",
        default="head",
        help="アップグレード先のリビジョン（デフォルト: head）"
    )

    # downgradeコマンド
    downgrade_parser = subparsers.add_parser("downgrade", help="データベースをダウングレードする")
    downgrade_parser.add_argument("revision", help="ダウングレード先���リビジョン")

    # historyコマンド
    subparsers.add_parser("history", help="マイグレーション履歴を表示する")

    # currentコマンド
    subparsers.add_parser("current", help="現在のリビジョンを表示する")

    args = parser.parse_args()

    if args.command == "create":
        create_migration(args.message)
    elif args.command == "upgrade":
        upgrade_database(args.revision)
    elif args.command == "downgrade":
        downgrade_database(args.revision)
    elif args.command == "history":
        show_history()
    elif args.command == "current":
        show_current()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()