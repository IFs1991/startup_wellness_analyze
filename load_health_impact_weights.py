#!/usr/bin/env python3
"""
業種・役職別健康影響度重み付け係数のデータベース初期化スクリプト

このスクリプトは:
1. SQLスキーマを適用してテーブルを作成
2. 初期データをロード
3. 最終重み係数を計算

Usage:
    python load_health_impact_weights.py --host hostname --dbname dbname --user username --password password
"""

import argparse
import os
import psycopg2
import logging
from pathlib import Path

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_db_connection(args):
    """データベース接続を作成"""
    connection = psycopg2.connect(
        host=args.host,
        dbname=args.dbname,
        user=args.user,
        password=args.password
    )
    connection.autocommit = False
    return connection

def execute_sql_file(connection, sql_file_path):
    """SQLファイルを実行"""
    try:
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        with connection.cursor() as cursor:
            cursor.execute(sql_content)

        connection.commit()
        logger.info(f"SQLファイル {sql_file_path} を実行しました")
        return True

    except Exception as e:
        connection.rollback()
        logger.error(f"SQLファイル {sql_file_path} の実行中にエラーが発生しました: {str(e)}")
        return False

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='業種・役職別健康影響度重み付け係数のデータベース初期化')
    parser.add_argument('--host', required=True, help='PostgreSQLホスト名')
    parser.add_argument('--dbname', required=True, help='データベース名')
    parser.add_argument('--user', required=True, help='データベースユーザー名')
    parser.add_argument('--password', required=True, help='データベースパスワード')
    args = parser.parse_args()

    # SQLファイルのパス
    current_dir = Path(__file__).parent
    schema_file = current_dir / 'health_impact_weights_schema.sql'
    data_file = current_dir / 'health_impact_weights_data.sql'

    # ファイルの存在確認
    if not schema_file.exists():
        logger.error(f"スキーマファイル {schema_file} が見つかりません")
        return False

    if not data_file.exists():
        logger.error(f"データファイル {data_file} が見つかりません")
        return False

    # データベース接続
    try:
        connection = create_db_connection(args)
        logger.info("データベースに接続しました")

        # スキーマ適用
        if not execute_sql_file(connection, schema_file):
            logger.error("スキーマの適用に失敗しました")
            return False

        # データロード
        if not execute_sql_file(connection, data_file):
            logger.error("データのロードに失敗しました")
            return False

        logger.info("業種・役職別健康影響度重み付け係数のデータベース初期化が完了しました")
        connection.close()
        return True

    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)