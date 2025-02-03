-- データベースの作成用スクリプト
-- 注意: このスクリプトは環境変数を使用します
-- 実行前に以下の環境変数が設定されていることを確認してください:
-- - POSTGRES_PASSWORD
-- - DB_NAME

-- パスワードの設定（環境変数から）
\set password `echo "$POSTGRES_PASSWORD"`
ALTER USER postgres WITH PASSWORD :'password';

-- 既存のデータベースを削除（存在する場合）
\set db_name `echo "$DB_NAME"`
DROP DATABASE IF EXISTS :db_name;
DROP DATABASE IF EXISTS test_:db_name;

-- データベースの作成
CREATE DATABASE :db_name;
CREATE DATABASE test_:db_name;