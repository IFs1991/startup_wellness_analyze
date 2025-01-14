-- パスワードの設定
ALTER USER postgres WITH PASSWORD 'postgres';

-- 既存のデータベースを削除（存在する場合）
DROP DATABASE IF EXISTS startup_wellness;
DROP DATABASE IF EXISTS test_startup_wellness;

-- データベースの作成
CREATE DATABASE startup_wellness;
CREATE DATABASE test_startup_wellness;