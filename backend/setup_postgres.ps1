# PostgreSQLの環境変数を設定
$env:PGPASSWORD = "postgres"

Write-Host "Setting up PostgreSQL databases..."

# パスワードの設定
psql -U postgres -c "ALTER USER postgres WITH PASSWORD 'postgres';"

# 既存のデータベースを削除（存在する場合）
psql -U postgres -c "DROP DATABASE IF EXISTS startup_wellness;"
psql -U postgres -c "DROP DATABASE IF EXISTS test_startup_wellness;"

# データベースの作成
psql -U postgres -c "CREATE DATABASE startup_wellness;"
psql -U postgres -c "CREATE DATABASE test_startup_wellness;"

Write-Host "Database setup completed."