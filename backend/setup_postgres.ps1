# PostgreSQLセットアップスクリプト
# .envファイルから環境変数を読み込む
$envFile = Join-Path $PSScriptRoot "../.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            $key = $matches[1]
            $value = $matches[2]
            [Environment]::SetEnvironmentVariable($key, $value)
        }
    }
}

# 環境変数の存在確認
if (-not $env:POSTGRES_PASSWORD) {
    Write-Error "POSTGRES_PASSWORD environment variable is not set"
    exit 1
}
if (-not $env:DB_NAME) {
    Write-Error "DB_NAME environment variable is not set"
    exit 1
}

# PostgreSQLの環境変数を設定
$env:PGPASSWORD = $env:POSTGRES_PASSWORD

Write-Host "Setting up PostgreSQL databases..."

# パスワードの設定
psql -U postgres -c "ALTER USER postgres WITH PASSWORD '$env:POSTGRES_PASSWORD';"

# 既存のデータベースを削除（存在する場合）
psql -U postgres -c "DROP DATABASE IF EXISTS $env:DB_NAME;"
psql -U postgres -c "DROP DATABASE IF EXISTS test_$env:DB_NAME;"

# データベースの作成
psql -U postgres -c "CREATE DATABASE $env:DB_NAME;"
psql -U postgres -c "CREATE DATABASE test_$env:DB_NAME;"

Write-Host "Database setup completed successfully."