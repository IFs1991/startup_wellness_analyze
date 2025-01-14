# PostgreSQLサービスを再起動
Write-Host "Restarting PostgreSQL service..."
Restart-Service postgresql-x64-17

# 少し待機
Start-Sleep -Seconds 5

# pg_hba.confを一時的に信頼モードに変更
Write-Host "Updating pg_hba.conf..."
$pg_hba_path = "C:\Program Files\PostgreSQL\17\data\pg_hba.conf"
$backup_path = "C:\Program Files\PostgreSQL\17\data\pg_hba.conf.backup"

# バックアップを作成
Copy-Item -Path $pg_hba_path -Destination $backup_path -Force

# 一時的に信頼モードに変更
$content = Get-Content $pg_hba_path
$content = $content -replace "host.*all.*all.*scram-sha-256", "host all all all trust"
$content | Set-Content $pg_hba_path

# PostgreSQLサービスを再起動
Write-Host "Restarting PostgreSQL service again..."
Restart-Service postgresql-x64-17

# 少し待機
Start-Sleep -Seconds 5

# パスワードを設定
Write-Host "Setting new password..."
psql -U postgres -c "ALTER USER postgres WITH PASSWORD 'postgres';"

# pg_hba.confを元に戻す
Write-Host "Restoring pg_hba.conf..."
Move-Item -Path $backup_path -Destination $pg_hba_path -Force

# 最後にサービスを再起動
Write-Host "Final PostgreSQL service restart..."
Restart-Service postgresql-x64-17

Write-Host "Password reset completed."