@echo off
echo Setting up PostgreSQL databases...
psql -U postgres -f setup_db.sql
if %ERRORLEVEL% EQU 0 (
    echo Database setup completed successfully.
) else (
    echo Error setting up databases.
)
pause