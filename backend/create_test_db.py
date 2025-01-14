import asyncpg
import asyncio
from dotenv import load_dotenv
import os

# .envファイルを読み込む
load_dotenv()

async def create_test_database():
    conn = None
    try:
        # PostgreSQLのデフォルトデータベースに接続
        conn = await asyncpg.connect(
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            database="postgres"
        )

        test_db_name = os.getenv("POSTGRES_TEST_DB", "test_startup_wellness")

        # 既存のデータベースを削除（存在する場合）
        try:
            # 既存の接続を切断
            await conn.execute(f'''
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{test_db_name}'
                AND pid <> pg_backend_pid();
            ''')
            await conn.execute(f'DROP DATABASE IF EXISTS {test_db_name}')
            print('Existing database dropped successfully')
        except Exception as e:
            print(f'Error dropping database: {e}')

        # 新しいデータベースを作成
        try:
            await conn.execute(f'CREATE DATABASE {test_db_name}')
            print('Database created successfully')
        except Exception as e:
            print(f'Error creating database: {e}')

    except Exception as e:
        print(f'Connection error: {e}')
    finally:
        if conn:
            try:
                await conn.close()
            except Exception as e:
                print(f'Error closing connection: {e}')

if __name__ == '__main__':
    asyncio.run(create_test_database())