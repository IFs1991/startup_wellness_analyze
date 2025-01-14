import asyncio
import sys
from pathlib import Path
from sqlalchemy import text

# プロジェクトルートをPythonパスに追加
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from database.database import get_db

async def test_connection():
    """データベース接続のテスト"""
    try:
        async for session in get_db():
            # テスト用のクエリを実行
            result = await session.execute(text("SELECT 1"))
            value = result.scalar()
            print("データベース接続成功!")
            print(f"テストクエリ結果: {value}")
            return
    except Exception as e:
        print(f"データベース接続エラー: {str(e)}")
        print(f"エラーの詳細: {type(e).__name__}")

if __name__ == "__main__":
    asyncio.run(test_connection())