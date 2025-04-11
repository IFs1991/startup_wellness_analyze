# -*- coding: utf-8 -*-
"""
リポジトリパターン使用例
新しいデータアクセス抽象化レイヤーの使用方法を示します。
"""
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..models import (
    UserEntity,
    StartupEntity,
    VASDataEntity,
    FinancialDataEntity,
    NoteEntity,
    ModelType
)
from ..repository import DataCategory, Repository
from ..repositories import repository_factory

async def create_startup_example():
    """スタートアップの作成例"""
    # スタートアップエンティティを作成
    startup = StartupEntity(
        name="健康テックスタートアップ",
        industry="ヘルスケア",
        founding_date=datetime(2022, 1, 1),
        description="従業員の健康状態をモニタリングするプラットフォーム",
        location="東京都渋谷区",
        website="https://healthtech-startup.example.com",
        owner_id="user123",
    )

    # モデルタイプを設定（実際には継承時に設定する）
    startup.__class__.model_type = ModelType.SQL

    # リポジトリの取得
    repo = repository_factory.get_repository(
        StartupEntity,
        data_category=DataCategory.COMPANY_MASTER
    )

    # データの保存
    saved_startup = repo.save(startup)
    print(f"スタートアップを保存しました: {saved_startup}")

    return saved_startup

async def find_startup_example(startup_id: str):
    """スタートアップの検索例"""
    # リポジトリの取得
    repo = repository_factory.get_repository(
        StartupEntity,
        data_category=DataCategory.COMPANY_MASTER
    )

    # IDによるデータ取得
    startup = repo.find_by_id(startup_id)
    if startup:
        print(f"スタートアップを見つけました: {startup.name}")
    else:
        print(f"スタートアップが見つかりませんでした: {startup_id}")

    return startup

async def update_startup_example(startup_id: str, new_data: Dict[str, Any]):
    """スタートアップの更新例"""
    # リポジトリの取得
    repo = repository_factory.get_repository(
        StartupEntity,
        data_category=DataCategory.COMPANY_MASTER
    )

    # データの更新
    try:
        updated_startup = repo.update(startup_id, new_data)
        print(f"スタートアップを更新しました: {updated_startup.name}")
        return updated_startup
    except Exception as e:
        print(f"更新エラー: {str(e)}")
        return None

async def find_startups_by_criteria_example(criteria: Dict[str, Any]):
    """条件によるスタートアップの検索例"""
    # リポジトリの取得
    repo = repository_factory.get_repository(
        StartupEntity,
        data_category=DataCategory.COMPANY_MASTER
    )

    # 条件によるデータ取得
    startups = repo.find_by_criteria(criteria)
    print(f"{len(startups)}件のスタートアップが条件に一致しました")

    for startup in startups:
        print(f" - {startup.name} ({startup.industry})")

    return startups

async def delete_startup_example(startup_id: str):
    """スタートアップの削除例"""
    # リポジトリの取得
    repo = repository_factory.get_repository(
        StartupEntity,
        data_category=DataCategory.COMPANY_MASTER
    )

    # データの削除
    try:
        result = repo.delete(startup_id)
        print(f"スタートアップを削除しました: {result}")
        return result
    except Exception as e:
        print(f"削除エラー: {str(e)}")
        return False

async def multi_database_example():
    """複数データベースを横断する処理の例"""
    # ユーザーデータはSQL（構造化）
    user_repo = repository_factory.get_repository(
        UserEntity,
        data_category=DataCategory.USER_MASTER
    )

    # 財務データはSQL（構造化）
    financial_repo = repository_factory.get_repository(
        FinancialDataEntity,
        data_category=DataCategory.FINANCIAL
    )

    # VASデータはFirestore（リアルタイム）
    vas_repo = repository_factory.get_repository(
        VASDataEntity,
        data_category=DataCategory.REALTIME
    )

    # スタートアップの分析情報を取得する例
    # 1. スタートアップIDでユーザーを検索
    startup_id = "startup123"
    users = user_repo.find_by_criteria({"company_id": startup_id})

    # 2. そのスタートアップの財務データを取得
    financial_data = financial_repo.find_by_criteria({"startup_id": startup_id})

    # 3. リアルタイムのVASデータを取得
    vas_data = vas_repo.find_by_criteria({"startup_id": startup_id})

    # 4. データの統合処理（実際にはここでさらに処理を行う）
    print(f"ユーザー数: {len(users)}")
    print(f"財務データポイント数: {len(financial_data)}")
    print(f"VASデータポイント数: {len(vas_data)}")

    return {
        "users": users,
        "financial_data": financial_data,
        "vas_data": vas_data
    }

async def main():
    """メイン関数"""
    print("リポジトリパターン使用例を開始します...")

    # スタートアップを作成
    startup = await create_startup_example()

    if startup:
        # スタートアップを取得
        await find_startup_example(startup.entity_id)

        # スタートアップを更新
        await update_startup_example(
            startup.entity_id,
            {"employee_count": 25, "funding_stage": "シリーズA"}
        )

        # 条件で検索
        await find_startups_by_criteria_example({"industry": "ヘルスケア"})

        # 削除
        await delete_startup_example(startup.entity_id)

    # 複数データベース処理例
    await multi_database_example()

    print("リポジトリパターン使用例を終了します")

if __name__ == "__main__":
    # 非同期関数を実行
    asyncio.run(main())