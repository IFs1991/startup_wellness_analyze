"""
データ移行スクリプト
FirestoreからPostgreSQLへのデータ移行を行うためのスクリプトです。
"""
import asyncio
from typing import List, Dict, Any
from datetime import datetime
from firebase_admin import firestore
from sqlalchemy.orm import Session

# 問題のインポートを修正
# from .database import get_collection_data
from .database import get_firestore_client, DatabaseManager, DataCategory
from .postgres import get_db, init_db
from . import models_sql
from . import crud_sql

# 必要な関数を独自に定義
async def get_collection_data(collection_name: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Firestoreコレクションからデータを取得する関数

    Args:
        collection_name: 取得するコレクション名
        **kwargs: 追加のフィルタリング条件

    Returns:
        ドキュメントデータのリスト
    """
    # Firestoreクライアントを取得
    db = get_firestore_client()
    if not db:
        print(f"Firestoreクライアントを取得できませんでした")
        return []

    collection_ref = db.collection(collection_name)

    # クエリ条件があれば適用
    query = collection_ref
    for key, value in kwargs.items():
        query = query.where(key, '==', value)

    # ドキュメントを取得
    try:
        docs = query.stream()
        results = []
        for doc in docs:
            data = doc.to_dict()
            # ドキュメントIDを追加
            if 'id' not in data:
                data['id'] = doc.id
            results.append(data)
        return results
    except Exception as e:
        print(f"コレクション {collection_name} の取得中にエラーが発生しました: {e}")
        return []

async def migrate_users():
    """ユーザーデータの移行"""
    print("ユーザーデータの移行を開始します...")
    users_data = await get_collection_data('users')

    with get_db() as db:
        for user_data in users_data:
            # IDフィールドの処理
            user_id = user_data.pop('id', None)
            if user_id:
                user_data['id'] = user_id

            # タイムスタンプの処理
            for field in ['created_at', 'updated_at']:
                if field in user_data and isinstance(user_data[field], firestore.SERVER_TIMESTAMP):
                    user_data[field] = datetime.utcnow()

            # 既存ユーザーの確認
            existing_user = crud_sql.get_user(db, user_id) if user_id else None

            if not existing_user:
                try:
                    crud_sql.create_user(db, user_data)
                    print(f"ユーザー {user_data.get('username', user_id)} を作成しました")
                except Exception as e:
                    print(f"ユーザー {user_data.get('username', user_id)} の作成に失敗しました: {str(e)}")
            else:
                print(f"ユーザー {user_data.get('username', user_id)} は既に存在します")

    print("ユーザーデータの移行が完了しました")

async def migrate_startups():
    """スタートアップデータの移行"""
    print("スタートアップデータの移行を開始します...")
    startups_data = await get_collection_data('startups')

    with get_db() as db:
        for startup_data in startups_data:
            # IDフィールドの処理
            startup_id = startup_data.pop('id', None)
            if startup_id:
                startup_data['id'] = startup_id

            # タイムスタンプの処理
            for field in ['created_at', 'updated_at', 'founding_date']:
                if field in startup_data and isinstance(startup_data[field], (firestore.SERVER_TIMESTAMP, dict)):
                    startup_data[field] = datetime.utcnow()

            # 既存スタートアップの確認
            existing_startup = crud_sql.get_startup(db, startup_id) if startup_id else None

            if not existing_startup:
                try:
                    crud_sql.create_startup(db, startup_data)
                    print(f"スタートアップ {startup_data.get('name', startup_id)} を作成しました")
                except Exception as e:
                    print(f"スタートアップ {startup_data.get('name', startup_id)} の作成に失敗しました: {str(e)}")
            else:
                print(f"スタートアップ {startup_data.get('name', startup_id)} は既に存在します")

    print("スタートアップデータの移行が完了しました")

async def migrate_vas_data():
    """VASデータの移行"""
    print("VASデータの移行を開始します...")
    vas_data_list = await get_collection_data('vas_data')

    with get_db() as db:
        for vas_data in vas_data_list:
            # IDフィールドの処理
            vas_id = vas_data.pop('id', None)
            if vas_id:
                vas_data['id'] = vas_id

            # タイムスタンプの処理
            for field in ['created_at', 'updated_at', 'timestamp']:
                if field in vas_data and isinstance(vas_data[field], (firestore.SERVER_TIMESTAMP, dict)):
                    vas_data[field] = datetime.utcnow()

            # 既存VASデータの確認
            existing_vas = crud_sql.get_vas_data(db, vas_id) if vas_id else None

            if not existing_vas:
                try:
                    crud_sql.create_vas_data(db, vas_data)
                    print(f"VASデータ {vas_id} を作成しました")
                except Exception as e:
                    print(f"VASデータ {vas_id} の作成に失敗しました: {str(e)}")
            else:
                print(f"VASデータ {vas_id} は既に存在します")

    print("VASデータの移行が完了しました")

async def migrate_financial_data():
    """財務データの移行"""
    print("財務データの移行を開始します...")
    financial_data_list = await get_collection_data('financial_data')

    with get_db() as db:
        for financial_data in financial_data_list:
            # IDフィールドの処理
            financial_id = financial_data.pop('id', None)
            if financial_id:
                financial_data['id'] = financial_id

            # タイムスタンプの処理
            for field in ['created_at', 'updated_at']:
                if field in financial_data and isinstance(financial_data[field], (firestore.SERVER_TIMESTAMP, dict)):
                    financial_data[field] = datetime.utcnow()

            # JSONフィールドの処理
            if 'kpis' in financial_data and financial_data['kpis'] is None:
                financial_data['kpis'] = {}

            # 既存財務データの確認
            existing_financial = crud_sql.get_financial_data(db, financial_id) if financial_id else None

            if not existing_financial:
                try:
                    crud_sql.create_financial_data(db, financial_data)
                    print(f"財務データ {financial_id} を作成しました")
                except Exception as e:
                    print(f"財務データ {financial_id} の作成に失敗しました: {str(e)}")
            else:
                print(f"財務データ {financial_id} は既に存在します")

    print("財務データの移行が完了しました")

async def migrate_notes():
    """メモデータの移行"""
    print("メモデータの移行を開始します...")
    notes_data_list = await get_collection_data('notes')

    with get_db() as db:
        for note_data in notes_data_list:
            # IDフィールドの処理
            note_id = note_data.pop('id', None)
            if note_id:
                note_data['id'] = note_id

            # タイムスタンプの処理
            for field in ['created_at', 'updated_at', 'timestamp']:
                if field in note_data and isinstance(note_data[field], (firestore.SERVER_TIMESTAMP, dict)):
                    note_data[field] = datetime.utcnow()

            # 既存メモの確認
            existing_note = crud_sql.get_note(db, note_id) if note_id else None

            if not existing_note:
                try:
                    crud_sql.create_note(db, note_data)
                    print(f"メモ {note_id} を作成しました")
                except Exception as e:
                    print(f"メモ {note_id} の作成に失敗しました: {str(e)}")
            else:
                print(f"メモ {note_id} は既に存在します")

    print("メモデータの移行が完了しました")

async def run_migration():
    """全データの移行を実行"""
    print("データベース移行を開始します...")

    # データベーススキーマを初期化
    init_db()

    # データ移行を実行
    await migrate_users()
    await migrate_startups()
    await migrate_vas_data()
    await migrate_financial_data()
    await migrate_notes()

    print("データベース移行が完了しました！")

if __name__ == "__main__":
    # 移行スクリプトの実行
    asyncio.run(run_migration())