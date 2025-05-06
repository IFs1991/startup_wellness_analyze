"""
アプリケーション起動時の設定

このモジュールではアプリケーション起動時に実行される様々な設定を行います。
- バックグラウンドタスクの開始
- キャッシュクリーンアップの設定
- その他の初期化処理
"""

import asyncio
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

# キャッシュクリーンアップ機能をインポート
from api.utils.caching import periodic_cache_cleanup

logger = logging.getLogger(__name__)

# バックグラウンドタスク
background_tasks = set()

# アプリケーションの起動と終了時に実行される非同期コンテキストマネージャ
@asynccontextmanager
async def lifespan(app: FastAPI):
    # アプリケーション起動時の処理
    # キャッシュクリーンアップタスクを開始
    cache_cleanup_task = asyncio.create_task(periodic_cache_cleanup(interval_seconds=600))
    background_tasks.add(cache_cleanup_task)
    cache_cleanup_task.add_done_callback(background_tasks.discard)

    logger.info("バックグラウンドタスクを開始しました: キャッシュクリーンアップ")

    yield

    # アプリケーション終了時の処理
    # すべてのバックグラウンドタスクをキャンセル
    for task in background_tasks:
        task.cancel()

    logger.info("すべてのバックグラウンドタスクを終了しました")