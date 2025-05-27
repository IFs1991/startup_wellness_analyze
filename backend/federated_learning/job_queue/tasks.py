"""
追加タスク定義
Task 4.3: 非同期ジョブキュー
"""

from .celery_app import celery_app
from .fl_jobs import *

# このファイルは celery_app.py の include 設定で必要
# 実際のタスクは fl_jobs.py で定義済み