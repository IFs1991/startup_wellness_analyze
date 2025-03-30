"""
環境変数の読み込みテスト

設定モジュールが.envファイルから正しく値を読み込んでいるかを確認します。
"""
import os
import json
from config import settings

def print_settings():
    """設定値を出力して.envからの読み込みを確認する"""
    print("=== 環境設定値の確認 ===")
    print(f"ENVIRONMENT: {settings.ENVIRONMENT} (期待値: development)")
    print(f"DEBUG: {settings.DEBUG} (期待値: True)")
    print(f"DEV_MODE: {settings.DEV_MODE} (期待値: True)")
    print(f"APP_ENV: {settings.APP_ENV} (期待値: development)")

    print("\n=== Google Cloud設定 ===")
    print(f"GCP_PROJECT_ID: {settings.GCP_PROJECT_ID} (期待値: startupwellnessanalyze-445505)")
    print(f"GCP_REGION: {settings.GCP_REGION} (期待値: asia-northeast1)")
    print(f"FIREBASE_PROJECT_ID: {settings.FIREBASE_PROJECT_ID} (期待値: startupwellnessanalyze-445505)")

    print("\n=== Google認証情報 ===")
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {settings.GOOGLE_APPLICATION_CREDENTIALS}")
    print(f"FIREBASE_ADMIN_CREDENTIALS取得: {'成功' if settings.FIREBASE_ADMIN_CREDENTIALS else '失敗'}")

    print("\n=== データベース設定 ===")
    print(f"DATABASE_URL: {settings.DATABASE_URL}")
    print(f"REDIS_HOST: {settings.REDIS_HOST} (期待値: startup-wellness-redis)")
    print(f"REDIS_PORT: {settings.REDIS_PORT} (期待値: 6379)")

    print("\n=== ワーカー設定 ===")
    print(f"WORKERS: {settings.WORKERS} (期待値: 4)")
    print(f"BACKLOG: {settings.BACKLOG} (期待値: 2048)")
    print(f"KEEP_ALIVE: {settings.KEEP_ALIVE} (期待値: 5)")

    print("\n=== モニタリング設定 ===")
    print(f"ENABLE_MEMORY_PROFILING: {settings.ENABLE_MEMORY_PROFILING} (期待値: False)")
    print(f"ENABLE_PSUTIL_MONITORING: {settings.ENABLE_PSUTIL_MONITORING} (期待値: True)")

if __name__ == "__main__":
    # テスト開始
    print("環境変数読み込みテストを開始します...\n")
    print_settings()

    # .env設定との一致を確認
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        print(f"\n.envファイルが見つかりました: {env_path}")
    else:
        print(f"\n.envファイルが見つかりません: {env_path}")

    print("\nテスト完了")