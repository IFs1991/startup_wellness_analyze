import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import datetime
import os
from dotenv import load_dotenv

# backendディレクトリへのパスを取得
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# backend/.envファイルのパスを設定
ENV_PATH = os.path.join(BACKEND_DIR, '.env')

# 環境変数を読み込み
if os.getenv("ENVIRONMENT") != "production":
    # backend/.envファイルを読み込む
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)
    else:
        # ENVファイルが見つからない場合はログ出力
        print(f"Warning: .env file not found at {ENV_PATH}")

def test_firestore_connection():
    """
    Firestoreの接続テストを行う関数
    """
    try:
        # すでに初期化されている場合は新たに初期化しない
        if not firebase_admin._apps:
            # 環境変数からクレデンシャルパスを取得
            cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

            if cred_path and os.path.exists(cred_path):
                print(f"Firebase認証情報を読み込みます: {cred_path}")
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            else:
                print("環境変数からFIREBASE認証情報が取得できません。デフォルト認証情報を使用します。")
                firebase_admin.initialize_app()

        # Firestoreクライアントの作成
        db = firestore.client()

        # テストコレクションにドキュメントを書き込み
        test_ref = db.collection('test').document('connection_test')
        test_ref.set({
            'timestamp': datetime.datetime.now(),
            'status': 'connection successful'
        })

        # 書き込んだドキュメントを読み取り
        doc = test_ref.get()
        if doc.exists:
            print("✅ Firestore connection test successful!")
            print(f"Retrieved data: {doc.to_dict()}")
            return True
        else:
            print("❌ Failed to retrieve test document")
            return False

    except Exception as e:
        print(f"❌ Firestore connection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_firestore_connection()