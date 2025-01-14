import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import datetime
import os

def test_firestore_connection():
    """
    Firestoreの接続テストを行う関数
    """
    try:
        # すでに初期化されている場合は新たに初期化しない
        if not firebase_admin._apps:
            # パスを修正：相対パスで指定
            cred = credentials.Certificate('../credentials/serviceAccountKey.json')  # ここを変更
            firebase_admin.initialize_app(cred)

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