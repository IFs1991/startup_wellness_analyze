from firebase_functions import https_fn
from firebase_functions import firestore_fn
from firebase_functions.firestore_fn import Event, Change, DocumentSnapshot
from firebase_admin import initialize_app, firestore
from typing import Optional
import google.cloud.firestore

# Firebase の初期化
initialize_app()

# Firestore クライアントの初期化
db = firestore.client()

@https_fn.on_request()
def hello_world(req: https_fn.Request) -> https_fn.Response:
    """動作確認用の簡単な関数"""
    return https_fn.Response("Hello from Firebase!")

@firestore_fn.on_document_created(document="csv_data/{documentId}")
def on_csv_data_created(event: Event[Optional[DocumentSnapshot]]) -> None:
    """新しいCSVデータが追加された時の処理"""
    if event and event.data:
        document_data = event.data.to_dict()
        print(f"New document: {document_data}")