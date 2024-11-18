# -*- coding: utf-8 -*-

"""
Startup Wellness データ分析システム バックエンド API

要件定義書と requirements.txt を元に作成された FastAPI アプリケーションです。
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from io import BytesIO
import uvicorn
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from google.cloud.firestore import Client as FirestoreClient

# 認証モジュール
from .auth import authenticate_user, create_access_token, get_current_user
from .auth import register_user, reset_password, logout_user

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore, auth, exceptions

# データ入力モジュールの定義
async def read_csv_data(file: UploadFile, db: FirestoreClient) -> Dict[str, Any]:
    """CSVファイルを読み込み、データベースに保存する"""
    try:
        contents = await file.read()
        # BytesIOを使用してバイトデータをストリームとして扱う
        csv_stream = BytesIO(contents)

        # エンコーディングを明示的に指定してCSVを読み込む
        df = pd.read_csv(csv_stream, encoding='utf-8')

        # メモリ効率を考慮してストリームを閉じる
        csv_stream.close()

        data_dict = df.to_dict('records')

        # Firestoreにバッチ処理で保存
        batch = db.batch()
        collection_ref = db.collection('csv_data')

        # バッチサイズの制限（500件）を考慮した処理
        for i in range(0, len(data_dict), 500):
            batch_chunk = data_dict[i:i + 500]
            current_batch = db.batch()

            for record in batch_chunk:
                doc_ref = collection_ref.document()
                current_batch.set(doc_ref, record)

            current_batch.commit()

        return {
            "status": "success",
            "records_processed": len(data_dict),
            "batches_processed": (len(data_dict) + 499) // 500
        }

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded CSV file is empty"
        )
    except pd.errors.ParserError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to parse the CSV file. Please check the format"
        )
    except Exception as e:
        logging.error(f"Error processing CSV file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the file: {str(e)}"
        )
    finally:
        # 確実にリソースを解放
        if 'csv_stream' in locals():
            csv_stream.close()

async def upload_files(files: List[UploadFile], db: FirestoreClient) -> Dict[str, Any]:
    """複数のファイルをアップロードし、データベースに保存する"""
    try:
        results = []
        for file in files:
            # ファイルをメモリ効率よく読み込む
            buffer = BytesIO()
            chunk_size = 8192  # 8KB chunks

            while chunk := await file.read(chunk_size):
                buffer.write(chunk)

            # ファイルメタデータの保存
            doc_ref = db.collection('uploaded_files').document()
            doc_ref.set({
                'filename': file.filename,
                'content_type': file.content_type,
                'size': buffer.tell(),
                'upload_timestamp': firebase_admin.firestore.SERVER_TIMESTAMP
            })

            results.append({
                'filename': file.filename,
                'status': 'success',
                'document_id': doc_ref.id
            })

            # リソースの解放
            buffer.close()

        return {
            "status": "success",
            "files_processed": len(results),
            "files": results
        }

    except Exception as e:
        logging.error(f"Error uploading files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )
    finally:
        # 確実にリソースを解放
        if 'buffer' in locals():
            buffer.close()

# 以下、既存のコードは変更なし...