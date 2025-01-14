# -*- coding: utf-8 -*-
"""
Firestoreセキュリティサービス
- データの暗号化/復号
- アクセス制御
- データの匿名化
- セキュリティ監査ログ
"""
from firebase_admin import firestore
import firebase_admin
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime
import logging
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import pandas as pd
import json
from fastapi import HTTPException

# プロジェクトの絶対インポートに変更
from backend.service.firestore.client import FirestoreService

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 定数定義
BLOCK_SIZE = 16  # AES block size in bytes
KEY_SIZE = 32    # 256-bit key size
AUDIT_COLLECTION = 'security_audit_logs'
ENCRYPTED_DOCS_COLLECTION = 'encrypted_documents'

class SecurityError(Exception):
    """セキュリティ操作に関するエラー"""
    pass

class FirestoreSecurityService:
    def __init__(self):
        """
        Firestoreセキュリティサービスを初期化します
        """
        try:
            # FirestoreServiceのインスタンス化
            self.firestore_service = FirestoreService()
            self.db = self.firestore_service.db
            self.encryption_key = get_random_bytes(KEY_SIZE)
            logger.info("Firestore Security Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize security service: {str(e)}")
            raise SecurityError(f"Initialization error: {str(e)}") from e

    async def encrypt_and_store(
        self,
        data: Dict[str, Any],
        collection_name: str,
        user_id: str
    ) -> str:
        """
        データを暗号化してFirestoreに保存します
        """
        try:
            # データの暗号化
            cipher = AES.new(self.encryption_key, AES.MODE_CBC)
            data_bytes = json.dumps(data).encode('utf-8')
            encrypted_data = cipher.encrypt(pad(data_bytes, BLOCK_SIZE))

            # 暗号化データの保存
            encrypted_doc = {
                'encrypted_data': encrypted_data,
                'iv': cipher.iv,
                'user_id': user_id,
                'created_at': datetime.now(),
                'collection': collection_name
            }

            doc_ids = await self.firestore_service.save_results(
                results=[encrypted_doc],
                collection_name=ENCRYPTED_DOCS_COLLECTION
            )

            # 監査ログの記録
            await self._log_security_event(
                event_type='data_encryption',
                user_id=user_id,
                details={'document_id': doc_ids[0], 'collection': collection_name}
            )

            return doc_ids[0]

        except Exception as e:
            error_msg = f"Error encrypting and storing data: {str(e)}"
            logger.error(error_msg)
            raise SecurityError(error_msg) from e

    async def decrypt_and_retrieve(
        self,
        document_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Firestoreから暗号化されたデータを取得して復号します
        """
        try:
            # 暗号化データの取得
            encrypted_docs = await self.firestore_service.fetch_documents(
                collection_name=ENCRYPTED_DOCS_COLLECTION,
                conditions=[
                    {'field': '__name__', 'operator': '==', 'value': document_id},
                    {'field': 'user_id', 'operator': '==', 'value': user_id}
                ],
                limit=1
            )

            if not encrypted_docs:
                raise SecurityError(f"Document not found or access denied: {document_id}")

            encrypted_doc = encrypted_docs[0]

            # データの復号
            cipher = AES.new(self.encryption_key, AES.MODE_CBC, iv=encrypted_doc['iv'])
            decrypted_data = unpad(cipher.decrypt(encrypted_doc['encrypted_data']), BLOCK_SIZE)

            # 監査ログの記録
            await self._log_security_event(
                event_type='data_decryption',
                user_id=user_id,
                details={'document_id': document_id}
            )

            return json.loads(decrypted_data.decode('utf-8'))

        except Exception as e:
            error_msg = f"Error decrypting and retrieving data: {str(e)}"
            logger.error(error_msg)
            raise SecurityError(error_msg) from e

    async def anonymize_and_store(
        self,
        data: pd.DataFrame,
        collection_name: str,
        sensitive_columns: List[str],
        user_id: str
    ) -> str:
        """
        データを匿名化してFirestoreに保存します
        """
        try:
            # データの匿名化
            anonymized_df = data.copy()
            for column in sensitive_columns:
                if column in anonymized_df.columns:
                    # 単純なハッシュ化（実際のプロジェクトではより高度な匿名化手法を使用）
                    anonymized_df[column] = anonymized_df[column].apply(
                        lambda x: hash(str(x)) if pd.notnull(x) else None
                    )

            # 匿名化データの保存
            anonymized_data = {
                'data': anonymized_df.to_dict('records'),
                'anonymized_columns': sensitive_columns,
                'user_id': user_id,
                'created_at': datetime.now()
            }

            doc_ids = await self.firestore_service.save_results(
                results=[anonymized_data],
                collection_name=collection_name
            )

            # 監査ログの記録
            await self._log_security_event(
                event_type='data_anonymization',
                user_id=user_id,
                details={
                    'document_id': doc_ids[0],
                    'collection': collection_name,
                    'anonymized_columns': sensitive_columns
                }
            )

            return doc_ids[0]

        except Exception as e:
            error_msg = f"Error anonymizing and storing data: {str(e)}"
            logger.error(error_msg)
            raise SecurityError(error_msg) from e

    async def _log_security_event(
        self,
        event_type: str,
        user_id: str,
        details: Dict[str, Any]
    ) -> None:
        """
        セキュリティイベントを監査ログに記録します
        """
        try:
            audit_log = {
                'event_type': event_type,
                'user_id': user_id,
                'timestamp': datetime.now(),
                'details': details,
                'ip_address': None  # 実装時にリクエストからIPアドレスを取得
            }

            await self.firestore_service.save_results(
                results=[audit_log],
                collection_name=AUDIT_COLLECTION
            )

        except Exception as e:
            logger.error(f"Failed to log security event: {str(e)}")
            # 監査ログの失敗は致命的ではないため、例外は発生させない

def get_security_service() -> FirestoreSecurityService:
    """
    FirestoreSecurityServiceのインスタンスを取得します
    """
    try:
        return FirestoreSecurityService()
    except Exception as e:
        logger.error(f"Failed to create security service: {str(e)}")
        raise SecurityError(f"Service creation error: {str(e)}") from e