# -*- coding: utf-8 -*-
"""
セキュリティ管理モジュール
Startup Wellness データアクセスのセキュリティを管理します。
"""
from typing import Dict, Any, Optional, List
import logging
from firebase_admin import firestore
import jwt
from datetime import datetime, timedelta
import hashlib
import os
from fastapi import HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import re
from passlib.context import CryptContext
# 循環インポートを避けるため、直接インポートせず関数内で遅延インポートします
# from service.firestore.client import FirestoreService
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import pandas as pd
import json
# anjanaライブラリは使用しない
# import anjana  # ライブラリを追加

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 定数
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "一時的な秘密鍵")  # 本番環境では環境変数から取得すべき
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
TOKEN_TYPE = "bearer"
K_ANONYMITY_VALUE = 5  # k匿名性のデフォルト値
L_DIVERSITY_VALUE = 3  # l多様性のデフォルト値

# SecurityErrorクラスの定義
class SecurityError(Exception):
    """セキュリティ関連のエラー"""
    pass


# FirestoreSecurityServiceクラスの定義
class FirestoreSecurityService:
    def __init__(self):
        """初期化"""
        # コンストラクタでは何もインポートせず、必要になったときに動的にインポートします
        self.firestore_service = None
        self.encryption_key = os.environ.get("ENCRYPTION_KEY") or get_random_bytes(32)
        self.security_log_collection = "security_logs"

        # 認証情報のパターン
        self.pii_patterns = {
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "phone": r'\+?[0-9]{10,15}',
            "name": r'[A-Z][a-z]+ [A-Z][a-z]+',
            "address": r'\d+ [A-Za-z]+ (St|Ave|Rd|Blvd)',
            "credit_card": r'\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}',
            "ssn": r'\d{3}-\d{2}-\d{4}'
        }

    async def encrypt_and_store(
        self,
        data: Dict[str, Any],
        collection_name: str,
        user_id: str
    ) -> str:
        """データを暗号化してFirestoreに保存"""
        # 遅延インポート - 必要になった時点でインポート
        if self.firestore_service is None:
            from service.firestore.client import FirestoreService
            self.firestore_service = FirestoreService()
            self.db = self.firestore_service.db
            logger.info("Firestore Security Service initialized dynamically")

        try:
            # データの暗号化
            json_data = json.dumps(data)
            cipher = AES.new(self.encryption_key, AES.MODE_CBC)
            ct_bytes = cipher.encrypt(pad(json_data.encode('utf-8'), AES.block_size))

            # 暗号化データの保存
            encrypted_doc = {
                'encrypted_data': ct_bytes,
                'iv': cipher.iv,
                'user_id': user_id,
                'collection_ref': collection_name,
                'created_at': datetime.now()
            }

            doc_ids = await self.firestore_service.save_results(
                results=[encrypted_doc],
                collection_name="encrypted_documents"
            )

            # セキュリティログの記録
            await self._log_security_event(
                event_type='data_encryption',
                user_id=user_id,
                details={'document_id': doc_ids[0], 'collection': collection_name}
            )

            return doc_ids[0]

        except Exception as e:
            error_msg = f"データの暗号化と保存に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise SecurityError(error_msg) from e

    async def decrypt_and_retrieve(
        self,
        document_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """暗号化されたデータを復号して取得"""
        # 遅延インポート - 必要になった時点でインポート
        if self.firestore_service is None:
            from service.firestore.client import FirestoreService
            self.firestore_service = FirestoreService()
            self.db = self.firestore_service.db
            logger.info("Firestore Security Service initialized dynamically")

        try:
            # Firestoreからドキュメントを取得
            encrypted_docs = await self.firestore_service.fetch_documents(
                collection_name="encrypted_documents",
                conditions=[
                    {'field': '__name__', 'operator': '==', 'value': document_id},
                    {'field': 'user_id', 'operator': '==', 'value': user_id}
                ],
                limit=1
            )

            if not encrypted_docs:
                raise SecurityError(f"ドキュメントが見つからないかアクセスが拒否されました: {document_id}")

            encrypted_doc = encrypted_docs[0]

            # データの復号
            cipher = AES.new(self.encryption_key, AES.MODE_CBC, iv=encrypted_doc['iv'])
            decrypted_data = unpad(cipher.decrypt(encrypted_doc['encrypted_data']), AES.block_size)

            # セキュリティログの記録
            await self._log_security_event(
                event_type='data_decryption',
                user_id=user_id,
                details={'document_id': document_id}
            )

            return json.loads(decrypted_data.decode('utf-8'))

        except Exception as e:
            error_msg = f"データの復号と取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise SecurityError(error_msg) from e

    async def anonymize_and_store(
        self,
        data: pd.DataFrame,
        collection_name: str,
        sensitive_columns: List[str],
        user_id: str,
        quasi_identifiers: Optional[List[str]] = None,
        k_value: int = K_ANONYMITY_VALUE,
        l_value: int = L_DIVERSITY_VALUE
    ) -> str:
        """データを匿名化してFirestoreに保存"""
        # 遅延インポート - 必要になった時点でインポート
        if self.firestore_service is None:
            from service.firestore.client import FirestoreService
            self.firestore_service = FirestoreService()
            self.db = self.firestore_service.db
            logger.info("Firestore Security Service initialized dynamically")

        try:
            # K匿名性の処理
            logger.info(f"開始: {len(data)}行のデータ匿名化処理")

            # quasi_identifiersが指定されていない場合はsensitive_columnsを使用
            if quasi_identifiers is None:
                quasi_identifiers = sensitive_columns

            # センシティブ属性と準識別子の設定
            for column in data.columns:
                if column in sensitive_columns:
                    # センシティブカラムを匿名化（値をマスク）
                    data[column] = data[column].apply(lambda x: "****" if x else x)
                elif column in quasi_identifiers:
                    # 準識別子カラムを一般化（カテゴリ化または丸め）
                    if pd.api.types.is_numeric_dtype(data[column]):
                        # 数値データの場合は丸める
                        data[column] = data[column].apply(lambda x: round(x / 10) * 10 if x is not None else x)
                    else:
                        # 文字列データの場合は最初の文字だけ残す
                        data[column] = data[column].apply(lambda x: x[0] + "***" if x and isinstance(x, str) else x)

            # 匿名化データを保存用のデータフレームにコピー
            anonymized_df = data.copy()

            # 匿名化メトリクス（シンプルな実装）
            metrics = {
                "information_loss": 35.0,  # 仮の値
                "suppression_rate": 0.0,   # 仮の値
                "generalization_levels": {}
            }

            logger.info(f"匿名化完了: センシティブカラム: {sensitive_columns}, 準識別子: {quasi_identifiers}")

            # 匿名化データの保存
            anonymized_data = {
                'data': anonymized_df.to_dict('records'),
                'anonymized_columns': sensitive_columns,
                'quasi_identifiers': quasi_identifiers,
                'k_value': k_value,
                'l_value': l_value,
                'metrics': metrics,
                'user_id': user_id,
                'created_at': datetime.now()
            }

            doc_ids = await self.firestore_service.save_results(
                results=[anonymized_data],
                collection_name=collection_name
            )

            # セキュリティログの記録
            await self._log_security_event(
                event_type='data_anonymization',
                user_id=user_id,
                details={
                    'document_id': doc_ids[0],
                    'collection': collection_name,
                    'anonymized_columns': sensitive_columns,
                    'quasi_identifiers': quasi_identifiers,
                    'k_value': k_value,
                    'l_value': l_value
                }
            )

            return doc_ids[0]

        except Exception as e:
            error_msg = f"データの匿名化と保存に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise SecurityError(error_msg) from e

    async def _log_security_event(
        self,
        event_type: str,
        user_id: str,
        details: Dict[str, Any]
    ) -> None:
        """セキュリティイベントをログに記録"""
        # 遅延インポート - 必要になった時点でインポート
        if self.firestore_service is None:
            from service.firestore.client import FirestoreService
            self.firestore_service = FirestoreService()
            self.db = self.firestore_service.db
            logger.info("Firestore Security Service initialized dynamically")

        try:
            # ログデータの作成
            audit_log = {
                'event_type': event_type,
                'user_id': user_id,
                'timestamp': datetime.now(),
                'details': details,
                'ip_address': None  # 実装時にリクエストからIPアドレスを取得
            }

            await self.firestore_service.save_results(
                results=[audit_log],
                collection_name=self.security_log_collection
            )
            logger.debug(f"セキュリティイベントがログに記録されました: {event_type}")

        except Exception as e:
            logger.error(f"セキュリティイベントの記録に失敗しました: {str(e)}")
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

# SecurityManagerとしてFirestoreSecurityServiceをエイリアス
SecurityManager = FirestoreSecurityService