import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, Any, List
import jwt

from core.security import (
    SecurityError,
    FirestoreSecurityService,
    get_security_service
)

@pytest.fixture
def mock_firestore_service():
    """FirestoreServiceのモックを提供します"""
    service = MagicMock()
    service.add_document = AsyncMock()
    service.get_document = AsyncMock()
    service.query_documents = AsyncMock()
    return service

@pytest.fixture
def sample_data():
    """テスト用のサンプルデータを提供します"""
    return {
        'company_id': 'comp1',
        'sensitive_data': 'This is sensitive information',
        'user_name': 'Test User',
        'email': 'test@example.com',
        'revenue': 1000000,
        'created_at': datetime.now().isoformat()
    }

@pytest.fixture
def sample_dataframe():
    """テスト用のサンプルDataFrameを提供します"""
    return pd.DataFrame({
        'company_id': ['comp1', 'comp1', 'comp2', 'comp2', 'comp3'],
        'name': ['John Doe', 'Jane Smith', 'Alice Brown', 'Bob Wilson', 'Carol Adams'],
        'age': [42, 38, 45, 36, 51],
        'email': ['john@example.com', 'jane@example.com', 'alice@example.com', 'bob@example.com', 'carol@example.com'],
        'salary': [85000, 92000, 78000, 88000, 95000],
        'department': ['Sales', 'IT', 'HR', 'IT', 'Finance'],
        'years_employed': [5, 7, 3, 4, 10]
    })

@pytest.mark.asyncio
async def test_encrypt_and_store(mock_firestore_service, sample_data):
    """データの暗号化と保存機能をテスト"""
    # Crypto.Cipherのモック
    with patch('core.security.AES') as mock_aes, \
         patch('core.security.get_random_bytes') as mock_get_random_bytes, \
         patch('core.security.pad') as mock_pad, \
         patch('core.security.FirestoreService', return_value=mock_firestore_service):

        # モックの設定
        mock_get_random_bytes.return_value = b'1234567890123456'  # 16バイトのダミーIV
        mock_cipher = MagicMock()
        mock_aes.new.return_value = mock_cipher
        mock_cipher.encrypt.return_value = b'encrypted_data'
        mock_pad.return_value = b'padded_data'

        # Firestoreへの保存結果をモック
        mock_firestore_service.add_document.return_value = 'encrypted_doc_id'

        # セキュリティサービスのインスタンスを作成
        security_service = FirestoreSecurityService()

        # データの暗号化と保存を実行
        document_id = await security_service.encrypt_and_store(
            data=sample_data,
            collection_name='test_collection',
            user_id='user123'
        )

        # 結果を検証
        assert document_id == 'encrypted_doc_id'

        # 暗号化が正しく呼び出されたか検証
        mock_aes.new.assert_called_once()
        mock_cipher.encrypt.assert_called_once()

        # Firestoreにデータが保存されたか検証
        mock_firestore_service.add_document.assert_called_once()

        # 監査ログが記録されたか検証
        mock_firestore_service.add_document.assert_any_call(
            'security_audit_logs',
            {'event_type': 'encrypt', 'user_id': 'user123', 'timestamp': pytest.approx(datetime.now().timestamp(), abs=60)}
        )

@pytest.mark.asyncio
async def test_decrypt_and_retrieve(mock_firestore_service):
    """暗号化されたデータの復号化と取得機能をテスト"""
    # 復号化のためのモックデータ
    encrypted_data = {
        'encrypted_data': b'encrypted_bytes',
        'initialization_vector': b'1234567890123456',
        'metadata': {
            'collection': 'test_collection',
            'created_by': 'user123',
            'created_at': datetime.now().timestamp()
        }
    }

    # Firestoreから取得するデータをモック
    mock_firestore_service.get_document.return_value = encrypted_data

    with patch('core.security.AES') as mock_aes, \
         patch('core.security.unpad') as mock_unpad, \
         patch('core.security.FirestoreService', return_value=mock_firestore_service):

        # モックの設定
        mock_cipher = MagicMock()
        mock_aes.new.return_value = mock_cipher

        # 復号化されたJSONデータ
        decrypted_json = '{"company_id": "comp1", "data": "secret"}'
        mock_cipher.decrypt.return_value = b'padded_decrypted_data'
        mock_unpad.return_value = decrypted_json.encode('utf-8')

        # セキュリティサービスのインスタンスを作成
        security_service = FirestoreSecurityService()

        # データの復号化と取得を実行
        decrypted_data = await security_service.decrypt_and_retrieve(
            document_id='encrypted_doc_id',
            user_id='user123'
        )

        # 結果を検証 - JSON文字列がパースされたオブジェクトになっていることを確認
        assert decrypted_data == {"company_id": "comp1", "data": "secret"}

        # 復号化が正しく呼び出されたか検証
        mock_aes.new.assert_called_once()
        mock_cipher.decrypt.assert_called_once_with(b'encrypted_bytes')

        # Firestoreからデータが取得されたか検証
        mock_firestore_service.get_document.assert_called_once_with(
            'encrypted_documents', 'encrypted_doc_id'
        )

        # 監査ログが記録されたか検証
        mock_firestore_service.add_document.assert_called_once()

@pytest.mark.asyncio
async def test_anonymize_and_store(mock_firestore_service, sample_dataframe):
    """データの匿名化と保存機能をテスト"""
    with patch('core.security.anjana') as mock_anjana, \
         patch('core.security.FirestoreService', return_value=mock_firestore_service):

        # anjanaモジュールの匿名化関数をモック
        mock_anonymizer = MagicMock()
        mock_anjana.Anonymizer.return_value = mock_anonymizer

        # 匿名化されたDataFrameをモック
        anonymized_df = sample_dataframe.copy()
        # 個人情報を匿名化したと仮定
        anonymized_df['name'] = ['Person1', 'Person2', 'Person3', 'Person4', 'Person5']
        anonymized_df['email'] = ['email1@anon.com', 'email2@anon.com', 'email3@anon.com', 'email4@anon.com', 'email5@anon.com']
        mock_anonymizer.anonymize.return_value = anonymized_df

        # Firestoreへの保存結果をモック
        mock_firestore_service.add_document.return_value = 'anonymized_doc_id'

        # セキュリティサービスのインスタンスを作成
        security_service = FirestoreSecurityService()

        # データの匿名化と保存を実行
        document_id = await security_service.anonymize_and_store(
            data=sample_dataframe,
            collection_name='anonymized_data',
            sensitive_columns=['name', 'email', 'salary'],
            user_id='user123',
            quasi_identifiers=['age', 'department']
        )

        # 結果を検証
        assert document_id == 'anonymized_doc_id'

        # 匿名化が正しく呼び出されたか検証
        mock_anjana.Anonymizer.assert_called_once()
        mock_anonymizer.anonymize.assert_called_once()

        # Firestoreにデータが保存されたか検証
        mock_firestore_service.add_document.assert_called()

        # 監査ログが記録されたか検証
        call_args_list = mock_firestore_service.add_document.call_args_list
        audit_log_call = False
        for call in call_args_list:
            if call[0][0] == 'security_audit_logs':
                audit_log_call = True
                break
        assert audit_log_call, "監査ログが記録されていません"

@pytest.mark.asyncio
async def test_encrypt_and_store_error_handling(mock_firestore_service, sample_data):
    """暗号化処理のエラーハンドリングをテスト"""
    # 暗号化処理でエラーを発生させる
    with patch('core.security.AES.new', side_effect=Exception("Encryption error")), \
         patch('core.security.FirestoreService', return_value=mock_firestore_service):

        # セキュリティサービスのインスタンスを作成
        security_service = FirestoreSecurityService()

        # 例外が発生することを確認
        with pytest.raises(SecurityError) as excinfo:
            await security_service.encrypt_and_store(
                data=sample_data,
                collection_name='test_collection',
                user_id='user123'
            )

        # エラーメッセージを検証
        assert "Failed to encrypt data" in str(excinfo.value)

@pytest.mark.asyncio
async def test_decrypt_and_retrieve_error_handling(mock_firestore_service):
    """復号化処理のエラーハンドリングをテスト"""
    # 無効なドキュメントIDを指定
    mock_firestore_service.get_document.side_effect = Exception("Document not found")

    with patch('core.security.FirestoreService', return_value=mock_firestore_service):

        # セキュリティサービスのインスタンスを作成
        security_service = FirestoreSecurityService()

        # 例外が発生することを確認
        with pytest.raises(SecurityError) as excinfo:
            await security_service.decrypt_and_retrieve(
                document_id='invalid_doc_id',
                user_id='user123'
            )

        # エラーメッセージを検証
        assert "Failed to decrypt data" in str(excinfo.value)

def test_security_service_factory():
    """セキュリティサービスのファクトリ関数をテスト"""
    with patch('core.security.FirestoreSecurityService') as mock_security_class:
        # モックインスタンスを設定
        mock_instance = MagicMock()
        mock_security_class.return_value = mock_instance

        # ファクトリ関数を呼び出し
        service = get_security_service()

        # 結果を検証
        assert service == mock_instance
        mock_security_class.assert_called_once()