"""
mTLS認証システムのテスト

TDD Phase 1, Task 1.3: mTLS認証システム
RED段階: 失敗するテストを最初に書く

TDD.yamlに基づく要件:
- test_client_certificate_validation: クライアント証明書検証
- test_certificate_rotation: 証明書ローテーション
- test_unauthorized_access_rejection: 不正アクセス拒否
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import ssl
import socket
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import asyncio


class TestMTLSAuthentication:
    """mTLS認証システムのテストクラス

    TDD.yamlに基づく要件実装:
    - 証明書管理システムのテスト
    - クライアント認証ミドルウェアのテスト
    - 証明書ローテーション機能のテスト
    - 不正アクセス拒否のテスト
    """

    def test_client_certificate_validation(self):
        """クライアント証明書検証のテスト

        要件:
        - 有効な証明書の検証が成功すること
        - 期限切れ証明書の検証が失敗すること
        - 無効な証明書の検証が失敗すること
        - 証明書チェーンの検証が正しく動作すること
        """
        # まだMTLSAuthenticatorクラスが実装されていないため、このテストは失敗する（RED段階）
        from backend.federated_learning.security.mtls_auth import MTLSAuthenticator

        authenticator = MTLSAuthenticator(
            ca_cert_path="test_ca.crt",
            server_cert_path="test_server.crt",
            server_key_path="test_server.key"
        )

        # 有効な証明書での検証
        valid_cert = self._create_test_certificate(
            common_name="valid_client",
            valid_days=30
        )

        result = authenticator.validate_client_certificate(valid_cert)
        assert result.is_valid, "有効な証明書の検証が成功すること"
        assert result.client_id == "valid_client", "クライアントIDが正しく抽出されること"
        assert result.permissions is not None, "権限情報が取得されること"

        # 期限切れ証明書での検証
        expired_cert = self._create_test_certificate(
            common_name="expired_client",
            valid_days=-1,  # 昨日期限切れ
            ca_signed=True  # CA署名証明書として作成
        )

        result = authenticator.validate_client_certificate(expired_cert)
        assert not result.is_valid, "期限切れ証明書の検証が失敗すること"
        assert "期限" in result.error_message or "expired" in result.error_message.lower(), "エラーメッセージに期限切れが含まれること"

    def test_invalid_certificate_rejection(self):
        """無効な証明書の拒否テスト"""
        from backend.federated_learning.security.mtls_auth import MTLSAuthenticator

        authenticator = MTLSAuthenticator(
            ca_cert_path="test_ca.crt",
            server_cert_path="test_server.crt",
            server_key_path="test_server.key"
        )

        # 自己署名証明書（CA署名でない）
        self_signed_cert = self._create_self_signed_certificate("malicious_client")

        result = authenticator.validate_client_certificate(self_signed_cert)
        assert not result.is_valid, "自己署名証明書の検証が失敗すること"
        assert "信頼" in result.error_message or "untrusted" in result.error_message.lower(), "信頼されていない証明書エラー"

        # 無効な形式の証明書
        with pytest.raises(ValueError, match="無効な証明書形式"):
            authenticator.validate_client_certificate("invalid_cert_data")

    def test_certificate_rotation(self):
        """証明書ローテーションのテスト

        要件:
        - 新しい証明書への自動切り替え
        - 古い証明書の無効化
        - ローテーション中の接続継続
        - ローテーション履歴の記録
        """
        from backend.federated_learning.security.mtls_auth import MTLSAuthenticator

        authenticator = MTLSAuthenticator(
            ca_cert_path="test_ca.crt",
            server_cert_path="test_server.crt",
            server_key_path="test_server.key",
            enable_auto_rotation=True,
            rotation_threshold_days=30
        )

        # 初期証明書の設定
        old_cert = self._create_test_certificate(
            common_name="client_1",
            valid_days=5,  # 5日後に期限切れ
            ca_signed=True
        )

        # 証明書をシステムに登録
        authenticator.register_client_certificate("client_1", old_cert)

        # ローテーションが必要かチェック
        needs_rotation = authenticator.check_certificate_rotation_needed("client_1")
        assert needs_rotation, "期限が近い証明書はローテーションが必要"

        # 新しい証明書を生成
        new_cert = self._create_test_certificate(
            common_name="client_1",
            valid_days=365,
            ca_signed=True
        )

        # ローテーション実行
        rotation_result = authenticator.rotate_client_certificate(
            client_id="client_1",
            new_certificate=new_cert
        )

        assert rotation_result.success, "証明書ローテーションが成功すること"
        assert rotation_result.old_cert_fingerprint != rotation_result.new_cert_fingerprint

        # 古い証明書は無効、新しい証明書は有効
        old_result = authenticator.validate_client_certificate(old_cert)
        new_result = authenticator.validate_client_certificate(new_cert)

        assert not old_result.is_valid, "古い証明書は無効になること"
        assert new_result.is_valid, "新しい証明書は有効であること"

        # ローテーション履歴の確認
        history = authenticator.get_rotation_history("client_1")
        assert len(history) == 1, "ローテーション履歴が記録されること"
        assert history[0]["event"] == "certificate_rotated"

    def test_unauthorized_access_rejection(self):
        """不正アクセス拒否のテスト

        要件:
        - 証明書なしのアクセス拒否
        - 無効な証明書でのアクセス拒否
        - ブラックリスト証明書の拒否
        - レート制限による拒否
        """
        from backend.federated_learning.security.mtls_auth import MTLSAuthenticator

        authenticator = MTLSAuthenticator(
            ca_cert_path="test_ca.crt",
            server_cert_path="test_server.crt",
            server_key_path="test_server.key",
            enable_rate_limiting=True,
            max_requests_per_minute=60
        )

        # 証明書なしのアクセス
        with pytest.raises(ssl.SSLError, match="証明書が提供されていません"):
            authenticator.authenticate_request(None, "127.0.0.1")

        # ブラックリスト証明書
        malicious_cert = self._create_test_certificate(
            common_name="malicious_client",
            valid_days=30,
            ca_signed=True  # CA署名証明書として作成
        )

        # 証明書をブラックリストに追加
        authenticator.add_to_blacklist("malicious_client", "security_violation")

        result = authenticator.validate_client_certificate(malicious_cert)
        assert not result.is_valid, "ブラックリスト証明書は拒否されること"
        assert "ブラックリスト" in result.error_message or "blacklisted" in result.error_message.lower()

        # レート制限テスト
        valid_cert = self._create_test_certificate("rate_test_client", 30, ca_signed=True)

        # 短時間に大量リクエスト
        for i in range(65):  # 制限を超える
            try:
                authenticator.authenticate_request(valid_cert, "127.0.0.1")
            except Exception as e:
                if "rate limit" in str(e).lower():
                    break
        else:
            pytest.fail("レート制限が機能していません")

    def test_mtls_middleware_integration(self):
        """mTLS認証ミドルウェアの統合テスト

        要件:
        - HTTPSリクエストでの証明書検証
        - 認証済みリクエストの通過
        - 認証失敗リクエストの拒否
        - 認証情報のリクエストコンテキストへの追加
        """
        from backend.federated_learning.security.mtls_auth import MTLSAuthenticationMiddleware

        # テスト用のWebアプリケーション作成
        middleware = MTLSAuthenticationMiddleware(
            ca_cert_path="test_ca.crt",
            server_cert_path="test_server.crt",
            server_key_path="test_server.key"
        )

        # モックリクエスト作成
        mock_request = Mock()
        mock_request.environ = {
            'SSL_CLIENT_CERT': self._create_test_certificate("test_client", 30, ca_signed=True).public_bytes(
                serialization.Encoding.PEM
            ).decode(),
            'REMOTE_ADDR': '127.0.0.1'
        }

        # 認証処理
        auth_result = middleware.authenticate(mock_request)

        assert auth_result.authenticated, "有効な証明書で認証が成功すること"
        assert auth_result.client_id == "test_client"
        assert hasattr(auth_result, 'permissions'), "権限情報が設定されること"

        # 認証失敗のケース
        mock_request.environ['SSL_CLIENT_CERT'] = "invalid_cert"

        auth_result = middleware.authenticate(mock_request)
        assert not auth_result.authenticated, "無効な証明書で認証が失敗すること"

    def test_certificate_management_system(self):
        """証明書管理システムのテスト

        要件:
        - 証明書の発行
        - 証明書の取り消し
        - 証明書の一覧表示
        - 証明書の有効性確認
        """
        from backend.federated_learning.security.mtls_auth import CertificateManager

        cert_manager = CertificateManager(
            ca_cert_path="test_ca.crt",
            ca_key_path="test_ca.key"
        )

        # 新しいクライアント証明書の発行
        cert_request = {
            "common_name": "new_client",
            "organization": "Test Org",
            "country": "JP",
            "validity_days": 365,
            "key_usage": ["digital_signature", "key_encipherment"]
        }

        issued_cert = cert_manager.issue_certificate(cert_request)

        assert issued_cert.success, "証明書発行が成功すること"
        assert issued_cert.certificate is not None, "証明書が生成されること"
        assert issued_cert.private_key is not None, "秘密鍵が生成されること"
        assert issued_cert.serial_number is not None, "シリアル番号が設定されること"

        # 発行した証明書の確認
        cert_info = cert_manager.get_certificate_info(issued_cert.serial_number)
        assert cert_info.common_name == "new_client"
        assert cert_info.status == "active"

        # 証明書の取り消し
        revocation_result = cert_manager.revoke_certificate(
            serial_number=issued_cert.serial_number,
            reason="testing"
        )

        assert revocation_result.success, "証明書の取り消しが成功すること"

        # 取り消し後の状態確認
        cert_info = cert_manager.get_certificate_info(issued_cert.serial_number)
        assert cert_info.status == "revoked", "証明書が取り消し状態になること"

    @pytest.mark.asyncio
    async def test_async_certificate_validation(self):
        """非同期証明書検証のテスト

        要件:
        - 並行証明書検証
        - 大量接続でのパフォーマンス
        - タイムアウト処理
        """
        from backend.federated_learning.security.mtls_auth import AsyncMTLSAuthenticator

        authenticator = AsyncMTLSAuthenticator(
            ca_cert_path="test_ca.crt",
            server_cert_path="test_server.crt",
            server_key_path="test_server.key",
            max_concurrent_validations=10
        )

        # 複数の証明書を並行検証
        certificates = [
            self._create_test_certificate(f"client_{i}", 30, ca_signed=True)
            for i in range(5)
        ]

        # 並行検証実行
        validation_tasks = [
            authenticator.validate_certificate_async(cert)
            for cert in certificates
        ]

        results = await asyncio.wait_for(
            asyncio.gather(*validation_tasks),
            timeout=10.0
        )

        # 全て成功することを確認
        for result in results:
            assert result.is_valid, "並行検証で全ての有効な証明書が検証されること"

        # タイムアウトテスト
        slow_cert = self._create_test_certificate("slow_client", 30, ca_signed=True)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                authenticator.validate_certificate_async(slow_cert, simulate_slow=True),
                timeout=1.0
            )

    def test_performance_benchmark(self):
        """mTLS認証のパフォーマンステスト

        要件:
        - 証明書検証時間 < 100ms
        - 1000件/秒の処理能力
        - メモリ使用量の制限
        """
        from backend.federated_learning.security.mtls_auth import MTLSAuthenticator
        import time

        authenticator = MTLSAuthenticator(
            ca_cert_path="test_ca.crt",
            server_cert_path="test_server.crt",
            server_key_path="test_server.key"
        )

        # 単体証明書検証のパフォーマンス
        test_cert = self._create_test_certificate("perf_test", 30, ca_signed=True)

        start_time = time.time()
        result = authenticator.validate_client_certificate(test_cert)
        validation_time = time.time() - start_time

        assert validation_time < 0.1, f"証明書検証時間が遅すぎます: {validation_time}秒"
        assert result.is_valid, "パフォーマンステスト用証明書が有効であること"

        # 大量検証のスループットテスト
        certificates = [
            self._create_test_certificate(f"bulk_test_{i}", 30, ca_signed=True)
            for i in range(100)
        ]

        start_time = time.time()
        valid_count = 0

        for cert in certificates:
            result = authenticator.validate_client_certificate(cert)
            if result.is_valid:
                valid_count += 1

        total_time = time.time() - start_time
        throughput = valid_count / total_time

        assert throughput > 100, f"スループットが低すぎます: {throughput:.2f} certs/sec"
        assert valid_count == 100, "全ての証明書が正しく検証されること"

    # ヘルパーメソッド
    def _create_test_certificate(self, common_name: str, valid_days: int, ca_signed: bool = True):
        """テスト用証明書の作成"""
        # 秘密鍵生成
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )

        # 日付の設定
        if valid_days < 0:
            # 期限切れ証明書：過去から昨日まで有効
            not_valid_before = datetime.utcnow() + timedelta(days=valid_days - 30)  # 30日前から
            not_valid_after = datetime.utcnow() + timedelta(days=valid_days)  # 指定日数前まで（期限切れ）
        else:
            # 通常の証明書：今から指定日数後まで有効
            not_valid_before = datetime.utcnow()
            not_valid_after = datetime.utcnow() + timedelta(days=valid_days)

        if ca_signed:
            # CA証明書を生成（テスト用）
            ca_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )

            # CA証明書作成
            ca_subject = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "JP"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Tokyo"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "Tokyo"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test CA"),
                x509.NameAttribute(NameOID.COMMON_NAME, "Test CA Root"),
            ])

            # クライアント証明書のsubject
            client_subject = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "JP"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Tokyo"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "Tokyo"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test Org"),
                x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            ])

            # CA署名済みクライアント証明書を作成
            cert = x509.CertificateBuilder().subject_name(
                client_subject
            ).issuer_name(
                ca_subject  # CA署名
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                not_valid_before
            ).not_valid_after(
                not_valid_after
            ).sign(ca_private_key, hashes.SHA256())  # CA鍵で署名
        else:
            # 自己署名証明書
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "JP"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Tokyo"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "Tokyo"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test Org"),
                x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            ])

            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                not_valid_before
            ).not_valid_after(
                not_valid_after
            ).sign(private_key, hashes.SHA256())

        return cert

    def _create_self_signed_certificate(self, common_name: str):
        """自己署名証明書の作成"""
        return self._create_test_certificate(common_name, 30, ca_signed=False)

@pytest.fixture
def temp_certificate_dir():
    """テスト用一時証明書ディレクトリ"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def mock_ssl_context():
    """モックSSLコンテキスト"""
    with patch('ssl.create_default_context') as mock:
        mock_context = Mock()
        mock.return_value = mock_context
        yield mock_context