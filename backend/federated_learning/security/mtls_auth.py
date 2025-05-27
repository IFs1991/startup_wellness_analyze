"""
mTLS認証システム実装

TDD Phase 1, Task 1.3: mTLS認証システム
GREEN段階: テストを通す最小限のコードを実装

実装要件（TDD.yamlより）:
- 証明書管理システム
- Nginx/Envoy設定
- クライアント認証ミドルウェア
"""

import logging
import ssl
import socket
import hashlib
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from collections import defaultdict, deque
from threading import Lock
import tempfile

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """証明書検証結果"""
    is_valid: bool
    client_id: Optional[str] = None
    permissions: Optional[List[str]] = None
    error_message: Optional[str] = None
    certificate_fingerprint: Optional[str] = None
    expiry_date: Optional[datetime] = None

@dataclass
class RotationResult:
    """証明書ローテーション結果"""
    success: bool
    old_cert_fingerprint: Optional[str] = None
    new_cert_fingerprint: Optional[str] = None
    rotation_time: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class IssuedCertificate:
    """発行済み証明書情報"""
    success: bool
    certificate: Optional[x509.Certificate] = None
    private_key: Optional[rsa.RSAPrivateKey] = None
    serial_number: Optional[int] = None
    error_message: Optional[str] = None

@dataclass
class CertificateInfo:
    """証明書情報"""
    serial_number: int
    common_name: str
    organization: str
    country: str
    status: str  # active, revoked, expired
    issued_date: datetime
    expiry_date: datetime
    fingerprint: str

@dataclass
class RevocationResult:
    """証明書取り消し結果"""
    success: bool
    serial_number: Optional[int] = None
    reason: Optional[str] = None
    revocation_time: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class AuthenticationResult:
    """認証結果"""
    authenticated: bool
    client_id: Optional[str] = None
    permissions: Optional[List[str]] = None
    error_message: Optional[str] = None

class RateLimiter:
    """レート制限機能"""

    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(deque)
        self.lock = Lock()

    def is_allowed(self, client_id: str) -> bool:
        """レート制限チェック"""
        with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]

            # 古いリクエストを削除
            while client_requests and client_requests[0] < now - self.time_window:
                client_requests.popleft()

            # 制限チェック
            if len(client_requests) >= self.max_requests:
                return False

            # 新しいリクエストを記録
            client_requests.append(now)
            return True

class MTLSAuthenticator:
    """mTLS認証システム

    クライアント証明書による相互認証を実装。
    証明書検証、ローテーション、ブラックリスト管理を含む。
    """

    def __init__(self,
                 ca_cert_path: str,
                 server_cert_path: str,
                 server_key_path: str,
                 enable_auto_rotation: bool = False,
                 rotation_threshold_days: int = 30,
                 enable_rate_limiting: bool = False,
                 max_requests_per_minute: int = 60):
        """初期化

        Args:
            ca_cert_path: CA証明書パス
            server_cert_path: サーバー証明書パス
            server_key_path: サーバー秘密鍵パス
            enable_auto_rotation: 自動ローテーション有効化
            rotation_threshold_days: ローテーション閾値日数
            enable_rate_limiting: レート制限有効化
            max_requests_per_minute: 分あたり最大リクエスト数
        """
        self.ca_cert_path = ca_cert_path
        self.server_cert_path = server_cert_path
        self.server_key_path = server_key_path
        self.enable_auto_rotation = enable_auto_rotation
        self.rotation_threshold_days = rotation_threshold_days

        # 証明書ストレージ
        self.registered_certificates = {}  # client_id -> certificate
        self.certificate_metadata = {}     # client_id -> metadata

        # ブラックリスト
        self.blacklist = set()  # client_ids
        self.blacklist_reasons = {}  # client_id -> reason

        # ローテーション履歴
        self.rotation_history = defaultdict(list)

        # 無効化された証明書のフィンガープリント
        self.revoked_fingerprints = set()

        # レート制限
        self.rate_limiter = None
        if enable_rate_limiting:
            self.rate_limiter = RateLimiter(max_requests_per_minute, 60)

        logger.info(f"mTLS認証システム初期化完了: "
                   f"auto_rotation={enable_auto_rotation}, "
                   f"rate_limiting={enable_rate_limiting}")

    def validate_client_certificate(self, certificate) -> ValidationResult:
        """クライアント証明書の検証

        Args:
            certificate: 検証する証明書

        Returns:
            検証結果
        """
        try:
            # 引数の型チェック
            if isinstance(certificate, str):
                raise ValueError("無効な証明書形式")

            if not isinstance(certificate, x509.Certificate):
                raise ValueError("無効な証明書形式")

            # 基本的な証明書情報を抽出
            subject = certificate.subject
            common_name = None

            for attribute in subject:
                if attribute.oid == NameOID.COMMON_NAME:
                    common_name = attribute.value
                    break

            if not common_name:
                return ValidationResult(
                    is_valid=False,
                    error_message="証明書にCommon Nameが設定されていません"
                )

            # ブラックリストチェック
            if common_name in self.blacklist:
                return ValidationResult(
                    is_valid=False,
                    client_id=common_name,
                    error_message=f"クライアントがブラックリストに登録されています: {self.blacklist_reasons.get(common_name, 'Unknown')}"
                )

            # 有効期限チェック
            now = datetime.utcnow()
            if certificate.not_valid_after < now:
                return ValidationResult(
                    is_valid=False,
                    client_id=common_name,
                    error_message="証明書の有効期限が切れています",
                    expiry_date=certificate.not_valid_after
                )

            if certificate.not_valid_before > now:
                return ValidationResult(
                    is_valid=False,
                    client_id=common_name,
                    error_message="証明書がまだ有効ではありません"
                )

            # 証明書フィンガープリント計算
            fingerprint = hashlib.sha256(
                certificate.public_bytes(serialization.Encoding.DER)
            ).hexdigest()

            # 無効化された証明書チェック
            if fingerprint in self.revoked_fingerprints:
                return ValidationResult(
                    is_valid=False,
                    client_id=common_name,
                    error_message="証明書は無効化されています",
                    certificate_fingerprint=fingerprint
                )

            # CA署名かチェック：issuerとsubjectが異なればCA署名
            is_ca_signed = certificate.issuer != certificate.subject

            # 自己署名証明書チェック（簡単な実装）
            if not is_ca_signed:
                return ValidationResult(
                    is_valid=False,
                    client_id=common_name,
                    error_message="自己署名証明書は信頼されていません",
                    certificate_fingerprint=fingerprint
                )

            # 権限情報の設定（デフォルト）
            permissions = ["federated_learning", "model_update"]

            return ValidationResult(
                is_valid=True,
                client_id=common_name,
                permissions=permissions,
                certificate_fingerprint=fingerprint,
                expiry_date=certificate.not_valid_after
            )

        except ValueError:
            # ValueErrorは再発生させる
            raise
        except Exception as e:
            logger.error(f"証明書検証エラー: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"証明書検証中にエラーが発生しました: {str(e)}"
            )

    def register_client_certificate(self, client_id: str, certificate: x509.Certificate):
        """クライアント証明書の登録

        Args:
            client_id: クライアントID
            certificate: クライアント証明書
        """
        self.registered_certificates[client_id] = certificate
        self.certificate_metadata[client_id] = {
            "registered_at": datetime.utcnow(),
            "fingerprint": hashlib.sha256(
                certificate.public_bytes(serialization.Encoding.DER)
            ).hexdigest()
        }
        logger.info(f"クライアント証明書を登録: {client_id}")

    def check_certificate_rotation_needed(self, client_id: str) -> bool:
        """証明書ローテーションが必要かチェック

        Args:
            client_id: クライアントID

        Returns:
            ローテーションが必要かどうか
        """
        if client_id not in self.registered_certificates:
            return False

        certificate = self.registered_certificates[client_id]
        expiry_date = certificate.not_valid_after
        threshold_date = datetime.utcnow() + timedelta(days=self.rotation_threshold_days)

        return expiry_date <= threshold_date

    def rotate_client_certificate(self,
                                client_id: str,
                                new_certificate: x509.Certificate) -> RotationResult:
        """クライアント証明書のローテーション

        Args:
            client_id: クライアントID
            new_certificate: 新しい証明書

        Returns:
            ローテーション結果
        """
        try:
            old_cert = self.registered_certificates.get(client_id)
            old_fingerprint = None

            if old_cert:
                old_fingerprint = hashlib.sha256(
                    old_cert.public_bytes(serialization.Encoding.DER)
                ).hexdigest()

            # 新しい証明書のフィンガープリント
            new_fingerprint = hashlib.sha256(
                new_certificate.public_bytes(serialization.Encoding.DER)
            ).hexdigest()

            # 証明書を更新
            self.registered_certificates[client_id] = new_certificate
            self.certificate_metadata[client_id].update({
                "last_rotated": datetime.utcnow(),
                "fingerprint": new_fingerprint
            })

            # 古い証明書を無効化
            if old_fingerprint:
                self.revoked_fingerprints.add(old_fingerprint)

            # ローテーション履歴に記録
            rotation_time = datetime.utcnow()
            self.rotation_history[client_id].append({
                "event": "certificate_rotated",
                "timestamp": rotation_time,
                "old_fingerprint": old_fingerprint,
                "new_fingerprint": new_fingerprint
            })

            logger.info(f"証明書ローテーション完了: {client_id}")

            return RotationResult(
                success=True,
                old_cert_fingerprint=old_fingerprint,
                new_cert_fingerprint=new_fingerprint,
                rotation_time=rotation_time
            )

        except Exception as e:
            logger.error(f"証明書ローテーションエラー: {e}")
            return RotationResult(
                success=False,
                error_message=str(e)
            )

    def get_rotation_history(self, client_id: str) -> List[Dict]:
        """ローテーション履歴の取得

        Args:
            client_id: クライアントID

        Returns:
            ローテーション履歴
        """
        return self.rotation_history.get(client_id, [])

    def add_to_blacklist(self, client_id: str, reason: str):
        """クライアントをブラックリストに追加

        Args:
            client_id: クライアントID
            reason: ブラックリスト理由
        """
        self.blacklist.add(client_id)
        self.blacklist_reasons[client_id] = reason
        logger.warning(f"クライアントをブラックリストに追加: {client_id}, 理由: {reason}")

    def authenticate_request(self, certificate, client_ip: str) -> AuthenticationResult:
        """リクエストの認証

        Args:
            certificate: クライアント証明書
            client_ip: クライアントIPアドレス

        Returns:
            認証結果

        Raises:
            ssl.SSLError: 証明書が提供されていない場合
        """
        if certificate is None:
            raise ssl.SSLError("証明書が提供されていません")

        # 証明書検証
        validation_result = self.validate_client_certificate(certificate)

        if not validation_result.is_valid:
            return AuthenticationResult(
                authenticated=False,
                error_message=validation_result.error_message
            )

        client_id = validation_result.client_id

        # レート制限チェック
        if self.rate_limiter and not self.rate_limiter.is_allowed(client_id):
            raise Exception(f"Rate limit exceeded for client: {client_id}")

        return AuthenticationResult(
            authenticated=True,
            client_id=client_id,
            permissions=validation_result.permissions
        )

class MTLSAuthenticationMiddleware:
    """mTLS認証ミドルウェア

    Webアプリケーション用のmTLS認証ミドルウェア。
    """

    def __init__(self,
                 ca_cert_path: str,
                 server_cert_path: str,
                 server_key_path: str):
        """初期化

        Args:
            ca_cert_path: CA証明書パス
            server_cert_path: サーバー証明書パス
            server_key_path: サーバー秘密鍵パス
        """
        self.authenticator = MTLSAuthenticator(
            ca_cert_path=ca_cert_path,
            server_cert_path=server_cert_path,
            server_key_path=server_key_path
        )

    def authenticate(self, request) -> AuthenticationResult:
        """リクエストの認証

        Args:
            request: HTTPリクエストオブジェクト

        Returns:
            認証結果
        """
        try:
            # SSL_CLIENT_CERT環境変数から証明書を取得
            cert_pem = request.environ.get('SSL_CLIENT_CERT')
            if not cert_pem:
                return AuthenticationResult(
                    authenticated=False,
                    error_message="クライアント証明書が提供されていません"
                )

            # PEM形式の証明書をパース
            try:
                certificate = x509.load_pem_x509_certificate(
                    cert_pem.encode() if isinstance(cert_pem, str) else cert_pem,
                    default_backend()
                )
            except Exception:
                return AuthenticationResult(
                    authenticated=False,
                    error_message="証明書の形式が無効です"
                )

            # クライアントIP取得
            client_ip = request.environ.get('REMOTE_ADDR', 'unknown')

            # 認証実行
            return self.authenticator.authenticate_request(certificate, client_ip)

        except Exception as e:
            logger.error(f"ミドルウェア認証エラー: {e}")
            return AuthenticationResult(
                authenticated=False,
                error_message=f"認証処理中にエラーが発生しました: {str(e)}"
            )

class CertificateManager:
    """証明書管理システム

    証明書の発行、取り消し、管理を行う。
    """

    def __init__(self, ca_cert_path: str, ca_key_path: str):
        """初期化

        Args:
            ca_cert_path: CA証明書パス
            ca_key_path: CA秘密鍵パス
        """
        self.ca_cert_path = ca_cert_path
        self.ca_key_path = ca_key_path

        # 証明書データベース（メモリ実装）
        self.certificate_db = {}  # serial_number -> CertificateInfo
        self.revoked_certificates = set()  # revoked serial numbers

        logger.info("証明書管理システム初期化完了")

    def issue_certificate(self, cert_request: Dict[str, Any]) -> IssuedCertificate:
        """証明書の発行

        Args:
            cert_request: 証明書リクエスト

        Returns:
            発行済み証明書
        """
        try:
            # 秘密鍵生成
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )

            # 証明書の設定
            subject_name = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, cert_request.get("country", "JP")),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, cert_request.get("organization", "Test Org")),
                x509.NameAttribute(NameOID.COMMON_NAME, cert_request["common_name"]),
            ])

            # CA証明書と秘密鍵の読み込み（モック実装）
            # 実際の実装では実際のCAファイルを読み込む
            ca_private_key = private_key  # テスト用

            # 証明書の作成
            serial_number = x509.random_serial_number()
            validity_days = cert_request.get("validity_days", 365)

            certificate = x509.CertificateBuilder().subject_name(
                subject_name
            ).issuer_name(
                subject_name  # テスト用（実際はCA名）
            ).public_key(
                private_key.public_key()
            ).serial_number(
                serial_number
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=validity_days)
            ).sign(ca_private_key, hashes.SHA256(), backend=default_backend())

            # 証明書情報をデータベースに保存
            fingerprint = hashlib.sha256(
                certificate.public_bytes(serialization.Encoding.DER)
            ).hexdigest()

            cert_info = CertificateInfo(
                serial_number=serial_number,
                common_name=cert_request["common_name"],
                organization=cert_request.get("organization", "Test Org"),
                country=cert_request.get("country", "JP"),
                status="active",
                issued_date=datetime.utcnow(),
                expiry_date=datetime.utcnow() + timedelta(days=validity_days),
                fingerprint=fingerprint
            )

            self.certificate_db[serial_number] = cert_info

            logger.info(f"証明書発行完了: {cert_request['common_name']}, シリアル: {serial_number}")

            return IssuedCertificate(
                success=True,
                certificate=certificate,
                private_key=private_key,
                serial_number=serial_number
            )

        except Exception as e:
            logger.error(f"証明書発行エラー: {e}")
            return IssuedCertificate(
                success=False,
                error_message=str(e)
            )

    def get_certificate_info(self, serial_number: int) -> Optional[CertificateInfo]:
        """証明書情報の取得

        Args:
            serial_number: シリアル番号

        Returns:
            証明書情報
        """
        return self.certificate_db.get(serial_number)

    def revoke_certificate(self, serial_number: int, reason: str) -> RevocationResult:
        """証明書の取り消し

        Args:
            serial_number: シリアル番号
            reason: 取り消し理由

        Returns:
            取り消し結果
        """
        try:
            if serial_number not in self.certificate_db:
                return RevocationResult(
                    success=False,
                    error_message="指定されたシリアル番号の証明書が見つかりません"
                )

            # 証明書の状態を更新
            cert_info = self.certificate_db[serial_number]
            cert_info.status = "revoked"

            # 取り消しリストに追加
            self.revoked_certificates.add(serial_number)

            revocation_time = datetime.utcnow()

            logger.info(f"証明書取り消し完了: シリアル {serial_number}, 理由: {reason}")

            return RevocationResult(
                success=True,
                serial_number=serial_number,
                reason=reason,
                revocation_time=revocation_time
            )

        except Exception as e:
            logger.error(f"証明書取り消しエラー: {e}")
            return RevocationResult(
                success=False,
                error_message=str(e)
            )

class AsyncMTLSAuthenticator:
    """非同期mTLS認証システム

    大量接続に対応した非同期版のmTLS認証。
    """

    def __init__(self,
                 ca_cert_path: str,
                 server_cert_path: str,
                 server_key_path: str,
                 max_concurrent_validations: int = 100):
        """初期化

        Args:
            ca_cert_path: CA証明書パス
            server_cert_path: サーバー証明書パス
            server_key_path: サーバー秘密鍵パス
            max_concurrent_validations: 最大同時検証数
        """
        self.authenticator = MTLSAuthenticator(
            ca_cert_path=ca_cert_path,
            server_cert_path=server_cert_path,
            server_key_path=server_key_path
        )
        self.semaphore = asyncio.Semaphore(max_concurrent_validations)

    async def validate_certificate_async(self,
                                       certificate: x509.Certificate,
                                       simulate_slow: bool = False) -> ValidationResult:
        """非同期証明書検証

        Args:
            certificate: 検証する証明書
            simulate_slow: 遅い処理のシミュレーション

        Returns:
            検証結果
        """
        async with self.semaphore:
            if simulate_slow:
                await asyncio.sleep(2.0)  # 遅い処理をシミュレート

            # 同期版を非同期実行
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.authenticator.validate_client_certificate,
                certificate
            )

            return result

    def __str__(self) -> str:
        """文字列表現"""
        return f"AsyncMTLSAuthenticator(max_concurrent={self.semaphore._value})"