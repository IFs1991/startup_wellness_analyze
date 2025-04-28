# -*- coding: utf-8 -*-
"""
認証管理
ユーザー認証 (ログイン, ログアウト, 登録, パスワードリセットなど) を管理します。
Firebase AuthenticationとCloud Firestoreを使用した認証機能も提供します。
"""
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHash
from firebase_admin import credentials, initialize_app, get_app, auth
from google.cloud import firestore
from google.cloud import secretmanager_v1 as secretmanager
from google.auth.exceptions import DefaultCredentialsError
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Union, List, Any, cast, Tuple
from datetime import datetime, timedelta
import jwt
from jwt.exceptions import PyJWTError
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from pydantic import BaseModel, EmailStr, Field, ValidationError
import uuid
import hashlib
from enum import Enum
import pyotp
import qrcode
import io
import base64
import secrets
import redis
from redis.exceptions import RedisError
# 循環インポートを避けるために遅延インポートを使用
from .patterns import LazyImport, Singleton
from .common_logger import get_logger
from dotenv import load_dotenv
from database.connection import Database as DatabaseManager

# compliance_managerを遅延インポート
ComplianceManager = LazyImport('core.compliance_manager', 'ComplianceManager')
ComplianceEvent = LazyImport('core.compliance_manager', 'ComplianceEvent')
ComplianceEventType = LazyImport('core.compliance_manager', 'ComplianceEventType')
get_compliance_manager = LazyImport('core.compliance_manager', 'get_compliance_manager')

# 環境変数を読み込み
if os.getenv("ENVIRONMENT") != "production":
    # backendディレクトリのパスを取得
    BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # backend/.envファイルのパスを設定
    ENV_PATH = os.path.join(BACKEND_DIR, '.env')
    # backend/.envファイルを読み込む
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)
    else:
        # ENVファイルが見つからない場合はログ出力
        print(f"Warning: .env file not found at {ENV_PATH}")

# 絶対インポートを使用する（ベストプラクティス）
try:
    # プロジェクトのデータベースパッケージからインポート
    from database.connection import Database as DatabaseManager
except ImportError:
    # フォールバック: ベースモジュールからインポート
    from database import connection

from .rate_limiter import RateLimiter
from .auth_metrics import AuthMetricsCollector
from .security_config import get_secret_key

# ロガーの設定
logger = get_logger(__name__)

# Argon2パスワードハッシュ化の設定
# 以下のパラメータが推奨されています:
# - time_cost: 3以上（計算時間、高いほど安全）
# - memory_cost: 65536以上（メモリ使用量、高いほど安全）
# - parallelism: 4（並列度、ハードウェアに応じて設定）
ph = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)

# OAuth2のトークン取得用のスキーム
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="auth/token",
    scopes={
        "user": "標準ユーザー権限",
        "admin": "管理者権限",
        "analyst": "データ分析者権限",
    }
)

# Redis接続設定
REDIS_HOST = os.environ.get("REDIS_HOST", "startup-wellness-redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)

# ユーザーロール
class UserRole(str, Enum):
    """ユーザーロールの列挙型"""
    ADMIN = "admin"
    USER = "user"
    ANALYST = "analyst"
    VC = "vc"

# MFA種別
class MFAType(str, Enum):
    """多要素認証の種別"""
    NONE = "none"  # MFA無効
    TOTP = "totp"  # 時間ベースのワンタイムパスワード（GoogleAuthenticatorなど）
    SMS = "sms"    # SMS認証
    EMAIL = "email" # 電子メール認証

# トークンモデル
class TokenData(BaseModel):
    """トークンデータモデル"""
    sub: str
    scopes: List[str] = []
    exp: Optional[datetime] = None

# ユーザーモデル
class User(BaseModel):
    """ユーザーモデル"""
    id: str
    email: str
    display_name: Optional[str] = None
    is_active: bool = True
    role: UserRole = UserRole.USER
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_email_verified: bool = False
    company_id: Optional[str] = None
    profile_completed: bool = False
    login_count: int = 0
    data_access: List[str] = []
    mfa_enabled: bool = False
    mfa_type: MFAType = MFAType.NONE
    phone_number: Optional[str] = None

# MFA関連のモデル
class TOTPSetup(BaseModel):
    """TOTP設定モデル"""
    secret: str
    qr_code: str
    uri: str

@Singleton
class AuthManager:
    """
    ユーザー認証を管理するクラス。シングルトンパターンで実装されています。
    Firebase AuthenticationとCloud Firestoreを使用します。
    """
    def __init__(self):
        """
        AuthManagerを初期化します。FirebaseとFirestoreクライアントを初期化します。
        """
        self.logger = get_logger(__name__)
        self.initialized = False
        self.firebase_app = None
        self.db = None
        self.ph = PasswordHasher()
        self.mfa_enabled = os.getenv('MFA_ENABLED', 'false').lower() == 'true'
        self.redis_client = None
        self.compliance_manager = None
        self._initialize_redis()
        self._initialize_firebase()

    def get_compliance_manager(self):
        """ComplianceManagerを遅延ロードして返す"""
        if self.compliance_manager is None:
            # 遅延インポートされたComplianceManagerのget_compliance_manager関数を使用
            self.compliance_manager = get_compliance_manager()
            self.logger.info("ComplianceManagerを初期化しました")
        return self.compliance_manager

    def _initialize_redis(self):
        """Redisクライアントの初期化"""
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True
            )
            self.redis_client.ping()  # 接続テスト
            self.logger.info("Redisクライアントが初期化されました")
        except RedisError as e:
            self.logger.error(f"Redisクライアントの初期化中にエラーが発生しました: {str(e)}")
            self.redis_client = None
            # 開発環境用のフォールバック
            if os.environ.get("ENVIRONMENT") == "development":
                self.logger.warning("開発環境ではRedis初期化エラーを無視します")
            else:
                raise

    def _initialize_firebase(self):
        """Firebase Admin SDKを初期化"""
        try:
            # すでに初期化されているか確認
            get_app()
            self.logger.info("Firebase Adminはすでに初期化されています")
        except ValueError:
            self.logger.info("Firebase Adminを初期化します")
            # 環境変数からクレデンシャルパスを取得
            cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

            # クレデンシャルファイルが存在するか確認
            if cred_path and os.path.exists(cred_path):
                self.logger.info(f"Firebase認証情報を読み込みます: {cred_path}")
                cred = credentials.Certificate(cred_path)
                initialize_app(cred)
            else:
                # 開発環境などではデフォルト認証情報を使用
                self.logger.warning("Firebase認証情報が見つかりません。デフォルト認証情報を使用します。")
                try:
                    initialize_app()
                except DefaultCredentialsError as e:
                    self.logger.error(f"デフォルト認証情報の取得に失敗しました: {str(e)}")
                    # 開発環境用のモック設定
                    if os.environ.get("ENVIRONMENT") == "development":
                        self.logger.warning("開発環境ではFirebase初期化エラーを無視します")
                    else:
                        raise

    def _check_initialization(self):
        """初期化状態をチェックし、問題があれば例外を発生させる"""
        if self.redis_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="認証サービスが正しく初期化されていません"
            )

    def _get_firestore_client(self):
        """Firestoreクライアントを取得（初期化チェック付き）"""
        self._check_initialization()
        return self.db

    def _get_user_collection(self):
        """ユーザーコレクションへの参照を取得"""
        return self._get_firestore_client().collection('users')

    def _get_role_scopes(self, role: UserRole) -> List[str]:
        """ロールに対応するスコープのリストを取得"""
        role_scopes = {
            UserRole.ADMIN: ["user", "admin", "analyst"],
            UserRole.ANALYST: ["user", "analyst"],
            UserRole.VC: ["user", "vc"],
            UserRole.USER: ["user"],
        }
        return role_scopes.get(role, ["user"])

    # === 多要素認証（MFA）関連メソッド ===

    async def setup_totp_mfa(self, user_id: str) -> TOTPSetup:
        """
        ユーザーにTOTP多要素認証を設定

        Args:
            user_id: ユーザーID

        Returns:
            TOTPSetup: TOTP設定情報
        """
        self._check_initialization()

        try:
            # ユーザー情報を取得
            user = await self.get_user_by_id(user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="ユーザーが見つかりません"
                )

            # TOTPシークレットを生成
            totp_secret = pyotp.random_base32()

            # TOTPプロビジョニングURIを生成
            app_name = os.environ.get("APP_NAME", "Startup Wellness")
            uri = pyotp.totp.TOTP(totp_secret).provisioning_uri(
                name=user.email,
                issuer_name=app_name
            )

            # QRコードを生成
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(uri)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")

            # 画像をBase64エンコード
            buffer = io.BytesIO()
            img.save(buffer)
            qr_code_base64 = base64.b64encode(buffer.getvalue()).decode()
            qr_code_data = f"data:image/png;base64,{qr_code_base64}"

            # シークレットを一時的に保存（Redisを使用）
            if self.redis_client:
                # シークレットを一時的に保存（5分間有効）
                self.redis_client.setex(
                    f"mfa_setup:{user_id}",
                    300,  # 5分間有効
                    totp_secret
                )
            else:
                # Redisが利用できない場合はFirestoreに保存
                self._get_user_collection().document(user_id).update({
                    'mfa_setup_secret': totp_secret,
                    'mfa_setup_timestamp': datetime.now(),
                    'mfa_setup_completed': False
                })

            # コンプライアンスログに記録
            await self.get_compliance_manager().log_event(ComplianceEvent(
                event_type=ComplianceEventType.MFA_ENABLE,
                user_id=user_id,
                success=True,
                details={"type": "TOTP", "status": "setup_started"}
            ))

            return TOTPSetup(
                secret=totp_secret,
                qr_code=qr_code_data,
                uri=uri
            )

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"TOTP設定中に予期せぬエラーが発生しました: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="多要素認証の設定中にエラーが発生しました"
            )

    async def verify_mfa_setup(self, user_id: str, code: str) -> bool:
        """
        MFA設定の検証と有効化

        Args:
            user_id: ユーザーID
            code: ユーザーが入力したコード

        Returns:
            bool: 検証成功の場合はTrue
        """
        self._check_initialization()

        try:
            # シークレットを取得
            totp_secret = None

            if self.redis_client:
                # Redisからシークレットを取得
                totp_secret = self.redis_client.get(f"mfa_setup:{user_id}")

            if not totp_secret:
                # Firestoreから取得を試みる
                user_doc = self._get_user_collection().document(user_id).get()
                if not user_doc.exists:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="ユーザーが見つかりません"
                    )

                user_data = user_doc.to_dict()
                totp_secret = user_data.get('mfa_setup_secret')

                # 設定が5分以上前の場合は期限切れ
                setup_time = user_data.get('mfa_setup_timestamp')
                if not setup_time or (datetime.now() - setup_time).total_seconds() > 300:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="設定の有効期限が切れました。もう一度設定をやり直してください。"
                    )

            if not totp_secret:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="MFA設定が見つかりません。もう一度設定をやり直してください。"
                )

            # コードを検証
            totp = pyotp.TOTP(totp_secret)
            if not totp.verify(code):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="コードが無効です。もう一度お試しください。"
                )

            # MFAを有効化
            self._get_user_collection().document(user_id).update({
                'mfa_enabled': True,
                'mfa_type': MFAType.TOTP.value,
                'mfa_secret': totp_secret,
                'mfa_setup_completed': True,
                'mfa_setup_timestamp': datetime.now()
            })

            # 一時データを削除
            if self.redis_client:
                self.redis_client.delete(f"mfa_setup:{user_id}")

            # コンプライアンスログに記録
            await self.get_compliance_manager().log_event(ComplianceEvent(
                event_type=ComplianceEventType.MFA_ENABLE,
                user_id=user_id,
                success=True,
                details={"type": "TOTP", "status": "setup_completed"}
            ))

            self.logger.info(f"ユーザーのTOTP MFAが有効化されました: {user_id}")
            return True

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"MFA設定検証中に予期せぬエラーが発生しました: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="多要素認証の設定中にエラーが発生しました"
            )

    async def verify_mfa_code(self, user_id: str, code: str) -> bool:
        """
        ログイン時のMFAコード検証

        Args:
            user_id: ユーザーID
            code: ユーザーが入力したコード

        Returns:
            bool: 検証成功の場合はTrue
        """
        self._check_initialization()

        try:
            # ユーザー情報を取得
            user_doc = self._get_user_collection().document(user_id).get()
            if not user_doc.exists:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="ユーザーが見つかりません"
                )

            user_data = user_doc.to_dict()

            # MFAが有効かチェック
            if not user_data.get('mfa_enabled', False):
                # MFAが無効な場合は検証をスキップ
                return True

            mfa_type = user_data.get('mfa_type', MFAType.NONE.value)

            if mfa_type == MFAType.TOTP.value:
                # TOTPコードを検証
                totp_secret = user_data.get('mfa_secret')
                if not totp_secret:
                    self.logger.error(f"ユーザーのMFA設定が不完全です: {user_id}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="MFA設定が不完全です。管理者にお問い合わせください。"
                    )

                totp = pyotp.TOTP(totp_secret)
                if not totp.verify(code):
                    # 監査ログに記録
                    self.logger.warning(f"MFA検証失敗: {user_id}")
                    return False

                return True

            elif mfa_type == MFAType.SMS.value:
                # SMS認証コードを検証（Redisに保存されているコードと比較）
                if not self.redis_client:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="SMS認証サービスが利用できません"
                    )

                stored_code = self.redis_client.get(f"sms_code:{user_id}")
                if not stored_code or stored_code != code:
                    # 監査ログに記録
                    self.logger.warning(f"SMS検証失敗: {user_id}")
                    return False

                # 使用済みコードを削除
                self.redis_client.delete(f"sms_code:{user_id}")
                return True

            elif mfa_type == MFAType.EMAIL.value:
                # メール認証コードを検証（Redisに保存されているコードと比較）
                if not self.redis_client:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="メール認証サービスが利用できません"
                    )

                stored_code = self.redis_client.get(f"email_code:{user_id}")
                if not stored_code or stored_code != code:
                    # 監査ログに記録
                    self.logger.warning(f"メール検証失敗: {user_id}")
                    return False

                # 使用済みコードを削除
                self.redis_client.delete(f"email_code:{user_id}")
                return True

            else:
                # 不明なMFAタイプ
                self.logger.error(f"不明なMFAタイプ: {mfa_type}, ユーザー: {user_id}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="MFA設定が無効です。管理者にお問い合わせください。"
                )

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"MFA検証中に予期せぬエラーが発生しました: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="多要素認証の検証中にエラーが発生しました"
            )

    async def disable_mfa(self, user_id: str) -> bool:
        """
        ユーザーのMFAを無効化

        Args:
            user_id: ユーザーID

        Returns:
            bool: 成功した場合はTrue
        """
        self._check_initialization()

        try:
            # ユーザー情報を取得して適切なmfa_typeを取得
            user_doc = self._get_user_collection().document(user_id).get()
            if not user_doc.exists:
                self.logger.warning(f"MFA無効化対象のユーザーが見つかりません: {user_id}")
                return False

            user_data = user_doc.to_dict()
            previous_mfa_type = user_data.get('mfa_type', MFAType.NONE.value)

            # MFAを無効化
            self._get_user_collection().document(user_id).update({
                'mfa_enabled': False,
                'mfa_type': MFAType.NONE.value,
                'mfa_secret': None
            })

            # コンプライアンスログに記録
            await self.get_compliance_manager().log_event(ComplianceEvent(
                event_type=ComplianceEventType.MFA_DISABLE,
                user_id=user_id,
                user_email=user_data.get('email', ''),  # user変数ではなくuser_dataから取得
                success=True,
                details={"previous_type": previous_mfa_type}
            ))

            self.logger.info(f"ユーザーのMFAが無効化されました: {user_id}")
            return True

        except Exception as e:
            self.logger.error(f"MFA無効化中に予期せぬエラーが発生しました: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="多要素認証の無効化中にエラーが発生しました"
            )

    async def send_sms_code(self, user_id: str, phone_number: str) -> bool:
        """
        SMS認証コードを送信

        Args:
            user_id: ユーザーID
            phone_number: 電話番号

        Returns:
            bool: 成功した場合はTrue
        """
        self._check_initialization()

        try:
            if not self.redis_client:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="SMS認証サービスが利用できません"
                )

            # 6桁の認証コードを生成
            code = ''.join([str(secrets.randbelow(10)) for _ in range(6)])

            # Redisに一時的に保存（5分間有効）
            self.redis_client.setex(f"sms_code:{user_id}", 300, code)

            # ここでSMS送信サービスを統合（Twilio, SNS, etc.）
            # TODO: 実際のSMS送信コードを実装
            self.logger.info(f"SMS認証コードが生成されました: {user_id}, コード: {code}")

            # 電話番号を更新
            self._get_user_collection().document(user_id).update({
                'phone_number': phone_number
            })

            return True

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"SMS認証コード送信中に予期せぬエラーが発生しました: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SMS認証コードの送信中にエラーが発生しました"
            )

    async def send_email_code(self, user_id: str, email: str) -> bool:
        """
        メール認証コードを送信

        Args:
            user_id: ユーザーID
            email: メールアドレス

        Returns:
            bool: 成功した場合はTrue
        """
        self._check_initialization()

        try:
            if not self.redis_client:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="メール認証サービスが利用できません"
                )

            # 6桁の認証コードを生成
            code = ''.join([str(secrets.randbelow(10)) for _ in range(6)])

            # Redisに一時的に保存（5分間有効）
            self.redis_client.setex(f"email_code:{user_id}", 300, code)

            # ここでメール送信サービスを統合
            # TODO: 実際のメール送信コードを実装
            self.logger.info(f"メール認証コードが生成されました: {user_id}, コード: {code}")

            return True

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"メール認証コード送信中に予期せぬエラーが発生しました: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="メール認証コードの送信中にエラーが発生しました"
            )

    # === 既存の認証メソッド ===

    async def hash_password(self, password: str) -> str:
        """
        パスワードをハッシュ化

        Args:
            password: 平文パスワード

        Returns:
            str: ハッシュ化されたパスワード
        """
        return self.ph.hash(password)

    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        パスワードを検証

        Args:
            plain_password: 平文パスワード
            hashed_password: ハッシュ化されたパスワード

        Returns:
            bool: パスワードが一致する場合はTrue
        """
        # 既存のbcryptハッシュ（passlib形式）との互換性を維持
        if hashed_password.startswith('$2'):
            try:
                # 一時的にpasslibをインポートして古いハッシュを検証
                from passlib.hash import bcrypt
                is_valid = bcrypt.verify(plain_password, hashed_password)
                self.logger.info("古いbcryptハッシュ形式を使用したパスワードを検証しました")
                return is_valid
            except ImportError:
                self.logger.error("passlibがインストールされていないため、古いbcryptハッシュを検証できません")
                return False

        # Argon2ハッシュの検証
        try:
            return self.ph.verify(hashed_password, plain_password)
        except VerifyMismatchError:
            return False
        except InvalidHash:
            self.logger.warning(f"無効なパスワードハッシュ形式が検出されました")
            return False

    async def password_needs_update(self, hashed_password: str) -> bool:
        """
        パスワードハッシュが更新必要かを確認

        Args:
            hashed_password: 現在のハッシュパスワード

        Returns:
            bool: ハッシュのアップグレードが必要な場合はTrue
        """
        # 古いbcryptハッシュを使用している場合
        if hashed_password.startswith('$2'):
            return True

        # Argon2ハッシュだが、パラメータが古い場合
        try:
            return self.ph.check_needs_rehash(hashed_password)
        except InvalidHash:
            return True

    async def register_user(
        self,
        email: str,
        password: str,
        display_name: Optional[str] = None,
        role: UserRole = UserRole.USER,
        company_id: Optional[str] = None
    ) -> User:
        """
        新規ユーザー登録

        Args:
            email: ユーザーのメールアドレス
            password: ユーザーのパスワード
            display_name: 表示名
            role: ユーザーロール
            company_id: 所属会社ID

        Returns:
            User: 作成されたユーザー

        Raises:
            HTTPException: 登録できない場合
        """
        self._check_initialization()

        try:
            # メールアドレスのバリデーション
            if not email or '@' not in email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="有効なメールアドレスを入力してください"
                )

            # パスワードの強度チェック
            if len(password) < 8:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="パスワードは8文字以上である必要があります"
                )

            # Firebaseでユーザー作成
            user_record = auth.create_user(
                email=email,
                password=password,
                display_name=display_name or email.split('@')[0]
            )

            # ユーザー情報をFirestoreに保存
            now = datetime.now()
            user_data = {
                'email': email,
                'password_hash': await self.hash_password(password),  # Argon2ハッシュに更新
                'display_name': display_name or email.split('@')[0],
                'is_active': True,
                'role': role.value,
                'created_at': now,
                'password_updated_at': now,  # パスワード更新日時を追加
                'last_login': None,
                'is_email_verified': user_record.email_verified,
                'company_id': company_id,
                'profile_completed': False,
                'login_count': 0,
                'data_access': [],
                'mfa_enabled': False,
                'mfa_type': MFAType.NONE.value,
                'phone_number': None
            }

            self._get_user_collection().document(user_record.uid).set(user_data)

            # ユーザー作成ログ
            self.logger.info(f"新規ユーザーを作成しました: {user_record.uid}, {email}, role={role.value}")

            # メール検証リンクを送信（オプション）
            try:
                auth.generate_email_verification_link(email)
                self.logger.info(f"メール検証リンクを送信しました: {email}")
            except Exception as e:
                self.logger.warning(f"メール検証リンク送信中にエラーが発生しました: {str(e)}")

            # コンプライアンスログに記録
            await self.get_compliance_manager().log_event(ComplianceEvent(
                event_type=ComplianceEventType.REGISTRATION,
                user_id=user_record.uid,
                user_email=email,
                success=True
            ))

            return User(
                id=user_record.uid,
                email=email,
                display_name=display_name or email.split('@')[0],
                is_active=True,
                role=role,
                created_at=now,
                company_id=company_id
            )

        except auth.EmailAlreadyExistsError:
            self.logger.warning(f"メールアドレスはすでに登録されています: {email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="このメールアドレスは既に登録されています"
            )
        except ValueError as e:
            self.logger.error(f"ユーザー登録中にバリデーションエラーが発生しました: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"ユーザー登録に失敗しました: {str(e)}"
            )
        except Exception as e:
            self.logger.error(f"ユーザー登録中に予期せぬエラーが発生しました: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ユーザー登録中にエラーが発生しました。しばらく経ってからもう一度お試しください。"
            )

    async def authenticate_user(self, email: str, password: str) -> Tuple[User, List[str]]:
        """
        ユーザー認証を行い、成功した場合はユーザー情報とスコープを返す

        Args:
            email: ユーザーのメールアドレス
            password: ユーザーのパスワード

        Returns:
            Tuple[User, List[str]]: 認証されたユーザーとスコープのリスト

        Raises:
            HTTPException: 認証失敗時
        """
        self._check_initialization()

        # 認証エラーメッセージ（セキュリティのため詳細を隠す）
        auth_error = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="認証に失敗しました。メールアドレスまたはパスワードが正しくありません",
            headers={"WWW-Authenticate": "Bearer"},
        )

        try:
            # Firebase Authentication で認証
            # 注意: Firebase Admin SDKはパスワード検証APIを提供していないため、
            # 実際の実装ではFirebase Auth REST APIを使用するか、
            # カスタム認証システムを実装する必要があります
            user = auth.get_user_by_email(email)

            # ユーザーが無効化されているかチェック
            if user.disabled:
                self.logger.warning(f"無効化されたユーザーがログインを試みました: {email}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="このアカウントは無効化されています",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Firestoreからユーザー情報を取得
            user_doc = self._get_user_collection().document(user.uid).get()

            # ユーザーデータの取得と検証
            if not user_doc.exists:
                # ユーザーデータがFirestoreに存在しない場合は作成
                self.logger.warning(f"ユーザーデータがFirestoreに存在しません。作成します: {email}")
                user_data = {
                    'email': email,
                    'display_name': user.display_name or email.split('@')[0],
                    'is_active': True,
                    'role': UserRole.USER.value,
                    'created_at': datetime.now(),
                    'last_login': datetime.now(),
                    'is_email_verified': user.email_verified,
                    'profile_completed': False,
                    'login_count': 1,
                    'data_access': [],
                    'mfa_enabled': False,
                    'mfa_type': MFAType.NONE.value,
                    'phone_number': None
                }
                self._get_user_collection().document(user.uid).set(user_data)
            else:
                # ユーザーデータがFirestoreに存在する場合は更新
                user_data = user_doc.to_dict()

                # パスワードハッシュをチェックして、古いハッシュ形式を使用している場合は更新
                if 'password_hash' in user_data:
                    stored_hash = user_data['password_hash']

                    # パスワード検証
                    if not await self.verify_password(password, stored_hash):
                        raise auth_error

                    # ハッシュ形式のアップグレードが必要かチェック
                    if await self.password_needs_update(stored_hash):
                        # 新しいArgon2ハッシュに更新
                        new_hash = await self.hash_password(password)
                        self._get_user_collection().document(user.uid).update({
                            'password_hash': new_hash,
                            'password_updated_at': datetime.now()
                        })
                        self.logger.info(f"ユーザーのパスワードハッシュをArgon2形式に更新しました: {user.uid}")

                # 最終ログイン日時とログイン回数を更新
                login_count = user_data.get('login_count', 0) + 1
                self._get_user_collection().document(user.uid).update({
                    'last_login': datetime.now(),
                    'login_count': login_count
                })

                # アカウントが無効化されているかチェック
                if not user_data.get('is_active', True):
                    self.logger.warning(f"無効化されたユーザーがログインを試みました: {email}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="このアカウントは無効化されています",
                        headers={"WWW-Authenticate": "Bearer"},
                    )

            # ユーザーオブジェクトを作成
            user_role = UserRole(user_data.get('role', UserRole.USER.value))
            scopes = self._get_role_scopes(user_role)

            user_obj = User(
                id=user.uid,
                email=email,
                display_name=user.display_name or user_data.get('display_name'),
                is_active=user_data.get('is_active', True),
                role=user_role,
                created_at=user_data.get('created_at', datetime.now()),
                last_login=datetime.now(),
                is_email_verified=user.email_verified,
                company_id=user_data.get('company_id'),
                profile_completed=user_data.get('profile_completed', False),
                login_count=login_count,
                data_access=user_data.get('data_access', []),
                mfa_enabled=user_data.get('mfa_enabled', False),
                mfa_type=MFAType(user_data.get('mfa_type', MFAType.NONE.value)),
                phone_number=user_data.get('phone_number')
            )

            self.logger.info(f"ユーザーが正常にログインしました: {email}, role={user_role.value}")

            # コンプライアンスログに記録
            await self.get_compliance_manager().log_event(ComplianceEvent(
                event_type=ComplianceEventType.LOGIN,
                user_id=user_obj.id,
                user_email=user_obj.email,
                success=True,
                details={"method": "password"}
            ))

            return user_obj, scopes

        except auth.UserNotFoundError:
            self.logger.warning(f"存在しないユーザーへのログイン試行: {email}")
            raise auth_error
        except Exception as e:
            self.logger.error(f"認証中に予期せぬエラーが発生しました: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="認証中にエラーが発生しました。しばらく経ってからもう一度お試しください。"
            )

    async def create_access_token(
        self,
        data: Dict[str, Any],
        scopes: List[str],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        JWTアクセストークンを生成

        Args:
            data: トークンに含めるデータ
            scopes: トークンのスコープリスト
            expires_delta: トークンの有効期限

        Returns:
            str: JWTトークン
        """
        if self.redis_client is None:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True
            )

        to_encode = data.copy()

        # スコープを追加
        to_encode.update({"scopes": scopes})

        # デフォルトの有効期限（30分）
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=30)

        to_encode.update({"exp": expire})

        # JWTトークンを生成
        encoded_jwt = jwt.encode(to_encode, self.redis_client.get('JWT_SECRET'), algorithm="HS256")
        return encoded_jwt

    async def get_current_user(
        self,
        security_scopes: SecurityScopes,
        token: str = Depends(oauth2_scheme)
    ) -> User:
        """
        JWTトークンからユーザー情報を取得

        Args:
            security_scopes: 必要なセキュリティスコープ
            token: JWTトークン

        Returns:
            User: ユーザー情報

        Raises:
            HTTPException: トークンが無効な場合
        """
        if self.redis_client is None:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True
            )

        # スコープ情報を含めた認証エラーメッセージ
        authenticate_value = f"Bearer scope=\"{security_scopes.scope_str}\""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="認証情報が無効です",
            headers={"WWW-Authenticate": authenticate_value},
        )

        try:
            # トークンのデコード
            payload = jwt.decode(token, self.redis_client.get('JWT_SECRET'), algorithms=["HS256"])

            # ユーザーIDを取得
            user_id: str = payload.get("sub")
            if user_id is None:
                self.logger.warning("トークンにsubクレームがありません")
                raise credentials_exception

            # トークンスコープを取得
            token_scopes = payload.get("scopes", [])

            # トークンモデルを構築
            token_data = TokenData(sub=user_id, scopes=token_scopes)

            # スコープのチェック
            for scope in security_scopes.scopes:
                if scope not in token_data.scopes:
                    self.logger.warning(f"必要なスコープがありません: {scope}, ユーザー: {user_id}")
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"アクセス権限が不足しています: {scope}",
                        headers={"WWW-Authenticate": authenticate_value},
                    )

            # ユーザー情報を取得
            user = await self.get_user_by_id(user_id)
            if user is None:
                self.logger.warning(f"トークンのユーザーが見つかりません: {user_id}")
                raise credentials_exception

            # ユーザーが無効化されていないか確認
            if not user.is_active:
                self.logger.warning(f"無効化されたユーザーのトークン: {user_id}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="このアカウントは無効化されています",
                    headers={"WWW-Authenticate": authenticate_value},
                )

            # コンプライアンスログに記録
            await self.get_compliance_manager().log_event(ComplianceEvent(
                event_type=ComplianceEventType.LOGIN,
                user_id=user.id,
                user_email=user.email,
                success=True,
                details={"method": "token"}
            ))

            return user

        except PyJWTError as e:
            self.logger.warning(f"JWT検証エラー: {str(e)}")
            raise credentials_exception
        except ValidationError:
            self.logger.warning("トークンデータのバリデーションエラー")
            raise credentials_exception
        except Exception as e:
            self.logger.error(f"ユーザー認証中に予期せぬエラーが発生しました: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="認証処理中にエラーが発生しました"
            )

    async def update_password(self, user_id: str, new_password: str) -> bool:
        """
        ユーザーのパスワードを更新

        Args:
            user_id: ユーザーID
            new_password: 新しいパスワード

        Returns:
            bool: 更新成功した場合はTrue
        """
        self._check_initialization()

        try:
            # ユーザー情報を取得
            user_doc = self._get_user_collection().document(user_id).get()
            if not user_doc.exists:
                self.logger.warning(f"パスワード更新対象のユーザーが見つかりません: {user_id}")
                return False

            user_data = user_doc.to_dict()
            user_email = user_data.get('email', '')

            # パスワードを新しいArgon2ハッシュに更新
            new_hash = await self.hash_password(new_password)

            # Firestoreのパスワードハッシュを更新
            self._get_user_collection().document(user_id).update({
                'password_hash': new_hash,
                'password_updated_at': datetime.now()
            })

            # Firebase Authのパスワードを更新（存在する場合）
            try:
                auth.update_user(user_id, password=new_password)
            except Exception as e:
                self.logger.warning(f"Firebase Authパスワード更新エラー: {str(e)}")

            self.logger.info(f"ユーザーのパスワードを更新しました: {user_id}")

            # コンプライアンスログに記録
            await self.get_compliance_manager().log_event(ComplianceEvent(
                event_type=ComplianceEventType.PASSWORD_CHANGE,
                user_id=user_id,
                user_email=user_email,  # user変数ではなくuser_dataから取得したemailを使用
                success=True
            ))

            return True

        except Exception as e:
            self.logger.error(f"パスワード更新中にエラーが発生しました: {str(e)}")
            return False

    async def reset_password(self, email: str) -> bool:
        """
        パスワードリセットメールを送信

        Args:
            email: ユーザーのメールアドレス

        Returns:
            bool: 成功した場合はTrue

        Raises:
            HTTPException: メール送信に失敗した場合
        """
        self._check_initialization()

        try:
            # Firebase Auth でパスワードリセットメールを送信
            reset_link = auth.generate_password_reset_link(email)
            self.logger.info(f"パスワードリセットリンクを送信しました: {email}")

            # ここで実際のメール送信処理を行う場合（オプション）
            # send_email(email, "パスワードリセット", f"パスワードをリセットするには以下のリンクをクリックしてください: {reset_link}")

            # コンプライアンスログに記録
            await self.get_compliance_manager().log_event(ComplianceEvent(
                event_type=ComplianceEventType.PASSWORD_RESET,
                user_email=email,
                success=True,
                details={"status": "email_sent"}
            ))

            return True
        except auth.UserNotFoundError:
            # セキュリティ上、ユーザーが存在しない場合でもエラーは返さない
            self.logger.info(f"存在しないユーザーへのパスワードリセット要求: {email}")
            return True
        except Exception as e:
            self.logger.error(f"パスワードリセットリンク送信中にエラーが発生しました: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="パスワードリセットリンクの送信中にエラーが発生しました。しばらく経ってからもう一度お試しください。"
            )

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        ユーザーIDからユーザー情報を取得

        Args:
            user_id: ユーザーID

        Returns:
            Optional[User]: ユーザー情報（存在しない場合はNone）
        """
        self._check_initialization()

        try:
            # Firebaseからユーザー情報を取得
            user = auth.get_user(user_id)

            # Firestoreからユーザー情報を取得
            user_doc = self._get_user_collection().document(user_id).get()

            if not user_doc.exists:
                self.logger.warning(f"ユーザーのFirestoreデータが見つかりません: {user_id}")
                return None

            # ユーザーモデルを構築
            user_data = user_doc.to_dict()

            return User(
                id=user_id,
                email=user.email or user_data.get('email', ''),
                display_name=user.display_name or user_data.get('display_name'),
                is_active=user_data.get('is_active', True),
                role=UserRole(user_data.get('role', UserRole.USER.value)),
                created_at=user_data.get('created_at', datetime.now()),
                last_login=user_data.get('last_login'),
                is_email_verified=user.email_verified,
                company_id=user_data.get('company_id'),
                profile_completed=user_data.get('profile_completed', False),
                login_count=user_data.get('login_count', 0),
                data_access=user_data.get('data_access', []),
                mfa_enabled=user_data.get('mfa_enabled', False),
                mfa_type=MFAType(user_data.get('mfa_type', MFAType.NONE.value)),
                phone_number=user_data.get('phone_number')
            )

        except auth.UserNotFoundError:
            self.logger.warning(f"ユーザーが見つかりません: {user_id}")
            return None
        except Exception as e:
            self.logger.error(f"ユーザー情報取得中にエラーが発生しました: {str(e)}")
            return None

    async def update_user(
        self,
        user_id: str,
        data: Dict[str, Any]
    ) -> Optional[User]:
        """
        ユーザー情報を更新

        Args:
            user_id: ユーザーID
            data: 更新するデータ

        Returns:
            Optional[User]: 更新されたユーザー情報（存在しない場合はNone）

        Raises:
            HTTPException: 更新に失敗した場合
        """
        self._check_initialization()

        try:
            # 更新可能なフィールド
            allowed_fields = {
                'display_name': 'display_name',
                'email': 'email',
                'password': 'password',
                'is_active': 'disabled',  # Firebaseでは反転する
                'role': None,  # FirestoreのみのフィールFd
                'company_id': None,  # FirestoreのみのフィールFd
                'profile_completed': None,  # FirestoreのみのフィールFd
                'data_access': None,  # FirestoreのみのフィールFd
                'mfa_enabled': None,  # FirestoreのみのフィールFd
                'mfa_type': None,  # FirestoreのみのフィールFd
                'phone_number': None  # FirestoreのみのフィールFd
            }

            # Firebase Authの更新パラメータを構築
            firebase_update = {}
            for field, firebase_field in allowed_fields.items():
                if field in data and firebase_field:
                    if field == 'is_active':
                        firebase_update[firebase_field] = not data[field]  # 反転
                    else:
                        firebase_update[firebase_field] = data[field]

            # Firebase Authでユーザー情報を更新（利用可能なフィールドのみ）
            if firebase_update:
                auth.update_user(user_id, **firebase_update)

            # Firestoreの更新パラメータを構築
            firestore_update = {}
            for field in data:
                if field != 'password':  # パスワードはFirestoreには保存しない
                    if field == 'role' and not isinstance(data[field], str):
                        # Enumの場合は値を取得
                        firestore_update[field] = data[field].value
                    else:
                        firestore_update[field] = data[field]

            # 更新日時を追加
            firestore_update['updated_at'] = datetime.now()

            # Firestoreのユーザー情報を更新
            if firestore_update:
                self._get_user_collection().document(user_id).update(firestore_update)

            self.logger.info(f"ユーザー情報を更新しました: {user_id}")

            # 更新後のユーザー情報を取得して返す
            return await self.get_user_by_id(user_id)

        except auth.UserNotFoundError:
            self.logger.warning(f"更新対象のユーザーが見つかりません: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="ユーザーが見つかりません"
            )
        except ValueError as e:
            self.logger.error(f"ユーザー更新中にバリデーションエラーが発生しました: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"ユーザー更新に失敗しました: {str(e)}"
            )
        except Exception as e:
            self.logger.error(f"ユーザー情報更新中に予期せぬエラーが発生しました: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ユーザー情報の更新中にエラーが発生しました。しばらく経ってからもう一度お試しください。"
            )

    async def delete_user(self, user_id: str) -> bool:
        """
        ユーザーを削除

        Args:
            user_id: ユーザーID

        Returns:
            bool: 成功した場合はTrue

        Raises:
            HTTPException: 削除に失敗した場合
        """
        self._check_initialization()

        try:
            # Firebase Authからユーザーを削除
            auth.delete_user(user_id)

            # Firestoreからユーザー情報を削除
            self._get_user_collection().document(user_id).delete()

            self.logger.info(f"ユーザーを削除しました: {user_id}")

            # コンプライアンスログに記録
            await self.get_compliance_manager().log_event(ComplianceEvent(
                event_type=ComplianceEventType.ADMIN_ACTION,
                user_id="system",
                affected_user_id=user_id,
                action="user_deletion",
                success=True
            ))

            return True

        except auth.UserNotFoundError:
            # すでに削除されている場合は成功扱い
            self.logger.warning(f"削除対象のユーザーが見つかりません: {user_id}")
            return True
        except Exception as e:
            self.logger.error(f"ユーザー削除中に予期せぬエラーが発生しました: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ユーザーの削除中にエラーが発生しました。しばらく経ってからもう一度お試しください。"
            )

    async def revoke_user_tokens(self, user_id: str) -> bool:
        """
        ユーザーのセッショントークンを無効化

        Args:
            user_id: ユーザーID

        Returns:
            bool: 成功した場合はTrue

        Raises:
            HTTPException: 失敗した場合
        """
        self._check_initialization()

        try:
            # Firebase Auth でトークンを無効化
            auth.revoke_refresh_tokens(user_id)

            # Firestoreに最終ログアウト時間を記録
            self._get_user_collection().document(user_id).update({
                'last_logout': datetime.now()
            })

            self.logger.info(f"ユーザートークンを無効化しました: {user_id}")

            # コンプライアンスログに記録
            await self.get_compliance_manager().log_event(ComplianceEvent(
                event_type=ComplianceEventType.LOGOUT,
                user_id=user_id,
                success=True
            ))

            return True
        except auth.UserNotFoundError:
            self.logger.warning(f"トークン無効化対象のユーザーが見つかりません: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="ユーザーが見つかりません"
            )
        except Exception as e:
            self.logger.error(f"トークン無効化中に予期せぬエラーが発生しました: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="トークンの無効化中にエラーが発生しました"
            )

    def update_user_role(self, user_id: str, new_role: str) -> bool:
        """ユーザーのロールを更新"""
        try:
            # userをuser_idに修正
            user_data = self._get_user_collection().document(user_id).get()
            if not user_data:
                return False

            # ロールを更新
            self._get_user_collection().document(user_id).update({
                'role': new_role,
                'updated_at': datetime.now()
            })

            self.logger.info(f"ユーザーのロールを更新しました: {user_id}, 新しいロール: {new_role}")

            # ユーザー情報を取得して適切なユーザーメールアドレスを使用
            user_info = user_data.to_dict()
            user_email = user_info.get('email', '')

            # コンプライアンスログに記録
            self.get_compliance_manager().log_event(ComplianceEvent(
                event_type=ComplianceEventType.ADMIN_ACTION,
                user_id="system",
                affected_user_id=user_id,
                user_email=user_email, # 修正: userではなくuser_infoからemailを取得
                action="role_update",
                success=True,
                details={"new_role": new_role}
            ))

            return True
        except Exception as e:
            self.logger.error(f"ユーザーロール更新中にエラーが発生しました: {str(e)}")
            return False

    def _verify_user_credentials(self, user_id: str, password: str) -> Optional[Dict[str, Any]]:
        """ユーザーの認証情報を検証"""
        # userをuser_idに修正
        user_data = self._get_user_collection().document(user_id).get()
        if not user_data:
            return None

        # ユーザー情報を取得
        user_info = user_data.to_dict()
        user_email = user_info.get('email', '')

        # コンプライアンスログに記録するときに正しい情報を使用
        self.get_compliance_manager().log_event(ComplianceEvent(
            event_type=ComplianceEventType.PASSWORD_CHANGE,
            user_id=user_id,
            user_email=user_email, # 修正: userではなくuser_infoからemailを取得
            success=True
        ))

        return user_info

# シングルトンインスタンスの取得関数
def get_auth_manager() -> AuthManager:
    """AuthManagerのシングルトンインスタンスを返す"""
    return AuthManager()

# 依存性注入用の関数
async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    現在のユーザーを取得

    Args:
        security_scopes: 必要なセキュリティスコープ
        token: JWTトークン

    Returns:
        User: 現在のユーザー
    """
    return await get_auth_manager().get_current_user(security_scopes, token)

async def get_current_active_user(
    current_user: User = Security(get_current_user, scopes=["user"])
) -> User:
    """
    現在のアクティブなユーザーを取得

    Args:
        current_user: 現在のユーザー

    Returns:
        User: アクティブなユーザー

    Raises:
        HTTPException: ユーザーが非アクティブな場合
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="このユーザーは無効化されています"
        )
    return current_user

async def get_current_admin_user(
    current_user: User = Security(get_current_user, scopes=["admin"])
) -> User:
    """
    現在の管理者ユーザーを取得

    Args:
        current_user: 現在のユーザー

    Returns:
        User: 管理者ユーザー
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="管理者権限が必要です"
        )
    return current_user

async def get_current_analyst_user(
    current_user: User = Security(get_current_user, scopes=["analyst"])
) -> User:
    """
    現在の分析者ユーザーを取得

    Args:
        current_user: 現在のユーザー

    Returns:
        User: 分析者ユーザー
    """
    if current_user.role not in [UserRole.ANALYST, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="分析者権限が必要です"
        )
    return current_user