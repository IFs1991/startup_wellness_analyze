"""
認証関連のユースケース
ユーザー認証に関するビジネスロジックを提供します。
"""
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import jwt

from domain.entities.user import User
from domain.repositories.user_repository import UserRepositoryInterface
from core.exceptions import AuthError, ValidationError
from core.common_logger import get_logger


class AuthUseCase:
    """認証関連のユースケースクラス"""

    def __init__(self, user_repository: UserRepositoryInterface):
        """
        初期化

        Args:
            user_repository: ユーザーリポジトリインターフェース
        """
        self.logger = get_logger(__name__)
        self.user_repository = user_repository

        # JWT設定
        self.jwt_secret = "temporary_secret_key"  # 実際には環境変数から取得すべき
        self.jwt_algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7

    async def authenticate_user(self, email: str, password: str) -> User:
        """
        ユーザーを認証

        Args:
            email: ユーザーのメールアドレス
            password: ユーザーのパスワード

        Returns:
            認証されたユーザー

        Raises:
            AuthError: 認証に失敗した場合
        """
        # メールアドレスでユーザーを検索
        user = await self.user_repository.get_by_email(email)
        if not user:
            self.logger.warning(f"認証失敗: メールアドレス {email} のユーザーが見つかりません")
            raise AuthError("メールアドレスまたはパスワードが無効です")

        # この例では実際のパスワード検証は省略しています
        # 実際の実装では、ハッシュ化されたパスワードを検証します
        # if not verify_password(password, user.password_hash):
        #     self.logger.warning(f"認証失敗: ユーザー {email} のパスワードが不一致")
        #     raise AuthError("メールアドレスまたはパスワードが無効です")

        # 最終ログイン時刻を更新
        user.record_login()
        await self.user_repository.update_last_login(user.id, user.last_login)

        self.logger.info(f"ユーザー {email} の認証に成功しました")
        return user

    async def register_user(self, email: str, password: str, display_name: Optional[str] = None) -> User:
        """
        新規ユーザーを登録

        Args:
            email: メールアドレス
            password: パスワード
            display_name: 表示名

        Returns:
            作成されたユーザー

        Raises:
            ValidationError: メールアドレスなどが無効な場合
            AuthError: ユーザー登録に失敗した場合
        """
        # メールアドレスの基本的なバリデーション
        if not email or "@" not in email:
            raise ValidationError("有効なメールアドレスを入力してください")

        # パスワードのバリデーション
        if not password or len(password) < 8:
            raise ValidationError("パスワードは8文字以上である必要があります")

        # メールアドレスの重複チェック
        existing_user = await self.user_repository.get_by_email(email)
        if existing_user:
            self.logger.warning(f"ユーザー登録失敗: メールアドレス {email} は既に使用されています")
            raise AuthError("このメールアドレスは既に使用されています")

        # 新しいユーザーを作成
        # 実際の実装では、パスワードはハッシュ化して保存します
        # password_hash = hash_password(password)

        new_user = User(
            id="",  # IDはリポジトリで生成
            email=email,
            display_name=display_name or email.split("@")[0]
        )

        # ユーザーをリポジトリに保存
        created_user = await self.user_repository.create(new_user)

        self.logger.info(f"ユーザー {email} を作成しました (ID: {created_user.id})")
        return created_user

    def create_access_token(self, user: User) -> str:
        """
        アクセストークンを生成

        Args:
            user: アクセストークンを生成するユーザー

        Returns:
            JWTアクセストークン
        """
        expires_delta = timedelta(minutes=self.access_token_expire_minutes)
        expire = datetime.utcnow() + expires_delta

        to_encode = {
            "sub": user.id,
            "email": user.email,
            "exp": expire,
            "type": "access"
        }

        encoded_jwt = jwt.encode(to_encode, self.jwt_secret, algorithm=self.jwt_algorithm)
        return encoded_jwt

    def create_refresh_token(self, user: User) -> str:
        """
        リフレッシュトークンを生成

        Args:
            user: リフレッシュトークンを生成するユーザー

        Returns:
            JWTリフレッシュトークン
        """
        expires_delta = timedelta(days=self.refresh_token_expire_days)
        expire = datetime.utcnow() + expires_delta

        to_encode = {
            "sub": user.id,
            "exp": expire,
            "type": "refresh"
        }

        encoded_jwt = jwt.encode(to_encode, self.jwt_secret, algorithm=self.jwt_algorithm)
        return encoded_jwt

    async def verify_token(self, token: str) -> Tuple[User, Dict[str, Any]]:
        """
        トークンを検証してユーザーを取得

        Args:
            token: 検証するJWTトークン

        Returns:
            (ユーザー, トークンデータ)のタプル

        Raises:
            AuthError: トークンが無効な場合
        """
        try:
            # トークンを検証してデコード
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])

            # トークンタイプを確認
            token_type = payload.get("type")
            if token_type not in ["access", "refresh"]:
                raise AuthError("無効なトークンタイプです")

            # ユーザーIDを取得
            user_id = payload.get("sub")
            if not user_id:
                raise AuthError("ユーザーIDが見つかりません")

            # ユーザーを取得
            user = await self.user_repository.get_by_id(user_id)
            if not user:
                raise AuthError("ユーザーが見つかりません")

            # ユーザーが非アクティブな場合
            if not user.is_active:
                raise AuthError("アカウントは非アクティブです")

            return user, payload

        except jwt.ExpiredSignatureError:
            raise AuthError("トークンの有効期限が切れています")
        except jwt.InvalidTokenError:
            raise AuthError("無効なトークンです")

    async def refresh_access_token(self, refresh_token: str) -> str:
        """
        リフレッシュトークンを使用して新しいアクセストークンを生成

        Args:
            refresh_token: リフレッシュトークン

        Returns:
            新しいアクセストークン

        Raises:
            AuthError: リフレッシュトークンが無効な場合
        """
        user, payload = await self.verify_token(refresh_token)

        # トークンタイプを確認
        if payload.get("type") != "refresh":
            raise AuthError("リフレッシュトークンではありません")

        # 新しいアクセストークンを生成
        new_access_token = self.create_access_token(user)

        return new_access_token