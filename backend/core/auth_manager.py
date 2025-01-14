# -*- coding: utf-8 -*-
"""
認証管理
ユーザー認証 (ログイン, ログアウト, 登録, パスワードリセットなど) を管理します。
Firebase AuthenticationとCloud Firestoreを使用した認証機能も提供します。
"""
from passlib.context import CryptContext
from firebase_admin import credentials, initialize_app, get_app, auth
from google.cloud import firestore
import os
from pathlib import Path
from typing import Optional, Dict, Union, Any
from datetime import datetime
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

# パスワードハッシュ化の設定
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthManager:
    """
    ユーザー認証を管理するためのクラス。
    従来のパスワードベース認証とFirebase認証の両方をサポートします。
    """
    _instance = None
    _firestore_client = None
    _executor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AuthManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初期化処理を行います。"""
        self._initialize_firebase()
        self._setup_logging()
        self._initialize_thread_pool()

    def _initialize_thread_pool(self):
        """スレッドプールを初期化します。"""
        if not self._executor:
            self._executor = ThreadPoolExecutor(max_workers=4)

    def _setup_logging(self):
        """ロギングを設定します。"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _initialize_firebase(self):
        """Firebase Adminの初期化"""
        try:
            self.app = get_app()
        except ValueError:
            cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if not cred_path:
                cred_path = str(Path(__file__).parent.parent / 'firebase-credentials.json')

            if not os.path.exists(cred_path):
                raise FileNotFoundError(
                    f"Firebase credentials file not found at {cred_path}. "
                    "Please ensure you have placed the credentials file in the correct location."
                )

            cred = credentials.Certificate(cred_path)
            self.app = initialize_app(cred)

    def hash_password(self, password: str) -> str:
        """
        パスワードをハッシュ化します。
        Args:
            password (str): 平文パスワード
        Returns:
            str: ハッシュ化されたパスワード
        """
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        平文パスワードとハッシュ化されたパスワードを比較検証します。
        Args:
            plain_password (str): 平文パスワード
            hashed_password (str): ハッシュ化されたパスワード
        Returns:
            bool: パスワードが一致する場合 True, そうでない場合 False
        """
        return pwd_context.verify(plain_password, hashed_password)

    @property
    def firestore_client(self):
        """Firestoreクライアントの取得（シングルトン）"""
        if not self._firestore_client:
            self._firestore_client = firestore.Client()
        return self._firestore_client

    async def create_firebase_user(
        self,
        email: str,
        password: str,
        display_name: Optional[str] = ""
    ) -> Dict[str, Any]:
        """
        Firebaseに新規ユーザーを作成します。

        Args:
            email (str): ユーザーのメールアドレス
            password (str): パスワード
            display_name (str, optional): 表示名

        Returns:
            Dict[str, Any]: 作成されたユーザーの情報

        Raises:
            ValueError: ユーザー作成に失敗した場合
        """
        try:
            self.logger.info(f"ユーザー作成開始: {email}")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._create_firebase_user_sync,
                email,
                password,
                display_name
            )
            self.logger.info(f"ユーザー作成成功: {email}")
            return result
        except Exception as e:
            self.logger.error(f"ユーザー作成失敗: {str(e)}")
            raise ValueError(f"ユーザー作成に失敗しました: {str(e)}")

    def _create_firebase_user_sync(
        self,
        email: str,
        password: str,
        display_name: Optional[str] = ""
    ) -> Dict[str, Any]:
        """
        同期的にFirebaseユーザーを作成します。

        Args:
            email (str): メールアドレス
            password (str): パスワード
            display_name (str, optional): 表示名

        Returns:
            Dict[str, Any]: ユーザー情報
        """
        try:
            create_args = {
                "email": email,
                "password": password
            }
            if display_name:
                create_args["display_name"] = display_name

            user_record = auth.create_user(**create_args)

            # Firestoreにユーザー情報を保存
            user_ref = self.firestore_client.collection('users').document(user_record.uid)
            user_data = {
                'email': email,
                'display_name': display_name if display_name else "",
                'created_at': firestore.SERVER_TIMESTAMP,
                'last_login': None,
                'auth_provider': 'firebase',
                'is_active': True
            }
            user_ref.set(user_data)

            return {
                'uid': user_record.uid,
                'email': email,
                'display_name': display_name if display_name else ""
            }
        except Exception as e:
            self.logger.error(f"同期的ユーザー作成中にエラー: {str(e)}")
            raise

    def create_custom_token(self, uid: str) -> str:
        """
        ユーザーIDに基づいてカスタムトークンを作成します。
        Args:
            uid (str): ユーザーID
        Returns:
            str: カスタム認証トークン
        """
        try:
            custom_token = auth.create_custom_token(uid)
            return custom_token.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to create custom token: {str(e)}")

    async def verify_firebase_token(self, token: str) -> Dict[str, Any]:
        """
        Firebaseトークンを検証します。

        Args:
            token (str): 検証するトークン

        Returns:
            Dict[str, Any]: デコードされたトークン情報

        Raises:
            ValueError: トークンが無効な場合
        """
        try:
            self.logger.info("トークン検証開始")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                auth.verify_id_token,
                token
            )
            self.logger.info("トークン検証成功")
            return result
        except Exception as e:
            self.logger.error(f"トークン検証失敗: {str(e)}")
            raise ValueError(f"無効なトークン: {str(e)}")

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        メールアドレスからユーザー情報を取得します。

        Args:
            email (str): ユーザーのメールアドレス

        Returns:
            Optional[Dict[str, Any]]: ユーザー情報、存在しない場合はNone
        """
        try:
            self.logger.info(f"ユーザー検索開始: {email}")
            loop = asyncio.get_event_loop()
            user = await loop.run_in_executor(
                self._executor,
                auth.get_user_by_email,
                email
            )
            if user:
                return {
                    'uid': user.uid,
                    'email': user.email,
                    'display_name': user.display_name if user.display_name else ""
                }
            return None
        except auth.UserNotFoundError:
            self.logger.info(f"ユーザーが見つかりません: {email}")
            return None
        except Exception as e:
            self.logger.error(f"ユーザー検索中にエラー: {str(e)}")
            raise

    async def update_user_last_login(self, uid: str):
        """
        ユーザーの最終ログイン時間を更新します。

        Args:
            uid (str): ユーザーID
        """
        try:
            self.logger.info(f"最終ログイン時間更新開始: {uid}")
            user_ref = self.firestore_client.collection('users').document(uid)
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: user_ref.update({
                    'last_login': firestore.SERVER_TIMESTAMP
                })
            )
            self.logger.info(f"最終ログイン時間更新成功: {uid}")
        except Exception as e:
            self.logger.error(f"最終ログイン時間更新失敗: {str(e)}")
            raise

    async def revoke_user_sessions(self, uid: str):
        """
        ユーザーの全てのセッションを無効化します。

        Args:
            uid (str): ユーザーID
        """
        try:
            self.logger.info(f"セッション無効化開始: {uid}")
            loop = asyncio.get_event_loop()

            # Firebase側でトークンを無効化
            await loop.run_in_executor(
                self._executor,
                auth.revoke_refresh_tokens,
                uid
            )

            # Firestoreの最終ログアウト時間を更新
            user_ref = self.firestore_client.collection('users').document(uid)
            await loop.run_in_executor(
                self._executor,
                lambda: user_ref.update({
                    'last_logout': firestore.SERVER_TIMESTAMP
                })
            )

            self.logger.info(f"セッション無効化成功: {uid}")
        except Exception as e:
            self.logger.error(f"セッション無効化失敗: {str(e)}")
            raise ValueError(f"セッション無効化に失敗しました: {str(e)}")

    def delete_user(self, uid: str):
        """
        ユーザーを削除します。
        Args:
            uid (str): 削除するユーザーのID
        """
        try:
            # Firebase Authenticationからユーザーを削除
            auth.delete_user(uid)

            # Firestoreからユーザーデータを削除
            user_ref = self.firestore_client.collection('users').document(uid)
            user_ref.delete()
        except Exception as e:
            raise ValueError(f"Failed to delete user: {str(e)}")

    def update_user_profile(self, uid: str, display_name: Optional[str] = None, photo_url: Optional[str] = None):
        """
        ユーザープロフィールを更新します。
        Args:
            uid (str): ユーザーID
            display_name (str, optional): 新しい表示名
            photo_url (str, optional): 新しいプロフィール写真URL
        """
        try:
            update_kwargs = {}
            if display_name is not None:
                update_kwargs['display_name'] = display_name
            if photo_url is not None:
                update_kwargs['photo_url'] = photo_url

            if update_kwargs:
                # Firebase Authentication側の更新
                auth.update_user(uid, **update_kwargs)

                # Firestore側の更新
                user_ref = self.firestore_client.collection('users').document(uid)
                user_ref.update(update_kwargs)
        except Exception as e:
            raise ValueError(f"Failed to update user profile: {str(e)}")

# シングルトンインスタンスの作成
auth_manager = AuthManager()