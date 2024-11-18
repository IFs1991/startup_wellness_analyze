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
from typing import Optional, Dict, Union
from datetime import datetime

# パスワードハッシュ化の設定
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthManager:
    """
    ユーザー認証を管理するためのクラス。
    従来のパスワードベース認証とFirebase認証の両方をサポートします。
    """
    _instance = None
    _firestore_client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AuthManager, cls).__new__(cls)
            cls._instance._initialize_firebase()
        return cls._instance

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

    def create_firebase_user(self, email: str, password: str, display_name: Optional[str] = "") -> Dict:
        """
        Firebaseに新規ユーザーを作成します。
        Args:
            email (str): ユーザーのメールアドレス
            password (str): パスワード
            display_name (str, optional): 表示名
        Returns:
            Dict: 作成されたユーザーの情報
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
            raise ValueError(f"Failed to create user: {str(e)}")

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

    def verify_firebase_token(self, token: str) -> Dict:
        """
        Firebaseトークンを検証します。
        Args:
            token (str): 検証するトークン
        Returns:
            Dict: デコードされたトークン情報
        """
        try:
            return auth.verify_id_token(token)
        except Exception as e:
            raise ValueError(f"Invalid token: {str(e)}")

    def get_user_by_email(self, email: str) -> Union[Dict, None]:
        """
        メールアドレスからユーザー情報を取得します。
        Args:
            email (str): ユーザーのメールアドレス
        Returns:
            Dict or None: ユーザー情報、存在しない場合はNone
        """
        try:
            user = auth.get_user_by_email(email)
            return {
                'uid': user.uid,
                'email': user.email,
                'display_name': user.display_name if user.display_name else ""
            }
        except auth.UserNotFoundError:
            return None

    def update_user_last_login(self, uid: str):
        """
        ユーザーの最終ログイン時間を更新します。
        Args:
            uid (str): ユーザーID
        """
        user_ref = self.firestore_client.collection('users').document(uid)
        user_ref.update({
            'last_login': firestore.SERVER_TIMESTAMP
        })

    def revoke_user_sessions(self, uid: str):
        """
        ユーザーの全てのセッションを無効化します。
        Args:
            uid (str): ユーザーID
        """
        try:
            # Firebase側でトークンを無効化
            auth.revoke_refresh_tokens(uid)

            # Firestoreの最終ログアウト時間を更新
            user_ref = self.firestore_client.collection('users').document(uid)
            user_ref.update({
                'last_logout': firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            raise ValueError(f"Failed to revoke user sessions: {str(e)}")

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