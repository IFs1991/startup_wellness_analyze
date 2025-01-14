from typing import Optional, Dict, List
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
from pydantic import BaseModel

class User(BaseModel):
    id: str
    username: str
    email: str
    hashed_password: str
    roles: List[str] = []
    permissions: List[str] = []

class AuthService:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self._sessions: Dict[str, Dict] = {}

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """ユーザー認証を行う"""
        user = self._get_user_by_username(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return user

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """パスワードの検証を行う"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """アクセストークンを生成する"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)

        to_encode = {
            "sub": user.id,
            "exp": expire,
            "roles": user.roles,
            "permissions": user.permissions
        }
        return jwt.encode(to_encode, self.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> Optional[Dict]:
        """トークンの検証を行う"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.PyJWTError:
            return None

    def has_permission(self, user: User, permission: str) -> bool:
        """ユーザーの権限を確認する"""
        return permission in user.permissions

    def has_role(self, user: User, role: str) -> bool:
        """ユーザーのロールを確認する"""
        return role in user.roles

    def create_session(self, user: User) -> str:
        """セッションを作成する"""
        session_id = self._generate_session_id()
        self._sessions[session_id] = {
            "user_id": user.id,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        return session_id

    def validate_session(self, session_id: str) -> bool:
        """セッションの有効性を確認する"""
        if session_id not in self._sessions:
            return False
        session = self._sessions[session_id]
        if (datetime.utcnow() - session["last_activity"]) > timedelta(hours=24):
            del self._sessions[session_id]
            return False
        session["last_activity"] = datetime.utcnow()
        return True

    def _get_user_by_username(self, username: str) -> Optional[User]:
        """ユーザー情報を取得する（実際のデータベース実装が必要）"""
        # TODO: データベースからユーザー情報を取得する実装
        pass

    def _generate_session_id(self) -> str:
        """セッションIDを生成する"""
        import uuid
        return str(uuid.uuid4())