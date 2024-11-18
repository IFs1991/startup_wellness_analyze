from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, TypeVar, Type
from pydantic import BaseModel, Field
from google.cloud import firestore
from enum import Enum

T = TypeVar('T', bound='FirestoreModel')

class FirestoreModel(BaseModel):
    """Firestoreベースモデル"""
    id: str = Field(default_factory=lambda: firestore.Client().collection('dummy').document().id)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> dict:
        """Firestoreドキュメント形式に変換"""
        return {k: v for k, v in self.dict(exclude_none=True).items()}

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """Firestoreドキュメントからモデルを生成"""
        return cls(**data)

    @classmethod
    def get_collection_name(cls) -> str:
        """コレクション名を取得"""
        return getattr(cls.Config, 'collection_name', cls.__name__.lower())

class User(FirestoreModel):
    """ユーザーモデル"""
    username: str
    email: str
    hashed_password: str
    is_active: bool = True
    is_vc: bool = False
    hr_system_user_id: Optional[str] = None

    class Config:
        collection_name = 'users'

# 他のモデルクラスは変更なし...

class FirestoreService:
    """Firestoreサービスクラス"""
    def __init__(self):
        self.db = firestore.Client()

    async def create_document(self, model: FirestoreModel) -> str:
        """ドキュメント作成"""
        collection_name = model.get_collection_name()
        doc_ref = self.db.collection(collection_name).document(model.id)
        doc_ref.set(model.to_dict())
        return model.id

    async def get_document(self, collection_name: str, doc_id: str) -> Optional[dict]:
        """ドキュメント取得"""
        doc_ref = self.db.collection(collection_name).document(doc_id)
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else None

    async def update_document(self, model: FirestoreModel) -> None:
        """ドキュメント更新"""
        collection_name = model.get_collection_name()
        doc_ref = self.db.collection(collection_name).document(model.id)
        model.updated_at = datetime.utcnow()
        doc_ref.update(model.to_dict())

    async def delete_document(self, collection_name: str, doc_id: str) -> None:
        """ドキュメント削除"""
        doc_ref = self.db.collection(collection_name).document(doc_id)
        doc_ref.delete()

    async def query_documents(
        self,
        collection_name: str,
        filters: Optional[List[Tuple[str, str, Any]]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[dict]:
        """ドキュメントのクエリ"""
        query = self.db.collection(collection_name)

        if filters:
            for field, op, value in filters:
                query = query.where(field, op, value)

        if order_by:
            query = query.order_by(order_by)

        if limit:
            query = query.limit(limit)

        docs = query.stream()
        return [doc.to_dict() for doc in docs]

class FirestoreRepository:
    """Firestoreリポジトリクラス"""
    def __init__(self):
        self.service = FirestoreService()

    async def create_user(self, user_data: dict) -> User:
        """ユーザー作成"""
        user = User(**user_data)
        await self.service.create_document(user)
        return user

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """ユーザー取得"""
        data = await self.service.get_document('users', user_id)
        return User.from_dict(data) if data else None

    async def get_startup_data(self, startup_id: str) -> Optional[Dict[str, Any]]:
        """スタートアップの全データ取得"""
        startup = await self.service.get_document('startups', startup_id)
        if not startup:
            return None

        # 関連データの取得
        vas_data = await self.service.query_documents(
            'vas_data',
            filters=[('startup_id', '==', startup_id)],
            order_by='timestamp'
        )

        financial_data = await self.service.query_documents(
            'financial_data',
            filters=[('startup_id', '==', startup_id)],
            order_by='year'
        )

        return {
            'startup': startup,
            'vas_data': vas_data,
            'financial_data': financial_data
        }