# -*- coding: utf-8 -*-
"""
データモデル定義
Startup Wellness データ分析システムで使用されるデータモデルを定義します。
Firestoreを使用したデータの永続化を提供します。
"""
from typing import Dict, Optional, Any, Sequence, TypeVar, Type, Mapping, List
from pydantic import BaseModel, EmailStr, Field, field_validator
from datetime import datetime
import asyncio
import logging
from service.firestore.client import get_firestore_client

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

T = TypeVar('T', bound='FirestoreModel')

class FirestoreModel(BaseModel):
    """Firestoreベースモデル"""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @classmethod
    def collection_name(cls) -> str:
        """コレクション名を返す（子クラスでオーバーライド）"""
        raise NotImplementedError

    @property
    def document_id(self) -> str:
        """ドキュメントIDを返す（子クラスでオーバーライド）"""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """モデルをディクショナリに変換"""
        data = self.dict(exclude_none=True)
        return {
            key: value if not isinstance(value, datetime) else value
            for key, value in data.items()
        }

    @classmethod
    def from_dict(cls: Type[T], data: Optional[Mapping[str, Any]]) -> Optional[T]:
        """ディクショナリからモデルを生成"""
        try:
            if not data:
                return None
            return cls(**dict(data))
        except Exception as e:
            logger.error(f"Error creating {cls.__name__} from dict: {str(e)}")
            return None

    @classmethod
    async def get_by_id(cls: Type[T], doc_id: str) -> Optional[T]:
        """IDによるドキュメント取得"""
        try:
            db = get_firestore_client()
            doc_ref = db.collection(cls.collection_name()).document(doc_id)

            loop = asyncio.get_event_loop()
            doc = await loop.run_in_executor(None, doc_ref.get)

            return cls.from_dict(doc.to_dict() if doc.exists else None)
        except Exception as e:
            logger.error(f"Error fetching document {doc_id}: {str(e)}")
            raise

    @classmethod
    async def fetch_all(
        cls: Type[T],
        conditions: Optional[Sequence[Dict[str, Any]]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        order_by: str = 'created_at',
        direction: str = 'desc'
    ) -> Sequence[T]:
        """条件に基づいて複数のドキュメントを取得"""
        try:
            db = get_firestore_client()
            query = db.collection(cls.collection_name())

            if conditions:
                for condition in conditions:
                    field = condition.get('field')
                    operator = condition.get('operator', '==')
                    value = condition.get('value')
                    if all(x is not None for x in [field, operator, value]):
                        query = query.where(field, operator, value)

            query = query.order_by(order_by, direction=direction)

            if offset > 0:
                query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)

            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, query.get)

            results: List[T] = []
            for doc in docs:
                if doc_dict := doc.to_dict():
                    if instance := cls.from_dict(doc_dict):
                        results.append(instance)
            return results

        except Exception as e:
            logger.error(f"Error fetching documents: {str(e)}")
            raise

    async def save(self) -> None:
        """ドキュメントの保存/更新"""
        try:
            db = get_firestore_client()
            self.updated_at = datetime.now()
            data = self.to_dict()

            doc_ref = db.collection(self.collection_name()).document(self.document_id)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: doc_ref.set(data, merge=True))

            logger.info(f"Successfully saved document {self.document_id} to {self.collection_name()}")
        except Exception as e:
            logger.error(f"Error saving document: {str(e)}")
            raise

    async def delete(self) -> None:
        """ドキュメントの削除"""
        try:
            db = get_firestore_client()
            doc_ref = db.collection(self.collection_name()).document(self.document_id)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, doc_ref.delete)

            logger.info(f"Successfully deleted document {self.document_id} from {self.collection_name()}")
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise

class UserModel(FirestoreModel):
    """ユーザーモデル"""
    username: str = Field(...)
    hashed_password: str = Field(...)
    email: EmailStr = Field(...)
    is_active: bool = Field(default=True)
    is_vc: bool = Field(default=False)

    @classmethod
    def collection_name(cls) -> str:
        return "users"

    @property
    def document_id(self) -> str:
        return self.email

class StartupModel(FirestoreModel):
    """スタートアップ企業モデル"""
    name: str = Field(...)
    industry: str = Field(...)
    founding_date: datetime = Field(...)

    @classmethod
    def collection_name(cls) -> str:
        return "startups"

    @property
    def document_id(self) -> str:
        return f"{self.name.lower().replace(' ', '-')}-{int(self.founding_date.timestamp())}"

class VASDataModel(FirestoreModel):
    """VAS データモデル"""
    startup_id: str = Field(...)
    user_id: str = Field(...)
    timestamp: datetime = Field(...)
    physical_symptoms: float = Field(...)
    mental_state: float = Field(...)
    motivation: float = Field(...)
    communication: float = Field(...)
    other: float = Field(...)
    free_text: Optional[str] = Field(default=None)

    @field_validator('physical_symptoms', 'mental_state', 'motivation', 'communication', 'other')
    @classmethod
    def validate_scores(cls, v: float) -> float:
        if not 0 <= v <= 100:
            raise ValueError("Score must be between 0 and 100")
        return v

    @classmethod
    def collection_name(cls) -> str:
        return "vas_data"

    @property
    def document_id(self) -> str:
        return f"{self.startup_id}-{self.user_id}-{int(self.timestamp.timestamp())}"

    @classmethod
    async def get_by_startup(cls, startup_id: str, limit: int = 100) -> Sequence['VASDataModel']:
        """スタートアップIDに基づくVASデータの取得"""
        conditions = [{'field': 'startup_id', 'operator': '==', 'value': startup_id}]
        return await cls.fetch_all(conditions=conditions, limit=limit)

class FinancialDataModel(FirestoreModel):
    """財務データモデル"""
    startup_id: str = Field(...)
    year: int = Field(...)
    revenue: float = Field(...)
    profit: float = Field(...)
    employee_count: int = Field(...)
    turnover_rate: float = Field(...)

    @field_validator('year')
    @classmethod
    def validate_year(cls, v: int) -> int:
        if not 1900 <= v <= datetime.now().year + 1:
            raise ValueError(f"Year must be between 1900 and {datetime.now().year + 1}")
        return v

    @classmethod
    def collection_name(cls) -> str:
        return "financial_data"

    @property
    def document_id(self) -> str:
        return f"{self.startup_id}-{self.year}"

    @classmethod
    async def get_by_startup_and_year(
        cls, startup_id: str, year: int
    ) -> Optional['FinancialDataModel']:
        """スタートアップIDと年度に基づく財務データの取得"""
        doc_id = f"{startup_id}-{year}"
        return await cls.get_by_id(doc_id)

class AnalysisSettingModel(FirestoreModel):
    """VC向け分析設定モデル"""
    user_id: str = Field(...)
    google_form_questions: Dict[str, Any] = Field(default_factory=dict)
    financial_data_items: Dict[str, Any] = Field(default_factory=dict)
    analysis_methods: Dict[str, Any] = Field(default_factory=dict)
    visualization_methods: Dict[str, Any] = Field(default_factory=dict)
    generative_ai_settings: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def collection_name(cls) -> str:
        return "analysis_settings"

    @property
    def document_id(self) -> str:
        return self.user_id

class NoteModel(FirestoreModel):
    """メモモデル"""
    user_id: str = Field(...)
    analysis_id: str = Field(...)
    content: str = Field(...)

    @classmethod
    def collection_name(cls) -> str:
        return "notes"

    @property
    def document_id(self) -> str:
        return f"{self.user_id}-{self.analysis_id}"

    @classmethod
    async def get_by_user(cls, user_id: str) -> Sequence['NoteModel']:
        """ユーザーIDに基づくメモの取得"""
        conditions = [{'field': 'user_id', 'operator': '==', 'value': user_id}]
        return await cls.fetch_all(conditions=conditions)