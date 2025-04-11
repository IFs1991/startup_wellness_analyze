# -*- coding: utf-8 -*-
"""
データモデル定義
Startup Wellness データ分析システムで使用されるデータモデルを定義します。
Firestoreを使用したデータの永続化を提供します。
"""
from typing import Dict, Optional, Any, Sequence, TypeVar, Type, Mapping, List, Union, Generic
from pydantic import BaseModel, EmailStr, Field, field_validator
from datetime import datetime
import asyncio
import logging
from service.firestore.client import get_firestore_client
from enum import Enum

# 自作ロギングユーティリティをインポート
from api.logging_utils import get_logger, trace_db_operation

# ロギングの設定
logger = get_logger(__name__)

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
            logger.error(
                f"Error creating {cls.__name__} from dict: {str(e)}",
                extra={"context": {"error": str(e), "model_class": cls.__name__}}
            )
            return None

    @classmethod
    @trace_db_operation("get_by_id")
    async def get_by_id(cls: Type[T], doc_id: str) -> Optional[T]:
        """
        IDによるドキュメント取得

        Args:
            doc_id: 取得するドキュメントのID

        Returns:
            取得したモデルインスタンスまたはNone

        Raises:
            Exception: ドキュメント取得中にエラーが発生した場合
        """
        try:
            db = get_firestore_client()
            doc_ref = db.collection(cls.collection_name()).document(doc_id)

            logger.debug(
                f"ドキュメント取得開始: {cls.collection_name()}/{doc_id}",
                extra={"context": {"collection": cls.collection_name(), "doc_id": doc_id}}
            )

            loop = asyncio.get_event_loop()
            doc = await loop.run_in_executor(None, doc_ref.get)

            exists = doc.exists

            logger.debug(
                f"ドキュメント取得完了: {cls.collection_name()}/{doc_id}, 存在: {exists}",
                extra={"context": {"collection": cls.collection_name(), "doc_id": doc_id, "exists": exists}}
            )

            return cls.from_dict(doc.to_dict() if exists else None)
        except Exception as e:
            logger.error(
                f"ドキュメント取得中にエラー発生: {cls.collection_name()}/{doc_id}: {str(e)}",
                extra={
                    "context": {
                        "collection": cls.collection_name(),
                        "doc_id": doc_id,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                }
            )
            raise

    @classmethod
    @trace_db_operation("fetch_all")
    async def fetch_all(
        cls: Type[T],
        conditions: Optional[Sequence[Dict[str, Any]]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        order_by: str = 'created_at',
        direction: str = 'desc'
    ) -> Sequence[T]:
        """
        条件に基づいて複数のドキュメントを取得

        Args:
            conditions: クエリ条件のリスト
            limit: 取得する最大ドキュメント数
            offset: スキップするドキュメント数
            order_by: ソートするフィールド
            direction: ソート方向 ('asc' or 'desc')

        Returns:
            取得したモデルインスタンスのリスト

        Raises:
            Exception: ドキュメント取得中にエラーが発生した場合
        """
        try:
            db = get_firestore_client()
            query = db.collection(cls.collection_name())

            query_desc = f"コレクション: {cls.collection_name()}"

            if conditions:
                for condition in conditions:
                    field = condition.get('field')
                    operator = condition.get('operator', '==')
                    value = condition.get('value')
                    if all(x is not None for x in [field, operator, value]):
                        query = query.where(field, operator, value)
                        query_desc += f", 条件: {field} {operator} {value}"

            query = query.order_by(order_by, direction=direction)
            query_desc += f", ソート: {order_by} {direction}"

            if offset > 0:
                query = query.offset(offset)
                query_desc += f", オフセット: {offset}"
            if limit is not None:
                query = query.limit(limit)
                query_desc += f", 制限: {limit}"

            logger.debug(
                f"複数ドキュメント取得開始: {query_desc}",
                extra={"context": {"collection": cls.collection_name(), "query": query_desc}}
            )

            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, query.get)

            results: List[T] = []
            for doc in docs:
                if doc_dict := doc.to_dict():
                    if instance := cls.from_dict(doc_dict):
                        results.append(instance)

            logger.debug(
                f"複数ドキュメント取得完了: {query_desc}, 結果数: {len(results)}",
                extra={"context": {"collection": cls.collection_name(), "query": query_desc, "result_count": len(results)}}
            )

            return results

        except Exception as e:
            logger.error(
                f"複数ドキュメント取得中にエラー発生: {cls.collection_name()}: {str(e)}",
                extra={
                    "context": {
                        "collection": cls.collection_name(),
                        "conditions": conditions,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                }
            )
            raise

    @trace_db_operation("save")
    async def save(self) -> None:
        """
        ドキュメントの保存/更新

        ドキュメントが存在しない場合は新規に作成し、存在する場合は更新します。

        Raises:
            Exception: ドキュメント保存中にエラーが発生した場合
        """
        try:
            db = get_firestore_client()
            self.updated_at = datetime.now()
            data = self.to_dict()

            doc_ref = db.collection(self.collection_name()).document(self.document_id)

            logger.debug(
                f"ドキュメント保存開始: {self.collection_name()}/{self.document_id}",
                extra={"context": {"collection": self.collection_name(), "doc_id": self.document_id}}
            )

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: doc_ref.set(data, merge=True))

            logger.info(
                f"ドキュメント保存完了: {self.collection_name()}/{self.document_id}",
                extra={"context": {"collection": self.collection_name(), "doc_id": self.document_id}}
            )
        except Exception as e:
            logger.error(
                f"ドキュメント保存中にエラー発生: {self.collection_name()}/{self.document_id}: {str(e)}",
                extra={
                    "context": {
                        "collection": self.collection_name(),
                        "doc_id": self.document_id,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                }
            )
            raise

    @trace_db_operation("delete")
    async def delete(self) -> None:
        """
        ドキュメントの削除

        Raises:
            Exception: ドキュメント削除中にエラーが発生した場合
        """
        try:
            db = get_firestore_client()
            doc_ref = db.collection(self.collection_name()).document(self.document_id)

            logger.debug(
                f"ドキュメント削除開始: {self.collection_name()}/{self.document_id}",
                extra={"context": {"collection": self.collection_name(), "doc_id": self.document_id}}
            )

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, doc_ref.delete)

            logger.info(
                f"ドキュメント削除完了: {self.collection_name()}/{self.document_id}",
                extra={"context": {"collection": self.collection_name(), "doc_id": self.document_id}}
            )
        except Exception as e:
            logger.error(
                f"ドキュメント削除中にエラー発生: {self.collection_name()}/{self.document_id}: {str(e)}",
                extra={
                    "context": {
                        "collection": self.collection_name(),
                        "doc_id": self.document_id,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                }
            )
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

# ========================================
# 共通レスポンスモデル
# ========================================

class StandardResponse(Generic[T], BaseModel):
    """標準APIレスポンスモデル

    すべてのAPIレスポンスの標準形式を定義します。
    successフラグとdataまたはerrorフィールドを持ちます。
    """
    success: bool = Field(..., description="リクエストの成功・失敗を示すフラグ")
    data: Optional[T] = Field(None, description="成功時のレスポンスデータ")
    error: Optional[Dict[str, Any]] = Field(None, description="エラー詳細（失敗時）")
    message: Optional[str] = Field(None, description="レスポンスメッセージ")

class PaginatedResponse(Generic[T], StandardResponse[List[T]]):
    """ページネーション対応レスポンスモデル

    ページネーションメタデータを含む標準レスポンス形式です。
    """
    meta: Optional[Dict[str, Any]] = Field(None, description="ページネーションメタデータ")

class ErrorDetail(BaseModel):
    """エラー詳細モデル"""
    code: str = Field(..., description="エラーコード")
    message: str = Field(..., description="エラーメッセージ")
    details: Optional[Dict[str, Any]] = Field(None, description="追加の詳細情報")
    request_id: Optional[str] = Field(None, description="リクエストID")

class ErrorResponse(StandardResponse[None]):
    """エラーレスポンスモデル"""
    success: bool = Field(False, description="リクエストの失敗")
    error: ErrorDetail = Field(..., description="エラー詳細")

# ========================================
# ユーザー関連モデル
# ========================================

class UserRole(str, Enum):
    """ユーザーロール定義"""
    ADMIN = "admin"
    USER = "user"
    VC = "vc"

class UserBase(BaseModel):
    """ユーザー基本情報モデル"""
    email: EmailStr = Field(..., description="ユーザーのメールアドレス")
    display_name: str = Field(..., description="表示名")
    role: UserRole = Field(UserRole.USER, description="ユーザーロール")

class UserCreate(UserBase):
    """ユーザー作成リクエストモデル"""
    password: str = Field(..., description="パスワード", min_length=8)

class User(UserBase):
    """ユーザーモデル"""
    id: str = Field(..., description="ユーザーID")
    created_at: datetime = Field(..., description="作成日時")
    is_active: bool = Field(True, description="アカウントがアクティブかどうか")
    company_id: Optional[str] = Field(None, description="所属会社ID")

    class Config:
        orm_mode = True

class UserInDB(User):
    """データベース内ユーザーモデル"""
    hashed_password: str = Field(..., description="ハッシュ化されたパスワード")

class Token(BaseModel):
    """認証トークンモデル"""
    access_token: str = Field(..., description="アクセストークン")
    token_type: str = Field("bearer", description="トークンタイプ")
    expires_in: int = Field(..., description="有効期限（秒）")
    user: User = Field(..., description="ユーザー情報")

class TokenData(BaseModel):
    """トークンデータモデル"""
    username: Optional[str] = None
    scopes: List[str] = []

# ========================================
# 可視化関連モデル
# ========================================

class ChartConfig(BaseModel):
    """チャート設定モデル"""
    chart_type: str = Field(..., description="チャートの種類 (bar, line, pie, scatter)")
    title: Optional[str] = Field(None, description="チャートのタイトル")
    x_axis_label: Optional[str] = Field(None, description="X軸のラベル")
    y_axis_label: Optional[str] = Field(None, description="Y軸のラベル")
    color_scheme: Optional[str] = Field(None, description="カラースキーム")
    show_legend: Optional[bool] = Field(True, description="凡例を表示するかどうか")
    width: Optional[int] = Field(800, description="チャートの幅")
    height: Optional[int] = Field(500, description="チャートの高さ")

class ChartDataset(BaseModel):
    """チャートデータセットモデル"""
    label: str = Field(..., description="データセットのラベル")
    data: List[float] = Field(..., description="データ値のリスト")
    color: Optional[str] = Field(None, description="データセットの色")

class ChartData(BaseModel):
    """チャートデータモデル"""
    labels: List[str] = Field(..., description="データラベルのリスト")
    datasets: List[ChartDataset] = Field(..., description="データセットのリスト")

class ChartRequest(BaseModel):
    """チャート生成リクエストモデル"""
    config: ChartConfig = Field(..., description="チャート設定")
    data: ChartData = Field(..., description="チャートデータ")
    format: Optional[str] = Field("png", description="出力フォーマット (png, svg, pdf)")
    template_id: Optional[str] = Field(None, description="テンプレートID")

class ChartMetadata(BaseModel):
    """チャートメタデータモデル"""
    created_at: str = Field(..., description="作成日時")
    user_id: Optional[str] = Field(None, description="ユーザーID")
    chart_type: str = Field(..., description="チャートタイプ")
    title: Optional[str] = Field(None, description="タイトル")
    template_id: Optional[str] = Field(None, description="テンプレートID")
    format: str = Field(..., description="フォーマット")

class ChartResponse(BaseModel):
    """チャート生成レスポンスモデル"""
    chart_id: str = Field(..., description="チャートID")
    url: str = Field(..., description="チャートURL")
    format: str = Field(..., description="フォーマット")
    thumbnail_url: Optional[str] = Field(None, description="サムネイルURL")
    metadata: ChartMetadata = Field(..., description="メタデータ")

class MultipleChartRequest(BaseModel):
    """複数チャート生成リクエストモデル"""
    charts: List[ChartRequest] = Field(..., description="生成するチャートのリスト")

class DashboardSection(BaseModel):
    """ダッシュボードセクションモデル"""
    title: str = Field(..., description="セクションのタイトル")
    charts: List[int] = Field(..., description="セクションに含めるチャートのインデックス")

class DashboardRequest(BaseModel):
    """ダッシュボード生成リクエストモデル"""
    title: str = Field(..., description="ダッシュボードのタイトル")
    description: Optional[str] = Field(None, description="ダッシュボードの説明")
    sections: List[DashboardSection] = Field(..., description="ダッシュボードのセクション")
    chart_ids: List[str] = Field(..., description="使用するチャートIDのリスト")
    theme: Optional[str] = Field("light", description="テーマ (light, dark, blue)")
    format: Optional[str] = Field("pdf", description="出力フォーマット (pdf, html)")

class DashboardMetadata(BaseModel):
    """ダッシュボードメタデータモデル"""
    created_at: str = Field(..., description="作成日時")
    user_id: Optional[str] = Field(None, description="ユーザーID")
    title: str = Field(..., description="タイトル")
    sections_count: int = Field(..., description="セクション数")
    charts_count: int = Field(..., description="チャート数")
    theme: str = Field(..., description="テーマ")

class DashboardResponse(BaseModel):
    """ダッシュボード生成レスポンスモデル"""
    dashboard_id: str = Field(..., description="ダッシュボードID")
    url: str = Field(..., description="ダッシュボードURL")
    format: str = Field(..., description="フォーマット")
    chart_ids: List[str] = Field(..., description="使用したチャートID")
    metadata: DashboardMetadata = Field(..., description="メタデータ")

class JobStatusResponse(BaseModel):
    """ジョブステータスレスポンス"""
    job_id: str = Field(..., description="ジョブID")
    status: str = Field(..., description="ステータス (pending, completed, failed)")
    result: Optional[Dict[str, Any]] = Field(None, description="結果データ")
    error: Optional[str] = Field(None, description="エラーメッセージ")
    created_at: str = Field(..., description="作成日時")
    completed_at: Optional[str] = Field(None, description="完了日時")

# ========================================
# レポート関連モデル
# ========================================

class ReportTemplate(BaseModel):
    """レポートテンプレートモデル"""
    id: str = Field(..., description="テンプレートID")
    name: str = Field(..., description="テンプレート名")
    description: str = Field(..., description="テンプレートの説明")
    sections: List[str] = Field(..., description="利用可能なセクション")

class ReportRequest(BaseModel):
    """レポート生成リクエストモデル"""
    template_id: str = Field(..., description="レポートテンプレートID")
    company_data: Dict[str, Any] = Field(..., description="企業データ")
    period: str = Field(..., description="レポート期間")
    include_sections: List[str] = Field(..., description="含めるセクション")
    customization: Optional[Dict[str, Any]] = Field(None, description="カスタマイズ設定")
    format: str = Field("pdf", description="出力フォーマット ('pdf', 'html')")

class ReportData(BaseModel):
    """レポートデータモデル"""
    report_id: str = Field(..., description="レポートID")
    report_url: str = Field(..., description="レポートURL")
    format: str = Field(..., description="フォーマット")

class ReportResponse(StandardResponse[ReportData]):
    """レポート生成レスポンスモデル"""
    pass

class TemplateData(BaseModel):
    """テンプレートデータモデル"""
    templates: List[ReportTemplate] = Field(..., description="テンプレートリスト")

class TemplateListResponse(StandardResponse[TemplateData]):
    """テンプレート一覧レスポンスモデル"""
    pass