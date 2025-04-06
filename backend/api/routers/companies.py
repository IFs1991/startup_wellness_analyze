# -*- coding: utf-8 -*-
"""
企業情報 API ルーター
----------------------
企業情報の取得 (GET) および追加 (POST) を処理します。
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status, Body
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

# サービスと依存関係のインポート (main.py や他のルーターを参考に)
# TODO: 実際の依存関係インポートパスを確認し、コメントアウトを解除する
# from service.firestore.client import FirestoreService, get_firestore_service
# from core.auth_manager import AuthManager, get_current_active_user # 認証用
# from models.user import User # 認証用

# --- 仮の依存関係 (実際のサービスに置き換える) ---

class MockFirestoreService:
    """FirestoreServiceの仮実装 (テスト用)"""
    async def query_documents(
        self,
        collection_path: str,
        conditions: Optional[List[tuple]] = None,
        limit: Optional[int] = None,
        order_by: Optional[str] = None,
        desc: bool = False,
        start_after: Optional[Any] = None # ページネーション用
    ) -> List[Dict[str, Any]]:
        """ドキュメントをクエリする (モック)"""
        logger.info(f"Mock Query: collection={collection_path}, conditions={conditions}, limit={limit}, order_by={order_by}, desc={desc}, start_after={start_after}")
        # サンプルデータ (フィルタリング/検索/ソート/ページネーションは別途実装が必要)
        all_companies = [
            {"id": "comp_1", "name": "株式会社サンプルA", "industry": "SaaS", "founded_date": datetime(2020, 1, 15), "employee_count": 50, "location": "東京都"},
            {"id": "comp_2", "name": "サンプルB有限会社", "industry": "FinTech", "founded_date": datetime(2019, 5, 10), "employee_count": 25, "location": "大阪府"},
            {"id": "comp_3", "name": "株式会社データ分析C", "industry": "SaaS", "founded_date": datetime(2021, 11, 1), "employee_count": 15, "location": "東京都"},
            {"id": "comp_4", "name": "イノベーションD", "industry": "HealthTech", "founded_date": datetime(2022, 2, 20), "employee_count": 8, "location": "福岡県"},
        ]
        # ここで conditions, order_by などに基づいたフィルタ/ソートを実装する
        # 簡単なフィルタ例 (完全一致のみ)
        filtered = all_companies
        if conditions:
            for cond in conditions:
                key, op, val = cond
                if op == "==":
                    filtered = [c for c in filtered if c.get(key) == val]
                # 他の演算子 (>, <, etc.) のサポートを追加

        # 簡単なソート例
        if order_by:
            filtered.sort(key=lambda x: x.get(order_by, None), reverse=desc)

        # 簡単な検索例 (name フィールドの部分一致) - クエリ後の処理として実装
        if hasattr(self, '_current_search_term') and self._current_search_term:
             filtered = [c for c in filtered if self._current_search_term.lower() in c.get('name', '').lower()]


        # 簡単なページネーション例 (limitのみ)
        if limit:
            return filtered[:limit]
        return filtered

    async def add_document(self, collection_path: str, data: Dict[str, Any], document_id: Optional[str] = None) -> str:
        """ドキュメントを追加する (モック)"""
        new_id = document_id or f"new_comp_{int(datetime.now().timestamp())}"
        logger.info(f"Mock Add: collection={collection_path}, id={new_id}, data={data}")
        # 実際のDB追加処理はここに記述
        return new_id

def get_firestore_service():
    """FirestoreService の依存関係プロバイダ (モック)"""
    # TODO: 実際の FirestoreService インスタンスを返すように修正
    return MockFirestoreService()

# 認証ユーザーの仮実装 (認証が不要な場合は削除)
class User(BaseModel):
    """仮のユーザーモデル"""
    id: str
    email: str
    is_active: bool

def get_current_active_user():
    """現在の認証済みアクティブユーザーを取得 (モック)"""
    # TODO: 実際の認証ロジックに置き換える
    return User(id="user123", email="test@example.com", is_active=True)

# --- ロガー設定 ---
logger = logging.getLogger(__name__)

# --- FastAPIルーターの初期化 ---
router = APIRouter(
    prefix="/companies",
    tags=["Companies"],
    responses={404: {"description": "Not found"}},
)

# --- Pydantic モデル定義 ---

class CompanyBase(BaseModel):
    """企業情報の基本モデル"""
    name: str = Field(..., description="企業名", min_length=1)
    industry: Optional[str] = Field(None, description="業界")
    founded_date: Optional[datetime] = Field(None, description="設立日")
    employee_count: Optional[int] = Field(None, description="従業員数", ge=0)
    location: Optional[str] = Field(None, description="所在地")
    # フロントエンド (CompaniesPage) で使用される可能性のある他のフィールド
    website: Optional[str] = Field(None, description="ウェブサイトURL")
    description: Optional[str] = Field(None, description="企業概要")

class CompanyCreate(CompanyBase):
    """企業作成時の入力モデル"""
    pass

class Company(CompanyBase):
    """企業情報のレスポンスモデル"""
    id: str = Field(..., description="企業ID (FirestoreドキュメントID)")
    created_at: Optional[datetime] = Field(None, description="作成日時") # Firestoreから取得する場合
    updated_at: Optional[datetime] = Field(None, description="最終更新日時") # Firestoreから取得する場合

    class Config:
        from_attributes = True # Pydantic V2 (旧 orm_mode)

# --- API エンドポイント定義 ---

@router.get(
    "/",
    response_model=List[Company],
    summary="企業リスト取得",
    description="検索条件やフィルタに基づいて企業リストを取得します。",
)
async def get_companies(
    search: Optional[str] = Query(None, description="検索キーワード (企業名など)"),
    filters: Optional[str] = Query(None, description="フィルタ条件 (例: industry=SaaS,location=東京都)"),
    limit: Optional[int] = Query(50, description="取得する最大件数", ge=1, le=1000), # 基本的な制限
    # TODO: 必要に応じてページネーションパラメータ (例: offset, last_doc_id) を追加
    db: MockFirestoreService = Depends(get_firestore_service),
    # current_user: User = Depends(get_current_active_user), # 認証が必要な場合
):
    """
    企業リストを取得するエンドポイント。

    - **search**: 企業名で部分一致検索 (現在はモック実装)。
    - **filters**: カンマ区切りのキー=値ペアで完全一致フィルタリング (例: `industry=SaaS,location=東京都`)。
    """
    logger.info(f"Fetching companies with search='{search}', filters='{filters}', limit={limit}")

    query_conditions = []
    parsed_filters = {}

    # フィルタ条件の解析
    if filters:
        try:
            for item in filters.split(','):
                if '=' not in item:
                    continue # 不正な形式は無視
                key, value = item.strip().split('=', 1)
                key = key.strip()
                value = value.strip()
                if key and value: # キーと値が空でないことを確認
                    # TODO: 値の型変換 (例: employee_count は数値) が必要になる場合がある
                    query_conditions.append((key, "==", value)) # Firestoreの完全一致クエリを想定
                    parsed_filters[key] = value
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid format for filters. Use 'key1=value1,key2=value2'.",
            )

    # 検索キーワードの処理 (Firestoreクエリでの直接的な部分一致は難しいため、クエリ後のフィルタリングで対応)
    # MockFirestoreService 内で検索キーワードを保持させる (仮の実装)
    db._current_search_term = search # この方法は良くないが、モックのため

    try:
        companies_data = await db.query_documents(
            collection_path="companies",
            conditions=query_conditions if query_conditions else None,
            limit=limit,
            order_by="name", # 名前でソート (例)
            desc=False
        )

        # FirestoreドキュメントをPydanticモデルに変換
        # 実際のデータでは、FirestoreのTimestampをPythonのdatetimeに変換する必要がある
        companies_list = []
        for company_dict in companies_data:
            # founded_date が datetime オブジェクトであることを確認 (モックデータは既にそうなっている)
            # if isinstance(company_dict.get('founded_date'), firestore.SERVER_TIMESTAMP): ...
            companies_list.append(Company(**company_dict))


        logger.info(f"Found {len(companies_list)} companies matching criteria.")
        return companies_list

    except Exception as e:
        logger.error(f"Error fetching companies: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch companies.",
        )
    finally:
         # MockFirestoreService の一時変数をクリア
         if hasattr(db, '_current_search_term'):
             delattr(db, '_current_search_term')


@router.post(
    "/",
    response_model=Company,
    status_code=status.HTTP_201_CREATED,
    summary="新規企業追加",
    description="新しい企業情報を登録します。",
)
async def add_company(
    company_data: CompanyCreate = Body(..., description="登録する企業情報"),
    db: MockFirestoreService = Depends(get_firestore_service),
    # current_user: User = Depends(get_current_active_user), # 認証が必要な場合
):
    """
    新しい企業情報を Firestore の `companies` コレクションに追加します。
    """
    logger.info(f"Attempting to add new company: {company_data.name}")

    try:
        # Pydanticモデルを辞書に変換 (Noneを除外, FirestoreはNoneを扱える)
        company_dict = company_data.model_dump(exclude_unset=True)

        # 日付をFirestoreが扱える形式に (既にdatetimeなら通常はSDKが処理)
        # if 'founded_date' in company_dict and isinstance(company_dict['founded_date'], datetime):
        #     pass # SDK should handle datetime objects

        # TODO: 重複チェックなど、ビジネスロジックを追加する (例: 同じ名前の会社が既に存在しないか)

        # Firestore にドキュメントを追加
        new_company_id = await db.add_document(
            collection_path="companies",
            data=company_dict
        )

        # 追加成功後、IDを含めてレスポンスを作成
        # Firestoreから追加したデータを再取得するのが確実だが、ここでは入力データを使う
        created_company_data = company_dict.copy()
        created_company_data['id'] = new_company_id
        # created_at/updated_at はFirestoreから取得するか、ここで設定
        now = datetime.now()
        created_company_data.setdefault('created_at', now)
        created_company_data.setdefault('updated_at', now)


        created_company = Company(**created_company_data)

        logger.info(f"Successfully added company '{created_company.name}' with id: {new_company_id}")
        return created_company

    except Exception as e:
        # TODO: 具体的なエラーハンドリング (例: Validation Error, DB Error)
        logger.error(f"Error adding company: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add company.",
        )

# TODO: 必要に応じて、特定の企業を取得 (GET /{company_id})、更新 (PUT /{company_id})、削除 (DELETE /{company_id}) するエンドポイントを追加