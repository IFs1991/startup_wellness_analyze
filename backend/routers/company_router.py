from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional, Dict, Any
from datetime import datetime

from auth import get_current_user
from schemas import CompanyCreate, CompanyUpdate, Company, StatusBase, Status, StageBase, Stage
from src.database.firestore.client import FirestoreClient

router = APIRouter(prefix="/api/companies", tags=["companies"])
firestore_client = FirestoreClient()

@router.post("/", response_model=Company, status_code=status.HTTP_201_CREATED)
async def create_company(
    company: CompanyCreate,
    current_user = Depends(get_current_user)
):
    """新規会社を作成"""
    try:
        company_id = company.id
        if not company_id:
            raise HTTPException(status_code=400, detail="企業IDは必須です")

        company.created_at = datetime.utcnow()
        company.updated_at = datetime.utcnow()

        await firestore_client.create_document(
            collection='companies',
            doc_id=company_id,
            data=company.dict()
        )

        return {
            "status": "success",
            "company_id": company_id
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{company_id}", response_model=Company)
async def get_company(
    company_id: str,
    current_user = Depends(get_current_user)
):
    """会社情報を取得"""
    try:
        company = await firestore_client.get_document(
            collection='companies',
            doc_id=company_id
        )
        if not company:
            raise HTTPException(status_code=404, detail="企業が見つかりません")
        return company
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{company_id}", response_model=Company)
async def update_company(
    company_id: str,
    company: CompanyUpdate,
    current_user = Depends(get_current_user)
):
    """会社情報を更新"""
    try:
        company.updated_at = datetime.utcnow()
        await firestore_client.update_document(
            collection='companies',
            doc_id=company_id,
            data=company.dict()
        )
        return {
            "status": "success",
            "company_id": company_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{company_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_company(
    company_id: str,
    current_user = Depends(get_current_user)
):
    """会社を削除"""
    try:
        await firestore_client.delete_document(
            collection='companies',
            doc_id=company_id
        )
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{company_id}/status", response_model=Status, status_code=status.HTTP_201_CREATED)
async def add_company_status(
    company_id: str,
    status: StatusBase,
    current_user = Depends(get_current_user)
):
    """会社のステータスを追加"""
    # 実装は省略
    return {
        "id": "dummy_status_id",
        **status.dict(),
        "created_at": datetime.now()
    }

@router.post("/{company_id}/stage", response_model=Stage, status_code=status.HTTP_201_CREATED)
async def add_company_stage(
    company_id: str,
    stage: StageBase,
    current_user = Depends(get_current_user)
):
    """会社のステージを追加"""
    # 実装は省略
    return {
        "id": "dummy_stage_id",
        **stage.dict(),
        "created_at": datetime.now()
    }

@router.get("/search", response_model=List[Company])
async def search_companies(
    industry: Optional[str] = None,
    min_employees: Optional[int] = None,
    max_employees: Optional[int] = None,
    current_user = Depends(get_current_user)
):
    """会社を検索"""
    # 実装は省略
    return [
        {
            "id": "dummy_id",
            "name": "Test Company",
            "industry": industry or "Technology",
            "description": "Test Description",
            "employee_count": min_employees or 100,
            "website": "https://example.com",
            "location": "Tokyo, Japan",
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
    ]

@router.get("/")
async def get_companies() -> List[Dict[str, Any]]:
    """企業一覧を取得"""
    try:
        companies = await firestore_client.query_documents(
            collection='companies',
            order_by=('created_at', 'desc')
        )
        return companies
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/")
async def create_company(company_data: Dict[str, Any]) -> Dict[str, Any]:
    """企業を作成"""
    try:
        company_id = company_data.get('id')
        if not company_id:
            raise HTTPException(status_code=400, detail="企業IDは必須です")

        company_data['created_at'] = datetime.utcnow()
        company_data['updated_at'] = datetime.utcnow()

        await firestore_client.create_document(
            collection='companies',
            doc_id=company_id,
            data=company_data
        )

        return {
            "status": "success",
            "company_id": company_id
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{company_id}")
async def update_company(company_id: str, company_data: Dict[str, Any]) -> Dict[str, Any]:
    """企業を更新"""
    try:
        company_data['updated_at'] = datetime.utcnow()
        await firestore_client.update_document(
            collection='companies',
            doc_id=company_id,
            data=company_data
        )
        return {
            "status": "success",
            "company_id": company_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{company_id}")
async def delete_company(company_id: str) -> Dict[str, str]:
    """企業を削除"""
    try:
        await firestore_client.delete_document(
            collection='companies',
            doc_id=company_id
        )
        return {
            "status": "success",
            "message": "企業が削除されました"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))