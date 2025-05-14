# -*- coding: utf-8 -*-
"""
企業情報 API ルーター
----------------------
企業情報の取得 (GET) および追加 (POST) を処理します。
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status, Body
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime

from backend.services.company_service import CompanyService
from backend.database.models.entities import CompanyEntity
from backend.database.connection import get_db
from backend.api.dependencies import get_company_service
from backend.database.repository import EntityNotFoundException, ValidationException

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
    website: Optional[str] = Field(None, description="ウェブサイトURL")
    description: Optional[str] = Field(None, description="企業概要")

class CompanyCreate(CompanyBase):
    """企業作成時の入力モデル"""
    pass

class Company(CompanyBase):
    """企業情報のレスポンスモデル"""
    id: str = Field(..., description="企業ID")
    created_at: Optional[datetime] = Field(None, description="作成日時")
    updated_at: Optional[datetime] = Field(None, description="最終更新日時")

    class Config:
        from_attributes = True  # Pydantic V2 (旧 orm_mode)

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
    limit: Optional[int] = Query(50, description="取得する最大件数", ge=1, le=1000),
    company_service: CompanyService = Depends(get_company_service)
):
    """
    企業リストを取得するエンドポイント。

    - **search**: 企業名で部分一致検索。
    - **filters**: カンマ区切りのキー=値ペアで完全一致フィルタリング (例: `industry=SaaS,location=東京都`)。
    """
    logger.info(f"Fetching companies with search='{search}', filters='{filters}', limit={limit}")

    try:
        # フィルタ条件の解析
        parsed_filters = {}
        if filters:
            try:
                for item in filters.split(','):
                    if '=' not in item:
                        continue
                    key, value = item.strip().split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key and value:
                        parsed_filters[key] = value
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid format for filters. Use 'key1=value1,key2=value2'.",
                )

        # サービスを使用してデータを取得
        companies = await company_service.get_companies(
            search=search,
            filters=parsed_filters,
            limit=limit
        )

        logger.info(f"Found {len(companies)} companies matching criteria.")
        return companies

    except Exception as e:
        logger.error(f"Error fetching companies: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch companies.",
        )

@router.post(
    "/",
    response_model=Company,
    status_code=status.HTTP_201_CREATED,
    summary="新規企業追加",
    description="新しい企業情報を登録します。",
)
async def add_company(
    company_data: CompanyCreate = Body(..., description="登録する企業情報"),
    company_service: CompanyService = Depends(get_company_service)
):
    """
    新しい企業情報をデータベースに追加します。
    """
    logger.info(f"Attempting to add new company: {company_data.name}")

    try:
        # Pydanticモデルを辞書に変換
        company_dict = company_data.model_dump(exclude_unset=True)

        # サービスを使用して企業を作成
        created_company = await company_service.create_company(company_dict)

        logger.info(f"Successfully added company '{created_company.name}' with id: {created_company.entity_id}")
        return created_company

    except ValidationException as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error adding company: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add company.",
        )

@router.get(
    "/{company_id}",
    response_model=Company,
    summary="企業情報取得",
    description="企業IDで企業情報を取得します。",
)
async def get_company(
    company_id: str,
    company_service: CompanyService = Depends(get_company_service)
):
    """
    指定されたIDの企業情報を取得します。
    """
    logger.info(f"Fetching company with id: {company_id}")

    try:
        company = await company_service.get_company_by_id(company_id)

        if not company:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Company with id {company_id} not found",
            )

        return company

    except EntityNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id {company_id} not found",
        )
    except Exception as e:
        logger.error(f"Error fetching company: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch company.",
        )

@router.put(
    "/{company_id}",
    response_model=Company,
    summary="企業情報更新",
    description="既存の企業情報を更新します。",
)
async def update_company(
    company_id: str,
    company_data: CompanyBase = Body(..., description="更新する企業情報"),
    company_service: CompanyService = Depends(get_company_service)
):
    """
    指定されたIDの企業情報を更新します。
    """
    logger.info(f"Updating company with id: {company_id}")

    try:
        # Pydanticモデルを辞書に変換
        company_dict = company_data.model_dump(exclude_unset=True)

        # サービスを使用して企業を更新
        updated_company = await company_service.update_company(company_id, company_dict)

        logger.info(f"Successfully updated company with id: {company_id}")
        return updated_company

    except EntityNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id {company_id} not found",
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error updating company: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update company.",
        )

@router.delete(
    "/{company_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="企業情報削除",
    description="企業情報を削除します。",
)
async def delete_company(
    company_id: str,
    company_service: CompanyService = Depends(get_company_service)
):
    """
    指定されたIDの企業情報を削除します。
    """
    logger.info(f"Deleting company with id: {company_id}")

    try:
        # サービスを使用して企業を削除
        success = await company_service.delete_company(company_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Company with id {company_id} not found",
            )

        logger.info(f"Successfully deleted company with id: {company_id}")
        return None

    except EntityNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with id {company_id} not found",
        )
    except Exception as e:
        logger.error(f"Error deleting company: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete company.",
        )