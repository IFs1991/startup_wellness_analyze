from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class CompanyBase(BaseModel):
    """企業の基本情報モデル"""
    name: str = Field(..., description="企業名")
    description: Optional[str] = Field(None, description="企業概要")
    founded_year: Optional[int] = Field(None, description="設立年")
    employees: Optional[int] = Field(None, description="従業員数")
    location: Optional[str] = Field(None, description="所在地")
    industry: Optional[str] = Field(None, description="業界")
    stage: Optional[str] = Field(None, description="企業ステージ")
    website: Optional[str] = Field(None, description="Webサイト")

class CompanyCreate(CompanyBase):
    """企業新規作成モデル"""
    pass

class CompanyUpdate(CompanyBase):
    """企業情報更新モデル"""
    pass

class Company(CompanyBase):
    """企業情報完全モデル"""
    id: str = Field(..., description="企業ID")
    created_at: datetime = Field(default_factory=datetime.now, description="作成日時")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新日時")
    is_active: bool = Field(default=True, description="アクティブ状態")

    class Config:
        from_attributes = True