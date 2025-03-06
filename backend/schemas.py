# -*- coding: utf-8 -*-
"""
Pydantic スキーマ

API リクエストやレスポンスのデータ構造を定義する Pydantic モデルを定義します。

"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, EmailStr


class Token(BaseModel):
    """トークンスキーマ"""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """トークンデータスキーマ"""
    username: Optional[str] = None


class UserBase(BaseModel):
    """ユーザーベーススキーマ"""
    email: EmailStr
    username: str
    role: str = "user"


class UserCreate(UserBase):
    """ユーザー作成スキーマ"""
    password: str


class UserUpdate(BaseModel):
    """ユーザー更新スキーマ"""
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    role: Optional[str] = None


class User(UserBase):
    """ユーザースキーマ"""
    id: str
    is_active: bool = True

    class Config:
        from_attributes = True


class DashboardConfig(BaseModel):
    """ダッシュボード設定モデル"""
    title: str
    description: Optional[str] = None
    widgets: List[Dict]
    layout: Dict


class GraphConfig(BaseModel):
    """グラフ設定モデル"""
    type: str
    title: str
    data_source: str
    settings: Dict
    filters: Optional[List[Dict]] = None


class VisualizationResponse(BaseModel):
    """可視化レスポンスモデル"""
    id: str
    created_at: datetime
    updated_at: datetime
    config: Dict
    data: Dict
    created_by: str

    model_config = {
        "from_attributes": True
    }


class ReportBase(BaseModel):
    """レポートベースモデル"""
    title: str
    description: Optional[str] = None
    report_type: str
    parameters: Optional[Dict] = None

    model_config = {
        "from_attributes": True
    }


class CompanyBase(BaseModel):
    """会社ベーススキーマ"""
    name: str
    industry: str
    description: Optional[str] = None
    employee_count: int
    website: Optional[str] = None
    location: Optional[str] = None


class CompanyCreate(CompanyBase):
    """会社作成スキーマ"""
    pass


class CompanyUpdate(CompanyBase):
    """会社更新スキーマ"""
    pass


class Company(CompanyBase):
    """会社スキーマ"""
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class StatusBase(BaseModel):
    """ステータスベーススキーマ"""
    type: str
    description: str


class Status(StatusBase):
    """ステータススキーマ"""
    id: str
    created_at: datetime

    class Config:
        from_attributes = True


class StageBase(BaseModel):
    """ステージベーススキーマ"""
    type: str
    description: str


class Stage(StageBase):
    """ステージスキーマ"""
    id: str
    created_at: datetime

    class Config:
        from_attributes = True


class GroupBase(BaseModel):
    """グループベーススキーマ"""
    name: str
    description: Optional[str] = None
    company_id: str


class GroupCreate(GroupBase):
    """グループ作成スキーマ"""
    pass


class GroupUpdate(GroupBase):
    """グループ更新スキーマ"""
    pass


class Group(GroupBase):
    """グループスキーマ"""
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MemberBase(BaseModel):
    """メンバーベーススキーマ"""
    user_id: str
    role: str


class Member(MemberBase):
    """メンバースキーマ"""
    id: str
    group_id: str
    created_at: datetime

    class Config:
        from_attributes = True


class ReportCreate(ReportBase):
    """レポート作成スキーマ"""
    pass


class ReportUpdate(ReportBase):
    """レポート更新スキーマ"""
    pass


class Report(ReportBase):
    """レポートスキーマ"""
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TemplateBase(BaseModel):
    """テンプレートベーススキーマ"""
    name: str
    description: Optional[str] = None
    format: str
    template_content: str
    variables: Dict[str, Any]


class Template(TemplateBase):
    """テンプレートスキーマ"""
    id: str
    created_at: datetime

    class Config:
        from_attributes = True


class AnalysisRequest(BaseModel):
    """分析リクエストスキーマ"""
    type: str
    data: Dict[str, Any]
    params: Dict[str, Any]

    model_config = {
        "from_attributes": True
    }


class AnalysisResponse(BaseModel):
    """分析レスポンススキーマ"""
    id: str
    type: str
    status: str
    created_at: datetime
    updated_at: datetime
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    user_id: str

    model_config = {
        "from_attributes": True
    }