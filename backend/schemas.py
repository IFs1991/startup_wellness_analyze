# -*- coding: utf-8 -*-
"""
Pydantic スキーマ

API リクエストやレスポンスのデータ構造を定義する Pydantic モデルを定義します。

"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel


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


class ReportBase(BaseModel):
    """レポートベースモデル"""
    title: str
    description: Optional[str] = None
    report_type: str
    parameters: Optional[Dict] = None


class CompanyBase(BaseModel):
    name: str
    industry: str
    stage: str
    employees: Optional[int] = None
    founded_year: Optional[int] = None
    location: Optional[str] = None
    ceo: Optional[str] = None
    investment: Optional[float] = None
    score: Optional[float] = None


class CompanyCreate(CompanyBase):
    pass


class Company(CompanyBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True