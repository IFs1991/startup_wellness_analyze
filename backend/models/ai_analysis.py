from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .company import Company
from .financial import FinancialData, FinancialRatios, FinancialGrowth
from .wellness import WellnessAggregateMetrics, WellnessTrend

class CompanyAnalysisContext(BaseModel):
    """企業分析コンテキストモデル"""
    company: Company
    financial_data: List[FinancialData]
    financial_ratios: List[FinancialRatios]
    financial_growth: List[FinancialGrowth]
    wellness_metrics: List[WellnessAggregateMetrics]
    wellness_trends: List[WellnessTrend]
    analysis_date: datetime = Field(default_factory=datetime.now)

class AIAnalysisRequest(BaseModel):
    """AI分析リクエストモデル"""
    company_id: str = Field(..., description="企業ID")
    analysis_type: str = Field(..., description="分析タイプ")
    time_range: Optional[str] = Field(None, description="時間範囲")
    metrics: Optional[List[str]] = Field(None, description="分析対象指標")
    additional_context: Optional[Dict[str, Any]] = Field(None, description="追加コンテキスト")

class AIAnalysisResponse(BaseModel):
    """AI分析レスポンスモデル"""
    company_id: str = Field(..., description="企業ID")
    analysis_type: str = Field(..., description="分析タイプ")
    analysis_date: datetime = Field(default_factory=datetime.now)
    insights: List[Dict[str, Any]] = Field(..., description="分析インサイト")
    recommendations: List[Dict[str, Any]] = Field(..., description="推奨アクション")
    risk_factors: Optional[List[Dict[str, Any]]] = Field(None, description="リスク要因")
    opportunity_areas: Optional[List[Dict[str, Any]]] = Field(None, description="機会領域")

class AIAssistantContext(BaseModel):
    """AIアシスタントコンテキストモデル"""
    company_context: CompanyAnalysisContext
    current_analysis: Optional[AIAnalysisResponse] = None
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    user_preferences: Optional[Dict[str, Any]] = Field(None)

    class Config:
        from_attributes = True