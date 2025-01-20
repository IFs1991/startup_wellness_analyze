from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator
from decimal import Decimal

class WellnessMetrics(BaseModel):
    """ウェルネス指標モデル"""
    engagement: Decimal = Field(..., ge=0, le=100, description="エンゲージメントスコア")
    satisfaction: Decimal = Field(..., ge=0, le=100, description="満足度スコア")
    work_life_balance: Decimal = Field(..., ge=0, le=100, description="ワークライフバランススコア")
    stress_level: Decimal = Field(..., ge=0, le=100, description="ストレスレベル")
    team_collaboration: Decimal = Field(..., ge=0, le=100, description="チームコラボレーション")
    personal_growth: Decimal = Field(..., ge=0, le=100, description="個人の成長")

    @validator('*')
    def round_decimal(cls, v):
        """小数点以下1桁に丸める"""
        if isinstance(v, Decimal):
            return round(v, 1)
        return v

class WellnessSurveyResponse(BaseModel):
    """ウェルネス調査回答モデル"""
    company_id: str = Field(..., description="企業ID")
    employee_id: str = Field(..., description="従業員ID")
    survey_date: datetime = Field(..., description="調査日")
    metrics: WellnessMetrics
    comments: Optional[str] = Field(None, description="コメント")
    department: Optional[str] = Field(None, description="部署")
    position_level: Optional[str] = Field(None, description="役職レベル")

    class Config:
        from_attributes = True

class WellnessAggregateMetrics(BaseModel):
    """ウェルネス集計指標モデル"""
    company_id: str = Field(..., description="企業ID")
    period: str = Field(..., description="期間")
    metrics: WellnessMetrics
    response_count: int = Field(..., description="回答数")
    department_scores: Optional[Dict[str, WellnessMetrics]] = Field(None, description="部署別スコア")
    trend_indicators: Dict[str, str] = Field(..., description="トレンド指標")

class WellnessTrend(BaseModel):
    """ウェルネストレンドモデル"""
    company_id: str = Field(..., description="企業ID")
    start_date: datetime = Field(..., description="開始日")
    end_date: datetime = Field(..., description="終了日")
    metrics_history: List[Dict[str, Decimal]] = Field(..., description="指標履歴")
    trend_analysis: Dict[str, str] = Field(..., description="トレンド分析")
    seasonal_patterns: Optional[Dict[str, str]] = Field(None, description="季節性パターン")

class WellnessAlert(BaseModel):
    """ウェルネスアラートモデル"""
    company_id: str = Field(..., description="企業ID")
    alert_date: datetime = Field(..., description="アラート日時")
    alert_type: str = Field(..., description="アラートタイプ")
    metric_name: str = Field(..., description="指標名")
    current_value: Decimal = Field(..., description="現在値")
    threshold_value: Decimal = Field(..., description="閾値")
    description: str = Field(..., description="アラート内容")
    priority: str = Field(..., description="優先度")
    status: str = Field(default="active", description="ステータス")