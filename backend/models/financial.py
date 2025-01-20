from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from decimal import Decimal

class FinancialMetrics(BaseModel):
    """財務指標モデル"""
    revenue: Decimal = Field(..., description="売上高")
    operating_profit: Decimal = Field(..., description="営業利益")
    net_profit: Decimal = Field(..., description="純利益")
    total_assets: Decimal = Field(..., description="総資産")
    total_liabilities: Decimal = Field(..., description="総負債")
    equity: Decimal = Field(..., description="純資産")
    cash_flow: Decimal = Field(..., description="営業キャッシュフロー")

    @validator('*')
    def round_decimal(cls, v):
        """小数点以下2桁に丸める"""
        if isinstance(v, Decimal):
            return round(v, 2)
        return v

class FinancialData(BaseModel):
    """財務データモデル"""
    company_id: str = Field(..., description="企業ID")
    fiscal_year: int = Field(..., description="会計年度")
    fiscal_quarter: Optional[int] = Field(None, description="会計四半期")
    metrics: FinancialMetrics
    report_date: datetime = Field(..., description="報告日")
    currency: str = Field(default="JPY", description="通貨")

    class Config:
        from_attributes = True

class FinancialRatios(BaseModel):
    """財務比率モデル"""
    company_id: str = Field(..., description="企業ID")
    period: str = Field(..., description="期間")
    profit_margin: Decimal = Field(..., description="利益率")
    roa: Decimal = Field(..., description="総資産利益率")
    roe: Decimal = Field(..., description="自己資本利益率")
    current_ratio: Decimal = Field(..., description="流動比率")
    debt_ratio: Decimal = Field(..., description="負債比率")
    asset_turnover: Decimal = Field(..., description="総資産回転率")

    @validator('*')
    def round_decimal(cls, v):
        """小数点以下2桁に丸める"""
        if isinstance(v, Decimal):
            return round(v, 2)
        return v

class FinancialGrowth(BaseModel):
    """成長率モデル"""
    company_id: str = Field(..., description="企業ID")
    period: str = Field(..., description="期間")
    revenue_growth: Decimal = Field(..., description="売上高成長率")
    profit_growth: Decimal = Field(..., description="利益成長率")
    asset_growth: Decimal = Field(..., description="資産成長率")
    employee_growth: Decimal = Field(..., description="従業員成長率")

    @validator('*')
    def round_decimal(cls, v):
        """小数点以下2桁に丸める"""
        if isinstance(v, Decimal):
            return round(v, 2)
        return v