from pydantic import BaseModel
from typing import Optional

class OpenAISettings(BaseModel):
    """OpenAI API設定モデル"""
    api_key: str
    model: str
    request_limit: Optional[int] = 1000
    limit_period: Optional[str] = "hour"  # hour, day, month
    notify_on_limit: Optional[bool] = True

class OpenAISettingsResponse(BaseModel):
    """OpenAI API設定のレスポンスモデル（APIキーを除く）"""
    model: str
    request_limit: Optional[int] = 1000
    limit_period: Optional[str] = "hour"
    notify_on_limit: Optional[bool] = True