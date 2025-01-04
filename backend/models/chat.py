from pydantic import BaseModel
from typing import Optional, Dict, Any

class ChatRequest(BaseModel):
    """チャットリクエストモデル"""
    message: str
    company_id: str
    model: str

class ChatResponse(BaseModel):
    """チャットレスポンスモデル"""
    response: str