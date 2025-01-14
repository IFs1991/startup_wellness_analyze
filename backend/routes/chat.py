from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from ..models.chat import ChatRequest, ChatResponse
from ..services.chat_service import ChatService
from ..services.company_service import CompanyService
from ..auth import get_current_user

router = APIRouter(
    prefix="/api/v1/chat",
    tags=["chat"],
    dependencies=[Depends(get_current_user)]
)

@router.post("", response_model=ChatResponse)
async def chat_with_company_context(request: ChatRequest):
    """企業コンテキストを含むAIチャット"""
    try:
        # 企業データを取得
        company_service = CompanyService()
        company_data = await company_service.get_company_details(request.company_id)

        if not company_data:
            raise HTTPException(status_code=404, detail="企業が見つかりません")

        # チャットサービスを初期化
        chat_service = ChatService()

        # 企業データを含めてAIに質問
        response = await chat_service.get_response(
            message=request.message,
            company_data=company_data,
            model=request.model
        )

        return ChatResponse(response=response)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))