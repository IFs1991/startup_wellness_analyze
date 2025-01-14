from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from ..models.api_settings import OpenAISettings, OpenAISettingsResponse
from ..services.api_settings_service import APISettingsService
from ..auth import get_current_user

router = APIRouter(
    prefix="/api/v1/settings",
    tags=["settings"],
    dependencies=[Depends(get_current_user)]
)

@router.post("/openai", response_model=OpenAISettingsResponse)
async def save_openai_settings(settings: OpenAISettings):
    """OpenAI API設定を保存"""
    try:
        service = APISettingsService()
        return await service.save_openai_settings(settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/openai", response_model=OpenAISettingsResponse)
async def get_openai_settings():
    """OpenAI API設定を取得"""
    try:
        service = APISettingsService()
        settings = await service.get_openai_settings()
        if not settings:
            raise HTTPException(status_code=404, detail="設定が見つかりません")
        return settings
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))