from typing import Optional, List, Dict
from datetime import datetime
from pydantic import BaseModel
from enum import Enum

class StageType(str, Enum):
    INITIAL = "initial"
    SCREENING = "screening"
    DUE_DILIGENCE = "due_diligence"
    NEGOTIATION = "negotiation"
    CLOSING = "closing"
    POST_INVESTMENT = "post_investment"

class Stage(BaseModel):
    id: str
    company_id: str
    stage_type: StageType
    status: str
    validation_status: str
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class StageService:
    def __init__(self, database):
        self.db = database

    async def update_stage(self, company_id: str, stage_data: dict) -> Stage:
        """ステージを更新する"""
        stage_data.update({
            "company_id": company_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
        stage = Stage(**stage_data)
        # TODO: データベースに保存する実装
        await self._validate_stage(stage)
        return stage

    async def get_stage_history(self, company_id: str) -> List[Stage]:
        """ステージ履歴を取得する"""
        # TODO: データベースからステージ履歴を取得する実装
        return []

    async def _validate_stage(self, stage: Stage) -> bool:
        """ステージの妥当性を検証する"""
        validation_rules = {
            StageType.INITIAL: lambda s: True,  # 初期ステージは常に有効
            StageType.SCREENING: self._validate_screening_stage,
            StageType.DUE_DILIGENCE: self._validate_due_diligence_stage,
            StageType.NEGOTIATION: self._validate_negotiation_stage,
            StageType.CLOSING: self._validate_closing_stage,
            StageType.POST_INVESTMENT: self._validate_post_investment_stage
        }

        validator = validation_rules.get(stage.stage_type)
        if validator:
            return await validator(stage)
        return False

    async def _validate_screening_stage(self, stage: Stage) -> bool:
        """スクリーニングステージの検証"""
        # TODO: スクリーニングステージの検証ロジックを実装
        return True

    async def _validate_due_diligence_stage(self, stage: Stage) -> bool:
        """デューデリジェンスステージの検証"""
        # TODO: デューデリジェンスステージの検証ロジックを実装
        return True

    async def _validate_negotiation_stage(self, stage: Stage) -> bool:
        """交渉ステージの検証"""
        # TODO: 交渉ステージの検証ロジックを実装
        return True

    async def _validate_closing_stage(self, stage: Stage) -> bool:
        """クロージングステージの検証"""
        # TODO: クロージングステージの検証ロジックを実装
        return True

    async def _validate_post_investment_stage(self, stage: Stage) -> bool:
        """ポストインベストメントステージの検証"""
        # TODO: ポストインベストメントステージの検証ロジックを実装
        return True