from datetime import datetime
from typing import List, Dict, Any
from uuid import uuid4

from backend.app.schemas import VisualizationResponse

class BaseVisualizationService:
    def __init__(self, db_service):
        self.db = db_service

    async def create_visualization(
        self,
        config: Dict[str, Any],
        visualization_type: str,
        user_id: str
    ) -> VisualizationResponse:
        """新しい可視化を作成"""
        visualization_id = str(uuid4())
        now = datetime.utcnow().isoformat()

        visualization_data = {
            "id": visualization_id,
            "user_id": user_id,
            "visualization_type": visualization_type,
            "config": config,
            "created_at": now,
            "updated_at": now
        }

        # Firestoreに保存
        await self.db.create_document(
            collection="visualizations",
            document_id=visualization_id,
            data=visualization_data
        )

        return VisualizationResponse(**visualization_data)

    async def get_user_visualizations(
        self,
        user_id: str
    ) -> List[VisualizationResponse]:
        """ユーザーの可視化一覧を取得"""
        visualizations = await self.db.query_documents(
            collection="visualizations",
            filters={"user_id": user_id}
        )

        return [VisualizationResponse(**v) for v in visualizations]