from datetime import datetime
from typing import Union, List

from backend.app.schemas import DashboardConfig, GraphConfig, VisualizationResponse
from backend.service.firestore.client import FirestoreService
from backend.service.tasks import process_visualization_data


class BaseVisualizationService:
    """可視化サービスの基本クラス"""
    def __init__(self, firestore_service: FirestoreService):
        self.firestore = firestore_service

    async def create_visualization(
        self,
        config: Union[DashboardConfig, GraphConfig],
        visualization_type: str,
        user_id: str
    ) -> VisualizationResponse:
        """可視化を作成する共通ロジック"""
        visualization_data = {
            'id': f"{visualization_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'type': visualization_type,
            'config': config.dict(),
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'created_by': user_id,
            'data': {},
            'status': 'processing'
        }

        await self.firestore.save_results(
            results=[visualization_data],
            collection_name='visualizations'
        )

        # バックグラウンドタスクの開始
        await process_visualization_data(
            visualization_data['id'],
            config.dict(),
            user_id
        )

        return VisualizationResponse(**visualization_data)

    async def get_user_visualizations(
        self,
        user_id: str
    ) -> List[VisualizationResponse]:
        """ユーザーの可視化一覧を取得"""
        conditions = [
            {'field': 'created_by', 'operator': '==', 'value': user_id}
        ]

        visualizations = await self.firestore.fetch_documents(
            collection_name='visualizations',
            conditions=conditions
        )

        return [VisualizationResponse(**v) for v in visualizations]