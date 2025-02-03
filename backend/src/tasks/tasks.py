import logging
from datetime import datetime
from typing import Dict, List, TypedDict, Any
from backend.src.database.firestore.client import FirestoreClient

logger = logging.getLogger(__name__)

class FilterConfig(TypedDict, total=False):
    field: str
    operator: str
    value: Any

def create_filter(filter_dict: Dict[str, Any]) -> FilterConfig:
    return {
        'field': filter_dict['field'],
        'operator': filter_dict['operator'],
        'value': filter_dict['value']
    }

class TaskProcessor:
    def __init__(self):
        self.firestore_client = FirestoreClient()

    async def process_visualization_data(
        self,
        visualization_id: str,
        config: Dict[str, Any],
        user_id: str
    ) -> None:
        """バックグラウンドで可視化データを処理"""
        try:
            # データソースからデータを取得
            filters: List[FilterConfig] = []
            if config.get('filters'):
                filters = [create_filter(f) for f in config['filters']]

            source_data = await self.firestore_client.query_documents(
                collection=config['data_source'],
                filters=filters
            )

            # データ処理と更新
            processed_data = {
                'id': visualization_id,
                'data': {
                    'source': source_data,
                    'processed': {},  # 実際のデータ処理ロジックを実装
                    'last_updated': datetime.utcnow()
                },
                'status': 'completed',
                'user_id': user_id
            }

            await self.firestore_client.create_document(
                collection='visualizations',
                doc_id=visualization_id,
                data=processed_data
            )

        except Exception as e:
            logger.error(f"Error processing visualization data: {str(e)}")
            error_data = {
                'id': visualization_id,
                'status': 'error',
                'error': str(e),
                'last_updated': datetime.utcnow(),
                'user_id': user_id
            }
            await self.firestore_client.create_document(
                collection='visualizations',
                doc_id=visualization_id,
                data=error_data
            )