import logging
from datetime import datetime
from typing import Dict

from backend.service.firestore.client import FirestoreService

logger = logging.getLogger(__name__)
firestore_service = FirestoreService()

async def process_visualization_data(
    visualization_id: str,
    config: Dict,
    user_id: str
) -> None:
    """バックグラウンドで可視化データを処理"""
    try:
        # データソースからデータを取得
        data_source_conditions = []
        if config.get('filters'):
            data_source_conditions = [
                {'field': f['field'], 'operator': f['operator'], 'value': f['value']}
                for f in config['filters']
            ]

        source_data = await firestore_service.fetch_documents(
            collection_name=config['data_source'],
            conditions=data_source_conditions
        )

        # データ処理と更新
        processed_data = {
            'id': visualization_id,
            'data': {
                'source': source_data,
                'processed': {},  # 実際のデータ処理ロジックを実装
                'last_updated': datetime.now()
            },
            'status': 'completed'
        }

        await firestore_service.save_results(
            results=[processed_data],
            collection_name='visualizations'
        )

    except Exception as e:
        logger.error(f"Error processing visualization data: {str(e)}")
        error_data = {
            'id': visualization_id,
            'status': 'error',
            'error': str(e),
            'last_updated': datetime.now()
        }
        await firestore_service.save_results(
            results=[error_data],
            collection_name='visualizations'
        )