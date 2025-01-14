# services/bigquery/__init__.py
from typing import Optional
import os
import logging
from .client import BigQueryService

# グローバル変数名の修正（アンダースコアの表記を修正）
_default_service = None
logger = logging.getLogger(__name__)

def get_bigquery_service(project_id: Optional[str] = None) -> BigQueryService:
    """
    BigQueryServiceのシングルトンインスタンスを取得

    Args:
        project_id (Optional[str]): プロジェクトID（省略時は環境変数から取得）

    Returns:
        BigQueryService: サービスインスタンス

    Raises:
        ValueError: 必要な環境変数が設定されていない場合
        Exception: サービスの初期化に失敗した場合
    """
    global _default_service

    try:
        if _default_service is None:
            logger.debug("Initializing new BigQuery service instance")

            # 環境変数の取得と検証
            project_id = project_id or os.getenv('GCP_PROJECT_ID')
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

            # バリデーション
            if not project_id:
                raise ValueError("GCP_PROJECT_ID is not set in environment variables")
            if not credentials_path:
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set in environment variables")
            if not os.path.exists(credentials_path):
                raise ValueError(f"Credentials file not found at: {credentials_path}")

            # サービスの初期化
            _default_service = BigQueryService(
                project_id=project_id,
                credentials_path=credentials_path
            )
            logger.info(f"BigQuery service initialized with project ID: {project_id}")

        return _default_service

    except Exception as e:
        logger.error(f"Failed to initialize BigQuery service: {str(e)}")
        raise

def reset_service() -> None:
    """
    サービスインスタンスをリセット
    主にテスト時やインスタンスの再初期化が必要な場合に使用
    """
    global _default_service
    if _default_service is not None:
        logger.debug("Resetting BigQuery service instance")
        _default_service = None