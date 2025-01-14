"""
ログ設定モジュール

このモジュールは、アプリケーション全体のログ設定を管理します。
Cloud Logging、Elasticsearchとの連携、各モジュール別のログ設定をサポートします。
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Any, Optional, cast

try:
    from google.cloud import logging as cloud_logging
    from google.cloud.logging_v2.client import Client as CloudLoggingClient
    HAS_CLOUD_LOGGING = True
except ImportError:
    cloud_logging = None
    CloudLoggingClient = Any
    HAS_CLOUD_LOGGING = False

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.client import Elasticsearch as ElasticsearchClient
    HAS_ELASTICSEARCH = True
except ImportError:
    Elasticsearch = None
    ElasticsearchClient = Any
    HAS_ELASTICSEARCH = False

class LogConfig:
    def __init__(self) -> None:
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.log_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        self.es_host: Optional[str] = os.getenv("ELASTICSEARCH_HOST")
        self.es_port: int = int(os.getenv("ELASTICSEARCH_PORT", "9200"))

    def setup_basic_logging(self):
        """基本的なログ設定をセットアップ"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format=self.log_format,
            handlers=[
                logging.StreamHandler(),  # コンソール出力
                logging.handlers.RotatingFileHandler(  # ファイル出力
                    os.path.join(self.log_dir, "app.log"),
                    maxBytes=10485760,  # 10MB
                    backupCount=5
                )
            ]
        )

    def setup_module_logger(self, module_name: str) -> logging.Logger:
        """
        モジュール固有のロガーをセットアップ

        Args:
            module_name: モジュール名

        Returns:
            logging.Logger: 設定されたロガーインスタンス
        """
        logger = logging.getLogger(module_name)
        logger.setLevel(getattr(logging, self.log_level))

        # モジュール固有のファイルハンドラを追加
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(self.log_dir, f"{module_name}.log"),
            maxBytes=5242880,  # 5MB
            backupCount=3
        )
        file_handler.setFormatter(logging.Formatter(self.log_format))
        logger.addHandler(file_handler)

        return logger

    def setup_cloud_logging(self, project_id: str):
        """
        Google Cloud Loggingの設定

        Args:
            project_id: Google Cloudプロジェクトのプロジェクトid
        """
        if cloud_logging is None:
            logging.warning("Google Cloud Loggingパッケージがインストールされていません")
            return

        client = cloud_logging.Client(project=project_id)
        client.setup_logging()

    def setup_elasticsearch_logging(self):
        """Elasticsearchへのログ転送設定"""
        if not self.es_host or Elasticsearch is None:
            return

        class ElasticsearchHandler(logging.Handler):
            def __init__(self, es_host: str, es_port: int):
                super().__init__()
                if Elasticsearch is not None:
                    self.es = Elasticsearch([f"http://{es_host}:{es_port}"])
                else:
                    self.es = None

            def emit(self, record):
                if self.es is None:
                    return
                try:
                    log_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'level': record.levelname,
                        'message': record.getMessage(),
                        'module': record.module,
                        'funcName': record.funcName,
                        'lineNo': record.lineno
                    }
                    self.es.index(
                        index=f"logs-{datetime.now():%Y.%m.%d}",
                        document=log_entry
                    )
                except Exception as e:
                    print(f"Failed to send log to Elasticsearch: {e}")

        # Elasticsearchハンドラを追加
        es_handler = ElasticsearchHandler(self.es_host, self.es_port)
        logging.getLogger().addHandler(es_handler)

def get_logger(
    module_name: str,
    setup_cloud_logging: bool = False,
    project_id: Optional[str] = None
) -> logging.Logger:
    """
    指定されたモジュール用のロガーを取得

    Args:
        module_name: モジュール名
        setup_cloud_logging: Cloud Loggingを使用するかどうか
        project_id: Google Cloudプロジェクトのプロジェクトid

    Returns:
        logging.Logger: 設定されたロガーインスタンス
    """
    config = LogConfig()
    config.setup_basic_logging()

    if setup_cloud_logging and project_id:
        config.setup_cloud_logging(project_id)

    if os.getenv("ELASTICSEARCH_HOST"):
        config.setup_elasticsearch_logging()

    return config.setup_module_logger(module_name)