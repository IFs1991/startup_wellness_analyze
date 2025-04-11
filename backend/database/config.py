"""
データベース設定の中央集権化モジュール
Startup Wellness Analyze プロジェクト

このモジュールは、すべてのデータベース接続と操作に関する設定を一元管理します。
"""

import os
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

# データベースタイプの列挙型
class DatabaseType(Enum):
    FIRESTORE = "firestore"
    POSTGRESQL = "postgresql"
    NEO4J = "neo4j"

# データカテゴリの列挙型
class DataCategory(Enum):
    USER_PROFILE = "user_profile"
    ACTIVITY_LOG = "activity_log"
    HEALTH_METRICS = "health_metrics"
    SOCIAL_GRAPH = "social_graph"
    RECOMMENDATIONS = "recommendations"
    SETTINGS = "settings"
    ANALYTICS = "analytics"

# データベースタイプとデータカテゴリのマッピング
DATABASE_CATEGORY_MAPPING = {
    DataCategory.USER_PROFILE: DatabaseType.POSTGRESQL,
    DataCategory.ACTIVITY_LOG: DatabaseType.FIRESTORE,
    DataCategory.HEALTH_METRICS: DatabaseType.POSTGRESQL,
    DataCategory.SOCIAL_GRAPH: DatabaseType.NEO4J,
    DataCategory.RECOMMENDATIONS: DatabaseType.NEO4J,
    DataCategory.SETTINGS: DatabaseType.FIRESTORE,
    DataCategory.ANALYTICS: DatabaseType.POSTGRESQL,
}

@dataclass
class DatabaseConfig:
    """データベース接続設定の基本クラス"""
    host: str
    port: int
    database_name: str
    username: Optional[str] = None
    password: Optional[str] = None
    connection_timeout: int = 30
    pool_size: int = 5
    ssl_enabled: bool = True
    additional_params: Dict[str, Any] = None

# 環境設定から値を取得するヘルパー関数
def get_env_or_default(key: str, default: Any) -> Any:
    """環境変数から値を取得し、ない場合はデフォルト値を返す"""
    return os.environ.get(key, default)

# 各データベースの設定
FIRESTORE_CONFIG = DatabaseConfig(
    host=get_env_or_default("FIRESTORE_HOST", "firestore.googleapis.com"),
    port=int(get_env_or_default("FIRESTORE_PORT", 443)),
    database_name=get_env_or_default("FIRESTORE_DB", "startup-wellness-analyze"),
    connection_timeout=int(get_env_or_default("FIRESTORE_TIMEOUT", 30)),
    additional_params={
        "project_id": get_env_or_default("FIRESTORE_PROJECT_ID", "startup-wellness-analyze"),
        "credentials_path": get_env_or_default("FIRESTORE_CREDENTIALS", "path/to/credentials.json"),
    }
)

POSTGRESQL_CONFIG = DatabaseConfig(
    host=get_env_or_default("POSTGRES_HOST", "localhost"),
    port=int(get_env_or_default("POSTGRES_PORT", 5432)),
    database_name=get_env_or_default("POSTGRES_DB", "wellness_db"),
    username=get_env_or_default("POSTGRES_USER", "wellness_user"),
    password=get_env_or_default("POSTGRES_PASSWORD", "wellness_password"),
    pool_size=int(get_env_or_default("POSTGRES_POOL_SIZE", 10)),
    connection_timeout=int(get_env_or_default("POSTGRES_TIMEOUT", 30)),
    ssl_enabled=get_env_or_default("POSTGRES_SSL", "true").lower() == "true"
)

NEO4J_CONFIG = DatabaseConfig(
    host=get_env_or_default("NEO4J_HOST", "localhost"),
    port=int(get_env_or_default("NEO4J_PORT", 7687)),
    database_name=get_env_or_default("NEO4J_DB", "wellness_graph"),
    username=get_env_or_default("NEO4J_USER", "neo4j"),
    password=get_env_or_default("NEO4J_PASSWORD", "neo4j_password"),
    connection_timeout=int(get_env_or_default("NEO4J_TIMEOUT", 30)),
    additional_params={
        "encryption": get_env_or_default("NEO4J_ENCRYPTION", "true").lower() == "true",
        "trust": get_env_or_default("NEO4J_TRUST", "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"),
    }
)

# データベースタイプに基づいて設定を取得する関数
def get_db_config(db_type: DatabaseType) -> DatabaseConfig:
    """データベースタイプに基づいて設定を取得する"""
    configs = {
        DatabaseType.FIRESTORE: FIRESTORE_CONFIG,
        DatabaseType.POSTGRESQL: POSTGRESQL_CONFIG,
        DatabaseType.NEO4J: NEO4J_CONFIG,
    }
    return configs.get(db_type)

# データカテゴリに基づいて適切なデータベースタイプを取得する関数
def get_db_type_for_category(category: DataCategory) -> DatabaseType:
    """データカテゴリに基づいて適切なデータベースタイプを取得する"""
    return DATABASE_CATEGORY_MAPPING.get(category, DatabaseType.POSTGRESQL)

# 全てのデータベース設定情報を取得する関数
def get_all_db_configs() -> Dict[DatabaseType, DatabaseConfig]:
    """全てのデータベース設定情報を返す"""
    return {
        DatabaseType.FIRESTORE: FIRESTORE_CONFIG,
        DatabaseType.POSTGRESQL: POSTGRESQL_CONFIG,
        DatabaseType.NEO4J: NEO4J_CONFIG,
    }
