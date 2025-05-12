"""
データベース設定の中央集権化モジュール
Startup Wellness Analyze プロジェクト

このモジュールは、すべてのデータベース接続と操作に関する設定を一元管理します。
"""

import os
from enum import Enum
from dataclasses import dataclass, field
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
    VAS_DATA = "vas_data"  # 追加: VASデータ
    BUSINESS_PERFORMANCE = "business_performance"  # 追加: 業績データ

# フォームタイプの列挙型
class FormType(Enum):
    INITIAL = "initial_consultation"
    LATEST = "latest_consultation"
    TREATMENT = "treatment_effect"
    VAS_HEALTH = "vas_health_performance"  # VAS健康・パフォーマンスフォーム

@dataclass
class GoogleFormsConfig:
    """Google Formsの設定クラス"""
    form_id: str
    sheet_id: str
    active: bool = True
    sync_frequency: int = 3600  # デフォルト1時間ごと（秒単位）
    field_mappings: Dict[str, str] = field(default_factory=dict)

# データベースタイプとデータカテゴリのマッピング
DATABASE_CATEGORY_MAPPING = {
    DataCategory.USER_PROFILE: DatabaseType.POSTGRESQL,
    DataCategory.ACTIVITY_LOG: DatabaseType.FIRESTORE,
    DataCategory.HEALTH_METRICS: DatabaseType.POSTGRESQL,
    DataCategory.SOCIAL_GRAPH: DatabaseType.NEO4J,
    DataCategory.RECOMMENDATIONS: DatabaseType.NEO4J,
    DataCategory.SETTINGS: DatabaseType.FIRESTORE,
    DataCategory.ANALYTICS: DatabaseType.POSTGRESQL,
    DataCategory.VAS_DATA: DatabaseType.POSTGRESQL,  # 追加: VASデータはPostgreSQLで管理
    DataCategory.BUSINESS_PERFORMANCE: DatabaseType.POSTGRESQL,  # 追加: 業績データはPostgreSQLで管理
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
    additional_params: Dict[str, Any] = field(default_factory=dict)

def get_env_or_default(key: str, default: Any) -> Any:
    """環境変数の値を取得し、存在しない場合はデフォルト値を返す"""
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

# Google Forms設定
GOOGLE_FORMS_CONFIGS = {
    FormType.INITIAL.value: GoogleFormsConfig(
        form_id=get_env_or_default("INITIAL_CONSULTATION_FORM_ID", ""),
        sheet_id=get_env_or_default("INITIAL_CONSULTATION_SHEET_ID", "")
    ),
    FormType.LATEST.value: GoogleFormsConfig(
        form_id=get_env_or_default("LATEST_CONSULTATION_FORM_ID", ""),
        sheet_id=get_env_or_default("LATEST_CONSULTATION_SHEET_ID", "")
    ),
    FormType.TREATMENT.value: GoogleFormsConfig(
        form_id=get_env_or_default("TREATMENT_EFFECT_FORM_ID", ""),
        sheet_id=get_env_or_default("TREATMENT_EFFECT_SHEET_ID", "")
    ),
    FormType.VAS_HEALTH.value: GoogleFormsConfig(
        form_id=get_env_or_default("VAS_HEALTH_PERFORMANCE_FORM_ID", ""),
        sheet_id=get_env_or_default("VAS_HEALTH_PERFORMANCE_SHEET_ID", "")
    )
}

# データベースタイプに基づいて設定を取得する関数
def get_db_config(db_type: DatabaseType) -> DatabaseConfig:
    """データベースタイプに基づいて設定を取得する"""
    configs = {
        DatabaseType.FIRESTORE: FIRESTORE_CONFIG,
        DatabaseType.POSTGRESQL: POSTGRESQL_CONFIG,
        DatabaseType.NEO4J: NEO4J_CONFIG,
    }
    return configs.get(db_type)

# データカテゴリからデータベースタイプを取得する関数
def get_db_type_for_category(category: DataCategory) -> DatabaseType:
    """データカテゴリに対応するデータベースタイプを取得する"""
    return DATABASE_CATEGORY_MAPPING.get(category)

# 全てのデータベース設定情報を取得する関数
def get_all_db_configs() -> Dict[DatabaseType, DatabaseConfig]:
    """全てのデータベース設定情報を返す"""
    return {
        DatabaseType.FIRESTORE: FIRESTORE_CONFIG,
        DatabaseType.POSTGRESQL: POSTGRESQL_CONFIG,
        DatabaseType.NEO4J: NEO4J_CONFIG,
    }

# フォームタイプに基づいてGoogle Forms設定を取得する関数
def get_google_forms_config(form_type: str) -> Optional[GoogleFormsConfig]:
    """フォームタイプに基づいてGoogle Forms設定を取得する"""
    return GOOGLE_FORMS_CONFIGS.get(form_type)

# グローバル設定
SYNC_LOG_ENABLED = get_env_or_default("SYNC_LOG_ENABLED", "true").lower() == "true"
DEFAULT_SYNC_INTERVAL = int(get_env_or_default("DEFAULT_SYNC_INTERVAL", 3600))  # デフォルト1時間
