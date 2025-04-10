# -*- coding: utf-8 -*-

"""
認証に関連するコンプライアンス、監査、ポリシー管理機能を提供するモジュール
"""

import logging
import json
import time
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import uuid

import redis
from fastapi import Request, HTTPException, status
from pydantic import BaseModel, Field, validator

from .common_logger import get_logger
from .patterns import Singleton, LazyImport

# 循環インポートを回避するための遅延インポート
AuthManager = LazyImport('core.auth_manager', 'AuthManager')

# ロギングの設定
logger = get_logger(__name__)

# Redisの設定
REDIS_HOST = os.environ.get("REDIS_HOST", "startup-wellness-redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_COMPLIANCE_DB", 2))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)

class ComplianceEventType(str, Enum):
    """コンプライアンスイベントタイプの列挙型"""
    LOGIN = "login"
    LOGOUT = "logout"
    REGISTRATION = "registration"
    PASSWORD_RESET = "password_reset"
    PASSWORD_CHANGE = "password_change"
    PROFILE_UPDATE = "profile_update"
    ROLE_CHANGE = "role_change"
    MFA_ENABLE = "mfa_enable"
    MFA_DISABLE = "mfa_disable"
    TOKEN_REFRESH = "token_refresh"
    ADMIN_ACTION = "admin_action"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_HIT = "rate_limit_hit"
    POLICY_VIOLATION = "policy_violation"
    DATA_ACCESS = "data_access"

class ComplianceEvent(BaseModel):
    """コンプライアンスイベントモデル"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: ComplianceEventType
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    success: bool = True
    details: Optional[Dict[str, Any]] = None
    affected_user_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PasswordPolicy(BaseModel):
    """パスワードポリシーモデル"""
    min_length: int = 8
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    max_age_days: int = 90
    prevent_reuse_count: int = 5

class SessionPolicy(BaseModel):
    """セッションポリシーモデル"""
    max_sessions_per_user: int = 5
    session_timeout_minutes: int = 60
    idle_timeout_minutes: int = 15
    enforce_ip_binding: bool = False
    enforce_device_binding: bool = False

class RetentionPolicy(BaseModel):
    """データ保持ポリシーモデル"""
    audit_log_retention_days: int = 365
    user_data_retention_days: int = 730
    inactive_account_deletion_days: int = 730

@Singleton
class ComplianceManager:
    """認証関連のコンプライアンス管理を行うクラス"""
    def __init__(self):
        """初期化メソッド"""
        self.redis_client = None

        # ポリシー設定
        self.password_policy = PasswordPolicy()
        self.session_policy = SessionPolicy()
        self.retention_policy = RetentionPolicy()

        # 特別なパスワードポリシーを持つロールのマッピング
        self.role_password_policies = {
            # 'admin': PasswordPolicy(min_length=12, max_age_days=60, prevent_reuse_count=10)
        }

        # コンプライアンスポリシー
        self.gdpr_enabled = True
        self.hipaa_enabled = False
        self.pci_dss_enabled = False

        # クライアントIPの地理情報キャッシュ
        self.geo_cache = {}

        # Redisクライアントを初期化
        self._init_redis()
        logger.info("ComplianceManagerが初期化されました")

    def _init_redis(self):
        """Redisクライアントの初期化"""
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True
            )
            logger.info("コンプライアンスマネージャーのRedisクライアントを初期化しました")
        except Exception as e:
            logger.warning(f"Redisクライアントの初期化に失敗しました: {str(e)}")
            self.redis_client = None

    async def log_event(self, event: ComplianceEvent) -> bool:
        """コンプライアンスイベントをログに記録"""
        try:
            event_dict = event.dict()
            event_json = json.dumps(event_dict, default=str)

            # Redisにログを保存
            if self.redis_client:
                # イベントをタイムラインに追加
                self.redis_client.zadd(
                    "compliance:timeline",
                    {event_json: int(time.time())}
                )

                # ユーザー別のイベントセットに追加
                if event.user_id:
                    self.redis_client.zadd(
                        f"compliance:user:{event.user_id}",
                        {event_json: int(time.time())}
                    )

                # イベントタイプ別のセットに追加
                self.redis_client.zadd(
                    f"compliance:event_type:{event.event_type.value}",
                    {event_json: int(time.time())}
                )

                # データ保持期間に基づいて有効期限を設定
                retention_seconds = self.retention_policy.audit_log_retention_days * 86400
                self.redis_client.expire(f"compliance:user:{event.user_id}", retention_seconds)
                self.redis_client.expire(f"compliance:event_type:{event.event_type.value}", retention_seconds)
                self.redis_client.expire("compliance:timeline", retention_seconds)

            # ログにも記録
            logger.info(f"コンプライアンスイベント: {event_json}")

            return True
        except Exception as e:
            logger.error(f"コンプライアンスイベントの記録に失敗: {str(e)}")
            return False

    async def get_user_events(
        self,
        user_id: str,
        event_type: Optional[ComplianceEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ComplianceEvent]:
        """ユーザーのイベント履歴を取得"""
        try:
            if not self.redis_client:
                return []

            # 時間範囲の設定
            min_score = "-inf"
            max_score = "+inf"

            if start_time:
                min_score = int(start_time.timestamp())
            if end_time:
                max_score = int(end_time.timestamp())

            # イベント取得のキー
            key = f"compliance:user:{user_id}"
            if event_type:
                key = f"compliance:event_type:{event_type.value}"

            # Redisからイベントを取得
            events_json = self.redis_client.zrangebyscore(
                key, min_score, max_score, 0, limit
            )

            # JSONからイベントオブジェクトに変換
            events = []
            for event_json in events_json:
                event_dict = json.loads(event_json)
                if user_id == event_dict.get("user_id") or user_id == event_dict.get("affected_user_id"):
                    events.append(ComplianceEvent(**event_dict))

            return events
        except Exception as e:
            logger.error(f"ユーザーイベント取得エラー: {str(e)}")
            return []

    async def get_all_events(
        self,
        event_type: Optional[ComplianceEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ComplianceEvent]:
        """すべてのイベント履歴を取得"""
        try:
            if not self.redis_client:
                return []

            # 時間範囲の設定
            min_score = "-inf"
            max_score = "+inf"

            if start_time:
                min_score = int(start_time.timestamp())
            if end_time:
                max_score = int(end_time.timestamp())

            # イベント取得のキー
            key = "compliance:timeline"
            if event_type:
                key = f"compliance:event_type:{event_type.value}"

            # Redisからイベントを取得
            events_json = self.redis_client.zrangebyscore(
                key, min_score, max_score, 0, limit
            )

            # JSONからイベントオブジェクトに変換
            events = []
            for event_json in events_json:
                event_dict = json.loads(event_json)
                events.append(ComplianceEvent(**event_dict))

            return events
        except Exception as e:
            logger.error(f"イベント取得エラー: {str(e)}")
            return []

    async def check_password_policy(self, password: str, role: str = "user") -> Tuple[bool, str]:
        """パスワードがポリシーに準拠しているか確認"""
        # ロール固有のパスワードポリシーを取得
        policy = self.role_password_policies.get(role.lower(), self.password_policy)

        # 長さのチェック
        if len(password) < policy.min_length:
            return False, f"パスワードは{policy.min_length}文字以上である必要があります"

        # 大文字の要件
        if policy.require_uppercase and not any(c.isupper() for c in password):
            return False, "パスワードには大文字を含める必要があります"

        # 小文字の要件
        if policy.require_lowercase and not any(c.islower() for c in password):
            return False, "パスワードには小文字を含める必要があります"

        # 数字の要件
        if policy.require_numbers and not any(c.isdigit() for c in password):
            return False, "パスワードには数字を含める必要があります"

        # 特殊文字の要件
        if policy.require_special_chars and not any(c in '!@#$%^&*()_-+={}[]|\:;"<>,.?/' for c in password):
            return False, "パスワードには特殊文字を含める必要があります"

        return True, "パスワードはポリシーに準拠しています"

    async def check_password_history(self, user_id: str, password: str) -> bool:
        """パスワード履歴をチェックして再利用を防止"""
        if not self.redis_client:
            return True

        role = "user"  # デフォルトのロール
        policy = self.role_password_policies.get(role, self.password_policy)

        # パスワード履歴の取得
        password_history_key = f"compliance:password_history:{user_id}"
        password_history = self.redis_client.lrange(password_history_key, 0, policy.prevent_reuse_count - 1)

        # パスワード履歴に存在するか確認
        for saved_hash in password_history:
            # ここでは単純なハッシュ比較ですが、実際にはArgon2などでハッシュ比較を行う必要があります
            if saved_hash == password:
                return False

        return True

    async def add_password_to_history(self, user_id: str, password_hash: str) -> bool:
        """パスワード履歴に新しいパスワードを追加"""
        if not self.redis_client:
            return False

        try:
            password_history_key = f"compliance:password_history:{user_id}"

            # パスワード履歴に追加
            self.redis_client.lpush(password_history_key, password_hash)

            # 保持するパスワード履歴の数を制限
            role = "user"  # デフォルトのロール
            policy = self.role_password_policies.get(role, self.password_policy)
            self.redis_client.ltrim(password_history_key, 0, policy.prevent_reuse_count - 1)

            # データ保持期間に基づいて有効期限を設定
            retention_seconds = self.retention_policy.user_data_retention_days * 86400
            self.redis_client.expire(password_history_key, retention_seconds)

            return True
        except Exception as e:
            logger.error(f"パスワード履歴の更新エラー: {str(e)}")
            return False

    async def check_password_expiration(self, user_id: str, last_password_change: datetime) -> bool:
        """パスワードの有効期限をチェック"""
        role = "user"  # デフォルトのロール
        policy = self.role_password_policies.get(role, self.password_policy)

        # パスワード変更からの経過日数を計算
        days_since_change = (datetime.now() - last_password_change).days

        # 有効期限を超えているかチェック
        return days_since_change <= policy.max_age_days

    async def enforce_session_policy(self, user_id: str, request: Request) -> Tuple[bool, str]:
        """セッションポリシーの強制適用"""
        if not self.redis_client:
            return True, ""

        try:
            # アクティブセッション数のチェック
            active_sessions_key = f"compliance:active_sessions:{user_id}"
            active_sessions = self.redis_client.scard(active_sessions_key)

            if active_sessions >= self.session_policy.max_sessions_per_user:
                return False, f"アクティブセッションの最大数（{self.session_policy.max_sessions_per_user}）に達しています"

            # IPバインディングのチェック
            if self.session_policy.enforce_ip_binding:
                client_ip = request.client.host
                bound_ip_key = f"compliance:ip_binding:{user_id}"
                bound_ip = self.redis_client.get(bound_ip_key)

                if bound_ip and bound_ip != client_ip:
                    return False, "異なるIPアドレスからのログインは制限されています"

            # デバイスバインディングのチェック
            if self.session_policy.enforce_device_binding:
                user_agent = request.headers.get("user-agent", "")
                bound_device_key = f"compliance:device_binding:{user_id}"
                bound_device = self.redis_client.get(bound_device_key)

                if bound_device and bound_device != user_agent:
                    return False, "異なるデバイスからのログインは制限されています"

            return True, ""
        except Exception as e:
            logger.error(f"セッションポリシー適用エラー: {str(e)}")
            return True, ""

    async def record_session_start(self, user_id: str, session_id: str, request: Request) -> bool:
        """セッション開始の記録"""
        if not self.redis_client:
            return False

        try:
            # セッション情報
            session_info = {
                "session_id": session_id,
                "ip_address": request.client.host,
                "user_agent": request.headers.get("user-agent", ""),
                "start_time": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            }

            # セッション情報の保存
            session_key = f"compliance:session:{session_id}"
            self.redis_client.hmset(session_key, session_info)

            # アクティブセッションの集合に追加
            active_sessions_key = f"compliance:active_sessions:{user_id}"
            self.redis_client.sadd(active_sessions_key, session_id)

            # IPバインディングの設定
            if self.session_policy.enforce_ip_binding:
                bound_ip_key = f"compliance:ip_binding:{user_id}"
                self.redis_client.set(bound_ip_key, request.client.host)

            # デバイスバインディングの設定
            if self.session_policy.enforce_device_binding:
                bound_device_key = f"compliance:device_binding:{user_id}"
                self.redis_client.set(bound_device_key, request.headers.get("user-agent", ""))

            # セッションのTTLを設定
            ttl = self.session_policy.session_timeout_minutes * 60
            self.redis_client.expire(session_key, ttl)
            self.redis_client.expire(active_sessions_key, ttl)

            return True
        except Exception as e:
            logger.error(f"セッション開始記録エラー: {str(e)}")
            return False

    async def update_session_activity(self, session_id: str) -> bool:
        """セッションの最終アクティビティ時間を更新"""
        if not self.redis_client:
            return False

        try:
            # セッションキー
            session_key = f"compliance:session:{session_id}"

            # セッションが存在するかチェック
            if not self.redis_client.exists(session_key):
                return False

            # 最終アクティビティ時間を更新
            self.redis_client.hset(session_key, "last_activity", datetime.now().isoformat())

            # TTLをリセット
            ttl = self.session_policy.session_timeout_minutes * 60
            self.redis_client.expire(session_key, ttl)

            return True
        except Exception as e:
            logger.error(f"セッションアクティビティ更新エラー: {str(e)}")
            return False

    async def record_session_end(self, user_id: str, session_id: str) -> bool:
        """セッション終了の記録"""
        if not self.redis_client:
            return False

        try:
            # セッション情報キー
            session_key = f"compliance:session:{session_id}"

            # セッション情報を取得
            session_info = self.redis_client.hgetall(session_key)

            if session_info:
                # セッション終了時間を記録
                session_info["end_time"] = datetime.now().isoformat()

                # 終了したセッションをログに記録
                ended_session_key = f"compliance:ended_session:{session_id}"
                self.redis_client.hmset(ended_session_key, session_info)

                # 終了したセッションの有効期限を設定
                retention_seconds = self.retention_policy.audit_log_retention_days * 86400
                self.redis_client.expire(ended_session_key, retention_seconds)

            # アクティブセッションの集合から削除
            active_sessions_key = f"compliance:active_sessions:{user_id}"
            self.redis_client.srem(active_sessions_key, session_id)

            # セッションデータを削除
            self.redis_client.delete(session_key)

            return True
        except Exception as e:
            logger.error(f"セッション終了記録エラー: {str(e)}")
            return False

    async def check_inactive_accounts(self) -> List[Dict]:
        """非アクティブアカウントを確認"""
        if not self.redis_client:
            return []

        try:
            # 非アクティブアカウントのチェック日数
            days = self.retention_policy.inactive_account_deletion_days
            cutoff_date = datetime.now() - timedelta(days=days)

            # TODO: 実際のユーザーストアと連携して非アクティブアカウントを検索
            inactive_accounts = []

            return inactive_accounts
        except Exception as e:
            logger.error(f"非アクティブアカウントチェックエラー: {str(e)}")
            return []

    async def apply_data_retention_policy(self) -> bool:
        """データ保持ポリシーに基づいてデータをクリーンアップ"""
        if not self.redis_client:
            return False

        try:
            # 監査ログの期限切れ日数
            audit_days = self.retention_policy.audit_log_retention_days
            audit_cutoff = int((datetime.now() - timedelta(days=audit_days)).timestamp())

            # イベントタイムラインをクリーンアップ
            self.redis_client.zremrangebyscore("compliance:timeline", "-inf", audit_cutoff)

            # イベントタイプ別セットをクリーンアップ
            for event_type in ComplianceEventType:
                self.redis_client.zremrangebyscore(
                    f"compliance:event_type:{event_type.value}", "-inf", audit_cutoff
                )

            # ユーザーセッションデータの期限切れ日数
            user_data_days = self.retention_policy.user_data_retention_days

            # TODO: 他のユーザーデータのクリーンアップ処理

            return True
        except Exception as e:
            logger.error(f"データ保持ポリシー適用エラー: {str(e)}")
            return False

    async def create_gdpr_data_export(self, user_id: str) -> Dict:
        """GDPRに準拠したユーザーデータのエクスポート"""
        if not self.gdpr_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="GDPRデータエクスポートは有効になっていません"
            )

        try:
            # ユーザーデータの収集
            user_data = {}

            # セッション履歴
            if self.redis_client:
                # ユーザーのイベント履歴
                events = await self.get_user_events(user_id, limit=1000)
                user_data["events"] = [event.dict() for event in events]

                # セッション履歴
                active_sessions_key = f"compliance:active_sessions:{user_id}"
                active_sessions = self.redis_client.smembers(active_sessions_key)
                session_data = []

                for session_id in active_sessions:
                    session_key = f"compliance:session:{session_id}"
                    session_info = self.redis_client.hgetall(session_key)
                    if session_info:
                        session_data.append(session_info)

                user_data["active_sessions"] = session_data

            # TODO: 他のユーザーデータソースからデータを収集

            return user_data
        except Exception as e:
            logger.error(f"GDPRデータエクスポートエラー: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="データエクスポートの作成中にエラーが発生しました"
            )

    async def execute_data_deletion_request(self, user_id: str) -> bool:
        """ユーザーデータ削除リクエストの実行"""
        if not self.gdpr_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="GDPRデータ削除は有効になっていません"
            )

        try:
            if self.redis_client:
                # ユーザーのイベント削除
                user_events_key = f"compliance:user:{user_id}"
                self.redis_client.delete(user_events_key)

                # アクティブセッションの削除
                active_sessions_key = f"compliance:active_sessions:{user_id}"
                active_sessions = self.redis_client.smembers(active_sessions_key)

                for session_id in active_sessions:
                    session_key = f"compliance:session:{session_id}"
                    self.redis_client.delete(session_key)

                self.redis_client.delete(active_sessions_key)

                # パスワード履歴の削除
                password_history_key = f"compliance:password_history:{user_id}"
                self.redis_client.delete(password_history_key)

                # IPバインディングとデバイスバインディングの削除
                bound_ip_key = f"compliance:ip_binding:{user_id}"
                bound_device_key = f"compliance:device_binding:{user_id}"
                self.redis_client.delete(bound_ip_key, bound_device_key)

            # TODO: 他のデータストアからのユーザーデータ削除

            # ユーザー削除イベントのログ記録
            await self.log_event(ComplianceEvent(
                event_type=ComplianceEventType.ADMIN_ACTION,
                user_id="system",
                action="user_data_deletion",
                affected_user_id=user_id,
                details={"reason": "GDPR削除リクエスト"}
            ))

            return True
        except Exception as e:
            logger.error(f"データ削除リクエスト実行エラー: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="データ削除リクエストの実行中にエラーが発生しました"
            )

# コンプライアンスマネージャーのシングルトンインスタンスを取得する関数
def get_compliance_manager() -> ComplianceManager:
    """コンプライアンスマネージャーのシングルトンインスタンスを取得"""
    return ComplianceManager()