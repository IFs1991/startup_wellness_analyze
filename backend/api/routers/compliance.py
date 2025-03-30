# -*- coding: utf-8 -*-

"""
コンプライアンス関連のエンドポイントを提供するルーター
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request, Query
from pydantic import BaseModel

from core.auth_manager import get_current_admin_user
from core.compliance_manager import get_compliance_manager, ComplianceEventType, ComplianceEvent
from core.rate_limiter import get_rate_limiter

router = APIRouter(prefix="/api/compliance", tags=["コンプライアンス"])

# ロガーの設定
logger = logging.getLogger(__name__)

# シングルトンのインスタンスを取得
compliance_manager = get_compliance_manager()
rate_limiter = get_rate_limiter()

# モデル定義
class EventResponse(BaseModel):
    """イベントレスポンスモデル"""
    id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    ip_address: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    success: bool
    details: Optional[Dict[str, Any]] = None
    affected_user_id: Optional[str] = None

class PasswordPolicyModel(BaseModel):
    """パスワードポリシーモデル"""
    min_length: int
    require_uppercase: bool
    require_lowercase: bool
    require_numbers: bool
    require_special_chars: bool
    max_age_days: int
    prevent_reuse_count: int

class SessionPolicyModel(BaseModel):
    """セッションポリシーモデル"""
    max_sessions_per_user: int
    session_timeout_minutes: int
    idle_timeout_minutes: int
    enforce_ip_binding: bool
    enforce_device_binding: bool

class RetentionPolicyModel(BaseModel):
    """データ保持ポリシーモデル"""
    audit_log_retention_days: int
    user_data_retention_days: int
    inactive_account_deletion_days: int

class ComplianceConfigModel(BaseModel):
    """コンプライアンス設定モデル"""
    password_policy: PasswordPolicyModel
    session_policy: SessionPolicyModel
    retention_policy: RetentionPolicyModel
    gdpr_enabled: bool
    hipaa_enabled: bool
    pci_dss_enabled: bool

class GDPRRequestModel(BaseModel):
    """GDPRリクエストモデル"""
    user_id: str
    request_type: str  # "export" または "delete"

@router.get("/events", response_model=List[EventResponse])
async def get_events(
    event_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(100, le=1000),
    current_user: dict = Depends(get_current_admin_user)
):
    """コンプライアンスイベント履歴を取得（管理者専用）"""
    try:
        event_type_enum = None
        if event_type:
            try:
                event_type_enum = ComplianceEventType(event_type)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"無効なイベントタイプ: {event_type}"
                )

        events = await compliance_manager.get_all_events(
            event_type=event_type_enum,
            start_time=start_date,
            end_time=end_date,
            limit=limit
        )

        # イベントをレスポンスモデルに変換
        return [
            EventResponse(
                id=event.id,
                timestamp=event.timestamp,
                event_type=event.event_type.value,
                user_id=event.user_id,
                user_email=event.user_email,
                ip_address=event.ip_address,
                resource=event.resource,
                action=event.action,
                success=event.success,
                details=event.details,
                affected_user_id=event.affected_user_id
            )
            for event in events
        ]
    except Exception as e:
        logger.error(f"イベント取得エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="イベント履歴の取得中にエラーが発生しました"
        )

@router.get("/user/{user_id}/events", response_model=List[EventResponse])
async def get_user_events(
    user_id: str,
    event_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(100, le=1000),
    current_user: dict = Depends(get_current_admin_user)
):
    """特定のユーザーのイベント履歴を取得（管理者専用）"""
    try:
        event_type_enum = None
        if event_type:
            try:
                event_type_enum = ComplianceEventType(event_type)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"無効なイベントタイプ: {event_type}"
                )

        events = await compliance_manager.get_user_events(
            user_id=user_id,
            event_type=event_type_enum,
            start_time=start_date,
            end_time=end_date,
            limit=limit
        )

        # イベントをレスポンスモデルに変換
        return [
            EventResponse(
                id=event.id,
                timestamp=event.timestamp,
                event_type=event.event_type.value,
                user_id=event.user_id,
                user_email=event.user_email,
                ip_address=event.ip_address,
                resource=event.resource,
                action=event.action,
                success=event.success,
                details=event.details,
                affected_user_id=event.affected_user_id
            )
            for event in events
        ]
    except Exception as e:
        logger.error(f"ユーザーイベント取得エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ユーザーイベント履歴の取得中にエラーが発生しました"
        )

@router.get("/config", response_model=ComplianceConfigModel)
async def get_compliance_config(
    current_user: dict = Depends(get_current_admin_user)
):
    """コンプライアンス設定を取得（管理者専用）"""
    try:
        config = ComplianceConfigModel(
            password_policy=PasswordPolicyModel(
                min_length=compliance_manager.password_policy.min_length,
                require_uppercase=compliance_manager.password_policy.require_uppercase,
                require_lowercase=compliance_manager.password_policy.require_lowercase,
                require_numbers=compliance_manager.password_policy.require_numbers,
                require_special_chars=compliance_manager.password_policy.require_special_chars,
                max_age_days=compliance_manager.password_policy.max_age_days,
                prevent_reuse_count=compliance_manager.password_policy.prevent_reuse_count
            ),
            session_policy=SessionPolicyModel(
                max_sessions_per_user=compliance_manager.session_policy.max_sessions_per_user,
                session_timeout_minutes=compliance_manager.session_policy.session_timeout_minutes,
                idle_timeout_minutes=compliance_manager.session_policy.idle_timeout_minutes,
                enforce_ip_binding=compliance_manager.session_policy.enforce_ip_binding,
                enforce_device_binding=compliance_manager.session_policy.enforce_device_binding
            ),
            retention_policy=RetentionPolicyModel(
                audit_log_retention_days=compliance_manager.retention_policy.audit_log_retention_days,
                user_data_retention_days=compliance_manager.retention_policy.user_data_retention_days,
                inactive_account_deletion_days=compliance_manager.retention_policy.inactive_account_deletion_days
            ),
            gdpr_enabled=compliance_manager.gdpr_enabled,
            hipaa_enabled=compliance_manager.hipaa_enabled,
            pci_dss_enabled=compliance_manager.pci_dss_enabled
        )

        return config
    except Exception as e:
        logger.error(f"コンプライアンス設定取得エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="コンプライアンス設定の取得中にエラーが発生しました"
        )

@router.put("/config/password-policy")
async def update_password_policy(
    policy: PasswordPolicyModel,
    current_user: dict = Depends(get_current_admin_user)
):
    """パスワードポリシーを更新（管理者専用）"""
    try:
        # ポリシーの更新
        compliance_manager.password_policy.min_length = policy.min_length
        compliance_manager.password_policy.require_uppercase = policy.require_uppercase
        compliance_manager.password_policy.require_lowercase = policy.require_lowercase
        compliance_manager.password_policy.require_numbers = policy.require_numbers
        compliance_manager.password_policy.require_special_chars = policy.require_special_chars
        compliance_manager.password_policy.max_age_days = policy.max_age_days
        compliance_manager.password_policy.prevent_reuse_count = policy.prevent_reuse_count

        # コンプライアンスログに記録
        await compliance_manager.log_event(ComplianceEvent(
            event_type=ComplianceEventType.ADMIN_ACTION,
            user_id=current_user["sub"],
            action="update_password_policy",
            success=True,
            details={"policy": policy.dict()}
        ))

        return {"detail": "パスワードポリシーが更新されました"}
    except Exception as e:
        logger.error(f"パスワードポリシー更新エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="パスワードポリシーの更新中にエラーが発生しました"
        )

@router.put("/config/session-policy")
async def update_session_policy(
    policy: SessionPolicyModel,
    current_user: dict = Depends(get_current_admin_user)
):
    """セッションポリシーを更新（管理者専用）"""
    try:
        # ポリシーの更新
        compliance_manager.session_policy.max_sessions_per_user = policy.max_sessions_per_user
        compliance_manager.session_policy.session_timeout_minutes = policy.session_timeout_minutes
        compliance_manager.session_policy.idle_timeout_minutes = policy.idle_timeout_minutes
        compliance_manager.session_policy.enforce_ip_binding = policy.enforce_ip_binding
        compliance_manager.session_policy.enforce_device_binding = policy.enforce_device_binding

        # コンプライアンスログに記録
        await compliance_manager.log_event(ComplianceEvent(
            event_type=ComplianceEventType.ADMIN_ACTION,
            user_id=current_user["sub"],
            action="update_session_policy",
            success=True,
            details={"policy": policy.dict()}
        ))

        return {"detail": "セッションポリシーが更新されました"}
    except Exception as e:
        logger.error(f"セッションポリシー更新エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="セッションポリシーの更新中にエラーが発生しました"
        )

@router.put("/config/retention-policy")
async def update_retention_policy(
    policy: RetentionPolicyModel,
    current_user: dict = Depends(get_current_admin_user)
):
    """データ保持ポリシーを更新（管理者専用）"""
    try:
        # ポリシーの更新
        compliance_manager.retention_policy.audit_log_retention_days = policy.audit_log_retention_days
        compliance_manager.retention_policy.user_data_retention_days = policy.user_data_retention_days
        compliance_manager.retention_policy.inactive_account_deletion_days = policy.inactive_account_deletion_days

        # コンプライアンスログに記録
        await compliance_manager.log_event(ComplianceEvent(
            event_type=ComplianceEventType.ADMIN_ACTION,
            user_id=current_user["sub"],
            action="update_retention_policy",
            success=True,
            details={"policy": policy.dict()}
        ))

        return {"detail": "データ保持ポリシーが更新されました"}
    except Exception as e:
        logger.error(f"データ保持ポリシー更新エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="データ保持ポリシーの更新中にエラーが発生しました"
        )

@router.post("/gdpr-request")
async def process_gdpr_request(
    request_data: GDPRRequestModel,
    current_user: dict = Depends(get_current_admin_user)
):
    """GDPRデータリクエストを処理（管理者専用）"""
    try:
        if request_data.request_type == "export":
            user_data = await compliance_manager.create_gdpr_data_export(request_data.user_id)

            # コンプライアンスログに記録
            await compliance_manager.log_event(ComplianceEvent(
                event_type=ComplianceEventType.ADMIN_ACTION,
                user_id=current_user["sub"],
                action="gdpr_export",
                affected_user_id=request_data.user_id,
                success=True
            ))

            return user_data

        elif request_data.request_type == "delete":
            result = await compliance_manager.execute_data_deletion_request(request_data.user_id)

            # コンプライアンスログに記録
            await compliance_manager.log_event(ComplianceEvent(
                event_type=ComplianceEventType.ADMIN_ACTION,
                user_id=current_user["sub"],
                action="gdpr_deletion",
                affected_user_id=request_data.user_id,
                success=result
            ))

            return {"detail": "ユーザーデータ削除リクエストが正常に処理されました"}

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="無効なリクエストタイプです。'export'または'delete'を指定してください。"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GDPRリクエスト処理エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GDPRリクエストの処理中にエラーが発生しました"
        )

@router.post("/apply-retention-policy")
async def apply_data_retention_policy(
    current_user: dict = Depends(get_current_admin_user)
):
    """データ保持ポリシーを適用（管理者専用）"""
    try:
        result = await compliance_manager.apply_data_retention_policy()

        # コンプライアンスログに記録
        await compliance_manager.log_event(ComplianceEvent(
            event_type=ComplianceEventType.ADMIN_ACTION,
            user_id=current_user["sub"],
            action="apply_retention_policy",
            success=result
        ))

        return {"detail": "データ保持ポリシーが適用されました"}
    except Exception as e:
        logger.error(f"データ保持ポリシー適用エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="データ保持ポリシーの適用中にエラーが発生しました"
        )

@router.get("/inactive-accounts")
async def get_inactive_accounts(
    current_user: dict = Depends(get_current_admin_user)
):
    """非アクティブなアカウントを取得（管理者専用）"""
    try:
        inactive_accounts = await compliance_manager.check_inactive_accounts()

        return inactive_accounts
    except Exception as e:
        logger.error(f"非アクティブアカウント取得エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="非アクティブアカウントの取得中にエラーが発生しました"
        )