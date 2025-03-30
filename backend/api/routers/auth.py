# -*- coding: utf-8 -*-
"""
認証 API ルーター
Firebase AuthenticationとCloud Firestoreを使用した認証機能を提供します。
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Security, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, EmailStr, Field, field_validator
from datetime import datetime, timedelta

# コアモジュールのインポート
from core.auth_manager import (
    AuthManager,
    User,
    UserRole,
    MFAType,
    TOTPSetup,
    get_current_active_user,
    get_current_admin_user
)
from core.security import SecurityManager
from core.rate_limiter import get_rate_limiter
from core.auth_metrics import get_auth_metrics

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
    responses={404: {"description": "Not found"}}
)

# シングルトンのAuthManagerインスタンスを取得
auth_manager = AuthManager()
security_manager = SecurityManager()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
rate_limiter = get_rate_limiter()
auth_metrics = get_auth_metrics()

# ロガーの設定
logger = logging.getLogger(__name__)

class Token(BaseModel):
    """トークンモデル"""
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    """ユーザー作成モデル"""
    email: EmailStr
    password: str = Field(..., min_length=8)
    display_name: Optional[str] = None
    role: UserRole = UserRole.USER
    company_id: Optional[str] = None

    @field_validator('password')
    @classmethod
    def password_strength(cls, v):
        """パスワード強度の検証"""
        if len(v) < 8:
            raise ValueError('パスワードは8文字以上である必要があります')
        if not any(c.isupper() for c in v):
            raise ValueError('パスワードには大文字を含める必要があります')
        if not any(c.islower() for c in v):
            raise ValueError('パスワードには小文字を含める必要があります')
        if not any(c.isdigit() for c in v):
            raise ValueError('パスワードには数字を含める必要があります')
        if not any(c in '!@#$%^&*()_-+={}[]|\:;"<>,.?/' for c in v):
            raise ValueError('パスワードには特殊文字を含める必要があります')
        return v

class UserResponse(BaseModel):
    """ユーザーレスポンスモデル"""
    id: str
    email: str
    display_name: Optional[str] = None
    is_active: bool = True
    role: UserRole = UserRole.USER
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    profile_completed: bool = False
    company_id: Optional[str] = None
    is_email_verified: bool = False
    mfa_enabled: bool = False
    mfa_type: MFAType = MFAType.NONE

class UserUpdate(BaseModel):
    """ユーザー更新モデル"""
    display_name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[UserRole] = None
    company_id: Optional[str] = None
    profile_completed: Optional[bool] = None
    data_access: Optional[List[str]] = None
    phone_number: Optional[str] = None

class PasswordReset(BaseModel):
    """パスワードリセットモデル"""
    email: EmailStr

class MFASetupResponse(BaseModel):
    """MFA設定レスポンスモデル"""
    secret: str
    qr_code: str
    uri: str

class MFAVerifyRequest(BaseModel):
    """MFA検証リクエストモデル"""
    code: str

class MFALoginRequest(BaseModel):
    """MFAログインリクエストモデル"""
    email: str
    code: str

class SMSSetupRequest(BaseModel):
    """SMS設定リクエストモデル"""
    phone_number: str

class LoginResponse(BaseModel):
    """ログインレスポンスモデル"""
    access_token: str
    token_type: str
    user: UserResponse
    requires_mfa: bool = False
    mfa_type: MFAType = MFAType.NONE

@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    新規ユーザー登録エンドポイント

    Args:
        user_data: 登録するユーザー情報

    Returns:
        UserResponse: 作成されたユーザー情報
    """
    client_ip = request.client.host

    # レート制限をチェック
    if rate_limiter.is_rate_limited(client_ip, "register"):
        # メトリクスの記録
        auth_metrics.track_rate_limit_hit(client_ip, "register")

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="登録リクエストの回数制限を超えました。しばらく経ってからお試しください。"
        )

    user = await auth_manager.register_user(
        email=user_data.email,
        password=user_data.password,
        display_name=user_data.display_name,
        role=user_data.role,
        company_id=user_data.company_id
    )

    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        is_active=user.is_active,
        role=user.role,
        created_at=user.created_at,
        profile_completed=user.profile_completed,
        company_id=user.company_id,
        is_email_verified=user.is_email_verified,
        mfa_enabled=user.mfa_enabled,
        mfa_type=user.mfa_type
    )

@router.post("/token", response_model=LoginResponse)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2互換のトークン取得エンドポイント

    Args:
        form_data: ログインフォームデータ

    Returns:
        LoginResponse: ログイン結果（トークンとユーザー情報）
    """
    # ユーザー認証
    user, scopes = await auth_manager.authenticate_user(
        email=form_data.username,  # OAuth2ではusernameフィールドを使う
        password=form_data.password
    )

    # MFAが有効かチェック
    if user.mfa_enabled:
        # MFAが必要な場合は特別なトークンを発行
        # このトークンは短時間のみ有効でMFA検証のみに使用可能
        mfa_token_data = {
            "sub": user.id,
            "email": user.email,
            "mfa_pending": True
        }

        # 5分の有効期限を持つMFA用トークンを作成
        mfa_token = await auth_manager.create_access_token(
            data=mfa_token_data,
            scopes=["mfa_verification"],
            expires_delta=timedelta(minutes=5)
        )

        # MFA検証が必要であることを示すレスポンスを返す
        return LoginResponse(
            access_token=mfa_token,
            token_type="bearer",
            user=UserResponse(
                id=user.id,
                email=user.email,
                display_name=user.display_name,
                role=user.role,
                mfa_enabled=user.mfa_enabled,
                mfa_type=user.mfa_type,
                is_active=user.is_active
            ),
            requires_mfa=True,
            mfa_type=user.mfa_type
        )

    # MFAが不要な場合は通常のトークンを発行
    token_data = {
        "sub": user.id,
        "email": user.email,
        "role": user.role.value
    }

    # 30分の有効期限を持つトークンを作成
    access_token = await auth_manager.create_access_token(
        data=token_data,
        scopes=scopes,
        expires_delta=timedelta(minutes=30)
    )

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            is_active=user.is_active,
            role=user.role,
            created_at=user.created_at,
            last_login=user.last_login,
            profile_completed=user.profile_completed,
            company_id=user.company_id,
            is_email_verified=user.is_email_verified,
            mfa_enabled=user.mfa_enabled,
            mfa_type=user.mfa_type
        ),
        requires_mfa=False
    )

@router.post("/mfa/verify", response_model=LoginResponse)
async def verify_mfa_code(mfa_request: MFAVerifyRequest, current_user: User = Depends(get_current_active_user)):
    """
    MFA検証エンドポイント

    Args:
        mfa_request: MFA検証リクエスト
        current_user: 現在のユーザー（MFA検証用トークンで認証）

    Returns:
        LoginResponse: 検証成功時にフルアクセストークンを返す
    """
    # MFAコードを検証
    is_valid = await auth_manager.verify_mfa_code(
        user_id=current_user.id,
        code=mfa_request.code
    )

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="無効な認証コードです"
        )

    # 検証成功したら通常のアクセストークンを発行
    token_data = {
        "sub": current_user.id,
        "email": current_user.email,
        "role": current_user.role.value
    }

    # ロールに応じたスコープを取得
    scopes = auth_manager._get_role_scopes(current_user.role)

    # 30分の有効期限を持つトークンを作成
    access_token = await auth_manager.create_access_token(
        data=token_data,
        scopes=scopes,
        expires_delta=timedelta(minutes=30)
    )

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=current_user.id,
            email=current_user.email,
            display_name=current_user.display_name,
            is_active=current_user.is_active,
            role=current_user.role,
            created_at=current_user.created_at,
            last_login=current_user.last_login,
            profile_completed=current_user.profile_completed,
            company_id=current_user.company_id,
            is_email_verified=current_user.is_email_verified,
            mfa_enabled=current_user.mfa_enabled,
            mfa_type=current_user.mfa_type
        ),
        requires_mfa=False
    )

@router.post("/mfa/setup/totp", response_model=MFASetupResponse)
async def setup_totp_mfa(current_user: User = Depends(get_current_active_user)):
    """
    TOTP MFA設定開始エンドポイント

    Args:
        current_user: 現在のユーザー

    Returns:
        MFASetupResponse: QRコードなどのTOTP設定情報
    """
    # TOTP MFA設定の初期化
    totp_setup = await auth_manager.setup_totp_mfa(current_user.id)

    return MFASetupResponse(
        secret=totp_setup.secret,
        qr_code=totp_setup.qr_code,
        uri=totp_setup.uri
    )

@router.post("/mfa/setup/verify", response_model=Dict[str, bool])
async def verify_mfa_setup(verify_request: MFAVerifyRequest, current_user: User = Depends(get_current_active_user)):
    """
    MFA設定検証エンドポイント

    Args:
        verify_request: 検証リクエスト
        current_user: 現在のユーザー

    Returns:
        Dict: 設定結果
    """
    # MFA設定を検証
    success = await auth_manager.verify_mfa_setup(
        user_id=current_user.id,
        code=verify_request.code
    )

    return {"success": success}

@router.post("/mfa/setup/sms", response_model=Dict[str, bool])
async def setup_sms_mfa(sms_request: SMSSetupRequest, current_user: User = Depends(get_current_active_user)):
    """
    SMS MFA設定エンドポイント

    Args:
        sms_request: SMS設定リクエスト
        current_user: 現在のユーザー

    Returns:
        Dict: 設定結果
    """
    # SMS認証コードを送信
    success = await auth_manager.send_sms_code(
        user_id=current_user.id,
        phone_number=sms_request.phone_number
    )

    return {"success": success}

@router.post("/mfa/disable", response_model=Dict[str, bool])
async def disable_mfa(current_user: User = Depends(get_current_active_user)):
    """
    MFA無効化エンドポイント

    Args:
        current_user: 現在のユーザー

    Returns:
        Dict: 無効化結果
    """
    # MFAを無効化
    success = await auth_manager.disable_mfa(current_user.id)

    return {"success": success}

@router.post("/password-reset", response_model=Dict[str, str])
async def reset_password(reset_data: PasswordReset):
    """
    パスワードリセットリンク送信エンドポイント

    Args:
        reset_data: リセット対象のメールアドレス

    Returns:
        Dict: 結果メッセージ
    """
    await auth_manager.reset_password(reset_data.email)
    return {"message": "パスワードリセットリンクを送信しました。メールをご確認ください。"}

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    現在ログイン中のユーザー情報取得エンドポイント

    Args:
        current_user: 現在のユーザー

    Returns:
        UserResponse: ユーザー情報
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        is_active=current_user.is_active,
        role=current_user.role,
        created_at=current_user.created_at,
        last_login=current_user.last_login,
        profile_completed=current_user.profile_completed,
        company_id=current_user.company_id,
        is_email_verified=current_user.is_email_verified,
        mfa_enabled=current_user.mfa_enabled,
        mfa_type=current_user.mfa_type
    )

@router.post("/logout", response_model=Dict[str, str])
async def logout(current_user: User = Depends(get_current_active_user)):
    """
    ログアウトエンドポイント

    Args:
        current_user: 現在のユーザー

    Returns:
        Dict: 結果メッセージ
    """
    success = await auth_manager.revoke_user_tokens(current_user.id)
    if success:
        return {"message": "ログアウトしました"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ログアウト処理中にエラーが発生しました"
        )

@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """
    ユーザー情報更新エンドポイント

    Args:
        user_id: 更新対象のユーザーID
        user_data: 更新するデータ
        current_user: 現在のユーザー

    Returns:
        UserResponse: 更新されたユーザー情報
    """
    # 権限チェック（自分自身または管理者のみ許可）
    if current_user.id != user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="他のユーザーの情報を更新する権限がありません"
        )

    # ロール変更の権限チェック
    if user_data.role is not None and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="ユーザーロールの変更は管理者のみ許可されています"
        )

    # 更新用データを作成
    update_data = {}
    for field, value in user_data.dict(exclude_unset=True).items():
        if value is not None:
            update_data[field] = value

    # 更新実行
    user = await auth_manager.update_user(user_id, update_data)

    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        is_active=user.is_active,
        role=user.role,
        created_at=user.created_at,
        last_login=user.last_login,
        profile_completed=user.profile_completed,
        company_id=user.company_id,
        is_email_verified=user.is_email_verified,
        mfa_enabled=user.mfa_enabled,
        mfa_type=user.mfa_type
    )

@router.delete("/users/{user_id}", response_model=Dict[str, str])
async def delete_user(
    user_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    ユーザー削除エンドポイント

    Args:
        user_id: 削除対象のユーザーID
        current_user: 現在のユーザー

    Returns:
        Dict: 結果メッセージ
    """
    # 権限チェック（自分自身または管理者のみ許可）
    if current_user.id != user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="他のユーザーを削除する権限がありません"
        )

    # 削除実行
    await auth_manager.delete_user(user_id)

    return {"message": "ユーザーを削除しました"}

@router.get("/users", response_model=List[UserResponse])
async def get_users(current_user: User = Security(get_current_admin_user)):
    """
    全ユーザー一覧取得エンドポイント（管理者のみ）

    Args:
        current_user: 現在のユーザー（管理者）

    Returns:
        List[UserResponse]: ユーザー一覧
    """
    # この関数は管理者のみがアクセス可能
    # TODO: 実際のユーザー一覧取得ロジックをauthマネージャーに実装する必要があります
    # ここではサンプルとして空のリストを返しています
    return []