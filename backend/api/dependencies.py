# -*- coding: utf-8 -*-
"""
API 依存関係の定義
-----------------
FastAPIエンドポイントで使用される依存関係を定義します。
レイヤー別に整理され、適切なスコープで依存関係を提供します。

このモジュールは、以下の主要なレイヤー向けに依存関係を提供します:
- データベース層: Firestoreなどのデータストアへのアクセス
- サービス層: ビジネスロジックの実装
- セキュリティ層: 認証と認可
- 設定層: アプリケーション設定

すべてのAPIルーターは、このモジュールから依存関係を注入することで、
一貫したサービスアクセスと依存関係の注入パターンを維持します。
"""

import logging
from typing import Optional, Annotated, Dict, Any
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import ValidationError

from database.models import UserModel
from service.firestore.client import get_firestore_client
from core.security_config import JWT_SECRET_KEY, JWT_ALGORITHM, verify_password, get_password_hash
from core.config import get_settings, Settings
from api.services.visualization import VisualizationService, get_visualization_service as get_vis_service_internal
from service.reports import ReportService, get_report_service as get_report_service_internal
from service.company_analysis import CompanyAnalysisService, get_company_analysis_service as get_company_service_internal
from backend.database.connection import get_db
from backend.services.company_service import CompanyService

# 構造化ロギング用のロガーを設定
from api.logging_utils import get_logger, log_function_call

# ロギングの設定
logger = get_logger(__name__)

# OAuth2認証の依存関係
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token",
    scopes={
        "user": "一般ユーザー権限",
        "admin": "管理者権限",
        "vc": "ベンチャーキャピタリスト権限"
    }
)

# =====================
# サービスプロバイダークラス
# =====================

class ServiceProvider:
    """
    サービスプロバイダークラス

    依存性注入のためのサービスプロバイダーを定義します。
    サービスクラスのインスタンスを管理し、必要なときに提供します。

    このクラスはシングルトンパターンを使用し、各サービスのインスタンスを
    キャッシュします。これにより、リソースの効率的な管理と
    一貫した依存性注入が可能になります。

    使用例:
        ```python
        # プロバイダーの取得
        provider = ServiceProvider()

        # サービスの登録
        provider.register_service("visualization", VisualizationService())

        # サービスの取得
        vis_service = provider.get_service("visualization")

        # サービスが登録されているか確認
        if provider.has_service("reports"):
            report_service = provider.get_service("reports")
        ```

    注意:
        このクラスはシングルトンとして実装されており、アプリケーション全体で
        単一のインスタンスを共有します。
    """
    _instance = None
    _services = {}

    def __new__(cls):
        """
        シングルトンパターンを実装するための__new__メソッド

        Returns:
            ServiceProvider: プロバイダーの唯一のインスタンス
        """
        if cls._instance is None:
            cls._instance = super(ServiceProvider, cls).__new__(cls)
            cls._instance._services = {}
        return cls._instance

    def register_service(self, service_key: str, service_instance) -> Any:
        """
        サービスをプロバイダーに登録します

        Args:
            service_key: サービスの一意の識別子
            service_instance: 登録するサービスインスタンス

        Returns:
            登録されたサービスインスタンス

        例:
            ```python
            provider = ServiceProvider()
            vis_service = VisualizationService()
            provider.register_service("visualization", vis_service)
            ```
        """
        self._services[service_key] = service_instance
        logger.info(
            f"サービスを登録しました: {service_key}",
            extra={"context": {"service_key": service_key, "service_type": type(service_instance).__name__}}
        )
        return service_instance

    def get_service(self, service_key: str) -> Any:
        """
        登録済みのサービスを取得します

        Args:
            service_key: 取得するサービスの識別子

        Returns:
            登録されたサービスインスタンス

        Raises:
            KeyError: 指定されたキーのサービスが登録されていない場合

        例:
            ```python
            provider = ServiceProvider()
            try:
                vis_service = provider.get_service("visualization")
            except KeyError:
                # サービスが登録されていない場合の処理
                pass
            ```
        """
        if service_key not in self._services:
            logger.error(
                f"サービスが登録されていません: {service_key}",
                extra={"context": {"service_key": service_key, "available_services": list(self._services.keys())}}
            )
            raise KeyError(f"サービス '{service_key}' が登録されていません")

        logger.debug(
            f"サービスを取得しました: {service_key}",
            extra={"context": {"service_key": service_key}}
        )
        return self._services[service_key]

    def has_service(self, service_key: str) -> bool:
        """
        サービスが登録されているか確認します

        Args:
            service_key: 確認するサービスの識別子

        Returns:
            サービスが登録されている場合はTrue、そうでない場合はFalse

        例:
            ```python
            provider = ServiceProvider()
            if provider.has_service("visualization"):
                vis_service = provider.get_service("visualization")
            else:
                # サービスが登録されていない場合の処理
                pass
            ```
        """
        return service_key in self._services

# グローバルなサービスプロバイダーのインスタンス
service_provider = ServiceProvider()

# =====================
# データベース層の依存関係
# =====================

@log_function_call()
def get_db_client():
    """
    Firestoreクライアントの取得

    Returns:
        firestore.Client: Firestoreクライアント

    例:
        ```python
        @app.get("/items")
        async def get_items(db = Depends(get_db_client)):
            # dbを使用してデータを取得
            pass
        ```
    """
    return get_firestore_client()

# =====================
# サービス層の依存関係
# =====================

@log_function_call()
async def get_user_by_email(email: str) -> Optional[UserModel]:
    """
    メールアドレスによるユーザー取得

    Args:
        email: ユーザーのメールアドレス

    Returns:
        Optional[UserModel]: 見つかったユーザーまたはNone

    例:
        ```python
        user = await get_user_by_email("user@example.com")
        if user:
            # ユーザーが見つかった場合の処理
            pass
        ```
    """
    try:
        db = get_db_client()
        user_ref = db.collection("users").where("email", "==", email).limit(1)
        users = user_ref.get()

        for user_doc in users:
            user_data = user_doc.to_dict()
            logger.debug(
                f"ユーザーが見つかりました: {email}",
                extra={"context": {"email": email, "user_id": user_doc.id}}
            )
            return UserModel.from_dict(user_data)

        logger.debug(
            f"ユーザーが見つかりません: {email}",
            extra={"context": {"email": email}}
        )
        return None
    except Exception as e:
        logger.error(
            f"ユーザー取得中にエラーが発生しました: {str(e)}",
            extra={"context": {"email": email, "error": str(e)}}
        )
        return None

@log_function_call()
async def authenticate_user(email: str, password: str) -> Optional[UserModel]:
    """
    ユーザー認証

    メールアドレスとパスワードを検証し、認証されたユーザーモデルを返します。

    Args:
        email: ユーザーのメールアドレス
        password: ユーザーのパスワード

    Returns:
        Optional[UserModel]: 認証されたユーザーまたはNone

    例:
        ```python
        user = await authenticate_user("user@example.com", "password123")
        if user:
            # 認証成功の処理
            pass
        else:
            # 認証失敗の処理
            pass
        ```
    """
    user = await get_user_by_email(email)
    if not user:
        logger.debug(
            f"認証失敗: ユーザーが存在しません: {email}",
            extra={"context": {"email": email, "reason": "user_not_found"}}
        )
        return None

    if not verify_password(password, user.hashed_password):
        logger.debug(
            f"認証失敗: パスワードが一致しません: {email}",
            extra={"context": {"email": email, "reason": "invalid_password"}}
        )
        return None

    logger.info(
        f"ユーザー認証成功: {email}",
        extra={"context": {"email": email}}
    )
    return user

# =====================
# セキュリティ層の依存関係
# =====================

@log_function_call()
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    アクセストークンの作成

    Args:
        data: トークンにエンコードするデータ
        expires_delta: トークンの有効期限（Noneの場合はデフォルトで15分）

    Returns:
        str: 作成されたJWTトークン

    例:
        ```python
        # 30分の有効期限を持つトークンを作成
        token_data = {"sub": user.email}
        token = create_access_token(
            data=token_data,
            expires_delta=timedelta(minutes=30)
        )
        ```
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    security_settings = get_security_settings()

    encoded_jwt = jwt.encode(
        to_encode,
        security_settings.secret_key,
        algorithm=security_settings.algorithm
    )

    logger.debug(
        "アクセストークンを生成しました",
        extra={"context": {"expires_at": expire.isoformat()}}
    )

    return encoded_jwt

@log_function_call()
async def get_current_user(
    security_scopes: SecurityScopes,
    token: Annotated[str, Depends(oauth2_scheme)]
) -> UserModel:
    """
    現在のユーザーを取得

    リクエストのアクセストークンからユーザーを認証し、返します。
    セキュリティスコープも検証します。

    Args:
        security_scopes: セキュリティスコープ
        token: アクセストークン

    Returns:
        UserModel: 現在のユーザー

    Raises:
        HTTPException: 認証エラーまたは権限エラーの場合

    例:
        ```python
        @app.get("/users/me")
        async def get_current_user_info(
            current_user: UserModel = Depends(get_current_user)
        ):
            return current_user
        ```
    """
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="認証情報を検証できませんでした",
        headers={"WWW-Authenticate": authenticate_value},
    )

    security_settings = get_security_settings()

    try:
        payload = jwt.decode(
            token,
            security_settings.secret_key,
            algorithms=[security_settings.algorithm]
        )
        email: str = payload.get("sub")

        if email is None:
            raise credentials_exception

        token_scopes = payload.get("scopes", [])

    except (JWTError, ValidationError):
        logger.exception("トークンの検証に失敗しました")
        raise credentials_exception

    user = await get_user_by_email(email)

    if user is None:
        raise credentials_exception

    for scope in security_scopes.scopes:
        if scope not in token_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"権限が不足しています: {scope}が必要です",
                headers={"WWW-Authenticate": authenticate_value},
            )

    return user

# =====================
# 設定層の依存関係
# =====================

@log_function_call()
def get_app_settings() -> Settings:
    """
    アプリケーション設定を取得

    Returns:
        Settings: アプリケーション設定

    例:
        ```python
        @app.get("/config")
        async def get_config(settings: Settings = Depends(get_app_settings)):
            return {"environment": settings.environment}
        ```
    """
    return get_settings()

@log_function_call()
def get_security_settings():
    """
    セキュリティ設定を取得

    Returns:
        Dict: セキュリティ設定を含む辞書

    例:
        ```python
        @app.get("/security/info")
        async def get_security_info(settings = Depends(get_security_settings)):
            return {"algorithm": settings.algorithm}
        ```
    """
    return {
        "secret_key": JWT_SECRET_KEY,
        "algorithm": JWT_ALGORITHM,
        "access_token_expire_minutes": 30
    }

# =====================
# サービス層の依存関係
# =====================

@log_function_call()
def get_visualization_service() -> VisualizationService:
    """
    可視化サービスを取得

    可視化機能を提供するサービスクラスのインスタンスを返します。
    必要な場合は新しいインスタンスを作成し、サービスプロバイダーに登録します。

    Returns:
        VisualizationService: 可視化サービスのインスタンス

    例:
        ```python
        @router.post("/chart")
        async def generate_chart(
            request: ChartRequest,
            vis_service: VisualizationService = Depends(get_visualization_service)
        ):
            return await vis_service.generate_chart(request)
        ```
    """
    if not service_provider.has_service("visualization"):
        service = get_vis_service_internal()
        service_provider.register_service("visualization", service)
        return service

    return service_provider.get_service("visualization")

@log_function_call()
def get_report_service() -> ReportService:
    """
    レポートサービスを取得

    レポート生成機能を提供するサービスクラスのインスタンスを返します。
    必要な場合は新しいインスタンスを作成し、サービスプロバイダーに登録します。

    Returns:
        ReportService: レポートサービスのインスタンス

    例:
        ```python
        @router.post("/generate")
        async def generate_report(
            request: ReportRequest,
            report_service: ReportService = Depends(get_report_service)
        ):
            return await report_service.generate_report(request)
        ```
    """
    if not service_provider.has_service("reports"):
        service = get_report_service_internal()
        service_provider.register_service("reports", service)
        return service

    return service_provider.get_service("reports")

@log_function_call()
def get_company_analysis_service() -> CompanyAnalysisService:
    """
    企業分析サービスを取得

    企業分析機能を提供するサービスクラスのインスタンスを返します。
    必要な場合は新しいインスタンスを作成し、サービスプロバイダーに登録します。

    Returns:
        CompanyAnalysisService: 企業分析サービスのインスタンス

    例:
        ```python
        @router.post("/analyze")
        async def analyze_company(
            company_id: str,
            analysis_service: CompanyAnalysisService = Depends(get_company_analysis_service)
        ):
            return await analysis_service.analyze_company(company_id)
        ```
    """
    if not service_provider.has_service("company_analysis"):
        service = get_company_service_internal()
        service_provider.register_service("company_analysis", service)
        return service

    return service_provider.get_service("company_analysis")

def get_company_service(session = Depends(get_db)) -> CompanyService:
    """
    企業情報サービスの依存性プロバイダ

    Args:
        session: データベースセッション

    Returns:
        CompanyService: 企業情報サービスのインスタンス
    """
    return CompanyService(session)