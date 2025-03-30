# -*- coding: utf-8 -*-
"""
連合学習API
-----------
連合学習システムのAPI定義。複数のクライアントが連携して
分散型でモデルを訓練するためのエンドポイントを提供します。
"""
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, UploadFile, File, Body, Query, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# 認証と権限管理
from core.auth_manager import get_current_user, get_current_active_user
from core.security import SecurityManager
from core.rate_limiter import RateLimiter, rate_limited

# 連合学習モジュール
from federated_learning.server.aggregator import FederatedAggregator
from federated_learning.models.registry import ModelRegistry
from federated_learning.security.secure_aggregation import SecureAggregation
from federated_learning.utils.differential_privacy import DifferentialPrivacy
from core.anonymization import AnonymizationService

# データベースとストレージ
from database.connection import get_db
from core.federated_learning import FederatedLearningManager, FederatedModel
from core.config import get_settings
from core.compliance_manager import get_compliance_manager
from core.auth_manager import UserRole

# ロギング設定
logger = logging.getLogger(__name__)

# ルーター定義
router = APIRouter(
    prefix="/api/v1/federated",
    tags=["federated_learning"],
    responses={404: {"description": "リソースが見つかりません"}},
)

# OAuth2認証スキーム
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# -----------------------------------------------------
# リクエスト/レスポンスモデル
# -----------------------------------------------------

class ModelMetadata(BaseModel):
    """連合学習モデルのメタデータ"""
    id: str
    name: str
    version: str
    description: Optional[str] = None
    architecture: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    status: str
    client_count: int
    rounds_completed: int
    metrics: Optional[Dict[str, Any]] = None

class ModelListResponse(BaseModel):
    """モデル一覧レスポンス"""
    models: List[ModelMetadata]
    count: int

class ModelUpdateRequest(BaseModel):
    """モデル更新リクエスト"""
    client_id: str = Field(..., description="クライアントID")
    round_number: int = Field(..., description="現在のラウンド番号")
    metrics: Dict[str, Any] = Field(default={}, description="ローカル訓練の評価指標")
    parameters_format: str = Field(default="tensorflow", description="モデルパラメータのフォーマット")
    samples_count: int = Field(..., description="訓練に使用したサンプル数")
    noise_multiplier: Optional[float] = Field(default=None, description="差分プライバシーのノイズ乗数")

class ModelUpdateResponse(BaseModel):
    """モデル更新レスポンス"""
    status: str
    message: str
    next_round: Optional[int] = None
    waiting_time: Optional[int] = None

class TrainingStatus(BaseModel):
    """訓練ステータスレスポンス"""
    model_id: str
    current_round: int
    status: str
    progress: float
    clients_required: int
    clients_submitted: int
    estimated_completion: Optional[datetime] = None
    next_aggregation: Optional[datetime] = None

# -----------------------------------------------------
# 依存関係
# -----------------------------------------------------

async def get_model_registry():
    """モデルレジストリの取得"""
    return ModelRegistry()

async def get_federated_aggregator():
    """連合学習アグリゲーターの取得"""
    return FederatedAggregator()

# -----------------------------------------------------
# API エンドポイント
# -----------------------------------------------------

@router.get("/models", response_model=ModelListResponse)
async def get_available_models(
    skip: int = 0,
    limit: int = 10,
    status: Optional[str] = None,
    current_user: Dict = Depends(get_current_active_user),
    registry: ModelRegistry = Depends(get_model_registry),
    db: Session = Depends(get_db)
):
    """
    利用可能な連合学習モデルの一覧を取得します。

    全ての利用可能なモデルまたはステータスでフィルタリングされたモデルを返します。
    """
    try:
        models = await registry.get_models(skip=skip, limit=limit, status=status)
        return {"models": models, "count": len(models)}
    except Exception as e:
        logger.error(f"モデル一覧取得エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="モデル一覧の取得中にエラーが発生しました"
        )

@router.get("/models/{model_id}", response_model=ModelMetadata)
async def get_model_details(
    model_id: str,
    current_user: Dict = Depends(get_current_active_user),
    registry: ModelRegistry = Depends(get_model_registry),
    db: Session = Depends(get_db)
):
    """
    特定の連合学習モデルの詳細情報を取得します。
    """
    try:
        model = await registry.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"モデルID {model_id} は見つかりません"
            )
        return model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"モデル詳細取得エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="モデル詳細の取得中にエラーが発生しました"
        )

@router.get("/models/{model_id}/download")
@RateLimiter(max_requests=30, window_seconds=60)
async def download_global_model(
    model_id: str,
    version: Optional[str] = None,
    current_user: Dict = Depends(get_current_active_user),
    registry: ModelRegistry = Depends(get_model_registry),
    db: Session = Depends(get_db)
):
    """
    グローバルモデルをダウンロードします。

    最新バージョンまたは特定バージョンのモデルを返します。
    """
    try:
        # ユーザーの認証情報と権限を確認
        client_id = current_user.get("uid")

        # クライアントIDをモデル参加者として記録
        await registry.register_client_participation(model_id, client_id)

        # モデルファイルへのパスを取得
        model_path = await registry.get_model_file_path(model_id, version)
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"モデルID {model_id} のファイルが見つかりません"
            )

        # モデルファイルをダウンロード
        return FileResponse(
            path=model_path,
            filename=f"model_{model_id}_{version or 'latest'}.h5",
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"モデルダウンロードエラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="モデルのダウンロード中にエラーが発生しました"
        )

@router.post("/models/{model_id}/update", response_model=ModelUpdateResponse)
@RateLimiter(max_requests=10, window_seconds=60)
async def submit_model_update(
    model_id: str,
    update_info: ModelUpdateRequest = Body(...),
    model_file: UploadFile = File(...),
    current_user: Dict = Depends(get_current_active_user),
    aggregator: FederatedAggregator = Depends(get_federated_aggregator),
    registry: ModelRegistry = Depends(get_model_registry),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    ローカルで訓練したモデル更新を提出します。

    クライアントの訓練結果をサーバーに送信し、グローバルモデルの更新に貢献します。
    """
    try:
        # ユーザーの認証情報と権限を確認
        client_id = current_user.get("uid")

        # クライアントIDと送信されたクライアントIDが一致するか確認
        if client_id != update_info.client_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="クライアントIDが認証情報と一致しません"
            )

        # 現在の訓練ステータスを確認
        training_status = await registry.get_training_status(model_id)
        if not training_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"モデルID {model_id} の訓練ステータスが見つかりません"
            )

        # ラウンド番号が合致するか確認
        if training_status.current_round != update_info.round_number:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "status": "outdated",
                    "message": "ラウンド番号が現在の訓練ラウンドと一致しません",
                    "next_round": training_status.current_round,
                    "waiting_time": 0
                }
            )

        # アップロードされたモデルファイルを保存
        file_location = f"/tmp/model_updates/{model_id}/{client_id}_{uuid.uuid4()}.h5"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)

        with open(file_location, "wb") as buffer:
            content = await model_file.read()
            buffer.write(content)

        # セキュアアグリゲーションと差分プライバシーの適用
        secure_agg = SecureAggregation()
        dp = DifferentialPrivacy()

        # バックグラウンドタスクでモデル更新を処理
        background_tasks.add_task(
            aggregator.process_model_update,
            model_id=model_id,
            client_id=client_id,
            model_path=file_location,
            metrics=update_info.metrics,
            samples_count=update_info.samples_count,
            noise_multiplier=update_info.noise_multiplier,
            secure_aggregation=secure_agg,
            differential_privacy=dp
        )

        # 次のラウンドのスケジュールを確認
        next_round = training_status.current_round
        waiting_time = 0

        if training_status.clients_submitted + 1 >= training_status.clients_required:
            # 十分なクライアントが更新を提出した場合
            next_round += 1
            waiting_time = 0  # すぐに次のラウンドが始まる
        else:
            # まだ十分なクライアントが集まっていない場合
            waiting_time = 3600  # 1時間待機（実際には動的に計算）

        return {
            "status": "accepted",
            "message": "モデル更新が正常に受け付けられました",
            "next_round": next_round,
            "waiting_time": waiting_time
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"モデル更新提出エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="モデル更新の提出中にエラーが発生しました"
        )

@router.get("/status", response_model=List[TrainingStatus])
async def get_training_status(
    model_id: Optional[str] = None,
    current_user: Dict = Depends(get_current_active_user),
    registry: ModelRegistry = Depends(get_model_registry),
    db: Session = Depends(get_db)
):
    """
    現在の訓練ステータスを確認します。

    全てのモデルまたは特定のモデルの訓練状況を返します。
    """
    try:
        if model_id:
            # 特定のモデルのステータスを取得
            status = await registry.get_training_status(model_id)
            if not status:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"モデルID {model_id} の訓練ステータスが見つかりません"
                )
            return [status]
        else:
            # 全てのアクティブなモデルのステータスを取得
            statuses = await registry.get_all_training_statuses()
            return statuses
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"訓練ステータス取得エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="訓練ステータスの取得中にエラーが発生しました"
        )

@router.post("/register-client")
async def register_federated_client(
    client_info: Dict = Body(...),
    current_user: Dict = Depends(get_current_active_user),
    registry: ModelRegistry = Depends(get_model_registry),
    anonymization_service: AnonymizationService = Depends(lambda: AnonymizationService()),
    db: Session = Depends(get_db)
):
    """
    新しい連合学習クライアントを登録します。

    クライアントの情報を登録し、匿名化IDを発行します。
    """
    try:
        # クライアント情報を取得
        company_id = current_user.get("uid")
        industry_type = client_info.get("industry_type", "unknown")
        data_size = client_info.get("data_size", 0)

        # クライアント情報を匿名化
        anonymous_id = await anonymization_service.anonymize_id(company_id)

        # クライアントを登録
        client_record = {
            "client_id": anonymous_id,
            "original_id": company_id,  # 実装時には暗号化して保存
            "industry_type": industry_type,
            "data_size": data_size,
            "registered_at": datetime.now(),
            "last_active": datetime.now(),
            "models_participated": []
        }

        await registry.register_client(client_record)

        return {
            "status": "success",
            "message": "連合学習クライアントが正常に登録されました",
            "client_id": anonymous_id
        }
    except Exception as e:
        logger.error(f"クライアント登録エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="クライアント登録中にエラーが発生しました"
        )