# -*- coding: utf-8 -*-
"""
API使用状況モニタリングモジュール
---------------------------------
非推奨APIパスの使用状況を追跡し、管理者向けに情報を提供します。
"""

import logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from datetime import datetime, timedelta

# 自作ロギングユーティリティをインポート
from api.logging_utils import get_logger, metrics, PROMETHEUS_AVAILABLE, ENABLE_METRICS
from api.dependencies import get_current_user, UserModel

# ロギングの設定
logger = get_logger(__name__)

# ルーターの設定
router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

# モデル定義
class DeprecatedApiUsage(BaseModel):
    """非推奨API使用状況モデル"""
    path: str
    method: str
    count: int
    last_used: Optional[datetime] = None
    client_ips: List[str]

class ApiUsageResponse(BaseModel):
    """API使用状況レスポンスモデル"""
    success: bool = True
    data: List[DeprecatedApiUsage]
    total_deprecated_calls: int
    period: str

# Prometheusメトリクスからデータを取得する関数
def get_deprecated_api_usage(days: int = 7) -> List[DeprecatedApiUsage]:
    """
    非推奨APIの使用状況を取得する

    Args:
        days: 何日分のデータを取得するか

    Returns:
        非推奨API使用状況のリスト
    """
    # 実際の環境ではPrometheusまたはデータベースからデータを取得する
    # このサンプル実装ではダミーデータを返す

    # Prometheusが利用可能ならメトリクスから集計
    if PROMETHEUS_AVAILABLE and ENABLE_METRICS and "deprecated_api_usage_total" in metrics:
        try:
            # Prometheusクライアントを使用してデータを取得する
            # この部分は実際の実装では、Prometheusクエリを発行する
            # ここではダミーデータを返す
            return [
                DeprecatedApiUsage(
                    path="/api/v1/visualizations/chart",
                    method="POST",
                    count=25,
                    last_used=datetime.now() - timedelta(hours=3),
                    client_ips=["192.168.1.1", "192.168.1.2"]
                ),
                DeprecatedApiUsage(
                    path="/api/v1/reports/generate",
                    method="POST",
                    count=14,
                    last_used=datetime.now() - timedelta(hours=5),
                    client_ips=["192.168.1.3"]
                ),
                DeprecatedApiUsage(
                    path="/api/v1/visualizations/dashboard",
                    method="POST",
                    count=8,
                    last_used=datetime.now() - timedelta(days=1),
                    client_ips=["192.168.1.4", "192.168.1.5", "192.168.1.6"]
                )
            ]
        except Exception as e:
            logger.error(f"Prometheusからのデータ取得エラー: {e}")
            return []
    else:
        # Prometheusが利用できない場合はダミーデータを返す
        logger.warning("Prometheusメトリクスが利用できないため、ダミーデータを返します")
        return [
            DeprecatedApiUsage(
                path="/api/v1/visualizations/chart",
                method="POST",
                count=10,
                last_used=datetime.now() - timedelta(hours=12),
                client_ips=["192.168.1.10"]
            ),
            DeprecatedApiUsage(
                path="/api/v1/reports/generate",
                method="POST",
                count=5,
                last_used=datetime.now() - timedelta(days=2),
                client_ips=["192.168.1.11"]
            )
        ]

# エンドポイント定義
@router.get("/deprecated-api-usage", response_model=ApiUsageResponse)
async def get_deprecated_api_usage_stats(
    days: int = Query(7, ge=1, le=30, description="取得する日数 (1-30)"),
    current_user: UserModel = Depends(get_current_user)
) -> ApiUsageResponse:
    """
    非推奨APIの使用状況を取得する

    Args:
        days: 何日分のデータを取得するか (1-30)
        current_user: 現在のユーザー (admin権限が必要)

    Returns:
        非推奨API使用状況

    Raises:
        HTTPException: 認証エラーまたは権限エラーの場合
    """
    # 管理者権限をチェック
    if not current_user.is_admin:
        logger.warning(
            f"権限のないユーザーがAPI使用状況を取得しようとしました: {current_user.email}",
            extra={"context": {"user_id": current_user.document_id}}
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="この操作には管理者権限が必要です"
        )

    # 非推奨API使用状況を取得
    usage_data = get_deprecated_api_usage(days)
    total_calls = sum(item.count for item in usage_data)

    logger.info(
        f"非推奨API使用状況を取得しました: {len(usage_data)}件, 合計{total_calls}回の呼び出し",
        extra={"context": {"days": days, "user_id": current_user.document_id}}
    )

    return ApiUsageResponse(
        success=True,
        data=usage_data,
        total_deprecated_calls=total_calls,
        period=f"過去{days}日間"
    )

# 非推奨APIリスト取得エンドポイント
@router.get("/deprecated-apis")
async def get_deprecated_apis(
    current_user: UserModel = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    非推奨APIのリストを取得する

    Args:
        current_user: 現在のユーザー (認証が必要)

    Returns:
        非推奨APIのリスト
    """
    # 非推奨APIのリストを返す
    return {
        "success": True,
        "data": {
            "deprecated_paths": [
                {
                    "prefix": "/api/v1/visualizations",
                    "new_prefix": "/api/visualization",
                    "deprecated_since": "2023-04-15",
                    "removal_date": "2023-07-15"
                },
                {
                    "prefix": "/api/v1/reports",
                    "new_prefix": "/api/reports",
                    "deprecated_since": "2023-04-15",
                    "removal_date": "2023-07-15"
                }
            ]
        }
    }