"""
サブスクリプションミドルウェア
ユーザーのサブスクリプション状態に基づいて機能アクセスを制御します。
"""
from fastapi import Request, HTTPException, status
from typing import Optional, List, Dict, Callable, Awaitable
import logging
from datetime import datetime
from src.database.firestore.client import FirestoreClient

logger = logging.getLogger(__name__)
firestore_client = FirestoreClient()

# 無料プランでの機能制限の定義
FREE_TIER_LIMITS = {
    'dashboard_widgets': {'max': 5},
    'reports': {'max': 5, 'period': 'month'},
    'data_uploads': {'max': 3, 'size_limit_mb': 5},
    'data_export_rows': {'max': 100},
    'analysis_types': {'allowed': ['basic_stats', 'correlation']},
    'historical_data_days': {'max': 30},
    'users': {'max': 2}
}

async def verify_subscription(
    request: Request,
    allowed_plans: Optional[List[str]] = None,
    allow_trial: bool = True
):
    """
    ユーザーがアクティブなサブスクリプションを持っているか確認するミドルウェア

    Args:
        request: FastAPIリクエスト
        allowed_plans: このエンドポイントにアクセスできるプランのリスト
        allow_trial: トライアル中のユーザーを許可するかどうか
    """
    # リクエストステートからユーザーを取得（認証ミドルウェアによって設定）
    user = request.state.user
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="認証されていません"
        )

    # サブスクリプションを確認
    subscriptions = await firestore_client.query_documents(
        collection='subscriptions',
        filters=[
            {'field': 'user_id', 'operator': '==', 'value': user.get('uid')},
            {'field': 'status', 'operator': 'in', 'value': ['active', 'trialing']}
        ],
        limit=1
    )

    # サブスクリプションがない場合はユーザー情報を確認
    if not subscriptions:
        # ユーザー情報を取得
        user_data = await firestore_client.get_document(
            collection='users',
            doc_id=user.get('uid')
        )

        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="ユーザー情報が見つかりません"
            )

        tier = user_data.get('tier')

        # 無料トライアル中かどうかを確認
        if tier == 'free':
            trial_end = user_data.get('trial_end')
            now = datetime.now()

            if not trial_end or trial_end < now:
                # トライアル終了または無料ユーザー
                if allowed_plans and 'free' not in allowed_plans:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"この機能には次のいずれかのプランが必要です: {', '.join(allowed_plans)}"
                    )
            else:
                # トライアル中
                if not allow_trial:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="この機能はトライアル期間中は利用できません"
                    )

            # ユーザー情報をリクエストステートに保存
            request.state.subscription = {
                'tier': 'free',
                'trial_end': trial_end
            }
            return
    else:
        # サブスクリプションがある場合
        subscription = subscriptions[0]

        # トライアル中かどうかとトライアルが許可されているかを確認
        if subscription['status'] == 'trialing' and not allow_trial:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="この機能はトライアル期間中は利用できません"
            )

        # プランタイプの制限を確認
        if allowed_plans and subscription['plan_type'] not in allowed_plans:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"この機能には次のいずれかのプランが必要です: {', '.join(allowed_plans)}"
            )

        # サブスクリプション情報をリクエストステートに保存
        request.state.subscription = subscription

def subscription_required(
    allowed_plans: Optional[List[str]] = None,
    allow_trial: bool = True
):
    """
    サブスクリプション確認デコレータ

    Args:
        allowed_plans: エンドポイントにアクセスできるプランのリスト
        allow_trial: トライアル中のユーザーを許可するかどうか
    """
    async def dependency(request: Request):
        await verify_subscription(request, allowed_plans, allow_trial)
        return True

    return dependency

async def check_feature_access(
    request: Request,
    feature: str,
    required_tier: str = "free"  # "free", "trial", "paid"
):
    """特定の機能へのアクセス権をチェック"""
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="認証が必要です")

    # ユーザーのサブスクリプション情報を取得
    subscription_data = await get_user_subscription(user.get('uid'))
    tier = subscription_data.get('tier', 'free')

    # 階層のレベルを数値に変換
    tier_levels = {
        'free': 0,
        'trial': 1,
        'paid': 2
    }

    user_tier_level = tier_levels.get(tier, 0)
    required_tier_level = tier_levels.get(required_tier, 0)

    # ユーザーの階層が必要な階層以上かチェック
    if user_tier_level < required_tier_level:
        raise HTTPException(
            status_code=403,
            detail=f"この機能を使用するには {required_tier} 以上のプランが必要です"
        )

    # 無料プランの場合は使用量制限をチェック
    if tier == 'free':
        # 機能ごとの使用量制限をチェック
        usage_limits = FREE_TIER_LIMITS.get(feature, {})
        if usage_limits:
            await check_usage_limits(user.get('uid'), feature, usage_limits)

    return True

def feature_access_required(
    feature: str,
    required_tier: str = "free"  # "free", "trial", "paid"
):
    """
    機能アクセス確認デコレータ

    Args:
        feature: アクセスする機能名
        required_tier: 必要なプランレベル
    """
    async def dependency(request: Request):
        return await check_feature_access(request, feature, required_tier)

    return dependency

async def get_user_subscription(user_id: str) -> Dict:
    """ユーザーのサブスクリプション情報を取得"""
    # サブスクリプションを確認
    subscriptions = await firestore_client.query_documents(
        collection='subscriptions',
        filters=[
            {'field': 'user_id', 'operator': '==', 'value': user_id},
            {'field': 'status', 'operator': 'in', 'value': ['active', 'trialing']}
        ],
        limit=1
    )

    if subscriptions:
        return {
            'tier': 'paid' if subscriptions[0]['status'] == 'active' else 'trial',
            'plan_type': subscriptions[0]['plan_type'],
            'subscription_id': subscriptions[0]['stripe_subscription_id'],
            'trial_end': subscriptions[0].get('trial_end')
        }

    # サブスクリプションがない場合はユーザー情報を確認
    user_data = await firestore_client.get_document(
        collection='users',
        doc_id=user_id
    )

    if not user_data:
        return {'tier': 'unknown'}

    tier = user_data.get('tier', 'free')
    trial_end = user_data.get('trial_end')

    # トライアル終了チェック
    if tier == 'free' and trial_end and trial_end > datetime.now():
        tier = 'trial'

    return {
        'tier': tier,
        'trial_end': trial_end
    }

# 使用量制限のチェック
async def check_usage_limits(user_id: str, feature: str, limits: Dict):
    """無料プランの使用量制限をチェック"""

    # 例: レポート数の制限をチェック
    if feature == 'reports':
        # 今月作成したレポート数を取得
        this_month_reports = await count_user_reports_this_month(user_id)
        if this_month_reports >= limits['max']:
            raise HTTPException(
                status_code=403,
                detail=f"無料プランでは月に最大{limits['max']}件のレポートまで作成できます"
            )

    # 例: ダッシュボードウィジェット数の制限をチェック
    elif feature == 'dashboard_widgets':
        widget_count = await count_user_dashboard_widgets(user_id)
        if widget_count >= limits['max']:
            raise HTTPException(
                status_code=403,
                detail=f"無料プランでは最大{limits['max']}個のウィジェットまで作成できます"
            )

    # 例: データアップロードの制限をチェック
    elif feature == 'data_uploads':
        # 今月のアップロード数を取得
        this_month_uploads = await count_user_uploads_this_month(user_id)
        if this_month_uploads >= limits['max']:
            raise HTTPException(
                status_code=403,
                detail=f"無料プランでは月に最大{limits['max']}個のファイルまでアップロードできます"
            )

    # 例: 分析タイプの制限をチェック
    elif feature == 'analysis_types':
        analysis_type = get_analysis_type_from_request()
        if analysis_type not in limits['allowed']:
            raise HTTPException(
                status_code=403,
                detail=f"無料プランでは次の分析タイプのみ利用可能です: {', '.join(limits['allowed'])}"
            )

async def count_user_reports_this_month(user_id: str) -> int:
    """ユーザーが今月作成したレポート数をカウント"""
    # 現在の月の最初の日
    from datetime import datetime
    now = datetime.now()
    first_day_of_month = datetime(now.year, now.month, 1)

    try:
        reports = await firestore_client.query_documents(
            collection='reports',
            filters=[
                {'field': 'user_id', 'operator': '==', 'value': user_id},
                {'field': 'created_at', 'operator': '>=', 'value': first_day_of_month}
            ]
        )
        return len(reports)
    except Exception as e:
        logger.error(f"レポート数取得エラー: {str(e)}")
        return 0

async def count_user_dashboard_widgets(user_id: str) -> int:
    """ユーザーのダッシュボードウィジェット数をカウント"""
    try:
        widgets = await firestore_client.query_documents(
            collection='dashboard_widgets',
            filters=[{'field': 'user_id', 'operator': '==', 'value': user_id}]
        )
        return len(widgets)
    except Exception as e:
        logger.error(f"ウィジェット数取得エラー: {str(e)}")
        return 0

async def count_user_uploads_this_month(user_id: str) -> int:
    """ユーザーが今月アップロードしたファイル数をカウント"""
    # 現在の月の最初の日
    from datetime import datetime
    now = datetime.now()
    first_day_of_month = datetime(now.year, now.month, 1)

    try:
        uploads = await firestore_client.query_documents(
            collection='data_uploads',
            filters=[
                {'field': 'user_id', 'operator': '==', 'value': user_id},
                {'field': 'created_at', 'operator': '>=', 'value': first_day_of_month}
            ]
        )
        return len(uploads)
    except Exception as e:
        logger.error(f"アップロード数取得エラー: {str(e)}")
        return 0

def get_analysis_type_from_request():
    """リクエストから分析タイプを取得する（実際の実装はフレームワークに依存）"""
    # 注: この関数は実際のリクエスト解析に合わせて実装する必要がある
    return "basic_stats"  # デフォルト値