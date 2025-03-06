"""
料金とプラン関連のAPIエンドポイント
パーソナライズされた料金表示やプラン案内機能を提供します。
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import Dict, Any, Optional, List
from datetime import datetime
from src.database.firestore.client import FirestoreClient
from service.analysis.usage.usage_analyzer import analyze_user_trial_usage
from service.promotion.offer_generator import recommend_plan, generate_conversion_offer
from routers.auth_router import get_current_user

router = APIRouter(
    prefix="/pricing",
    tags=["pricing"],
    responses={404: {"description": "Not found"}}
)

firestore_client = FirestoreClient()

@router.get("/personalized")
async def get_personalized_pricing(current_user: dict = Depends(get_current_user)):
    """ユーザーの使用状況に基づいてパーソナライズされた価格ページのデータを取得"""
    try:
        # ユーザーの使用状況分析
        usage_stats = await analyze_user_trial_usage(current_user['uid'])
        # ユーザーの地域と言語の取得
        user_region = await get_user_region(current_user['uid'])
        # ユーザーの使用パターンに基づく最適プランの選定
        recommended_plan = await recommend_plan(current_user['uid'], usage_stats)
        # 地域に基づく価格設定の調整
        localized_plans = await get_localized_plans(user_region)
        # ユーザーの活用状況に基づくROI計算
        roi_metrics = calculate_potential_roi(usage_stats, recommended_plan['plan']['id'])
        # セールスポイントの選定（ユーザーが最も利用した機能に基づく）
        key_features = usage_stats.get('primary_features', [])
        # 特別オファーの生成
        special_offer = await generate_conversion_offer(current_user['uid'])
        # トライアル終了日の取得
        trial_end_date = await get_trial_end_date(current_user['uid'])
        days_remaining = (trial_end_date - datetime.now()).days if trial_end_date else 0
        return {
            "user_usage_summary": usage_stats,
            "recommended_plan": recommended_plan,
            "plans": localized_plans,
            "roi_projection": roi_metrics,
            "key_features": key_features,
            "testimonials": select_relevant_testimonials(key_features),
            "special_offer": special_offer,
            "trial_status": {
                "end_date": trial_end_date,
                "days_remaining": max(0, days_remaining)
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"パーソナライズ価格データ取得エラー: {str(e)}"
        )

@router.get("/trial-status")
async def get_trial_status(current_user: dict = Depends(get_current_user)):
    """ユーザーのトライアル状態とメッセージを取得"""
    try:
        user_data = await firestore_client.get_document(
            collection='users',
            doc_id=current_user['uid']
        )
        if not user_data:
            raise HTTPException(status_code=404, detail="ユーザーが見つかりません")
        trial_end = user_data.get('trial_end')
        tier = user_data.get('tier', 'free')
        if not trial_end:
            return {"has_trial_message": False}
        now = datetime.now()
        days_remaining = (trial_end - now).days
        # トライアル終了が近い場合、または終了した場合にメッセージを表示
        if days_remaining <= 3:
            # プラン情報を取得
            plans_data = await get_subscription_plans_for_user(current_user['uid'])
            return {
                "has_trial_message": True,
                "days_remaining": max(0, days_remaining),
                "trial_ended": days_remaining <= 0,
                "message_type": "trial_ending" if days_remaining > 0 else "trial_ended",
                "tier": tier,
                "recommended_plan": plans_data["recommended_plan"],
                "all_plans": plans_data["plans"],
                "special_offer": await generate_conversion_offer(current_user['uid']) if days_remaining <= 1 else None
            }
        return {"has_trial_message": False}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"トライアル状態取得エラー: {str(e)}"
        )

async def get_user_region(user_id: str) -> str:
    """ユーザーの地域を取得"""
    from service.promotion.offer_generator import get_user_region
    return await get_user_region(user_id)

async def get_localized_plans(region: str) -> List[Dict[str, Any]]:
    """地域に基づいてプラン価格を調整"""
    # 地域ごとの価格調整係数
    region_factors = {
        'north_america': 1.0,
        'europe': 1.1,
        'asia': 0.9,
        'japan': 1.0,
        'other': 1.0
    }
    # 利用可能なプラン（実際にはDBから取得）
    from routers.subscription_router import SUBSCRIPTION_PLANS
    # 基本プラン
    all_plans = list(SUBSCRIPTION_PLANS.values())
    # 地域係数を取得
    factor = region_factors.get(region, 1.0)
    # プランのコピーを作成して価格を調整
    localized_plans = []
    for plan in all_plans:
        plan_dict = plan.dict()
        plan_dict['price'] = round(plan.price * factor, 2)
        localized_plans.append(plan_dict)
    return localized_plans

def calculate_potential_roi(usage_stats: Dict[str, Any], plan_id: str) -> Dict[str, Any]:
    """ユーザーの使用状況に基づくROI計算"""
    # 月間時間節約の予測値（現在の使用パターンを基に）
    monthly_hours_saved = usage_stats.get('trial_value', {}).get('total_time_saved_hours', 0) * 4  # トライアル期間の4倍と仮定
    # プラン価格（基本値）
    plan_pricing = {
        'basic': 9.99,
        'premium': 29.99
    }
    plan_price = plan_pricing.get(plan_id, 9.99)
    # ROIの計算
    hourly_value = 50  # 時給の想定値（ドル）
    monthly_value = monthly_hours_saved * hourly_value
    roi_percentage = (monthly_value / plan_price) * 100 if plan_price > 0 else 0
    payback_period_days = (plan_price / monthly_value) * 30 if monthly_value > 0 else 0
    return {
        'monthly_hours_saved': round(monthly_hours_saved, 1),
        'monthly_value_usd': round(monthly_value, 2),
        'roi_percentage': round(roi_percentage, 1),
        'payback_period_days': round(payback_period_days, 1),
        'hourly_rate_used': hourly_value
    }

def select_relevant_testimonials(key_features: List[str]) -> List[Dict[str, str]]:
    """ユーザーの重視機能に基づく関連証言を選択"""
    # 機能ごとの証言（実際にはDBから取得）
    all_testimonials = {
        'reporting': [
            {
                'quote': "レポート機能により、毎週のレポート作成が半分の時間で済むようになりました。",
                'author': "山田 太郎",
                'company': "山田テクノロジー株式会社",
                'role': "CTO"
            }
        ],
        'time_series_analysis': [
            {
                'quote': "時系列分析により、売上の季節変動を正確に予測できるようになりました。",
                'author': "鈴木 一郎",
                'company': "経営コンサルティング株式会社",
                'role': "データアナリスト"
            }
        ],
        'dashboarding': [
            {
                'quote': "ダッシュボードをチーム全体で共有することで、データドリブンな意思決定が可能になりました。",
                'author': "佐藤 花子",
                'company': "佐藤マーケティング",
                'role': "マーケティングディレクター"
            }
        ],
        'data_storage': [
            {
                'quote': "大量のデータも簡単に管理・分析できるようになり、業務効率が大幅に向上しました。",
                'author': "高橋 健太",
                'company': "タカハシ物流株式会社",
                'role': "オペレーションマネージャー"
            }
        ]
    }
    # ユーザーの主要機能に関連する証言を選択
    selected_testimonials = []
    for feature in key_features:
        if feature in all_testimonials:
            selected_testimonials.extend(all_testimonials[feature])
    # 証言が見つからない場合はデフォルトを返す
    if not selected_testimonials:
        return [
            {
                'quote': "このサービスのおかげで、業務効率が30%向上しました。",
                'author': "佐々木 健太",
                'company': "Startupテクノロジー株式会社",
                'role': "CEO"
            }
        ]
    # 最大3つの証言に制限
    return selected_testimonials[:3]

async def get_trial_end_date(user_id: str) -> Optional[datetime]:
    """ユーザーのトライアル終了日を取得"""
    # ユーザーデータを取得
    user_data = await firestore_client.get_document(
        collection='users',
        doc_id=user_id
    )
    if not user_data or 'trial_end' not in user_data:
        return None
    return user_data['trial_end']

async def get_subscription_plans_for_user(user_id: str) -> Dict[str, Any]:
    """ユーザー向けのサブスクリプションプラン情報を取得"""
    # 使用状況を分析
    usage_stats = await analyze_user_trial_usage(user_id)
    # 地域を取得
    user_region = await get_user_region(user_id)
    # 推奨プランを取得
    recommended_plan = await recommend_plan(user_id, usage_stats)
    # 地域に合わせたプラン一覧を取得
    localized_plans = await get_localized_plans(user_region)
    return {
        "recommended_plan": recommended_plan,
        "plans": localized_plans
    }