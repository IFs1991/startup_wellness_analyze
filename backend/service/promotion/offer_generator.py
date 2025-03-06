"""
特別オファー生成サービス
ユーザーの特性に基づいて変換オファーを生成します。
"""
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging
from src.database.firestore.client import FirestoreClient
from service.analysis.usage.usage_analyzer import analyze_user_trial_usage
from database.models.subscription import SubscriptionPlan

logger = logging.getLogger(__name__)
firestore_client = FirestoreClient()

async def generate_conversion_offer(user_id: str) -> Dict[str, Any]:
    """ユーザーの特性に基づいて変換オファーを生成"""

    # ユーザーの使用状況と地域
    usage = await analyze_user_trial_usage(user_id)
    user_region = await get_user_region(user_id)

    offers = {
        # 基本的なオファー（すべてのユーザーに適用可能）
        'basic': {
            'type': 'discount',
            'value': '20%',
            'duration': '最初の3ヶ月間',
            'code': 'TRIAL20',
            'expires_in_days': 7
        },

        # ヘビーユーザー向け
        'heavy_user': {
            'type': 'discount',
            'value': '25%',
            'duration': '最初の6ヶ月間',
            'code': 'POWER25',
            'expires_in_days': 7
        },

        # チームユーザー向け
        'team': {
            'type': 'seats',
            'value': '追加3ユーザー無料',
            'duration': '1年間',
            'code': 'TEAM3FREE',
            'expires_in_days': 7
        },

        # 年間契約インセンティブ
        'annual': {
            'type': 'discount',
            'value': '33%',
            'duration': '年間契約時',
            'code': 'ANNUAL33',
            'expires_in_days': 7
        }
    }

    # ユーザーの使用パターンに基づいて最適なオファーを選択
    if usage.get('session_count', 0) > 25 or usage.get('total_active_time_minutes', 0) > 300:
        offer_type = 'heavy_user'
    elif len(await get_team_members(user_id)) > 1:
        offer_type = 'team'
    else:
        # 地域に基づいて判断（特定の地域では年間契約が好まれる）
        if user_region in ['north_america', 'europe']:
            offer_type = 'annual'
        else:
            offer_type = 'basic'

    selected_offer = offers[offer_type]

    # オファーの有効期限を設定
    expiry_date = datetime.now() + timedelta(days=selected_offer['expires_in_days'])
    selected_offer['expires_at'] = expiry_date

    # オファーをデータベースに保存
    offer_data = {
        'user_id': user_id,
        'offer_type': offer_type,
        'offer_details': selected_offer,
        'created_at': datetime.now(),
        'expires_at': expiry_date,
        'status': 'active'
    }

    await firestore_client.create_document(
        collection='special_offers',
        doc_id=None,
        data=offer_data
    )

    return selected_offer

async def get_user_region(user_id: str) -> str:
    """ユーザーの地域を取得"""
    # ユーザーデータを取得
    user_data = await firestore_client.get_document(
        collection='users',
        doc_id=user_id
    )

    # デフォルト値
    default_region = 'other'

    if not user_data:
        return default_region

    # 地域情報を取得（存在する場合）
    return user_data.get('region', default_region)

async def get_team_members(user_id: str) -> list:
    """ユーザーのチームメンバーを取得"""
    try:
        # ユーザーのチーム情報を取得
        user_data = await firestore_client.get_document(
            collection='users',
            doc_id=user_id
        )

        if not user_data or 'company_id' not in user_data:
            return []

        # 同じ会社に所属するユーザーを検索
        company_id = user_data['company_id']
        team_members = await firestore_client.query_documents(
            collection='users',
            filters=[{'field': 'company_id', 'operator': '==', 'value': company_id}]
        )

        return team_members
    except Exception as e:
        logger.error(f"チームメンバー取得エラー: {str(e)}")
        return []

async def recommend_plan(user_id: str, usage_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """ユーザーの使用状況に基づいて最適なプランを推奨"""

    # 使用状況データが提供されていない場合は取得
    if usage_data is None:
        usage_data = await analyze_user_trial_usage(user_id)

    # プラン推奨ロジック
    if determine_plan_preference(usage_data) == 'premium':
        plan_id = 'premium'
    else:
        plan_id = 'basic'

    # プラン定義（実際にはデータベースから取得するべき）
    plans = {
        'basic': SubscriptionPlan(
            id="basic",
            name="ベーシックプラン",
            price_id="price_basic123",
            price=9.99,
            currency="USD",
            interval="month",
            description="基本的な分析機能",
            features=["制限付きデータ分析", "基本ダッシュボード", "週次レポート"]
        ),
        'premium': SubscriptionPlan(
            id="premium",
            name="プレミアムプラン",
            price_id="price_premium456",
            price=29.99,
            currency="USD",
            interval="month",
            description="すべての機能を含む高度な分析",
            features=["無制限のデータ分析", "高度なダッシュボード", "日次レポート", "APIアクセス"]
        )
    }

    recommended_plan = plans[plan_id]

    # 使用状況に基づいた推奨理由を生成
    reasons = generate_recommendation_reasons(usage_data, plan_id)

    return {
        'plan': recommended_plan.dict(),
        'reasons': reasons
    }

def determine_plan_preference(usage_data: Dict[str, Any]) -> str:
    """使用状況に基づいてプラン選好を判断"""

    score = 0

    # スコアリングロジック
    if usage_data.get('reports_count', 0) > 10:
        score += 3
    elif usage_data.get('reports_count', 0) > 5:
        score += 1

    if usage_data.get('widget_count', 0) > 8:
        score += 3
    elif usage_data.get('widget_count', 0) > 4:
        score += 1

    if usage_data.get('session_count', 0) > 20:
        score += 2

    if 'advanced_analytics' in usage_data.get('primary_features', []):
        score += 3

    if usage_data.get('upload_size_total', 0) > 50 * 1024 * 1024:  # 50MB
        score += 2

    # プレミアムプランの閾値
    if score >= 5:
        return 'premium'
    else:
        return 'basic'

def generate_recommendation_reasons(usage_data: Dict[str, Any], plan_id: str) -> List[str]:
    """プラン推奨の理由を生成"""
    reasons = []

    if plan_id == 'premium':
        if usage_data.get('reports_count', 0) > 10:
            reasons.append("あなたは多くのレポートを作成しており、プレミアムプランでは無制限のレポート生成が可能です")

        if usage_data.get('widget_count', 0) > 8:
            reasons.append("現在の複雑なダッシュボード構成には、プレミアムプランの高度な可視化機能が最適です")

        if 'advanced_analytics' in usage_data.get('primary_features', []):
            reasons.append("あなたは高度な分析機能を活用しており、プレミアムプランならすべての分析タイプに無制限にアクセスできます")

    else:  # basic
        if usage_data.get('reports_count', 0) <= 5:
            reasons.append("あなたの現在のレポート使用量には、ベーシックプランが最適です")

        if usage_data.get('session_count', 0) < 15:
            reasons.append("現在の使用頻度では、ベーシックプランで十分なパフォーマンスを発揮できます")

    # 金銭的価値も追加
    time_saved = usage_data.get('trial_value', {}).get('total_time_saved_hours', 0)
    if time_saved > 0:
        if plan_id == 'premium':
            reasons.append(f"トライアル期間中に{time_saved}時間の時間を節約しました。プレミアムプランでさらに生産性を向上できます")
        else:
            reasons.append(f"トライアル期間中に{time_saved}時間の時間を節約しました。ベーシックプランでこの節約を継続できます")

    return reasons