"""
サブスクリプション関連のAPIエンドポイント
サブスクリプション登録、プラン取得、トライアル登録などの機能を提供します。
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, EmailStr
import os
import time
from datetime import datetime, timedelta
import stripe
from src.database.firestore.client import FirestoreClient
from src.payment.stripe_client import StripeClient

# 環境に応じて適切なパスからインポート
try:
    # ローカル環境 - 相対インポート
    from ..database.models.subscription import SubscriptionModel, SubscriptionPlan
except (ImportError, ValueError):
    try:
        # ローカル環境 - 絶対インポート
        from backend.database.models.subscription import SubscriptionModel, SubscriptionPlan
    except ImportError:
        try:
            # Dockerコンテナ内
            from database.models.subscription import SubscriptionModel, SubscriptionPlan
        except ImportError:
            # 最後の手段
            import sys
            import os
            models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'models')
            if models_path not in sys.path:
                sys.path.append(models_path)
            from database.models.subscription import SubscriptionModel, SubscriptionPlan

# 認証関連の依存関係をインポート
# auth_router.pyで定義されている関数を使う想定
from routers.auth_router import get_current_user
router = APIRouter(
    prefix="/subscriptions",
    tags=["subscriptions"],
    responses={404: {"description": "Not found"}}
)
# サービスの初期化
firestore_client = FirestoreClient()
stripe_service = StripeClient()
# リクエスト/レスポンスモデル
class UserRegistration(BaseModel):
    """ユーザー登録リクエストモデル"""
    email: EmailStr
    password: str
    name: str
    company_name: Optional[str] = None
# プラン設定
SUBSCRIPTION_PLANS = {
    "basic": SubscriptionPlan(
        id="basic",
        name="ベーシックプラン",
        price_id="price_basic123",  # 実際のStripeの価格IDに置き換える
        price=9.99,
        currency="USD",
        interval="month",
        description="基本的な分析機能",
        features=["制限付きデータ分析", "基本ダッシュボード", "週次レポート"]
    ),
    "premium": SubscriptionPlan(
        id="premium",
        name="プレミアムプラン",
        price_id="price_premium456",  # 実際のStripeの価格IDに置き換える
        price=29.99,
        currency="USD",
        interval="month",
        description="すべての機能を含む高度な分析",
        features=["無制限のデータ分析", "高度なダッシュボード", "日次レポート", "APIアクセス"]
    )
}
@router.get("/plans")
async def get_subscription_plans():
    """利用可能なサブスクリプションプランを取得"""
    return list(SUBSCRIPTION_PLANS.values())
@router.post("/register/free-trial")
async def register_free_trial(user_data: UserRegistration):
    """無料トライアルに登録（クレジットカード不要）"""
    try:
        # Firebase Authでユーザー作成（実際の実装はauth_routerに依存）
        from routers.auth_router import create_firebase_user
        user = await create_firebase_user(user_data.email, user_data.password)
        # 無料プランのユーザーデータを作成
        user_profile = {
            'uid': user.uid,
            'email': user_data.email,
            'name': user_data.name,
            'company_name': user_data.company_name,
            'tier': 'free',
            'trial_start': datetime.now(),
            'trial_end': datetime.now() + timedelta(days=7),  # 7日間の無料トライアル
            'created_at': datetime.now()
        }
        # Firestoreに保存
        await firestore_client.create_document(
            collection='users',
            doc_id=user.uid,
            data=user_profile
        )
        return {"status": "success", "message": "無料トライアルにご登録いただきました"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"登録処理に失敗しました: {str(e)}"
        )
@router.post("/checkout")
async def create_checkout_session(
    plan_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Stripeのチェックアウトセッションを作成（有料トライアル向け）"""
    try:
        # プランの検証
        if plan_id not in SUBSCRIPTION_PLANS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"無効なプランID: {plan_id}"
            )
        plan = SUBSCRIPTION_PLANS[plan_id]
        # ユーザーのStripeカスタマーIDを取得または作成
        user_doc = await firestore_client.get_document(
            collection='users',
            doc_id=current_user['uid']
        )
        stripe_customer_id = None
        if user_doc and 'stripe_customer_id' in user_doc:
            stripe_customer_id = user_doc['stripe_customer_id']
        else:
            # Stripeカスタマーを作成
            customer = await stripe_service.create_customer(
                email=current_user.get('email', ''),
                name=current_user.get('name', ''),
                metadata={'firebase_uid': current_user['uid']}
            )
            stripe_customer_id = customer['id']
            # ユーザー情報を更新
            await firestore_client.update_document(
                collection='users',
                doc_id=current_user['uid'],
                data={'stripe_customer_id': stripe_customer_id}
            )
        # 14日間の有料トライアル付きチェックアウトセッションを作成
        session = await stripe_service.create_checkout_session(
            price_id=plan.price_id,
            customer_id=stripe_customer_id,
            success_url=f"https://your-website.com/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url="https://your-website.com/plans",
            trial_days=14  # 14日間の有料トライアル
        )
        # ユーザープロフィールを更新
        await firestore_client.update_document(
            collection='users',
            doc_id=current_user['uid'],
            data={
                'stripe_customer_id': stripe_customer_id,
                'tier': 'trial'
            }
        )
        return {"checkout_url": session['url']}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"チェックアウトセッション作成失敗: {str(e)}"
        )
@router.post("/webhook", include_in_schema=False)
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks):
    """StripeウェブフックエンドポイントでイベントをハンドリングするAPI"""
    # Webhookシークレットの取得
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    try:
        # リクエストボディを取得
        payload = await request.body()
        sig_header = request.headers.get('stripe-signature')
        # イベントを検証
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
        # イベントタイプに応じた処理
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            background_tasks.add_task(handle_successful_subscription, session)
        elif event['type'] == 'customer.subscription.updated':
            subscription = event['data']['object']
            background_tasks.add_task(update_subscription_status, subscription)
        elif event['type'] == 'customer.subscription.deleted':
            subscription = event['data']['object']
            background_tasks.add_task(handle_subscription_deleted, subscription)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Webhookエラー: {str(e)}"
        )
@router.get("/status")
async def get_subscription_status(current_user: dict = Depends(get_current_user)):
    """ユーザーのサブスクリプション状態を取得"""
    try:
        # ユーザーのサブスクリプションを検索
        subscriptions = await firestore_client.query_documents(
            collection='subscriptions',
            filters=[
                {'field': 'user_id', 'operator': '==', 'value': current_user['uid']}
            ],
            limit=1
        )
        if not subscriptions:
            # サブスクリプションがない場合はユーザー情報だけを返す
            user_doc = await firestore_client.get_document(
                collection='users',
                doc_id=current_user['uid']
            )
            if not user_doc:
                raise HTTPException(status_code=404, detail="ユーザーが見つかりません")
            return {
                "has_subscription": False,
                "tier": user_doc.get('tier', 'free'),
                "trial_end": user_doc.get('trial_end')
            }
        # サブスクリプション情報を返す
        subscription = subscriptions[0]
        return {
            "has_subscription": True,
            "subscription": SubscriptionModel(**subscription)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"サブスクリプション情報取得エラー: {str(e)}"
        )
@router.post("/cancel")
async def cancel_subscription(
    subscription_id: str,
    at_period_end: bool = True,
    current_user: dict = Depends(get_current_user)
):
    """ユーザーのサブスクリプションをキャンセル"""
    try:
        # サブスクリプションの所有者を確認
        subscription_docs = await firestore_client.query_documents(
            collection='subscriptions',
            filters=[
                {'field': 'stripe_subscription_id', 'operator': '==', 'value': subscription_id},
                {'field': 'user_id', 'operator': '==', 'value': current_user['uid']}
            ],
            limit=1
        )
        if not subscription_docs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="指定されたサブスクリプションが見つかりません"
            )
        # Stripeでキャンセル処理
        result = await stripe_service.cancel_subscription(
            subscription_id=subscription_id,
            at_period_end=at_period_end
        )
        # データベースを更新
        subscription_doc = subscription_docs[0]
        await firestore_client.update_document(
            collection='subscriptions',
            doc_id=subscription_doc['id'],
            data={
                'status': result['status'],
                'cancel_at': datetime.fromtimestamp(result['cancel_at']) if result.get('cancel_at') else None,
                'canceled_at': datetime.fromtimestamp(result['canceled_at']) if result.get('canceled_at') else None,
                'updated_at': datetime.now()
            }
        )
        return {"status": "success", "message": "サブスクリプションがキャンセルされました"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"サブスクリプションキャンセルエラー: {str(e)}"
        )
# バックグラウンドタスク処理関数
async def handle_successful_subscription(session: Dict[str, Any]):
    """サブスクリプション開始処理"""
    try:
        subscription_id = session['subscription']
        customer_id = session['customer']
        # サブスクリプション詳細を取得
        subscription = await stripe_service.retrieve_subscription(subscription_id)
        # ユーザーを取得
        users = await firestore_client.query_documents(
            collection='users',
            filters=[{'field': 'stripe_customer_id', 'operator': '==', 'value': customer_id}],
            limit=1
        )
        if not users:
            # エラーログを記録
            print(f"StripeカスタマーIDに対応するユーザーが見つかりません: {customer_id}")
            return
        user_id = users[0]['id']
        # サブスクリプションデータを保存
        subscription_data = {
            'user_id': user_id,
            'stripe_customer_id': customer_id,
            'stripe_subscription_id': subscription_id,
            'plan_type': subscription['items']['data'][0]['price']['metadata'].get('plan_type', 'basic'),
            'status': subscription['status'],
            'trial_start': datetime.fromtimestamp(subscription['trial_start']) if subscription.get('trial_start') else None,
            'trial_end': datetime.fromtimestamp(subscription['trial_end']) if subscription.get('trial_end') else None,
            'current_period_start': datetime.fromtimestamp(subscription['current_period_start']),
            'current_period_end': datetime.fromtimestamp(subscription['current_period_end']),
            'cancel_at': datetime.fromtimestamp(subscription['cancel_at']) if subscription.get('cancel_at') else None,
            'canceled_at': datetime.fromtimestamp(subscription['canceled_at']) if subscription.get('canceled_at') else None,
            'created_at': datetime.now()
        }
        # サブスクリプションをFirestoreに保存
        await firestore_client.create_document(
            collection='subscriptions',
            doc_id=None,  # 自動ID生成
            data=subscription_data
        )
        # ユーザー情報を更新
        await firestore_client.update_document(
            collection='users',
            doc_id=user_id,
            data={
                'tier': 'paid' if subscription['status'] == 'active' else 'trial',
                'updated_at': datetime.now()
            }
        )
    except Exception as e:
        # エラーログを記録
        print(f"サブスクリプション処理エラー: {str(e)}")
async def update_subscription_status(subscription: Dict[str, Any]):
    """サブスクリプションステータス更新処理"""
    try:
        subscription_id = subscription['id']
        # Firestoreのサブスクリプションを検索
        subscription_docs = await firestore_client.query_documents(
            collection='subscriptions',
            filters=[{'field': 'stripe_subscription_id', 'operator': '==', 'value': subscription_id}],
            limit=1
        )
        if not subscription_docs:
            print(f"サブスクリプションがFirestoreに見つかりません: {subscription_id}")
            return
        subscription_doc = subscription_docs[0]
        # 更新データを準備
        update_data = {
            'status': subscription['status'],
            'current_period_start': datetime.fromtimestamp(subscription['current_period_start']),
            'current_period_end': datetime.fromtimestamp(subscription['current_period_end']),
            'cancel_at': datetime.fromtimestamp(subscription['cancel_at']) if subscription.get('cancel_at') else None,
            'canceled_at': datetime.fromtimestamp(subscription['canceled_at']) if subscription.get('canceled_at') else None,
            'updated_at': datetime.now()
        }
        # Firestoreを更新
        await firestore_client.update_document(
            collection='subscriptions',
            doc_id=subscription_doc['id'],
            data=update_data
        )
        # ユーザーのtierも更新
        user_id = subscription_doc['user_id']
        new_tier = 'paid' if subscription['status'] == 'active' else 'trial'
        await firestore_client.update_document(
            collection='users',
            doc_id=user_id,
            data={
                'tier': new_tier,
                'updated_at': datetime.now()
            }
        )
    except Exception as e:
        print(f"サブスクリプション更新エラー: {str(e)}")
async def handle_subscription_deleted(subscription: Dict[str, Any]):
    """サブスクリプション削除処理"""
    try:
        subscription_id = subscription['id']
        # Firestoreのサブスクリプションを検索
        subscription_docs = await firestore_client.query_documents(
            collection='subscriptions',
            filters=[{'field': 'stripe_subscription_id', 'operator': '==', 'value': subscription_id}],
            limit=1
        )
        if not subscription_docs:
            print(f"サブスクリプションがFirestoreに見つかりません: {subscription_id}")
            return
        subscription_doc = subscription_docs[0]
        # 更新データを準備
        update_data = {
            'status': 'canceled',
            'canceled_at': datetime.now(),
            'updated_at': datetime.now()
        }
        # Firestoreを更新
        await firestore_client.update_document(
            collection='subscriptions',
            doc_id=subscription_doc['id'],
            data=update_data
        )
        # ユーザーのtierも更新
        user_id = subscription_doc['user_id']
        await firestore_client.update_document(
            collection='users',
            doc_id=user_id,
            data={
                'tier': 'free',
                'updated_at': datetime.now()
            }
        )
    except Exception as e:
        print(f"サブスクリプション削除エラー: {str(e)}")