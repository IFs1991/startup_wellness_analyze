import stripe
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class StripeService:
    def __init__(self, api_key: Optional[str] = None):
        """
        Stripeサービスを初期化

        Args:
            api_key: Stripe APIキー（デフォルトでは環境変数から取得）
        """
        try:
            self.api_key = api_key or os.getenv("STRIPE_API_KEY")
            if not self.api_key:
                raise ValueError("Stripe APIキーが設定されていません")

            stripe.api_key = self.api_key
            logger.info("Stripeサービスが正常に初期化されました")
        except Exception as e:
            logger.error(f"Stripeサービスの初期化に失敗しました: {str(e)}")
            raise

    async def create_customer(self, email: str, name: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        新しいStripeカスタマーを作成

        Args:
            email: カスタマーのメールアドレス
            name: カスタマーの名前
            metadata: 追加のメタデータ

        Returns:
            Dict: Stripeカスタマーオブジェクト
        """
        try:
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata=metadata or {}
            )
            return customer
        except Exception as e:
            logger.error(f"Stripeカスタマーの作成エラー: {str(e)}")
            raise

    async def create_subscription(
        self,
        customer_id: str,
        price_id: str,
        trial_end: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        カスタマーのサブスクリプションを作成

        Args:
            customer_id: Stripeカスタマーのid
            price_id: Stripe価格のid
            trial_end: トライアル終了のタイムスタンプ（Noneで無料トライアルなし）
            metadata: 追加のメタデータ

        Returns:
            Dict: Stripeサブスクリプションオブジェクト
        """
        try:
            subscription = stripe.Subscription.create(
                customer=customer_id,
                items=[{"price": price_id}],
                trial_end=trial_end,
                metadata=metadata or {}
            )
            return subscription
        except Exception as e:
            logger.error(f"サブスクリプション作成エラー: {str(e)}")
            raise

    async def create_checkout_session(
        self,
        price_id: str,
        customer_id: Optional[str] = None,
        success_url: str = "https://your-website.com/success",
        cancel_url: str = "https://your-website.com/cancel",
        trial_days: int = 0
    ) -> Dict[str, Any]:
        """
        サブスクリプション登録用のチェックアウトセッションを作成

        Args:
            price_id: Stripe価格ID
            customer_id: オプションのStripeカスタマーID
            success_url: 支払い成功時のリダイレクトURL
            cancel_url: ユーザーがキャンセルした場合のリダイレクトURL
            trial_days: トライアル期間の日数（0でトライアルなし）

        Returns:
            Dict: チェックアウトセッションの詳細
        """
        try:
            subscription_data = {"items": [{"price": price_id}]}

            if trial_days > 0:
                subscription_data["trial_period_days"] = trial_days

            session = stripe.checkout.Session.create(
                customer=customer_id,
                success_url=success_url,
                cancel_url=cancel_url,
                payment_method_types=["card"],
                mode="subscription",
                subscription_data=subscription_data
            )
            return session
        except Exception as e:
            logger.error(f"チェックアウトセッション作成エラー: {str(e)}")
            raise

    async def retrieve_subscription(self, subscription_id: str) -> Dict[str, Any]:
        """
        サブスクリプション情報を取得

        Args:
            subscription_id: StripeサブスクリプションID

        Returns:
            Dict: サブスクリプション詳細
        """
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            return subscription
        except Exception as e:
            logger.error(f"サブスクリプション取得エラー: {str(e)}")
            raise

    async def cancel_subscription(self, subscription_id: str, at_period_end: bool = True) -> Dict[str, Any]:
        """
        サブスクリプションをキャンセル

        Args:
            subscription_id: StripeサブスクリプションID
            at_period_end: 期間終了時にキャンセルするかどうか

        Returns:
            Dict: 更新されたサブスクリプション
        """
        try:
            subscription = stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=at_period_end
            )
            return subscription
        except Exception as e:
            logger.error(f"サブスクリプションキャンセルエラー: {str(e)}")
            raise