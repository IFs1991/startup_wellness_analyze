"""
Stripe決済サービスクライアント

このモジュールはStripe APIとの通信を管理し、決済処理、サブスクリプション管理、
および顧客情報の処理を行います。
"""

import stripe
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class StripeClient:
    def __init__(self, api_key: Optional[str] = None, webhook_secret: Optional[str] = None):
        """
        Stripeクライアントを初期化

        Args:
            api_key: Stripe APIキー（デフォルトでは環境変数から取得）
            webhook_secret: Stripeウェブフックシークレット
        """
        try:
            self.api_key = api_key or os.getenv("STRIPE_API_KEY")
            if not self.api_key:
                logger.warning("Stripe APIキーが設定されていません")

            stripe.api_key = self.api_key
            self.webhook_secret = webhook_secret or os.getenv("STRIPE_WEBHOOK_SECRET")
            logger.info("Stripeクライアントが正常に初期化されました")
        except Exception as e:
            logger.error(f"Stripeクライアントの初期化に失敗しました: {str(e)}")
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

    async def cancel_subscription(self, subscription_id: str, at_period_end: bool = True) -> Dict[str, Any]:
        """
        サブスクリプションをキャンセル

        Args:
            subscription_id: キャンセルするサブスクリプションのID
            at_period_end: 現在の期間終了時にキャンセルするかどうか（Trueの場合は期間終了時、Falseの場合は即時）

        Returns:
            Dict: 更新されたサブスクリプションオブジェクト
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

    async def verify_webhook_signature(self, payload: bytes, sig_header: str) -> Dict[str, Any]:
        """
        Stripeウェブフックの署名を検証

        Args:
            payload: リクエストボディ
            sig_header: Stripeシグネチャヘッダー

        Returns:
            Dict: 検証されたイベントオブジェクト
        """
        try:
            if not self.webhook_secret:
                raise ValueError("ウェブフックシークレットが設定されていません")

            event = stripe.Webhook.construct_event(
                payload, sig_header, self.webhook_secret
            )
            return event
        except Exception as e:
            logger.error(f"ウェブフック検証エラー: {str(e)}")
            raise

    async def get_subscription(self, subscription_id: str) -> Dict[str, Any]:
        """
        サブスクリプションの詳細を取得

        Args:
            subscription_id: 取得するサブスクリプションのID

        Returns:
            Dict: サブスクリプションオブジェクト
        """
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            return subscription
        except Exception as e:
            logger.error(f"サブスクリプション取得エラー: {str(e)}")
            raise

    async def list_invoices(self, customer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        顧客の請求書一覧を取得

        Args:
            customer_id: Stripeカスタマーのid
            limit: 取得する請求書の最大数

        Returns:
            List[Dict]: 請求書オブジェクトのリスト
        """
        try:
            invoices = stripe.Invoice.list(
                customer=customer_id,
                limit=limit
            )
            return invoices.data
        except Exception as e:
            logger.error(f"請求書リスト取得エラー: {str(e)}")
            raise

    async def get_payment_method(self, payment_method_id: str) -> Dict[str, Any]:
        """
        支払い方法の詳細を取得

        Args:
            payment_method_id: 支払い方法のID

        Returns:
            Dict: 支払い方法オブジェクト
        """
        try:
            payment_method = stripe.PaymentMethod.retrieve(payment_method_id)
            return payment_method
        except Exception as e:
            logger.error(f"支払い方法取得エラー: {str(e)}")
            raise