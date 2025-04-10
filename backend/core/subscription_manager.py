import payjp
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Firestoreサービスクラスをインポート
from service.firestore.client import FirestoreService
# 循環インポートを回避するための遅延インポート
from .patterns import LazyImport

# Auth Managerを遅延インポート
AuthManager = LazyImport('core.auth_manager', 'AuthManager')
get_auth_manager = LazyImport('core.auth_manager', 'get_auth_manager')

logger = logging.getLogger(__name__)

class SubscriptionManagerError(Exception):
    """サブスクリプションマネージャー固有のエラー"""
    pass

class SubscriptionManager:
    """Pay.jpと連携してサブスクリプションを管理するクラス"""

    def __init__(self, firestore_service: FirestoreService, auth_manager: Any):
        """コンストラクタ

        Args:
            firestore_service (FirestoreService): Firestoreとの対話用サービス
            auth_manager (AuthManager): ユーザー情報管理用マネージャー
        """
        self.firestore_service = firestore_service
        self.auth_manager = auth_manager
        self.payjp_secret_key = os.environ.get("PAYJP_SECRET_KEY")
        self.payjp_webhook_secret = os.environ.get("PAYJP_WEBHOOK_SECRET") # Webhook検証用

        if not self.payjp_secret_key:
            logger.critical("環境変数 PAYJP_SECRET_KEY が設定されていません！")
            raise SubscriptionManagerError("Pay.jpの秘密鍵が設定されていません。")
        if not self.payjp_webhook_secret:
            logger.warning("環境変数 PAYJP_WEBHOOK_SECRET が設定されていません。Webhook検証が無効になります。")
            # 本番環境ではエラーにするべき
            # raise SubscriptionManagerError("Pay.jpのWebhook秘密鍵が設定されていません。")

        payjp.api_key = self.payjp_secret_key
        self.user_collection = "users" # Firestoreのユーザーコレクション名 (仮)
        self.company_collection = "companies" # Firestoreの企業コレクション名 (仮)

    async def get_or_create_customer(self, user_id: str, email: str, metadata: Optional[Dict] = None) -> payjp.Customer:
        """Pay.jpの顧客を取得または作成する"""
        try:
            # まずFirestoreからpayjp_customer_idを検索
            user_data = await self.firestore_service.get_document(self.user_collection, user_id)
            payjp_customer_id = user_data.get("payjp_customer_id") if user_data else None

            if payjp_customer_id:
                try:
                    customer = payjp.Customer.retrieve(payjp_customer_id)
                    logger.info(f"既存のPay.jp顧客を取得しました: {payjp_customer_id} (ユーザー: {user_id})")
                    # メタデータやEmailが変更されている可能性があれば更新
                    if customer.email != email or customer.metadata != metadata:
                         payjp.Customer.modify(
                             payjp_customer_id,
                             email=email,
                             metadata=metadata if metadata else {}
                         )
                         logger.info(f"Pay.jp顧客情報を更新しました: {payjp_customer_id}")
                    return customer
                except payjp.error.InvalidRequestError as e:
                    # Pay.jp側に顧客が存在しない場合 (IDはあるが削除されたなど)
                    logger.warning(f"Firestoreに顧客ID {payjp_customer_id} がありますが、Pay.jpで見つかりません。新規作成します。エラー: {e}")
                    # FirestoreのIDをクリアする処理を追加しても良い
                    await self.firestore_service.update_document(self.user_collection, user_id, {"payjp_customer_id": None})


            # Pay.jpに顧客が存在しない場合は新規作成
            logger.info(f"Pay.jp顧客を新規作成します (ユーザー: {user_id}, Email: {email})")
            customer = payjp.Customer.create(
                email=email,
                description=f"User ID: {user_id}",
                metadata=metadata if metadata else {}
            )
            logger.info(f"Pay.jp顧客を作成しました: {customer.id} (ユーザー: {user_id})")

            # Firestoreにpayjp_customer_idを保存
            await self.firestore_service.update_document(self.user_collection, user_id, {"payjp_customer_id": customer.id})

            return customer
        except payjp.error.PayjpError as e:
            logger.error(f"Pay.jp顧客の取得/作成中にエラー: {e}", exc_info=True)
            raise SubscriptionManagerError(f"Pay.jp顧客の処理に失敗しました: {e}")
        except Exception as e:
            logger.error(f"顧客処理中に予期せぬエラー: {e}", exc_info=True)
            raise SubscriptionManagerError(f"顧客処理中に予期せぬエラーが発生しました。")

    async def list_plans(self) -> List[payjp.Plan]:
        """Pay.jpに登録されているプラン一覧を取得する"""
        try:
            logger.info("Pay.jpからプラン一覧を取得します")
            plans = payjp.Plan.list(limit=100) # 必要に応じてページネーションを実装
            logger.info(f"{len(plans.data)} 件のプランを取得しました")
            return plans.data
        except payjp.error.PayjpError as e:
            logger.error(f"Pay.jpプラン一覧の取得中にエラー: {e}", exc_info=True)
            raise SubscriptionManagerError(f"Pay.jpプラン一覧の取得に失敗しました: {e}")
        except Exception as e:
            logger.error(f"プラン一覧取得中に予期せぬエラー: {e}", exc_info=True)
            raise SubscriptionManagerError("プラン一覧取得中に予期せぬエラーが発生しました。")

    async def create_or_update_subscription(self, user_id: str, plan_id: str, payment_method_id: Optional[str] = None) -> payjp.Subscription:
        """サブスクリプションを作成または更新する

        Args:
            user_id (str): ユーザーID
            plan_id (str): Pay.jpのプランID
            payment_method_id (Optional[str]): Pay.jpの支払い方法ID (カードトークンなど)。初回登録時に必要。

        Returns:
            payjp.Subscription: 作成または更新されたサブスクリプションオブジェクト
        """
        try:
            user_data = await self.firestore_service.get_document(self.user_collection, user_id)
            if not user_data:
                raise SubscriptionManagerError(f"ユーザーが見つかりません: {user_id}")

            payjp_customer_id = user_data.get("payjp_customer_id")
            if not payjp_customer_id:
                # 顧客が存在しない場合は作成する (Email情報が必要)
                 email = user_data.get("email") # AuthManager経由で取得する方が良いかも
                 if not email:
                      raise SubscriptionManagerError(f"ユーザー {user_id} のEmailが見つかりません。顧客を作成できません。")
                 customer = await self.get_or_create_customer(user_id, email)
                 payjp_customer_id = customer.id

            current_subscription_id = user_data.get("payjp_subscription_id")

            if current_subscription_id:
                # 既存サブスクリプションを更新 (プラン変更)
                logger.info(f"既存のサブスクリプション {current_subscription_id} をプラン {plan_id} に変更します (ユーザー: {user_id})")
                subscription = payjp.Subscription.retrieve(current_subscription_id)
                subscription.plan = plan_id
                # subscription.prorate = True # 日割り計算を有効にするかなど
                subscription.save()
                logger.info(f"サブスクリプションを更新しました: {subscription.id}")
            else:
                # 新規サブスクリプションを作成
                logger.info(f"新規サブスクリプションを作成します (ユーザー: {user_id}, プラン: {plan_id})")
                if not payment_method_id:
                    # 支払い方法が指定されていない場合は、顧客のデフォルト支払い方法を使用
                    # customer = payjp.Customer.retrieve(payjp_customer_id)
                    # payment_method_id = customer.default_card # 注意: default_cardは廃止予定の可能性
                    # 代わりに SetupIntent などで事前にカード登録が必要
                    raise SubscriptionManagerError("新規サブスクリプション作成には支払い方法IDが必要です。")

                subscription = payjp.Subscription.create(
                    customer=payjp_customer_id,
                    plan=plan_id,
                    # card=payment_method_id # カードトークンを直接指定する場合 (非推奨)
                    # trial_end=... # トライアル期間を設定する場合
                )
                logger.info(f"新規サブスクリプションを作成しました: {subscription.id} (ユーザー: {user_id})")
                 # FirestoreにサブスクリプションIDを保存
                await self.firestore_service.update_document(self.user_collection, user_id, {"payjp_subscription_id": subscription.id})


            # Firestoreのプラン情報も更新
            await self.update_firestore_subscription_status(user_id, subscription)

            return subscription

        except payjp.error.PayjpError as e:
            logger.error(f"Pay.jpサブスクリプションの作成/更新中にエラー: {e}", exc_info=True)
            raise SubscriptionManagerError(f"サブスクリプションの作成/更新に失敗しました: {e}")
        except Exception as e:
            logger.error(f"サブスクリプション処理中に予期せぬエラー: {e}", exc_info=True)
            raise SubscriptionManagerError(f"サブスクリプション処理中に予期せぬエラーが発生しました。")

    async def cancel_subscription(self, user_id: str, at_period_end: bool = True) -> payjp.Subscription:
        """サブスクリプションをキャンセルする

        Args:
            user_id (str): ユーザーID
            at_period_end (bool): Trueの場合、期間終了時にキャンセル。Falseの場合、即時キャンセル。

        Returns:
            payjp.Subscription: キャンセルされたサブスクリプションオブジェクト
        """
        try:
            user_data = await self.firestore_service.get_document(self.user_collection, user_id)
            if not user_data:
                raise SubscriptionManagerError(f"ユーザーが見つかりません: {user_id}")

            subscription_id = user_data.get("payjp_subscription_id")
            if not subscription_id:
                raise SubscriptionManagerError(f"ユーザー {user_id} に有効なサブスクリプションが見つかりません。")

            logger.info(f"サブスクリプション {subscription_id} をキャンセルします (ユーザー: {user_id}, 期間終了時: {at_period_end})")
            subscription = payjp.Subscription.retrieve(subscription_id)

            if at_period_end:
                # 期間終了時にキャンセル (推奨)
                subscription.cancel_at_period_end = True
                updated_subscription = subscription.save()
                 # Firestoreのステータスを更新 (pending_cancellationなど)
                await self.firestore_service.update_document(self.user_collection, user_id, {
                    "subscription_status": "pending_cancellation",
                    "subscription_ends_at": datetime.fromtimestamp(updated_subscription.current_period_end).isoformat()
                 })

                logger.info(f"サブスクリプション {subscription_id} は期間終了時 ({updated_subscription.current_period_end}) にキャンセルされます")
                return updated_subscription
            else:
                # 即時キャンセル (データ喪失の可能性あり、通常は非推奨)
                deleted_subscription = subscription.cancel()
                 # Firestoreのステータスを更新 (cancelled) とID削除
                await self.firestore_service.update_document(self.user_collection, user_id, {
                     "payjp_subscription_id": None,
                     "subscription_plan_id": None,
                     "subscription_status": "cancelled",
                     "subscription_ends_at": datetime.now().isoformat()
                 })

                logger.info(f"サブスクリプション {subscription_id} は即時キャンセルされました")
                return deleted_subscription

        except payjp.error.InvalidRequestError as e:
             if "No such subscription" in str(e):
                  logger.warning(f"キャンセルしようとしたサブスクリプション {subscription_id} が見つかりません。Firestoreデータをクリアします。")
                  await self.firestore_service.update_document(self.user_collection, user_id, {
                     "payjp_subscription_id": None,
                     "subscription_plan_id": None,
                     "subscription_status": "error" # または cancelled
                 })
                  raise SubscriptionManagerError(f"キャンセル対象のサブスクリプションが見つかりませんでした。")
             else:
                  logger.error(f"Pay.jpサブスクリプションのキャンセル中にエラー: {e}", exc_info=True)
                  raise SubscriptionManagerError(f"サブスクリプションのキャンセルに失敗しました: {e}")
        except payjp.error.PayjpError as e:
            logger.error(f"Pay.jpサブスクリプションのキャンセル中にエラー: {e}", exc_info=True)
            raise SubscriptionManagerError(f"サブスクリプションのキャンセルに失敗しました: {e}")
        except Exception as e:
            logger.error(f"サブスクリプションキャンセル中に予期せぬエラー: {e}", exc_info=True)
            raise SubscriptionManagerError(f"サブスクリプションキャンセル中に予期せぬエラーが発生しました。")


    async def get_subscription_details(self, user_id: str) -> Optional[Dict[str, Any]]:
        """ユーザーの現在のサブスクリプション詳細を取得する (Firestore + Pay.jp)"""
        try:
            user_data = await self.firestore_service.get_document(self.user_collection, user_id)
            if not user_data:
                logger.warning(f"サブスクリプション詳細取得: ユーザー {user_id} が見つかりません")
                return None

            subscription_id = user_data.get("payjp_subscription_id")
            firestore_status = user_data.get("subscription_status")
            firestore_plan_id = user_data.get("subscription_plan_id")

            if not subscription_id:
                logger.info(f"ユーザー {user_id} はアクティブなPay.jpサブスクリプションを持っていません (Firestore)")
                # Firestoreにステータスがあればそれを返す
                return {
                    "status": firestore_status or "inactive",
                    "plan_id": firestore_plan_id,
                    "source": "firestore"
                }

            try:
                # Pay.jpから最新情報を取得
                subscription = payjp.Subscription.retrieve(subscription_id)
                logger.info(f"Pay.jpからサブスクリプション {subscription_id} の詳細を取得しました")

                # Firestoreも更新しておく
                await self.update_firestore_subscription_status(user_id, subscription)

                return {
                    "id": subscription.id,
                    "plan_id": subscription.plan.id,
                    "plan_name": subscription.plan.name,
                    "status": subscription.status, # active, canceled, trial, paused
                    "current_period_start": datetime.fromtimestamp(subscription.current_period_start),
                    "current_period_end": datetime.fromtimestamp(subscription.current_period_end),
                    "cancel_at_period_end": subscription.cancel_at_period_end,
                    "trial_end": datetime.fromtimestamp(subscription.trial_end) if subscription.trial_end else None,
                    "source": "payjp"
                }
            except payjp.error.InvalidRequestError as e:
                 logger.warning(f"サブスクリプション {subscription_id} がPay.jpで見つかりません (ユーザー: {user_id})。Firestoreデータを更新します。エラー: {e}")
                 await self.firestore_service.update_document(self.user_collection, user_id, {
                     "payjp_subscription_id": None,
                     "subscription_plan_id": None,
                     "subscription_status": "error" # または inactive
                 })
                 return {
                    "status": "error",
                    "message": "Pay.jp subscription not found.",
                    "source": "error"
                 }
            except payjp.error.PayjpError as e:
                 logger.error(f"Pay.jpからのサブスクリプション詳細取得エラー: {e}")
                 # Firestoreのキャッシュ情報を返すことも検討
                 return {
                    "status": firestore_status or "unknown",
                    "plan_id": firestore_plan_id,
                    "source": "firestore_cache",
                    "error": str(e)
                 }

        except Exception as e:
            logger.error(f"サブスクリプション詳細取得中に予期せぬエラー: {e}", exc_info=True)
            raise SubscriptionManagerError("サブスクリプション詳細取得中に予期せぬエラーが発生しました。")


    async def handle_webhook(self, payload: bytes, signature: str) -> bool:
        """Pay.jp Webhookイベントを処理する"""
        if not self.payjp_webhook_secret:
             logger.error("Webhook秘密鍵が未設定のため、Webhookを検証できません。")
             return False # 本番では検証失敗として扱うべき

        try:
            event = payjp.Webhook.construct_event(
                payload, signature, self.payjp_webhook_secret
            )
            logger.info(f"Pay.jp Webhook受信: イベントタイプ {event.type}, ID: {event.id}")

            # イベントタイプに応じて処理を分岐
            event_type = event.type
            event_data = event.data.object # イベントに関連するオブジェクト (Charge, Subscriptionなど)

            user_id = None
            customer_id = event_data.get("customer")

            if customer_id:
                 # customer_idからuser_idを特定 (Firestoreを検索)
                 # この検索が効率的でない場合、Webhookイベントのメタデータにuser_idを含めることを検討
                 user_docs = await self.firestore_service.query_documents(
                      self.user_collection,
                      filters=[("payjp_customer_id", "==", customer_id)],
                      limit=1
                 )
                 if user_docs:
                      user_id = user_docs[0].get("id") # ドキュメントIDをuser_idとする場合
                      logger.info(f"Webhook: 顧客ID {customer_id} に対応するユーザー {user_id} を特定しました")
                 else:
                      logger.warning(f"Webhook: 顧客ID {customer_id} に対応するユーザーが見つかりません。イベント処理をスキップします。")
                      return True # イベント自体は正常に受信したが、対象ユーザー不明

            # --- イベント処理の例 ---
            if event_type == 'charge.succeeded':
                # 支払い成功
                logger.info(f"Webhook: 支払い成功 (Charge ID: {event_data.id}, Customer: {customer_id})")
                # 必要に応じて領収書発行やサービスアクセス権更新など
                pass

            elif event_type == 'charge.failed':
                # 支払い失敗
                logger.error(f"Webhook: 支払い失敗 (Charge ID: {event_data.id}, Customer: {customer_id}, Reason: {event_data.failure_message})")
                # ユーザーへの通知、サブスクリプションの停止処理など
                if user_id:
                    # 例: 支払い失敗回数を記録、ステータス変更
                     await self.firestore_service.update_document(self.user_collection, user_id, {"payment_failed_count": payjp.firestore.Increment(1)})
                pass

            elif event_type.startswith('customer.subscription.'):
                 # サブスクリプション関連イベント (created, updated, deleted)
                 subscription = event_data # event_data自体がSubscriptionオブジェクト
                 logger.info(f"Webhook: サブスクリプションイベント {event_type} (Subscription ID: {subscription.id}, Customer: {customer_id})")
                 if user_id:
                      await self.update_firestore_subscription_status(user_id, subscription)

            elif event_type == 'invoice.payment_succeeded':
                 # 定期課金の請求書支払い成功
                 logger.info(f"Webhook: 定期課金支払い成功 (Invoice ID: {event_data.id}, Subscription: {event_data.subscription})")
                 # サブスクリプションの更新は customer.subscription.updated で処理されることが多い
                 pass

            elif event_type == 'invoice.payment_failed':
                 # 定期課金の請求書支払い失敗
                 logger.error(f"Webhook: 定期課金支払い失敗 (Invoice ID: {event_data.id}, Subscription: {event_data.subscription})")
                 # ユーザー通知、サブスクリプション停止処理
                 if user_id and event_data.subscription:
                      # サブスクリプションステータスを更新 (例: past_due)
                       sub_obj = payjp.Subscription.retrieve(event_data.subscription) # 最新状態を取得
                       await self.update_firestore_subscription_status(user_id, sub_obj)
                 pass

            # 他の必要なイベントタイプを追加...
            # 'customer.created', 'customer.updated', 'customer.deleted'
            # 'plan.created', 'plan.updated', 'plan.deleted'

            else:
                logger.info(f"Webhook: 未処理のイベントタイプ {event_type}")

            return True # イベント処理成功

        except ValueError as e:
            # 不正なペイロード
            logger.error(f"Webhookエラー: 不正なペイロード - {e}")
            return False
        except payjp.error.SignatureVerificationError as e:
            # 不正な署名
            logger.error(f"Webhookエラー: 署名検証失敗 - {e}")
            return False
        except payjp.error.PayjpError as e:
             logger.error(f"Webhook処理中のPay.jpエラー: {e}", exc_info=True)
             return False # エラー発生時は再送を期待してFalseを返すか検討
        except Exception as e:
            logger.error(f"Webhook処理中に予期せぬエラー: {e}", exc_info=True)
            return False # エラー発生時は再送を期待してFalseを返すか検討


    async def update_firestore_subscription_status(self, user_id: str, subscription: payjp.Subscription):
        """Pay.jpのSubscriptionオブジェクトに基づいてFirestoreのユーザードキュメントを更新する"""
        try:
            update_data = {
                "payjp_subscription_id": subscription.id,
                "subscription_plan_id": subscription.plan.id,
                "subscription_status": subscription.status, # active, trial, canceled, etc.
                "subscription_started_at": datetime.fromtimestamp(subscription.created).isoformat(),
                "subscription_current_period_start": datetime.fromtimestamp(subscription.current_period_start).isoformat(),
                "subscription_current_period_end": datetime.fromtimestamp(subscription.current_period_end).isoformat(),
                "subscription_cancel_at_period_end": subscription.cancel_at_period_end,
                "subscription_ended_at": datetime.fromtimestamp(subscription.canceled_at).isoformat() if subscription.canceled_at else None,
                 # トライアル情報
                 "subscription_trial_start": datetime.fromtimestamp(subscription.trial_start).isoformat() if subscription.trial_start else None,
                 "subscription_trial_end": datetime.fromtimestamp(subscription.trial_end).isoformat() if subscription.trial_end else None,
            }
            # Noneの値を削除 (Firestoreでフィールドを削除するため)
            update_data = {k: v for k, v in update_data.items() if v is not None}

            await self.firestore_service.update_document(self.user_collection, user_id, update_data)
            logger.info(f"Firestoreのユーザー {user_id} のサブスクリプション情報を更新しました (ステータス: {subscription.status})")

        except Exception as e:
             logger.error(f"Firestoreのサブスクリプション情報更新中にエラー (ユーザー: {user_id}, Sub ID: {subscription.id}): {e}", exc_info=True)
             # ここでのエラーはWebhook処理などに影響するため、注意深く扱う


# シングルトンインスタンスを取得する関数 (依存性注入用)
_subscription_manager_instance = None

def get_subscription_manager(
    firestore_service: Optional[FirestoreService] = None,
    auth_manager: Optional[Any] = None
) -> SubscriptionManager:
    """SubscriptionManagerのシングルトンインスタンスを取得"""
    global _subscription_manager_instance
    if _subscription_manager_instance is None:
        if not firestore_service:
            from service.firestore.client import FirestoreService
            firestore_service = FirestoreService()
            logger.warning("SubscriptionManager用にデフォルトのFirestoreServiceを使用します")
        if not auth_manager:
             # 遅延インポートしたget_auth_managerを使用してAuthManagerを取得
             auth_manager = get_auth_manager()
             logger.info("SubscriptionManager用にAuthManagerを取得しました")

        try:
            _subscription_manager_instance = SubscriptionManager(firestore_service, auth_manager)
            logger.info("SubscriptionManagerインスタンスが作成されました")
        except SubscriptionManagerError as e:
             logger.critical(f"SubscriptionManagerの初期化に失敗: {e}")
             # アプリケーションの起動を停止するなどの処理が必要
             raise
    return _subscription_manager_instance