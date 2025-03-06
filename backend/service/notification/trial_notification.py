"""
トライアル終了通知とプラン案内サービス
ユーザーのトライアル期間終了に関する通知を管理するサービスです。
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
from src.database.firestore.client import FirestoreClient
from service.analysis.usage.usage_analyzer import analyze_user_trial_usage

logger = logging.getLogger(__name__)
firestore_client = FirestoreClient()

class EmailService:
    """メール送信サービス（実際の実装はプロジェクトに依存）"""

    async def send_email(self, to_email: str, subject: str, template_id: str, template_data: Dict[str, Any]):
        """
        メールを送信する

        Args:
            to_email: 宛先メールアドレス
            subject: メールの件名
            template_id: メールテンプレートID
            template_data: テンプレート用のデータ
        """
        # 実際のメール送信処理を実装
        # 例: SendGrid, SES, SMTPなどを使用
        logger.info(f"メール送信: {to_email}, 件名: {subject}, テンプレート: {template_id}")

        # この実装は仮のもので、実際のプロジェクトでは適切なメール送信サービスを使用する
        pass

# メールサービスのインスタンス
email_service = EmailService()

async def schedule_trial_notifications(user_id: str, trial_end_date: datetime):
    """ユーザーのトライアル終了通知をスケジュール"""

    now = datetime.now()
    days_remaining = (trial_end_date - now).days

    notifications = [
        # 無料トライアル（7日間）の通知スケジュール
        {'days_before': 3, 'template': 'free_trial_ending_soon'},
        {'days_before': 1, 'template': 'free_trial_last_day'},
        {'days_before': 0, 'template': 'free_trial_ended'},

        # 有料トライアル（14日間）の通知スケジュール
        {'days_before': 7, 'template': 'paid_trial_halfway'},
        {'days_before': 3, 'template': 'paid_trial_ending_soon'},
        {'days_before': 1, 'template': 'paid_trial_last_day'},
        {'days_before': 0, 'template': 'paid_trial_ended'}
    ]

    for notification in notifications:
        if days_remaining == notification['days_before']:
            await send_trial_notification(
                user_id=user_id,
                template=notification['template'],
                trial_end_date=trial_end_date
            )

async def send_trial_notification(user_id: str, template: str, trial_end_date: datetime):
    """トライアル関連の通知メールを送信"""

    # ユーザーデータの取得
    user_data = await firestore_client.get_document(
        collection='users',
        doc_id=user_id
    )

    if not user_data:
        logger.error(f"通知メール送信失敗: ユーザー {user_id} が見つかりません")
        return

    email = user_data.get('email')

    if not email:
        logger.error(f"通知メール送信失敗: ユーザー {user_id} のメールアドレスがありません")
        return

    # ユーザーの使用状況データ
    usage_data = await analyze_user_trial_usage(user_id)

    # テンプレートに基づいて適切なメール内容を構築
    email_content = await build_email_content(
        template=template,
        user=user_data,
        usage_data=usage_data,
        trial_end_date=trial_end_date
    )

    # 特典オファーの生成（必要に応じて）
    if template in ['free_trial_ended', 'paid_trial_ended', 'paid_trial_last_day']:
        from backend.service.promotion.offer_generator import generate_conversion_offer
        special_offer = await generate_conversion_offer(user_id)
        email_content['data']['special_offer'] = special_offer

    # メール送信
    await email_service.send_email(
        to_email=email,
        subject=email_content['subject'],
        template_id=email_content['template_id'],
        template_data=email_content['data']
    )

    # 送信記録の保存
    notification_data = {
        'user_id': user_id,
        'email': email,
        'template': template,
        'sent_at': datetime.now(),
        'status': 'sent'
    }

    await firestore_client.create_document(
        collection='email_notifications',
        doc_id=None,
        data=notification_data
    )

async def build_email_content(template: str, user: Dict, usage_data: Dict, trial_end_date: datetime) -> Dict:
    """テンプレートに基づいてメール内容を構築"""

    # デフォルト値
    template_map = {
        'free_trial_ending_soon': {
            'subject': '無料トライアルが終了まであと3日です',
            'template_id': 'free-trial-ending-soon',
        },
        'free_trial_last_day': {
            'subject': '無料トライアルが明日終了します',
            'template_id': 'free-trial-last-day',
        },
        'free_trial_ended': {
            'subject': '無料トライアルが終了しました',
            'template_id': 'free-trial-ended',
        },
        'paid_trial_halfway': {
            'subject': 'トライアル期間の折り返し地点です',
            'template_id': 'paid-trial-halfway',
        },
        'paid_trial_ending_soon': {
            'subject': 'トライアルが終了まであと3日です',
            'template_id': 'paid-trial-ending-soon',
        },
        'paid_trial_last_day': {
            'subject': 'トライアルが明日終了します',
            'template_id': 'paid-trial-last-day',
        },
        'paid_trial_ended': {
            'subject': 'トライアルが終了しました',
            'template_id': 'paid-trial-ended',
        },
    }

    # 基本データ
    template_data = {
        'user_name': user.get('name', ''),
        'company_name': user.get('company_name', ''),
        'trial_end_date': trial_end_date.strftime('%Y年%m月%d日'),
        'dashboard_url': 'https://your-website.com/dashboard',
        'usage_summary': {
            'reports_count': usage_data.get('reports_count', 0),
            'analyses_performed': sum(usage_data.get('analyses', {}).values()),
            'time_saved': usage_data.get('trial_value', {}).get('total_time_saved_hours', 0)
        }
    }

    # 推奨プランを追加（トライアル終了時）
    if template in ['free_trial_ended', 'paid_trial_ended', 'paid_trial_last_day']:
        from backend.service.promotion.offer_generator import recommend_plan
        recommended_plan = await recommend_plan(user['id'], usage_data)
        template_data['recommended_plan'] = recommended_plan

    # メール内容を返す
    return {
        'subject': template_map[template]['subject'],
        'template_id': template_map[template]['template_id'],
        'data': template_data
    }

async def run_trial_notification_scheduler():
    """トライアル通知のスケジューラーを実行"""

    # トライアル中のユーザーを取得
    trial_users = await firestore_client.query_documents(
        collection='users',
        filters=[
            {'field': 'tier', 'operator': 'in', 'value': ['free', 'trial']},
            {'field': 'trial_end', 'operator': '>', 'value': datetime.now() - timedelta(days=1)}
        ]
    )

    # 各ユーザーに対する通知をスケジュール
    for user in trial_users:
        await schedule_trial_notifications(
            user_id=user['id'],
            trial_end_date=user['trial_end']
        )