"""
トライアル期間中のユーザー使用状況分析サービス
ユーザーのトライアル期間中の使用パターンを分析し、最適なプラン提案やコンバージョン戦略に活用します。
"""
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging
from src.database.firestore.client import FirestoreClient

logger = logging.getLogger(__name__)
firestore_client = FirestoreClient()

async def analyze_user_trial_usage(user_id: str) -> Dict[str, Any]:
    """ユーザーのトライアル期間中の使用状況を分析"""

    # 各機能の使用状況を集計
    usage_metrics = {}

    # レポート生成回数
    reports = await count_user_generated_reports(user_id)
    usage_metrics['reports_count'] = reports

    # データアップロード量
    uploads = await get_user_data_uploads(user_id)
    usage_metrics['upload_count'] = len(uploads)
    usage_metrics['upload_size_total'] = sum(upload.get('size', 0) for upload in uploads)

    # 分析実行数（タイプ別）
    analyses = await get_user_analyses(user_id)
    analysis_types = {}
    for analysis in analyses:
        analysis_type = analysis.get('type', 'unknown')
        analysis_types[analysis_type] = analysis_types.get(analysis_type, 0) + 1
    usage_metrics['analyses'] = analysis_types

    # ダッシュボード使用状況
    dashboard_usage = await get_user_dashboard_usage(user_id)
    usage_metrics['dashboard_views'] = dashboard_usage.get('views', 0)
    usage_metrics['widget_count'] = dashboard_usage.get('widget_count', 0)

    # ユーザーセッション数とアクティブ時間
    sessions = await get_user_sessions(user_id)
    usage_metrics['session_count'] = len(sessions)
    usage_metrics['total_active_time_minutes'] = sum(session.get('duration_minutes', 0) for session in sessions)

    # 主要な使用パターンの識別
    usage_metrics['primary_features'] = identify_primary_features(usage_metrics)

    # トライアルの価値計算
    usage_metrics['trial_value'] = calculate_trial_value(usage_metrics)

    return usage_metrics

async def count_user_generated_reports(user_id: str) -> int:
    """ユーザーが生成したレポート数をカウント"""
    try:
        reports = await firestore_client.query_documents(
            collection='reports',
            filters=[{'field': 'user_id', 'operator': '==', 'value': user_id}]
        )
        return len(reports)
    except Exception as e:
        logger.error(f"レポート数取得エラー: {str(e)}")
        return 0

async def get_user_data_uploads(user_id: str) -> List[Dict[str, Any]]:
    """ユーザーがアップロードしたデータを取得"""
    try:
        uploads = await firestore_client.query_documents(
            collection='data_uploads',
            filters=[{'field': 'user_id', 'operator': '==', 'value': user_id}]
        )
        return uploads
    except Exception as e:
        logger.error(f"データアップロード取得エラー: {str(e)}")
        return []

async def get_user_analyses(user_id: str) -> List[Dict[str, Any]]:
    """ユーザーが実行した分析を取得"""
    try:
        analyses = await firestore_client.query_documents(
            collection='analyses',
            filters=[{'field': 'user_id', 'operator': '==', 'value': user_id}]
        )
        return analyses
    except Exception as e:
        logger.error(f"分析データ取得エラー: {str(e)}")
        return []

async def get_user_dashboard_usage(user_id: str) -> Dict[str, Any]:
    """ユーザーのダッシュボード使用状況を取得"""
    try:
        # ダッシュボード閲覧数
        views = await firestore_client.query_documents(
            collection='dashboard_views',
            filters=[{'field': 'user_id', 'operator': '==', 'value': user_id}]
        )

        # ウィジェット数
        widgets = await firestore_client.query_documents(
            collection='dashboard_widgets',
            filters=[{'field': 'user_id', 'operator': '==', 'value': user_id}]
        )

        return {
            'views': len(views),
            'widget_count': len(widgets)
        }
    except Exception as e:
        logger.error(f"ダッシュボード使用状況取得エラー: {str(e)}")
        return {'views': 0, 'widget_count': 0}

async def get_user_sessions(user_id: str) -> List[Dict[str, Any]]:
    """ユーザーのセッション履歴を取得"""
    try:
        # 直近30日間のセッション
        thirty_days_ago = datetime.now() - timedelta(days=30)
        sessions = await firestore_client.query_documents(
            collection='user_sessions',
            filters=[
                {'field': 'user_id', 'operator': '==', 'value': user_id},
                {'field': 'timestamp', 'operator': '>=', 'value': thirty_days_ago}
            ]
        )
        return sessions
    except Exception as e:
        logger.error(f"セッションデータ取得エラー: {str(e)}")
        return []

def identify_primary_features(usage_metrics: Dict[str, Any]) -> List[str]:
    """ユーザーが最も利用している機能を特定"""
    primary_features = []

    # レポート使用の評価
    if usage_metrics.get('reports_count', 0) > 3:
        primary_features.append('reporting')

    # 分析使用の評価
    analyses = usage_metrics.get('analyses', {})
    if analyses.get('time_series', 0) > 2:
        primary_features.append('time_series_analysis')
    if analyses.get('correlation', 0) > 2:
        primary_features.append('correlation_analysis')
    if analyses.get('pca', 0) > 1:
        primary_features.append('advanced_analytics')

    # ダッシュボード使用の評価
    if usage_metrics.get('widget_count', 0) > 5:
        primary_features.append('dashboarding')

    # データ量の評価
    if usage_metrics.get('upload_size_total', 0) > 10 * 1024 * 1024:  # 10MB
        primary_features.append('data_storage')

    # セッション数の評価
    if usage_metrics.get('session_count', 0) > 10:
        primary_features.append('regular_usage')

    return primary_features

def calculate_trial_value(usage_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """トライアル期間中に生成された価値を計算"""

    value_metrics = {}

    # 作成されたレポート数
    reports = usage_metrics.get('reports_count', 0)
    value_metrics['reports_saved'] = reports
    value_metrics['reports_value'] = reports * 30  # 1レポートあたり30分の時間節約と仮定

    # 実行された分析数
    analyses = usage_metrics.get('analyses', {})
    total_analyses = sum(analyses.values())
    value_metrics['analyses_performed'] = total_analyses
    value_metrics['analyses_value'] = total_analyses * 45  # 1分析あたり45分の時間節約と仮定

    # ダッシュボード使用による洞察
    widget_count = usage_metrics.get('widget_count', 0)
    value_metrics['dashboard_insights'] = widget_count * 3  # 1ウィジェットあたり3つの洞察と仮定

    # 合計時間節約（分）
    total_time_saved = (
        value_metrics['reports_value'] +
        value_metrics['analyses_value']
    )
    value_metrics['total_time_saved_hours'] = round(total_time_saved / 60, 1)

    # 金銭的価値の概算（時給$50と仮定）
    value_metrics['estimated_value_usd'] = round(total_time_saved / 60 * 50, 2)

    return value_metrics