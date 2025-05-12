# -*- coding: utf-8 -*-
"""
VASデータリポジトリモジュール
VAS健康・パフォーマンスデータのデータアクセスを提供します。
"""
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc, extract
from ..models_sql import VASHealthPerformance, GoogleFormsConfiguration, GoogleFormsSyncLog
from ..repository import Repository, EntityNotFoundException, ValidationException


class VASRepository(Repository):
    """
    VASデータリポジトリクラス
    VAS健康・パフォーマンスデータの検索、保存、更新、削除機能を提供します。
    """
    def __init__(self, session: Session):
        """初期化"""
        self.session = session

    async def find_by_id(self, record_id: int) -> VASHealthPerformance:
        """
        IDによるVASデータの検索

        Args:
            record_id: 記録ID

        Returns:
            VASHealthPerformance: VASデータレコード

        Raises:
            EntityNotFoundException: レコードが見つからない場合
        """
        result = self.session.query(VASHealthPerformance).filter(
            VASHealthPerformance.record_id == record_id
        ).first()

        if not result:
            raise EntityNotFoundException(f"VASレコードが見つかりません: ID {record_id}")

        return result

    async def find_by_company_id(
        self,
        company_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[VASHealthPerformance]:
        """
        企業IDによるVASデータの検索

        Args:
            company_id: 企業ID
            start_date: 開始日時（デフォルト: なし）
            end_date: 終了日時（デフォルト: なし）
            limit: 結果の最大数（デフォルト: 100）
            offset: 結果のオフセット（デフォルト: 0）

        Returns:
            List[VASHealthPerformance]: VASデータレコードのリスト
        """
        query = self.session.query(VASHealthPerformance).filter(
            VASHealthPerformance.company_id == company_id
        )

        if start_date:
            query = query.filter(VASHealthPerformance.record_date >= start_date)

        if end_date:
            query = query.filter(VASHealthPerformance.record_date <= end_date)

        query = query.order_by(desc(VASHealthPerformance.record_date))

        if limit:
            query = query.limit(limit)

        if offset:
            query = query.offset(offset)

        return query.all()

    async def find_by_user_id(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[VASHealthPerformance]:
        """
        ユーザーIDによるVASデータの検索

        Args:
            user_id: ユーザーID
            start_date: 開始日時（デフォルト: なし）
            end_date: 終了日時（デフォルト: なし）
            limit: 結果の最大数（デフォルト: 100）
            offset: 結果のオフセット（デフォルト: 0）

        Returns:
            List[VASHealthPerformance]: VASデータレコードのリスト
        """
        query = self.session.query(VASHealthPerformance).filter(
            VASHealthPerformance.user_id == user_id
        )

        if start_date:
            query = query.filter(VASHealthPerformance.record_date >= start_date)

        if end_date:
            query = query.filter(VASHealthPerformance.record_date <= end_date)

        query = query.order_by(desc(VASHealthPerformance.record_date))

        if limit:
            query = query.limit(limit)

        if offset:
            query = query.offset(offset)

        return query.all()

    async def find_by_user_and_company(
        self,
        user_id: str,
        company_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[VASHealthPerformance]:
        """
        ユーザーIDと企業IDによるVASデータの検索

        Args:
            user_id: ユーザーID
            company_id: 企業ID
            start_date: 開始日時（デフォルト: なし）
            end_date: 終了日時（デフォルト: なし）

        Returns:
            List[VASHealthPerformance]: VASデータレコードのリスト
        """
        query = self.session.query(VASHealthPerformance).filter(
            VASHealthPerformance.user_id == user_id,
            VASHealthPerformance.company_id == company_id
        )

        if start_date:
            query = query.filter(VASHealthPerformance.record_date >= start_date)

        if end_date:
            query = query.filter(VASHealthPerformance.record_date <= end_date)

        query = query.order_by(desc(VASHealthPerformance.record_date))

        return query.all()

    async def get_monthly_average(
        self,
        company_id: str,
        year: Optional[int] = None,
        month: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        企業の月間平均VASスコアを取得

        Args:
            company_id: 企業ID
            year: 年（デフォルト: 現在の年）
            month: 月（デフォルト: 現在の月）

        Returns:
            Dict[str, Any]: 平均スコアの辞書
        """
        if not year:
            year = datetime.now().year

        if not month:
            month = datetime.now().month

        query = self.session.query(
            func.avg(VASHealthPerformance.physical_health).label('avg_physical_health'),
            func.avg(VASHealthPerformance.mental_health).label('avg_mental_health'),
            func.avg(VASHealthPerformance.work_performance).label('avg_work_performance'),
            func.avg(VASHealthPerformance.work_satisfaction).label('avg_work_satisfaction'),
            func.count(VASHealthPerformance.record_id).label('count')
        ).filter(
            VASHealthPerformance.company_id == company_id,
            extract('year', VASHealthPerformance.record_date) == year,
            extract('month', VASHealthPerformance.record_date) == month
        )

        result = query.one()

        return {
            'company_id': company_id,
            'year': year,
            'month': month,
            'avg_physical_health': float(result.avg_physical_health or 0),
            'avg_mental_health': float(result.avg_mental_health or 0),
            'avg_work_performance': float(result.avg_work_performance or 0),
            'avg_work_satisfaction': float(result.avg_work_satisfaction or 0),
            'record_count': int(result.count or 0)
        }

    async def create(self, data: Dict[str, Any]) -> VASHealthPerformance:
        """
        VASデータの作成

        Args:
            data: VASデータの辞書

        Returns:
            VASHealthPerformance: 作成されたVASデータレコード

        Raises:
            ValidationException: バリデーションエラーの場合
        """
        # 必須フィールドの検証
        required_fields = ['user_id', 'company_id', 'record_date',
                          'physical_health', 'mental_health',
                          'work_performance', 'work_satisfaction']

        for field in required_fields:
            if field not in data:
                raise ValidationException(f"必須フィールドがありません: {field}")

        # スコア範囲の検証
        score_fields = ['physical_health', 'mental_health',
                       'work_performance', 'work_satisfaction']

        for field in score_fields:
            if field in data and (data[field] < 0 or data[field] > 100):
                raise ValidationException(f"{field}は0から100の間である必要があります")

        # 重複レコードのチェック
        existing = self.session.query(VASHealthPerformance).filter(
            VASHealthPerformance.user_id == data['user_id'],
            VASHealthPerformance.company_id == data['company_id'],
            VASHealthPerformance.record_date == data['record_date']
        ).first()

        if existing:
            raise ValidationException("この日付のレコードはすでに存在します")

        # レコードの作成
        record = VASHealthPerformance(**data)
        self.session.add(record)
        self.session.flush()

        return record

    async def update(self, record_id: int, data: Dict[str, Any]) -> VASHealthPerformance:
        """
        VASデータの更新

        Args:
            record_id: 記録ID
            data: 更新データの辞書

        Returns:
            VASHealthPerformance: 更新されたVASデータレコード

        Raises:
            EntityNotFoundException: レコードが見つからない場合
            ValidationException: バリデーションエラーの場合
        """
        record = await self.find_by_id(record_id)

        # スコア範囲の検証
        score_fields = ['physical_health', 'mental_health',
                       'work_performance', 'work_satisfaction']

        for field in score_fields:
            if field in data and (data[field] < 0 or data[field] > 100):
                raise ValidationException(f"{field}は0から100の間である必要があります")

        # フィールドの更新
        for key, value in data.items():
            if hasattr(record, key):
                setattr(record, key, value)

        # 自動的にupdated_atが更新される

        return record

    async def delete(self, record_id: int) -> bool:
        """
        VASデータの削除

        Args:
            record_id: 記録ID

        Returns:
            bool: 削除が成功した場合はTrue

        Raises:
            EntityNotFoundException: レコードが見つからない場合
        """
        record = await self.find_by_id(record_id)

        self.session.delete(record)

        return True

    async def get_sync_configurations(
        self,
        company_id: Optional[str] = None,
        form_type: Optional[str] = None,
        active_only: bool = True
    ) -> List[GoogleFormsConfiguration]:
        """
        Google Forms同期設定の取得

        Args:
            company_id: 企業ID（デフォルト: なし）
            form_type: フォームタイプ（デフォルト: なし）
            active_only: アクティブな設定のみ（デフォルト: True）

        Returns:
            List[GoogleFormsConfiguration]: 同期設定のリスト
        """
        query = self.session.query(GoogleFormsConfiguration)

        if company_id:
            query = query.filter(GoogleFormsConfiguration.company_id == company_id)

        if form_type:
            query = query.filter(GoogleFormsConfiguration.form_type == form_type)

        if active_only:
            query = query.filter(GoogleFormsConfiguration.active == True)

        return query.all()

    async def save_sync_log(self, log_data: Dict[str, Any]) -> GoogleFormsSyncLog:
        """
        同期ログの保存

        Args:
            log_data: ログデータの辞書

        Returns:
            GoogleFormsSyncLog: 保存された同期ログ

        Raises:
            ValidationException: バリデーションエラーの場合
        """
        # 必須フィールドの検証
        required_fields = ['config_id', 'sync_start_time', 'sync_end_time', 'status']

        for field in required_fields:
            if field not in log_data:
                raise ValidationException(f"必須フィールドがありません: {field}")

        # レコードの作成
        log = GoogleFormsSyncLog(**log_data)
        self.session.add(log)
        self.session.flush()

        return log

    async def get_sync_logs(
        self,
        config_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10
    ) -> List[GoogleFormsSyncLog]:
        """
        同期ログの取得

        Args:
            config_id: 設定ID
            start_date: 開始日時（デフォルト: なし）
            end_date: 終了日時（デフォルト: なし）
            limit: 結果の最大数（デフォルト: 10）

        Returns:
            List[GoogleFormsSyncLog]: 同期ログのリスト
        """
        query = self.session.query(GoogleFormsSyncLog).filter(
            GoogleFormsSyncLog.config_id == config_id
        )

        if start_date:
            query = query.filter(GoogleFormsSyncLog.sync_start_time >= start_date)

        if end_date:
            query = query.filter(GoogleFormsSyncLog.sync_end_time <= end_date)

        query = query.order_by(desc(GoogleFormsSyncLog.sync_start_time))

        if limit:
            query = query.limit(limit)

        return query.all()