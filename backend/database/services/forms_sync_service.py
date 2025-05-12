# -*- coding: utf-8 -*-
"""
Google Forms同期サービス
VASデータの自動収集とデータベース同期を管理します。
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
from ..connectors.google_forms_connector import (
    GoogleFormsConnector, GoogleFormsError, ResponseStatus
)
from ..repositories.vas_repository import VASRepository
from ..config import FormType, get_google_forms_config, SYNC_LOG_ENABLED
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SyncStatus(Enum):
    """同期ステータス"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"

class SyncResult(BaseModel):
    """同期結果のデータモデル"""
    form_id: str
    form_type: str
    status: SyncStatus
    start_time: datetime
    end_time: datetime
    total_records: int = 0
    new_records: int = 0
    updated_records: int = 0
    errors: List[str] = Field(default_factory=list)

class FormsSyncService:
    """
    Google Formsデータ同期サービス
    VASデータの自動収集と同期を管理します。
    """
    def __init__(
        self,
        session: Session,
        connector: Optional[GoogleFormsConnector] = None,
        service_account_file: Optional[str] = None
    ):
        """
        初期化

        Args:
            session: データベースセッション
            connector: GoogleFormsConnectorインスタンス（デフォルト: None）
            service_account_file: サービスアカウントファイルパス（デフォルト: None）
        """
        self.session = session
        self.vas_repository = VASRepository(session)
        self.connector = connector or GoogleFormsConnector(service_account_file)

    async def initialize(self) -> None:
        """コネクタの初期化"""
        if not getattr(self.connector, 'service', None):
            await self.connector.initialize()

    async def sync_vas_form(
        self,
        company_id: str,
        form_config_id: Optional[int] = None,
        form_id: Optional[str] = None,
        force_sync: bool = False
    ) -> SyncResult:
        """
        VASフォームデータの同期

        Args:
            company_id: 企業ID
            form_config_id: フォーム設定ID（デフォルト: None）
            form_id: 直接指定するフォームID（デフォルト: None）
            force_sync: 強制同期フラグ（デフォルト: False）

        Returns:
            SyncResult: 同期結果

        Raises:
            ValueError: 設定とフォームIDの両方が指定されていない場合
        """
        await self.initialize()

        start_time = datetime.now()
        sync_result = SyncResult(
            form_id="",
            form_type=FormType.VAS_HEALTH.value,
            status=SyncStatus.FAILED,
            start_time=start_time,
            end_time=start_time,
            total_records=0,
            new_records=0,
            updated_records=0,
            errors=[]
        )

        try:
            # フォーム設定の取得
            if form_config_id:
                config_list = await self.vas_repository.get_sync_configurations(
                    company_id=company_id,
                    form_type=FormType.VAS_HEALTH.value,
                    active_only=True
                )

                config = next((c for c in config_list if c.config_id == form_config_id), None)
                if not config:
                    raise ValueError(f"フォーム設定が見つかりません: ID {form_config_id}")

                form_id = config.form_id
                field_mappings = config.field_mappings or {}
                sync_result.form_id = form_id

            elif form_id:
                # 直接フォームIDが指定された場合
                config = None
                field_mappings = {}
                sync_result.form_id = form_id

            else:
                raise ValueError("フォーム設定IDかフォームIDのどちらかを指定してください")

            # 前回の同期時刻を取得
            last_sync_time = None
            if config and not force_sync:
                last_sync_time = config.last_sync_time

            # フォームレスポンスの取得
            all_responses = []
            next_page_token = None

            while True:
                response_data = await self.connector.get_form_responses(
                    form_id=form_id,
                    page_size=100,
                    page_token=next_page_token
                )

                responses = response_data.get('responses', [])
                all_responses.extend(responses)
                sync_result.total_records += len(responses)

                next_page_token = response_data.get('next_page_token')
                if not next_page_token:
                    break

            # レスポンスのフィルタリング
            if last_sync_time and not force_sync:
                # 前回の同期以降のレスポンスのみを処理
                filtered_responses = [
                    r for r in all_responses
                    if r.get('metadata', {}).get('last_submitted_time') and
                    datetime.fromisoformat(r['metadata']['last_submitted_time'].rstrip('Z')) > last_sync_time
                ]
            else:
                filtered_responses = all_responses

            logger.info(f"処理するレスポンス数: {len(filtered_responses)}/{len(all_responses)}")

            # データの保存
            for response in filtered_responses:
                try:
                    # レスポンスをVASデータに変換
                    vas_data = self._transform_response_to_vas_data(
                        response, company_id, field_mappings
                    )

                    # VASデータの存在チェック
                    existing_records = await self.vas_repository.find_by_user_and_company(
                        user_id=vas_data['user_id'],
                        company_id=vas_data['company_id'],
                        start_date=vas_data['record_date'],
                        end_date=vas_data['record_date']
                    )

                    if existing_records:
                        # 更新
                        await self.vas_repository.update(
                            existing_records[0].record_id, vas_data
                        )
                        sync_result.updated_records += 1
                    else:
                        # 新規作成
                        await self.vas_repository.create(vas_data)
                        sync_result.new_records += 1

                except Exception as e:
                    error_msg = f"レスポンス処理エラー: {str(e)}"
                    logger.error(error_msg)
                    sync_result.errors.append(error_msg)

            # 同期ステータスの更新
            if sync_result.errors:
                if len(sync_result.errors) == len(filtered_responses):
                    sync_result.status = SyncStatus.FAILED
                else:
                    sync_result.status = SyncStatus.PARTIAL
            else:
                sync_result.status = SyncStatus.SUCCESS

            # 設定の更新
            if config:
                config.last_sync_time = datetime.now()
                self.session.flush()

            # 同期ログの保存
            if SYNC_LOG_ENABLED and config:
                await self._save_sync_log(sync_result, config.config_id)

            sync_result.end_time = datetime.now()
            return sync_result

        except Exception as e:
            end_time = datetime.now()
            error_msg = f"同期処理エラー: {str(e)}"
            logger.error(error_msg)

            sync_result.status = SyncStatus.FAILED
            sync_result.errors.append(error_msg)
            sync_result.end_time = end_time

            # 同期ログの保存（エラー時）
            if SYNC_LOG_ENABLED and config:
                await self._save_sync_log(sync_result, config.config_id)

            return sync_result

    def _transform_response_to_vas_data(
        self,
        response: Dict[str, Any],
        company_id: str,
        field_mappings: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        フォームレスポンスをVASデータに変換

        Args:
            response: フォームレスポンス
            company_id: 企業ID
            field_mappings: フィールドマッピング

        Returns:
            Dict[str, Any]: VASデータ

        Raises:
            ValueError: 必須フィールドがない場合
        """
        # デフォルトマッピング
        default_mappings = {
            'user_id': 'user_id',
            'physical_health': 'physical_health',
            'mental_health': 'mental_health',
            'work_performance': 'work_performance',
            'work_satisfaction': 'work_satisfaction',
            'additional_comments': 'comments'
        }

        # マッピングの適用
        mappings = {**default_mappings, **(field_mappings or {})}

        # 変換データの初期化
        vas_data = {
            'company_id': company_id,
            'record_date': datetime.fromisoformat(response.get('submit_time', datetime.now().isoformat()).rstrip('Z'))
        }

        # ユーザーIDの取得
        respondent_email = response.get('respondent_email', '')
        if respondent_email:
            vas_data['user_id'] = respondent_email.split('@')[0]
        else:
            vas_data['user_id'] = response.get('metadata', {}).get('respondent_id', f"unknown_{datetime.now().timestamp()}")

        # 回答データの抽出
        answers = response.get('answers', {})
        for target_field, source_field in mappings.items():
            if source_field not in answers:
                continue

            value = answers[source_field]
            if value is None:
                continue

            # 数値フィールドの変換
            if target_field in ['physical_health', 'mental_health',
                              'work_performance', 'work_satisfaction']:
                try:
                    # 文字列から数値に変換
                    if isinstance(value, str):
                        if value.isdigit():
                            value = int(value)
                        else:
                            try:
                                value = int(float(value))
                            except ValueError:
                                # スケール（1-10）を0-100に変換
                                if '/' in value:
                                    parts = value.split('/')
                                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                                        value = int(float(parts[0]) / float(parts[1]) * 100)
                except (ValueError, TypeError):
                    logger.warning(f"数値変換エラー: {value}")
                    continue

            vas_data[target_field] = value

        # 必須フィールドの検証
        required_fields = ['user_id', 'physical_health', 'mental_health',
                         'work_performance', 'work_satisfaction']

        for field in required_fields:
            if field not in vas_data:
                raise ValueError(f"必須フィールドがありません: {field}")

        return vas_data

    async def _save_sync_log(self, result: SyncResult, config_id: int) -> None:
        """
        同期ログの保存

        Args:
            result: 同期結果
            config_id: 設定ID
        """
        try:
            log_data = {
                'config_id': config_id,
                'sync_start_time': result.start_time,
                'sync_end_time': result.end_time,
                'records_processed': result.total_records,
                'records_created': result.new_records,
                'records_updated': result.updated_records,
                'status': result.status.value,
                'error_details': '\n'.join(result.errors) if result.errors else None
            }

            await self.vas_repository.save_sync_log(log_data)

        except Exception as e:
            logger.error(f"同期ログ保存エラー: {str(e)}")

    async def synchronize_all_active_forms(self, company_id: Optional[str] = None) -> List[SyncResult]:
        """
        すべてのアクティブなフォームを同期

        Args:
            company_id: 企業ID（デフォルト: すべての企業）

        Returns:
            List[SyncResult]: 同期結果のリスト
        """
        # アクティブな設定の取得
        active_configs = await self.vas_repository.get_sync_configurations(
            company_id=company_id,
            active_only=True
        )

        # 同期結果
        results = []

        # 各設定に対して同期を実行
        for config in active_configs:
            # 前回の同期から最小間隔（1分）以上経過しているか確認
            if config.last_sync_time:
                time_since_last_sync = datetime.now() - config.last_sync_time
                if time_since_last_sync.total_seconds() < 60:  # 1分未満ならスキップ
                    logger.info(f"スキップ: 前回の同期から1分経過していません: {config.config_id}")
                    continue

            # 同期の実行
            result = await self.sync_vas_form(
                company_id=config.company_id,
                form_config_id=config.config_id
            )
            results.append(result)

        return results