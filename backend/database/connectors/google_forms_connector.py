# -*- coding: utf-8 -*-
"""
統合Google Formsコネクタ
Google Forms APIを使用してVASデータと業績データを取得・同期するためのクラス
"""
import os
import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Union, Sequence, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import cachetools
from tenacity import retry, stop_after_attempt, wait_exponential

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field, field_validator

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# スコープ定義
SCOPES = ['https://www.googleapis.com/auth/forms.responses.readonly',
          'https://www.googleapis.com/auth/spreadsheets.readonly']
MAX_PAGE_SIZE = 100
MAX_RETRIES = 3

class GoogleFormsError(Exception):
    """Google Forms操作に関するエラー"""
    pass

class ResponseStatus(Enum):
    """回答ステータス"""
    NEW = "new"               # 新規回答
    UPDATED = "updated"       # 更新された回答
    DUPLICATE = "duplicate"   # 重複回答
    INVALID = "invalid"       # 無効な回答

class FormResponse(BaseModel):
    """Google Forms回答モデル"""
    response_id: str
    form_id: str
    submit_time: datetime
    respondent_email: Optional[str] = None
    answers: Dict[str, Any]
    status: ResponseStatus = ResponseStatus.NEW
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('submit_time', mode='before')
    def parse_datetime(cls, v):
        """日時文字列をdatetimeオブジェクトに変換"""
        if isinstance(v, str):
            return datetime.fromisoformat(v.rstrip('Z'))
        return v

class FormsRequestParams(BaseModel):
    """Forms API リクエストパラメータ"""
    form_id: str
    page_size: Optional[str] = None
    page_token: Optional[str] = None

class GoogleFormsConnector:
    """
    統合Google Formsコネクタ
    VASデータと業績データの同期を管理します
    """
    def __init__(self, service_account_file: Optional[str] = None):
        """
        初期化

        Args:
            service_account_file: サービスアカウントキーのパス（省略時は環境変数から取得）
        """
        self.service = None
        self.sheets_service = None
        self.credentials = None
        self.service_account_file = service_account_file
        self._cache = cachetools.TTLCache(maxsize=100, ttl=3600)  # 1時間キャッシュ

    async def initialize(self) -> None:
        """
        Google Forms APIとSheets APIの初期化
        """
        try:
            if not self.service_account_file:
                self.service_account_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

            if not self.service_account_file:
                raise GoogleFormsError("サービスアカウント認証情報が見つかりません")

            self.credentials = await asyncio.to_thread(
                service_account.Credentials.from_service_account_file,
                self.service_account_file,
                scopes=SCOPES
            )

            # Forms API初期化
            self.service = await asyncio.to_thread(
                build,
                'forms',
                'v1',
                credentials=self.credentials
            )

            # Sheets API初期化
            self.sheets_service = await asyncio.to_thread(
                build,
                'sheets',
                'v4',
                credentials=self.credentials
            )

            logger.info("Google Forms API・Sheets APIが初期化されました")

        except Exception as e:
            logger.error(f"Google APIs初期化エラー: {str(e)}")
            raise GoogleFormsError(f"Google APIs初期化エラー: {str(e)}")

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_form_responses(
        self,
        form_id: str,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        指定されたGoogle Formのアンケート結果を取得

        Args:
            form_id: Google FormのID
            page_size: 1ページあたりの回答数
            page_token: 次ページのトークン
            use_cache: キャッシュを使用するかどうか

        Returns:
            Dict[str, Any]: レスポンスデータ
            - responses: フォーム回答のリスト
            - next_page_token: 次ページのトークン（ある場合）

        Raises:
            GoogleFormsError: API操作に失敗した場合
        """
        if not self.service:
            await self.initialize()

        cache_key = f"form_responses:{form_id}:{page_size}:{page_token}"
        if use_cache and cache_key in self._cache:
            logger.info(f"キャッシュからフォーム回答を取得: {form_id}")
            return self._cache[cache_key]

        try:
            logger.info(f"フォーム回答取得: {form_id}")

            # リクエストパラメータの設定
            request_params = {'formId': form_id}
            if page_size is not None:
                request_params['pageSize'] = str(min(page_size, MAX_PAGE_SIZE))
            if page_token:
                request_params['pageToken'] = page_token

            # API呼び出し
            response = await asyncio.to_thread(
                lambda: self.service.forms().responses().list(**request_params).execute()
            )

            # レスポンスの整形
            form_responses = []
            for res in response.get('responses', []):
                try:
                    answers = {}
                    for question_id, answer_data in res.get('answers', {}).items():
                        # 回答タイプに応じた処理
                        answer_type = next(iter(answer_data.keys()))
                        answer_value = answer_data[answer_type]
                        answers[question_id] = answer_value

                    # 回答オブジェクトの作成
                    form_response = FormResponse(
                        response_id=res.get('responseId', ''),
                        form_id=form_id,
                        submit_time=res.get('createTime', datetime.now().isoformat()),
                        answers=answers,
                        metadata={
                            'last_submitted_time': res.get('lastSubmittedTime'),
                            'respondent_id': res.get('respondentEmail', '').split('@')[0]
                        }
                    )
                    form_responses.append(form_response.model_dump())
                except Exception as e:
                    logger.warning(f"回答データの解析エラー: {str(e)}")
                    continue

            result = {
                'responses': form_responses,
                'next_page_token': response.get('nextPageToken')
            }

            # キャッシュに保存
            if use_cache:
                self._cache[cache_key] = result

            return result

        except HttpError as e:
            if e.resp.status == 404:
                raise GoogleFormsError(f"フォームが見つかりません: {form_id}")
            raise GoogleFormsError(f"HTTPエラー: {str(e)}")
        except Exception as e:
            raise GoogleFormsError(f"フォーム回答取得エラー: {str(e)}")

    async def get_form_metadata(self, form_id: str) -> Dict[str, Any]:
        """
        フォームのメタデータを取得します。

        Args:
            form_id: Google FormのID

        Returns:
            Dict[str, Any]: フォームのメタデータ

        Raises:
            GoogleFormsError: API操作に失敗した場合
        """
        if not self.service:
            await self.initialize()

        try:
            logger.info(f"フォームメタデータ取得: {form_id}")

            form = await asyncio.to_thread(
                lambda: self.service.forms().get(formId=form_id).execute()
            )

            metadata = {
                'form_id': form_id,
                'title': form.get('info', {}).get('title'),
                'description': form.get('info', {}).get('description'),
                'settings': form.get('settings', {}),
                'updated_time': datetime.fromisoformat(form['updateTime'].rstrip('Z'))
            }

            logger.info(f"フォームメタデータ取得完了: {form_id}")
            return metadata

        except Exception as e:
            error_msg = f"フォームメタデータ取得エラー: {str(e)}"
            logger.error(error_msg)
            raise GoogleFormsError(error_msg)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_sheet_data(
        self,
        sheet_id: str,
        range_name: str = 'A1:Z1000',
        include_headers: bool = True,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Google Sheetsからデータを取得

        Args:
            sheet_id: スプレッドシートID
            range_name: 範囲（A1表記）
            include_headers: 1行目をヘッダーとして扱うかどうか
            use_cache: キャッシュを使用するかどうか

        Returns:
            List[Dict[str, Any]]: シートデータ（辞書のリスト）

        Raises:
            GoogleFormsError: API操作に失敗した場合
        """
        if not self.sheets_service:
            await self.initialize()

        cache_key = f"sheet_data:{sheet_id}:{range_name}"
        if use_cache and cache_key in self._cache:
            logger.info(f"キャッシュからシートデータを取得: {sheet_id}")
            return self._cache[cache_key]

        try:
            logger.info(f"シートデータ取得: {sheet_id}, 範囲: {range_name}")

            result = await asyncio.to_thread(
                lambda: self.sheets_service.spreadsheets().values().get(
                    spreadsheetId=sheet_id,
                    range=range_name,
                    valueRenderOption='UNFORMATTED_VALUE'
                ).execute()
            )

            values = result.get('values', [])
            if not values:
                logger.warning(f"データが見つかりません: {sheet_id}, 範囲: {range_name}")
                return []

            if include_headers and len(values) > 0:
                headers = values[0]
                data = []
                for row in values[1:]:
                    # 行の長さがヘッダーより短い場合、Noneで埋める
                    row_extended = row + [None] * (len(headers) - len(row))
                    data.append(dict(zip(headers, row_extended[:len(headers)])))
            else:
                data = values

            # キャッシュに保存
            if use_cache:
                self._cache[cache_key] = data

            return data

        except Exception as e:
            error_msg = f"シートデータ取得エラー: {str(e)}"
            logger.error(error_msg)
            raise GoogleFormsError(error_msg)

    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()
        logger.info("キャッシュをクリアしました")

    def invalidate_cache_item(self, key: str) -> None:
        """特定のキャッシュアイテムを無効化"""
        if key in self._cache:
            del self._cache[key]
            logger.info(f"キャッシュアイテムを無効化: {key}")

def create_forms_connector(service_account_file: Optional[str] = None) -> GoogleFormsConnector:
    """
    GoogleFormsConnectorのインスタンスを作成します。

    Args:
        service_account_file: サービスアカウントキーファイルのパス

    Returns:
        GoogleFormsConnector: コネクターインスタンス
    """
    return GoogleFormsConnector(service_account_file)