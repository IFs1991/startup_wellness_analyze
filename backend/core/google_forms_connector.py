# -*- coding: utf-8 -*-
"""
Google Forms コネクター
Google Forms API を使用してアンケート結果を取得し、
Firestoreとの連携機能を提供します。
"""
from googleapiclient.discovery import build
from google.oauth2 import service_account
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
import asyncio
from pydantic import BaseModel, Field, validator

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 定数定義
SCOPES = ['https://www.googleapis.com/auth/forms.responses.readonly']
MAX_PAGE_SIZE = 100

class FormsAPIError(Exception):
    """Google Forms API操作に関するエラー"""
    pass

class FormResponse(BaseModel):
    """フォームレスポンスのデータモデル"""
    form_id: str
    response_id: str
    created_time: datetime
    last_submitted_time: datetime
    answers: Dict[str, Any]

    class Config:
        arbitrary_types_allowed = True

class FormsRequestParams(BaseModel):
    """フォームリクエストパラメータのバリデーションモデル"""
    form_id: str
    page_size: Optional[str] = None
    page_token: Optional[str] = None

    @validator('page_size')
    def validate_page_size(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                page_size = int(v)
                if not 1 <= page_size <= MAX_PAGE_SIZE:
                    raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}")
            except ValueError as e:
                raise ValueError(f"Invalid page_size: {str(e)}")
        return v

class GoogleFormsConnector:
    """
    Google Forms APIとの接続を管理し、アンケート結果を取得するためのクラスです。
    """
    def __init__(self, service_account_file: str):
        """
        Google Forms APIクライアントを初期化します。

        Args:
            service_account_file (str): サービスアカウントキーファイルのパス
        """
        try:
            self.credentials = service_account.Credentials.from_service_account_file(
                service_account_file,
                scopes=SCOPES
            )
            self.service = build('forms', 'v1', credentials=self.credentials)
            logger.info("Google Forms API client initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize Google Forms API client: {str(e)}"
            logger.error(error_msg)
            raise FormsAPIError(error_msg) from e

    async def get_form_responses(
        self,
        form_id: str,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None
    ) -> List[FormResponse]:
        """
        指定されたGoogle Formのアンケート結果を非同期で取得します。

        Args:
            form_id (str): Google FormのID
            page_size (Optional[int]): 1ページあたりの回答数
            page_token (Optional[str]): 次ページのトークン

        Returns:
            List[FormResponse]: 整形されたアンケート回答のリスト

        Raises:
            FormsAPIError: API操作に失敗した場合
        """
        try:
            logger.info(f"Fetching responses for form: {form_id}")

            # リクエストパラメータの検証と変換
            request_params = {'formId': form_id}
            if page_size is not None:
                # 整数値を文字列に変換
                request_params['pageSize'] = str(min(page_size, MAX_PAGE_SIZE))
            if page_token:
                request_params['pageToken'] = page_token

            # パラメータのバリデーション
            validated_params = FormsRequestParams(
                form_id=form_id,
                page_size=request_params.get('pageSize'),
                page_token=page_token
            )

            # 非同期でAPI呼び出しを実行
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.service.forms().responses().list(**request_params).execute()
            )

            # レスポンスの整形
            form_responses = []
            for response_data in response.get('responses', []):
                try:
                    form_response = FormResponse(
                        form_id=form_id,
                        response_id=response_data['responseId'],
                        created_time=datetime.fromisoformat(response_data['createTime'].rstrip('Z')),
                        last_submitted_time=datetime.fromisoformat(response_data['lastSubmittedTime'].rstrip('Z')),
                        answers=self._process_answers(response_data.get('answers', {}))
                    )
                    form_responses.append(form_response)
                except Exception as e:
                    logger.warning(f"Failed to process response {response_data.get('responseId')}: {str(e)}")
                    continue

            logger.info(f"Successfully fetched {len(form_responses)} responses")
            return form_responses

        except Exception as e:
            error_msg = f"Error fetching form responses: {str(e)}"
            logger.error(error_msg)
            raise FormsAPIError(error_msg) from e

    async def get_form_metadata(self, form_id: str) -> Dict[str, Any]:
        """
        フォームのメタデータを取得します。

        Args:
            form_id (str): Google FormのID

        Returns:
            Dict[str, Any]: フォームのメタデータ

        Raises:
            FormsAPIError: API操作に失敗した場合
        """
        try:
            logger.info(f"Fetching metadata for form: {form_id}")

            loop = asyncio.get_event_loop()
            form = await loop.run_in_executor(
                None,
                lambda: self.service.forms().get(formId=form_id).execute()
            )

            metadata = {
                'form_id': form_id,
                'title': form.get('info', {}).get('title'),
                'description': form.get('info', {}).get('description'),
                'settings': form.get('settings', {}),
                'updated_time': datetime.fromisoformat(form['updateTime'].rstrip('Z'))
            }

            logger.info(f"Successfully fetched metadata for form: {form_id}")
            return metadata

        except Exception as e:
            error_msg = f"Error fetching form metadata: {str(e)}"
            logger.error(error_msg)
            raise FormsAPIError(error_msg) from e

    def _process_answers(self, raw_answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        回答データを処理して整形します。

        Args:
            raw_answers (Dict[str, Any]): 生の回答データ

        Returns:
            Dict[str, Any]: 整形された回答データ
        """
        processed_answers = {}
        for question_id, answer_data in raw_answers.items():
            answer_value = self._extract_answer_value(answer_data)
            if answer_value is not None:
                processed_answers[question_id] = {
                    'question_id': question_id,
                    'type': self._determine_answer_type(answer_data),
                    'value': answer_value
                }
        return processed_answers

    def _determine_answer_type(self, answer_data: Dict[str, Any]) -> str:
        """
        回答タイプを判定します。

        Args:
            answer_data (Dict[str, Any]): 回答データ

        Returns:
            str: 回答タイプ
        """
        if 'textAnswers' in answer_data:
            return 'text'
        elif 'fileUploadAnswers' in answer_data:
            return 'file'
        elif 'questionGroupAnswers' in answer_data:
            return 'group'
        return 'unknown'

    def _extract_answer_value(self, answer_data: Dict[str, Any]) -> Any:
        """
        回答の値を抽出します。

        Args:
            answer_data (Dict[str, Any]): 回答データ

        Returns:
            Any: 抽出された回答値
        """
        if 'textAnswers' in answer_data:
            answers = answer_data['textAnswers'].get('answers', [])
            return [a.get('value', '') for a in answers] if answers else None
        elif 'fileUploadAnswers' in answer_data:
            files = answer_data['fileUploadAnswers'].get('answers', [])
            return [a.get('fileId', '') for a in files] if files else None
        elif 'questionGroupAnswers' in answer_data:
            group_answers = {}
            for qa in answer_data['questionGroupAnswers']:
                question_id = qa.get('questionId')
                if question_id:
                    group_answers[question_id] = self._extract_answer_value(qa)
            return group_answers if group_answers else None
        return None

    async def close(self) -> None:
        """
        APIクライアントの接続を閉じます。
        """
        try:
            await asyncio.get_event_loop().run_in_executor(None, self.service.close)
            logger.info("Google Forms API client connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing Google Forms API client connection: {str(e)}")
            raise

def create_forms_connector(service_account_file: str) -> GoogleFormsConnector:
    """
    GoogleFormsConnectorのインスタンスを作成します。

    Args:
        service_account_file (str): サービスアカウントキーファイルのパス

    Returns:
        GoogleFormsConnector: コネクターインスタンス
    """
    return GoogleFormsConnector(service_account_file)