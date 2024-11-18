"""
外部データ入力モジュール
Google FormsとAPIからのデータ取得と変換を管理します。
データの永続化はFirestoreServiceに委譲します。
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import aiohttp
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import cachetools
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel

# ロギングの設定
logger = logging.getLogger(__name__)

class DataInputError(Exception):
    """データ入力操作に関するエラー"""
    pass

class FormResponse(BaseModel):
    """フォームレスポンスのPydanticモデル"""
    response_id: str
    created_time: Optional[str]
    last_submitted_time: Optional[str]
    answers: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class GoogleFormsConnector:
    """
    Google Formsとの連携を管理するクラス
    Firestoreへの保存はFirestoreServiceに委譲します。
    """
    def __init__(self):
        self.service = None
        self.credentials = None
        self._cache = cachetools.TTLCache(maxsize=100, ttl=3600)

    async def initialize(self, service_account_file: str) -> None:
        """Google Forms APIの初期化"""
        try:
            self.credentials = await asyncio.to_thread(
                service_account.Credentials.from_service_account_file,
                service_account_file,
                scopes=['https://www.googleapis.com/auth/forms.responses.readonly']
            )
            self.service = await asyncio.to_thread(
                build,
                'forms',
                'v1',
                credentials=self.credentials
            )
            logger.info("Successfully initialized Google Forms API")
        except Exception as e:
            logger.error(f"Failed to initialize Google Forms API: {str(e)}")
            raise DataInputError(f"Failed to initialize Google Forms API: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_form_responses(
        self,
        form_id: str,
        use_cache: bool = True
    ) -> List[FormResponse]:
        """フォームの回答を取得して構造化データに変換"""
        if not self.service:
            raise DataInputError("Google Forms API has not been initialized")

        cache_key = f"responses_{form_id}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            response = await asyncio.to_thread(
                self.service.forms().responses().list(formId=form_id).execute
            )

            if not response or 'responses' not in response:
                return []

            responses = []
            for item in response['responses']:
                try:
                    formatted = await self._format_response(item)
                    responses.append(FormResponse(**formatted))
                except Exception as e:
                    logger.warning(f"Skipping invalid response: {str(e)}")
                    continue

            if use_cache:
                self._cache[cache_key] = responses

            return responses

        except HttpError as e:
            if e.resp.status == 404:
                raise DataInputError(f"Form not found: {form_id}")
            raise DataInputError(f"HTTP error occurred: {str(e)}")
        except Exception as e:
            raise DataInputError(f"Error fetching form responses: {str(e)}")

    async def _format_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """APIレスポンスを整形"""
        if not isinstance(response, dict):
            raise DataInputError("Invalid response format: expected dictionary")

        try:
            formatted = {
                'response_id': response.get('responseId'),
                'created_time': response.get('createTime'),
                'last_submitted_time': response.get('lastSubmittedTime'),
                'answers': {},
                'metadata': {
                    'source': 'google_forms',
                    'timestamp': datetime.now().isoformat()
                }
            }

            answers = response.get('answers', {})
            if answers and isinstance(answers, dict):
                for question_id, answer in answers.items():
                    if not answer or not isinstance(answer, dict):
                        continue

                    formatted_answer = {
                        'question_id': question_id,
                        'type': answer.get('type', ''),
                        'value': self._extract_answer_value(answer)
                    }
                    formatted['answers'][question_id] = formatted_answer

            return formatted

        except Exception as e:
            raise DataInputError(f"Failed to format response: {str(e)}")

    def _extract_answer_value(self, answer: Dict[str, Any]) -> str:
        """回答値を抽出"""
        try:
            if not isinstance(answer, dict):
                return ''

            text_answers = answer.get('textAnswers')
            if not text_answers or not isinstance(text_answers, dict):
                return ''

            answers_list = text_answers.get('answers', [])
            if not answers_list or not isinstance(answers_list, list):
                return ''

            first_answer = answers_list[0] if answers_list else None
            if not first_answer or not isinstance(first_answer, dict):
                return ''

            return first_answer.get('value', '')
        except Exception:
            return ''

    async def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()
        logger.info("Cache cleared")

class ExternalDataFetcher:
    """
    外部APIからのデータ取得を管理するクラス
    Firestoreへの保存はFirestoreServiceに委譲します。
    """
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_keys: Dict[str, str] = {}
        self._cache = cachetools.TTLCache(maxsize=100, ttl=1800)

    async def initialize(self, api_keys: Optional[Dict[str, str]] = None) -> None:
        """外部データフェッチャーの初期化"""
        try:
            if api_keys:
                self.api_keys = api_keys

            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'ExternalDataFetcher/1.0'}
            )
            logger.info("ExternalDataFetcher initialized")
        except Exception as e:
            raise DataInputError(f"Failed to initialize ExternalDataFetcher: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def fetch_data(
        self,
        source_name: str,
        use_cache: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """外部ソースからデータを取得"""
        if not self.session:
            raise DataInputError("ExternalDataFetcher has not been initialized")

        cache_key = f"{source_name}_{json.dumps(kwargs, sort_keys=True)}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            data = await self._fetch_from_source(source_name, **kwargs)

            if use_cache:
                self._cache[cache_key] = data

            return data

        except aiohttp.ClientError as e:
            raise DataInputError(f"HTTP error occurred: {str(e)}")
        except Exception as e:
            raise DataInputError(f"Failed to fetch data: {str(e)}")

    async def _fetch_from_source(
        self,
        source_name: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """データソースに応じた取得処理"""
        if not self.session:
            raise DataInputError("Session not initialized")

        try:
            endpoint = kwargs.get('endpoint')
            if not endpoint:
                raise DataInputError(f"No endpoint specified for {source_name}")

            async with self.session.get(endpoint) as response:
                response.raise_for_status()
                data = await response.json()

                if not isinstance(data, dict):
                    raise DataInputError("Invalid response format")

                items = data.get('items', [])
                if not isinstance(items, list):
                    raise DataInputError("Invalid items format")

                return [{
                    **item,
                    'metadata': {
                        'source': source_name,
                        'timestamp': datetime.now().isoformat(),
                        'endpoint': endpoint
                    }
                } for item in items]

        except Exception as e:
            raise DataInputError(f"Error fetching from {source_name}: {str(e)}")

    async def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()
        logger.info("Cache cleared")

    async def close(self) -> None:
        """セッションをクローズ"""
        if self.session:
            await self.session.close()
            logger.info("Session closed")