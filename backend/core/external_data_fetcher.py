# -*- coding: utf-8 -*-
"""
外部データ フェッチャー
外部データソース（業界ベンチマーク、経済指標など）からデータを取得し、
Firestoreに保存する機能を提供します。
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Union, List
import asyncio
import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ハンドラーの設定
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 定数定義
CACHE_EXPIRY: int = 3600  # キャッシュの有効期限（秒）
MAX_RETRY_ATTEMPTS: int = 3
RETRY_DELAY: int = 1  # 秒
REQUEST_TIMEOUT: ClientTimeout = ClientTimeout(total=30)  # aiohttp用のタイムアウト設定

class ExternalDataError(Exception):
    """外部データ取得に関するエラー"""
    pass

class StorageError(Exception):
    """ストレージ操作に関するエラー"""
    pass

class ExternalDataFetcher:
    """
    外部データソースからデータを取得し、Firestoreに保存するクラス
    """
    def __init__(self) -> None:
        """
        Firestoreクライアントとセッションを初期化
        """
        try:
            if not firebase_admin._apps:
                firebase_admin.initialize_app()
            self.db = firestore.client()
            self.collection_name: str = 'external_data'
            logger.info("Firestore client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore client: {str(e)}")
            raise StorageError("Firestore initialization failed") from e

    async def fetch_and_store(
        self,
        source_name: str,
        force_refresh: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        外部データを取得し、Firestoreに保存します

        Args:
            source_name: データソース識別子
            force_refresh: キャッシュを無視して強制的に再取得するかどうか
            **kwargs: データソース固有のパラメータ

        Returns:
            Dict[str, Any]: 取得したデータ
        """
        try:
            if not force_refresh:
                cached_data = await self._get_cached_data(source_name)
                if cached_data:
                    logger.info(f"Returning cached data for {source_name}")
                    return cached_data

            if source_name.endswith('_api'):
                data = await self._fetch_api_data(source_name, **kwargs)
            elif source_name.endswith('_webpage'):
                data = await self._fetch_webpage_data(source_name, **kwargs)
            else:
                raise ValueError(f"Unsupported source name: {source_name}")

            await self._store_data(source_name, data)
            return data

        except Exception as e:
            error_msg = f"Error fetching data from {source_name}: {str(e)}"
            logger.error(error_msg)
            raise ExternalDataError(error_msg) from e

    async def _get_cached_data(
        self,
        source_name: str
    ) -> Dict[str, Any]:
        """
        Firestoreからキャッシュされたデータを取得
        """
        try:
            loop = asyncio.get_running_loop()
            query = self.db.collection(self.collection_name)\
                .where('source_name', '==', source_name)\
                .where('timestamp', '>=',
                      datetime.now() - timedelta(seconds=CACHE_EXPIRY))\
                .limit(1)

            docs = await loop.run_in_executor(None, lambda: list(query.stream()))

            if docs:
                doc_dict = docs[0].to_dict()
                if doc_dict is not None and 'data' in doc_dict:
                    return doc_dict['data']

            return {}

        except Exception as e:
            logger.warning(f"Cache retrieval failed for {source_name}: {str(e)}")
            return {}

    async def _store_data(
        self,
        source_name: str,
        data: Dict[str, Any]
    ) -> None:
        """
        データをFirestoreに保存
        """
        try:
            doc_data = {
                'source_name': source_name,
                'data': data,
                'timestamp': datetime.now(),
                'status': 'success'
            }

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.db.collection(self.collection_name).add(doc_data)
            )

            logger.info(f"Successfully stored data for {source_name}")

        except Exception as e:
            logger.error(f"Failed to store data for {source_name}: {str(e)}")
            raise StorageError(f"Failed to store external data: {str(e)}") from e

    async def _fetch_api_data(
        self,
        api_endpoint: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        APIからデータを非同期に取得
        """
        async with aiohttp.ClientSession() as session:
            for attempt in range(MAX_RETRY_ATTEMPTS):
                try:
                    headers = kwargs.get('headers', {})
                    params = kwargs.get('params', {})

                    async with session.get(
                        api_endpoint,
                        headers=headers,
                        params=params,
                        timeout=REQUEST_TIMEOUT
                    ) as response:
                        response.raise_for_status()
                        return await response.json()

                except aiohttp.ClientError as e:
                    if attempt == MAX_RETRY_ATTEMPTS - 1:
                        raise ExternalDataError(f"API request failed: {str(e)}")
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))

            return {}

    async def _fetch_webpage_data(
        self,
        url: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Webページからデータを非同期に取得
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=REQUEST_TIMEOUT) as response:
                    response.raise_for_status()
                    content = await response.text()

                    soup = BeautifulSoup(content, 'html.parser')

                    selector = str(kwargs.get('selector', ''))
                    if not selector:
                        raise ValueError("CSS selector is required for webpage scraping")

                    elements = soup.select(selector)
                    if not elements:
                        raise ExternalDataError(f"No data found for selector: {selector}")

                    return {
                        'content': [element.text.strip() for element in elements],
                        'url': url,
                        'timestamp': datetime.now().isoformat()
                    }

            except Exception as e:
                raise ExternalDataError(f"Webpage scraping failed: {str(e)}") from e

    async def close(self) -> None:
        """
        リソースのクリーンアップ
        """
        logger.info("External data fetcher closed successfully")