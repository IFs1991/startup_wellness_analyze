# -*- coding: utf-8 -*-
"""
Firestoreスケーラビリティサービス
データ量増加に伴う処理の最適化と効率的なクエリ実行を提供します。
"""
from typing import (
    List, Dict, Any, Optional, AsyncGenerator,
    TypeVar, Callable, Union, cast, Protocol
)
import asyncio
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from google.cloud import firestore
from google.cloud.firestore_v1 import (
    Client,
    AsyncClient,
    DocumentReference,
    DocumentSnapshot,
    CollectionReference,
    Query,
    WriteBatch
)
import firebase_admin
from firebase_admin import credentials

# 型変数の定義
T = TypeVar('T')
DocumentType = Dict[str, Any]

class ProcessorProtocol(Protocol):
    def __call__(self, data: DocumentType, doc_id: str) -> Any: ...

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class ScalingError(Exception):
    """スケーリング操作に関するエラー"""
    pass

class FirestoreScalabilityService:
    """
    Firestoreのスケーラビリティを管理するサービスクラス
    """
    def __init__(self) -> None:
        """
        Firestoreクライアントとスケーリング設定を初期化
        """
        try:
            if not firebase_admin._apps:
                firebase_admin.initialize_app()
            self.db: Client = firestore.Client()
            self.executor = ThreadPoolExecutor(max_workers=10)
            self.batch_size = 500
            logger.info("Firestore Scalability Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore Scalability Service: {str(e)}")
            raise ScalingError(f"Initialization error: {str(e)}")

    async def process_large_collection(
        self,
        collection_name: str,
        processor: ProcessorProtocol,
        query_conditions: Optional[List[Dict[str, Any]]] = None,
        chunk_size: int = 100
    ) -> Dict[str, Any]:
        """
        大規模コレクションを効率的に処理
        """
        try:
            start_time = datetime.now()
            processed_count = 0
            error_count = 0

            # クエリの構築
            collection_ref: CollectionReference = self.db.collection(collection_name)
            query: Union[CollectionReference, Query] = collection_ref

            if query_conditions:
                for condition in query_conditions:
                    field = condition.get('field')
                    operator = condition.get('operator', '==')
                    value = condition.get('value')
                    if all([field, operator, value is not None]):
                        query = cast(Query, query).where(field, operator, value)

            # チャンク単位での非同期処理
            async for chunk in self._get_document_chunks(query, chunk_size):
                try:
                    tasks = []
                    for doc in chunk:
                        doc_data = doc.to_dict()
                        if doc_data is not None:
                            tasks.append(self._process_document(doc_data, doc.id, processor))

                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        for result in results:
                            if isinstance(result, Exception):
                                error_count += 1
                            else:
                                processed_count += 1

                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
                    error_count += len(chunk)

            # 処理統計の作成
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            stats = {
                'processed_count': processed_count,
                'error_count': error_count,
                'duration_seconds': duration,
                'documents_per_second': processed_count / duration if duration > 0 else 0
            }

            logger.info(f"Collection processing completed: {stats}")
            return stats

        except Exception as e:
            error_msg = f"Error in large collection processing: {str(e)}"
            logger.error(error_msg)
            raise ScalingError(error_msg)

    async def _get_document_chunks(
        self,
        query: Union[CollectionReference, Query],
        chunk_size: int
    ) -> AsyncGenerator[List[DocumentSnapshot], None]:
        """
        ドキュメントをチャンク単位で取得するジェネレータ
        """
        try:
            last_doc = None
            while True:
                current_query = cast(Query, query.limit(chunk_size))

                if last_doc:
                    current_query = current_query.start_after(last_doc)

                docs = await self._execute_query(current_query)

                if not docs:
                    break

                last_doc = docs[-1]
                yield docs
        except Exception as e:
            logger.error(f"Error in document chunk generation: {str(e)}")
            raise ScalingError(f"Chunk generation error: {str(e)}")

    async def _execute_query(
        self,
        query: Query
    ) -> List[DocumentSnapshot]:
        """
        クエリを非同期実行
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, query.get)
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            raise ScalingError(f"Query execution error: {str(e)}")

    async def _process_document(
        self,
        doc_data: DocumentType,
        doc_id: str,
        processor: ProcessorProtocol
    ) -> Any:
        """
        個別ドキュメントの処理を実行
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                processor,
                doc_data,
                doc_id
            )
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            raise ScalingError(f"Document processing error: {str(e)}")

    async def create_distributed_counter(
        self,
        collection_name: str,
        counter_id: str,
        num_shards: int = 5
    ) -> None:
        """
        分散カウンターを作成
        """
        try:
            batch: WriteBatch = self.db.batch()
            counter_ref: DocumentReference = self.db.collection(collection_name).document(counter_id)

            for i in range(num_shards):
                shard_ref = counter_ref.collection('shards').document(str(i))
                batch.set(shard_ref, {'count': 0})

            await self._execute_batch(batch)
            logger.info(f"Created distributed counter {counter_id} with {num_shards} shards")

        except Exception as e:
            error_msg = f"Error creating distributed counter: {str(e)}"
            logger.error(error_msg)
            raise ScalingError(error_msg)

    async def _execute_batch(self, batch: WriteBatch) -> None:
        """
        バッチ処理を非同期実行
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, batch.commit)
        except Exception as e:
            logger.error(f"Batch execution error: {str(e)}")
            raise ScalingError(f"Batch execution error: {str(e)}")

    async def close(self) -> None:
        """
        リソースの解放
        """
        try:
            self.executor.shutdown(wait=True)
            logger.info("Firestore Scalability Service shutdown completed")
        except Exception as e:
            logger.error(f"Error during service shutdown: {str(e)}")
            raise ScalingError(f"Shutdown error: {str(e)}")

def get_scalability_service() -> FirestoreScalabilityService:
    """
    FirestoreScalabilityServiceのインスタンスを取得
    """
    return FirestoreScalabilityService()