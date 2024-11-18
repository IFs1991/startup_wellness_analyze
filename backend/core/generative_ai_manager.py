"""
Gemini APIとの通信およびレスポンスの永続化を管理するサービス
"""
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
import requests
from fastapi import HTTPException, status

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class GenerativeAIManager:
    """
    Gemini APIを使用した生成AIの管理とレスポンスの永続化を行うクラス
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        GenerativeAIManagerを初期化します

        Args:
            api_key (Optional[str]): Gemini APIキー。指定がない場合は環境変数から取得。

        Raises:
            ValueError: APIキーが設定されていない場合
        """
        try:
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
            if self.api_key is None:
                raise ValueError("GEMINI_API_KEY が設定されていません。")

            self.api_endpoint = os.environ.get("GEMINI_API_ENDPOINT")
            if self.api_endpoint is None:
                raise ValueError("GEMINI_API_ENDPOINT が設定されていません。")

            from backend.service.firestore.client import FirestoreService
            self.firestore_service = FirestoreService()
            logger.info("GenerativeAIManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize GenerativeAIManager: {str(e)}")
            raise

    async def generate_text(
        self,
        prompt: str,
        user_id: str,
        model: str = "gemini-pro",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Gemini APIを使用してテキストを生成し、結果をFirestoreに保存します

        Args:
            prompt (str): 生成AIへのプロンプト
            user_id (str): リクエストを行うユーザーのID
            model (str, optional): 使用するモデル名。デフォルトは"gemini-pro"
            metadata (Dict[str, Any], optional): 追加のメタデータ

        Returns:
            Dict[str, Any]: 生成されたテキストとメタデータを含む辞書

        Raises:
            HTTPException: API通信やデータ保存に失敗した場合
        """
        try:
            if not isinstance(self.api_endpoint, str):
                raise ValueError("GEMINI_API_ENDPOINT must be a string")

            # APIリクエストの準備
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "prompt": prompt,
                "model": model,
            }

            # API呼び出し
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=data
            )
            response.raise_for_status()

            # レスポンスの処理
            result = response.json()
            generated_text = result.get("text", "")

            # Firestoreに保存するデータの準備
            storage_data = {
                "prompt": prompt,
                "response": generated_text,
                "model": model,
                "user_id": user_id,
                "timestamp": datetime.now(),
                "metadata": metadata or {}
            }

            # Firestoreへの保存
            doc_ids = await self.firestore_service.save_results(
                results=[storage_data],
                collection_name="ai_generations"
            )

            # 結果の返却
            return {
                "text": generated_text,
                "document_id": doc_ids[0],
                "timestamp": storage_data["timestamp"],
                "metadata": storage_data["metadata"]
            }

        except requests.exceptions.RequestException as e:
            error_msg = f"Error communicating with Gemini API: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=error_msg
            )

        except Exception as e:
            error_msg = f"Error in text generation process: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )

    async def get_generation_history(
        self,
        user_id: str,
        limit: Optional[int] = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        ユーザーの生成履歴を取得します

        Args:
            user_id (str): ユーザーID
            limit (int, optional): 取得する履歴の最大数。デフォルトは10
            offset (int, optional): 取得開始位置。デフォルトは0

        Returns:
            List[Dict[str, Any]]: 生成履歴のリスト
        """
        try:
            conditions = [{
                "field": "user_id",
                "operator": "==",
                "value": user_id
            }]

            history = await self.firestore_service.fetch_documents(
                collection_name="ai_generations",
                conditions=conditions,
                limit=limit,
                offset=offset,
                order_by="timestamp",
                direction="desc"
            )

            return history

        except Exception as e:
            error_msg = f"Error fetching generation history: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )