# -*- coding: utf-8 -*-
"""
テキスト分析サービス
アンケート自由記述項目から有益な情報を抽出し、Firestoreに保存します。
Google Gemini APIを使用して、より高度な感情分析と情報抽出を行います。
"""
from typing import Dict, List, Any, Optional, NamedTuple
import logging
from datetime import datetime
from google.cloud import firestore
import asyncio
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
import json

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TextAnalysisError(Exception):
    """テキスト分析に関するエラー"""
    pass

class AnalysisResult(NamedTuple):
    """分析結果を格納する型"""
    sentiment_score: float  # -1.0 から 1.0
    keywords: List[str]
    categories: List[str]
    summary: str
    main_topics: List[str]
    word_count: int

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "sentiment_score": self.sentiment_score,
            "keywords": self.keywords,
            "categories": self.categories,
            "summary": self.summary,
            "main_topics": self.main_topics,
            "statistics": {
                "word_count": self.word_count
            }
        }

class TextMiner:
    """テキストデータを分析し、結果をFirestoreに保存するクラスです。"""
    def __init__(self, db: firestore.Client, gemini_api_key: str):
        """
        初期化
        Args:
            db (firestore.Client): Firestoreクライアントのインスタンス
            gemini_api_key (str): Gemini APIキー
        """
        self.db = db
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.executor = ThreadPoolExecutor(max_workers=3)
        logger.info("TextMiner initialized successfully")

    async def _analyze_with_gemini(self, text: str) -> Dict[str, Any]:
        """
        Geminiを使用してテキスト分析を実行

        Args:
            text (str): 分析対象テキスト

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            prompt = f"""
            以下のテキストを分析し、JSONフォーマットで結果を返してください。
            返却値は必ず以下のJSONスキーマに準拠してください。

            テキスト: {text}

            JSONスキーマ:
            {{
                "sentiment_score": float,  // -1.0(ネガティブ) から 1.0(ポジティブ)
                "keywords": string[],      // 重要なキーワード（最大5つ）
                "categories": string[],    // テキストのカテゴリ（最大3つ）
                "summary": string,         // 1-2文での要約
                "main_topics": string[]    // 主要なトピック（最大3つ）
            }}
            """

            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.model.generate_content(prompt)
            )

            # レスポンスからJSONを抽出して解析
            result = json.loads(response.text)
            return result

        except Exception as e:
            error_msg = f"Error in Gemini analysis: {str(e)}"
            logger.error(error_msg)
            raise TextAnalysisError(error_msg) from e

    async def analyze_text(
        self,
        text: str,
        user_id: str,
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        テキスト分析を実行し、結果をFirestoreに保存します。

        Args:
            text (str): 分析対象のテキストデータ
            user_id (str): ユーザーID
            source_id (Optional[str]): 元データのID
            metadata (Optional[Dict[str, Any]]): 追加のメタデータ

        Returns:
            Dict[str, Any]: 分析結果と保存されたドキュメントID
        """
        try:
            logger.info(f"Starting text analysis for user: {user_id}")

            # Geminiによる分析
            gemini_analysis = await self._analyze_with_gemini(text)

            # 結果の構造化
            analysis_result = AnalysisResult(
                sentiment_score=gemini_analysis["sentiment_score"],
                keywords=gemini_analysis["keywords"],
                categories=gemini_analysis["categories"],
                summary=gemini_analysis["summary"],
                main_topics=gemini_analysis["main_topics"],
                word_count=len(text.split())
            )

            # Firestoreに保存するデータの準備
            document_data = {
                'text': text,
                'analysis_result': analysis_result.to_dict(),
                'user_id': user_id,
                'source_id': source_id,
                'metadata': metadata or {},
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }

            # Firestoreへの保存
            doc_ref = self.db.collection('text_analysis').document()
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: doc_ref.set(document_data)
            )

            result = {
                'document_id': doc_ref.id,
                'analysis': analysis_result.to_dict()
            }

            logger.info(f"Text analysis completed and saved with ID: {doc_ref.id}")
            return result

        except Exception as e:
            error_msg = f"Error in text analysis: {str(e)}"
            logger.error(error_msg)
            raise TextAnalysisError(error_msg) from e

    async def get_analysis_history(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        ユーザーの分析履歴を取得します。

        Args:
            user_id (str): ユーザーID
            limit (int): 取得する履歴の最大数

        Returns:
            List[Dict[str, Any]]: 分析履歴のリスト
        """
        try:
            logger.info(f"Fetching analysis history for user: {user_id}")

            query = (self.db.collection('text_analysis')
                    .where('user_id', '==', user_id)
                    .order_by('created_at', direction=firestore.Query.DESCENDING)
                    .limit(limit))

            docs = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: list(query.stream())
            )

            history = []
            for doc in docs:
                doc_dict = doc.to_dict()
                if doc_dict is not None:
                    doc_dict['id'] = doc.id
                    history.append(doc_dict)

            logger.info(f"Successfully retrieved {len(history)} analysis records")
            return history

        except Exception as e:
            error_msg = f"Error fetching analysis history: {str(e)}"
            logger.error(error_msg)
            raise TextAnalysisError(error_msg) from e

    def __del__(self):
        """クリーンアップ処理"""
        self.executor.shutdown(wait=False)

def get_text_miner(
    db: firestore.Client,
    gemini_api_key: str
) -> TextMiner:
    """
    TextMinerインスタンスを取得します。

    Args:
        db (firestore.Client): Firestoreクライアントのインスタンス
        gemini_api_key (str): Gemini APIキー

    Returns:
        TextMiner: 初期化されたTextMinerインスタンス
    """
    return TextMiner(db, gemini_api_key)