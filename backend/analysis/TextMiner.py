# -*- coding: utf-8 -*-

"""
テキスト分析

Firestore からテキストデータを取得し、テキスト分析を実行します。
分析結果は Firestore に保存し、集計用データは BigQuery に保存します。
時系列分析は別モジュールで実行されます。
"""

import logging
from typing import Optional, Any, Dict, List, Tuple
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from dataclasses import dataclass
from service.bigquery.queries.data_queries import DataQueries
from service.firestore.client import FirestoreService

# NLTKの必要なリソースをダウンロード
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# ロガーの設定
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class TextAnalysisConfig:
    """テキスト分析の設定を保持するデータクラス"""
    collection_name: str
    save_results: bool = True
    output_collection_name: Optional[str] = None
    save_to_bigquery: bool = False
    bigquery_dataset: Optional[str] = None
    bigquery_table: Optional[str] = None


class TextMiner:
    """
    テキスト分析を実行するクラスです。
    基本的なテキスト分析機能を提供し、結果を保存します。
    """

    def __init__(self, firestore_service: FirestoreService):
        """
        Args:
            firestore_service (FirestoreService): Firestore操作用のサービスインスタンス
        """
        self.firestore_service = firestore_service
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.data_queries = DataQueries()

    def _analyze_single_text(self, text: str) -> Dict[str, Any]:
        """
        単一のテキストを分析します。

        Args:
            text (str): 分析対象のテキスト

        Returns:
            Dict[str, Any]: 分析結果（感情分析、トークン化、単語数など）
        """
        try:
            tokens = word_tokenize(text)
            filtered_tokens = [w for w in tokens if not w.lower() in self.stop_words]
            sentiment = self.sia.polarity_scores(text)

            return {
                'tokens': tokens,
                'filtered_tokens': filtered_tokens,
                'sentiment': sentiment,
                'word_count': len(tokens),
                'filtered_word_count': len(filtered_tokens),
                'sentiment_scores': {
                    'compound': sentiment['compound'],
                    'positive': sentiment['pos'],
                    'negative': sentiment['neg'],
                    'neutral': sentiment['neu']
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            raise

    def _prepare_analysis_data(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析結果を保存用に整形します。

        Args:
            raw_result (Dict[str, Any]): 生の分析結果

        Returns:
            Dict[str, Any]: 整形された分析結果
        """
        return {
            'id': raw_result.get('id'),
            'timestamp': pd.Timestamp.now().isoformat(),
            'word_count': raw_result['word_count'],
            'filtered_word_count': raw_result['filtered_word_count'],
            'sentiment_scores': raw_result['sentiment_scores'],
            'tokens_sample': raw_result['tokens'][:100] if raw_result.get('tokens') else [],
            'analysis_metadata': {
                'analyzer_version': '1.0',
                'language': 'english',
            }
        }

    async def analyze(self, config: TextAnalysisConfig) -> Dict[str, Any]:
        """
        テキスト分析を実行します。

        Args:
            config (TextAnalysisConfig): テキスト分析の設定

        Returns:
            Dict[str, Any]: 分析結果

        Raises:
            RuntimeError: 計算実行中のエラー
        """
        try:
            logger.info(f"Starting analysis for collection: {config.collection_name}")

            # Firestoreからドキュメントを取得
            documents = await self.firestore_service.fetch_documents(config.collection_name)
            logger.info(f"Retrieved {len(documents)} documents for analysis")

            # テキスト分析を実行
            results = []
            for doc in documents:
                text = doc.get('text', '')
                if text:
                    raw_result = self._analyze_single_text(text)
                    processed_result = self._prepare_analysis_data({
                        'id': doc.get('id'),
                        **raw_result
                    })
                    results.append(processed_result)

            # 分析結果の集計
            analysis_summary = {
                "results": results,
                "metadata": {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "total_documents": len(documents),
                    "analyzed_documents": len(results),
                    "average_sentiment": sum(r['sentiment_scores']['compound'] for r in results) / len(results) if results else 0
                }
            }

            # 分析結果をFirestoreに保存
            if config.save_results and config.output_collection_name:
                logger.info(f"Saving results to Firestore collection: {config.output_collection_name}")
                await self.firestore_service.save_results(results, config.output_collection_name)

            # BigQueryにデータを保存
            if config.save_to_bigquery and config.bigquery_dataset and config.bigquery_table:
                logger.info(f"Saving analysis data to BigQuery: {config.bigquery_dataset}.{config.bigquery_table}")
                # BigQueryへの保存は別途実装予定

            return analysis_summary

        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            raise


async def analyze_text(request: Any) -> Tuple[Dict[str, Any], int]:
    """
    Cloud Functions用のエントリーポイント関数

    Args:
        request: Cloud Functionsのリクエストオブジェクト

    Returns:
        Tuple[Dict[str, Any], int]: (レスポンス, ステータスコード)
    """
    try:
        request_json = request.get_json()

        if not request_json:
            logger.error("No request data provided")
            return {'error': 'リクエストデータがありません'}, 400

        # 必須パラメータのバリデーション
        required_params = ['collection_name']
        for param in required_params:
            if param not in request_json:
                logger.error(f"Missing required parameter: {param}")
                return {'error': f"必須パラメータ '{param}' が指定されていません"}, 400

        # 設定オブジェクトの作成
        config = TextAnalysisConfig(
            collection_name=request_json['collection_name'],
            output_collection_name=request_json.get('output_collection_name'),
            save_results=request_json.get('save_results', True),
            save_to_bigquery=request_json.get('save_to_bigquery', False),
            bigquery_dataset=request_json.get('bigquery_dataset'),
            bigquery_table=request_json.get('bigquery_table')
        )

        # サービスの初期化
        firestore_service = FirestoreService()
        text_miner = TextMiner(firestore_service)

        # テキスト分析の実行
        results = await text_miner.analyze(config)

        logger.info("Analysis completed successfully")
        return {
            'status': 'success',
            'results': results
        }, 200

    except Exception as e:
        logger.error(f"Error in analyze_text: {str(e)}")
        return {
            'status': 'error',
            'type': 'internal_error',
            'message': str(e)
        }, 500