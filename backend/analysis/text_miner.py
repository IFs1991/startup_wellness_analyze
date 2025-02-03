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
from backend.src.database.bigquery.queries.data_queries import DataQueries
from backend.src.database.firestore.client import FirestoreClient
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import MeCab
from . import BaseAnalyzer

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


class TextMiner(BaseAnalyzer):
    """テキストマイニングを行うクラス"""

    def __init__(self, text_column: str, n_topics: int = 5, max_features: int = 1000):
        """
        初期化メソッド

        Args:
            text_column (str): テキストデータのカラム名
            n_topics (int): トピック数
            max_features (int): 使用する特徴量（単語）の最大数
        """
        super().__init__("text_mining")
        self.text_column = text_column
        self.n_topics = n_topics
        self.max_features = max_features
        self.mecab = MeCab.Tagger("-Ochasen")

    def _preprocess_text(self, text: str) -> str:
        """
        テキストの前処理を行う

        Args:
            text (str): 前処理対象のテキスト

        Returns:
            str: 前処理済みのテキスト
        """
        try:
            # MeCabで形態素解析
            node = self.mecab.parseToNode(str(text))
            words = []

            while node:
                # 品詞情報を取得
                pos = node.feature.split(',')[0]

                # 名詞、動詞、形容詞のみを抽出
                if pos in ['名詞', '動詞', '形容詞']:
                    words.append(node.surface)

                node = node.next

            return ' '.join(words)
        except Exception as e:
            self.logger.error(f"Error in text preprocessing: {str(e)}")
            return ""

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        テキストマイニングを実行

        Args:
            data (pd.DataFrame): 分析対象データ

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            # テキストの前処理
            processed_texts = [
                self._preprocess_text(text)
                for text in data[self.text_column].fillna('')
            ]

            # TF-IDF変換
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english'  # 日本語の場合は前処理で対応
            )
            tfidf_matrix = vectorizer.fit_transform(processed_texts)

            # 頻出語の抽出
            word_freq = {
                word: tfidf_matrix.getcol(idx).sum()
                for word, idx in vectorizer.vocabulary_.items()
            }
            top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50])

            # トピックモデリング（LDA）
            lda = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42
            )
            lda.fit(tfidf_matrix)

            # トピックごとの特徴語を抽出
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-10:-1]  # 上位10語を取得
                topics.append({
                    'topic_id': topic_idx,
                    'words': [
                        {
                            'word': feature_names[i],
                            'weight': float(topic[i])
                        }
                        for i in top_words_idx
                    ]
                })

            # 文書ごとのトピック分布
            doc_topics = lda.transform(tfidf_matrix)

            # 結果の整形
            result = {
                'word_frequencies': top_words,
                'topics': topics,
                'document_topic_distribution': doc_topics.tolist(),
                'summary': {
                    'total_documents': len(data),
                    'vocabulary_size': len(vectorizer.vocabulary_),
                    'n_topics': self.n_topics,
                    'sparsity': float(tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in text mining analysis: {str(e)}")
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
        firestore_client = FirestoreClient()
        text_miner = TextMiner(firestore_client.collection_name, 5, 1000)

        # テキスト分析の実行
        results = text_miner.analyze(firestore_client.fetch_documents())

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