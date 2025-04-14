# -*- coding: utf-8 -*-

"""
テキスト分析

Firestore からテキストデータを取得し、テキスト分析を実行します。
分析結果は Firestore に保存し、集計用データは BigQuery に保存します。
時系列分析は別モジュールで実行されます。
"""

import logging
from typing import Optional, Any, Dict, List, Tuple, Union
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from dataclasses import dataclass
from service.bigquery.queries.data_queries import DataQueries
from service.firestore.client import FirestoreService
import traceback
import gc
import contextlib

# NLTKの必要なリソースをダウンロード
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"NLTKリソースのダウンロードに失敗しました: {str(e)}")

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
        self._resources = []  # リソース追跡用

    def __del__(self):
        """
        デストラクタ：リソースの自動解放
        """
        self.release_resources()

    def release_resources(self):
        """
        使用したリソースを解放する
        """
        try:
            # 明示的にガベージコレクションを実行
            gc.collect()
            logger.debug("リソースを解放しました")
        except Exception as e:
            logger.error(f"リソース解放中にエラーが発生しました: {str(e)}")

    def _analyze_single_text(self, text: str) -> Dict[str, Any]:
        """
        単一のテキストを分析します。

        Args:
            text (str): 分析対象のテキスト

        Returns:
            Dict[str, Any]: 分析結果（感情分析、トークン化、単語数など）

        Raises:
            ValueError: テキストが空または無効な場合
        """
        if not text or not isinstance(text, str):
            raise ValueError("分析対象のテキストが無効です")

        try:
            # テキストのトークン化と前処理
            tokens = self._tokenize_text(text)

            # ストップワード除去
            filtered_tokens = self._remove_stopwords(tokens)

            # 感情分析
            sentiment = self._analyze_sentiment(text)

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
            logger.error(f"テキスト分析中にエラーが発生しました: {str(e)}")
            logger.debug(traceback.format_exc())
            raise ValueError(f"テキスト分析に失敗しました: {str(e)}")

    def _tokenize_text(self, text: str) -> List[str]:
        """
        テキストをトークン化します。

        Args:
            text (str): トークン化するテキスト

        Returns:
            List[str]: トークンのリスト
        """
        return word_tokenize(text)

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        トークンからストップワードを除去します。

        Args:
            tokens (List[str]): トークンのリスト

        Returns:
            List[str]: ストップワードが除去されたトークンのリスト
        """
        return [w for w in tokens if w.lower() not in self.stop_words]

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        テキストの感情分析を実行します。

        Args:
            text (str): 分析対象のテキスト

        Returns:
            Dict[str, float]: 感情分析スコア
        """
        return self.sia.polarity_scores(text)

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

    @contextlib.contextmanager
    def _managed_results(self):
        """
        分析結果を管理するコンテキストマネージャー
        """
        results = []
        try:
            yield results
        finally:
            # 明示的なクリーンアップ
            if results:
                for r in results:
                    # 大きなトークンリストへの参照を削除
                    if 'tokens' in r:
                        del r['tokens']
                    if 'filtered_tokens' in r:
                        del r['filtered_tokens']

            # メモリ使用量のログ記録（デバッグ用）
            logger.debug("結果管理コンテキスト終了")
            gc.collect()

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
        if not config or not config.collection_name:
            raise ValueError("有効な分析設定が必要です")

        try:
            logger.info(f"コレクション {config.collection_name} の分析を開始します")

            # Firestoreからドキュメントを取得
            documents = await self._fetch_documents(config.collection_name)
            if not documents:
                logger.warning("分析対象のドキュメントがありません")
                return self._create_empty_results()

            # 分析の実行とプロセス
            results_summary = await self._process_documents(documents, config)

            # 結果の保存（設定に応じて）
            if config.save_results and results_summary.get("results"):
                await self._save_results(results_summary["results"], config)

            return results_summary

        except Exception as e:
            logger.error(f"テキスト分析に失敗しました: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"テキスト分析の実行中にエラーが発生しました: {str(e)}")

    async def _fetch_documents(self, collection_name: str) -> List[Dict]:
        """
        Firestoreからドキュメントを取得します。

        Args:
            collection_name (str): 取得するコレクション名

        Returns:
            List[Dict]: 取得したドキュメントのリスト
        """
        documents = await self.firestore_service.fetch_documents(collection_name)
        logger.info(f"分析のために {len(documents)} 件のドキュメントを取得しました")
        return documents

    def _create_empty_results(self) -> Dict[str, Any]:
        """
        空の結果セットを作成します。

        Returns:
            Dict[str, Any]: 空の分析結果
        """
        return {
            "results": [],
            "metadata": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "total_documents": 0,
                "analyzed_documents": 0,
                "average_sentiment": 0
            }
        }

    async def _process_documents(self, documents: List[Dict], config: TextAnalysisConfig) -> Dict[str, Any]:
        """
        ドキュメントを処理して分析結果を作成します。

        Args:
            documents (List[Dict]): 処理するドキュメント
            config (TextAnalysisConfig): 分析設定

        Returns:
            Dict[str, Any]: 分析結果の要約
        """
        with self._managed_results() as results:
            # 各ドキュメントを処理
            for doc in documents:
                text = doc.get('text', '')
                if text:
                    try:
                        # 単一テキストの分析
                        raw_result = self._analyze_single_text(text)

                        # IDの追加
                        raw_result['id'] = doc.get('id')

                        # 保存用に整形
                        processed_result = self._prepare_analysis_data(raw_result)
                        results.append(processed_result)
                    except Exception as e:
                        # 個別ドキュメントのエラーはスキップして続行
                        logger.warning(f"ドキュメント {doc.get('id')} の処理中にエラー: {str(e)}")

            # 結果の集計と要約の作成
            summary = self._create_analysis_summary(results, documents)

            return summary

    def _create_analysis_summary(self, results: List[Dict], documents: List[Dict]) -> Dict[str, Any]:
        """
        分析結果の要約を作成します。

        Args:
            results (List[Dict]): 分析結果のリスト
            documents (List[Dict]): 元のドキュメントのリスト

        Returns:
            Dict[str, Any]: 要約情報を含む辞書
        """
        # 平均感情スコアの計算
        avg_sentiment = 0
        if results:
            avg_sentiment = sum(r['sentiment_scores']['compound'] for r in results) / len(results)

        # 要約の作成
        return {
            "results": results,
            "metadata": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "total_documents": len(documents),
                "analyzed_documents": len(results),
                "average_sentiment": avg_sentiment
            }
        }

    async def _save_results(self, results: List[Dict], config: TextAnalysisConfig) -> None:
        """
        分析結果を保存します。

        Args:
            results (List[Dict]): 保存する分析結果
            config (TextAnalysisConfig): 保存設定
        """
        # Firestoreへの保存
        if config.output_collection_name:
            logger.info(f"結果を Firestore コレクション {config.output_collection_name} に保存します")
            await self.firestore_service.save_results(results, config.output_collection_name)

        # BigQueryへの保存（未実装）
        if config.save_to_bigquery and config.bigquery_dataset and config.bigquery_table:
            logger.info(f"BigQuery {config.bigquery_dataset}.{config.bigquery_table} への保存はまだ実装されていません")


async def analyze_text(request: Any) -> Tuple[Dict[str, Any], int]:
    """
    Cloud Functions用のエントリーポイント関数

    Args:
        request: Cloud Functionsのリクエストオブジェクト

    Returns:
        Tuple[Dict[str, Any], int]: (レスポンス, ステータスコード)
    """
    try:
        # リクエストの検証
        request_json = request.get_json()
        if not request_json:
            logger.error("リクエストデータがありません")
            return {'error': 'リクエストデータがありません'}, 400

        # 必須パラメータの検証
        if 'collection_name' not in request_json:
            logger.error("必須パラメータ 'collection_name' がありません")
            return {'error': "必須パラメータ 'collection_name' が指定されていません"}, 400

        # 設定オブジェクトの作成
        config = TextAnalysisConfig(
            collection_name=request_json['collection_name'],
            output_collection_name=request_json.get('output_collection_name'),
            save_results=request_json.get('save_results', True),
            save_to_bigquery=request_json.get('save_to_bigquery', False),
            bigquery_dataset=request_json.get('bigquery_dataset'),
            bigquery_table=request_json.get('bigquery_table')
        )

        # サービスの初期化と分析の実行
        firestore_service = FirestoreService()
        text_miner = TextMiner(firestore_service)

        try:
            results = await text_miner.analyze(config)
            logger.info("分析が正常に完了しました")
            return {
                'status': 'success',
                'results': results
            }, 200
        finally:
            # リソースの解放
            text_miner.release_resources()

    except ValueError as e:
        # 検証エラー
        logger.error(f"検証エラー: {str(e)}")
        return {
            'status': 'error',
            'type': 'validation_error',
            'message': str(e)
        }, 400
    except Exception as e:
        # その他のエラー
        logger.error(f"analyze_text 関数でエラーが発生しました: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            'status': 'error',
            'type': 'internal_error',
            'message': str(e)
        }, 500