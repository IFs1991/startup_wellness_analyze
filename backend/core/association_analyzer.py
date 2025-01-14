# -*- coding: utf-8 -*-

"""
アソシエーション分析

特定の健康状態と関連性の高い行動や属性を特定します。
マルチスレッドと非同期処理に対応しています。
MLxtendライブラリを使用して、データセット内のパターンや関連性を発見します。
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import asyncio
import logging
from typing import Dict, List, Union
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# ロギングの設定
# アプリケーションの動作状況を詳細に記録するためのロガーを設定します
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class AssociationAnalyzer:
    """
    アソシエーション分析を実行するためのクラス。
    マルチスレッドと非同期処理に対応しており、大規模なデータセットの効率的な分析が可能です。
    """
    def __init__(self, max_workers: int = 4):
        """
        初期化メソッド

        Args:
            max_workers (int): スレッドプールの最大ワーカー数。
                             データセットのサイズに応じて適切な値を設定してください。
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logger

    async def analyze(
        self,
        data: pd.DataFrame,
        min_support: float,
        metric: str = "lift",
        min_threshold: float = 1.0
    ) -> Dict[str, Union[pd.DataFrame, List[Dict]]]:
        """
        アソシエーション分析を非同期で実行します。

        Args:
            data (pd.DataFrame): ワンホットエンコーディングされたデータ。
                               各列は特徴を表し、値は0または1である必要があります。
            min_support (float): 最小サポート値。0から1の間の値を指定してください。
            metric (str): 評価指標（デフォルト: "lift"）。
                         "lift"、"confidence"、"support"などが指定可能です。
            min_threshold (float): 最小しきい値（デフォルト: 1.0）。
                                 評価指標の最小値を指定します。

        Returns:
            Dict[str, Union[pd.DataFrame, List[Dict]]]: 分析結果
                - rules: アソシエーションルール（DataFrame形式）
                - summary: 分析サマリー（List[Dict]形式）

        Raises:
            ValueError: 入力データが不正な場合
            RuntimeError: 分析実行中にエラーが発生した場合
        """
        try:
            # 入力データの検証
            if not isinstance(data, pd.DataFrame):
                raise ValueError("データはpandas DataFrameである必要があります")

            if not (0 < min_support < 1):
                raise ValueError("min_supportは0から1の間である必要があります")

            self.logger.info("アソシエーション分析を開始します")

            # 非同期実行のためのループを取得
            loop = asyncio.get_event_loop()

            # 同期的な分析処理を非同期的に実行
            result = await loop.run_in_executor(
                self.executor,
                partial(self._analyze_sync, data, min_support, metric, min_threshold)
            )

            self.logger.info("アソシエーション分析が完了しました")
            return result

        except Exception as e:
            self.logger.error(f"分析中にエラーが発生しました: {str(e)}")
            raise RuntimeError(f"アソシエーション分析に失敗しました: {str(e)}") from e

    def _analyze_sync(
        self,
        data: pd.DataFrame,
        min_support: float,
        metric: str,
        min_threshold: float
    ) -> Dict[str, Union[pd.DataFrame, List[Dict]]]:
        """
        同期的にアソシエーション分析を実行します。

        Args:
            data (pd.DataFrame): 入力データ
            min_support (float): 最小サポート値
            metric (str): 評価指標
            min_threshold (float): 最小しきい値

        Returns:
            Dict[str, Union[pd.DataFrame, List[Dict]]]: 分析結果
        """
        try:
            # 頻出アイテムセットの抽出
            # データセットから頻繁に出現するアイテムの組み合わせを見つけます
            frequent_itemsets = apriori(
                data,
                min_support=min_support,
                use_colnames=True,
                max_len=None  # アイテムセットの最大長に制限を設けない
            )

            # 全アイテムセット数を計算
            # これにより、生成可能な全てのルールを対象とすることができます
            total_itemsets = len(frequent_itemsets)

            # アソシエーションルールの生成
            # 頻出アイテムセットから意味のあるルールを導出します
            rules = association_rules(
                frequent_itemsets,
                metric=metric,
                min_threshold=min_threshold,
                num_itemsets=total_itemsets  # 全てのアイテムセットを対象にする
            )

            # 結果のサマリー作成
            summary = self._create_summary(rules)

            return {
                "rules": rules,
                "summary": summary
            }

        except Exception as e:
            self.logger.error(f"同期処理中にエラーが発生しました: {str(e)}")
            raise

    def _create_summary(self, rules: pd.DataFrame) -> List[Dict]:
        """
        分析結果のサマリーを作成します。
        各ルールの重要な指標を抽出し、わかりやすい形式に整理します。

        Args:
            rules (pd.DataFrame): アソシエーションルール

        Returns:
            List[Dict]: サマリー情報。各要素は以下の情報を含みます：
                - antecedents: 前提条件となるアイテム
                - consequents: 結論となるアイテム
                - support: サポート値
                - confidence: 信頼度
                - lift: リフト値
        """
        return [
            {
                "antecedents": list(rule.antecedents),
                "consequents": list(rule.consequents),
                "support": rule.support,
                "confidence": rule.confidence,
                "lift": rule.lift
            }
            for _, rule in rules.iterrows()
        ]