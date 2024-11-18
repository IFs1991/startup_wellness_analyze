# -*- coding: utf-8 -*-

"""
アソシエーション分析

特定の健康状態と関連性の高い行動や属性を特定します。

"""


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

class AssociationAnalyzer:
    """
    アソシエーション分析を実行するためのクラスです。

    """
    def analyze(self, data: pd.DataFrame, min_support: float) -> pd.DataFrame:
        """
        アソシエーション分析を実行します。

        Args:
            data (pd.DataFrame): ワンホットエンコーディングされたデータ
            min_support (float): 最小サポート値

        Returns:
            pd.DataFrame: アソシエーションルール

        """
        frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

        return rules