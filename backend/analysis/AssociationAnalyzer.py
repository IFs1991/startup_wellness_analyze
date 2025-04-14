from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import traceback
import gc
import contextlib

# ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class AggregationConfig:
    """集計設定を表すデータクラス"""
    table_name: str
    group_by_columns: List[str]
    agg_functions: Dict[str, str]
    conditions: Optional[str] = None


class AssociationAnalyzer:
    """
    アソシエーション分析を行うクラス
    """
    def __init__(self):
        """
        コンストラクタ
        """
        self.logger = logger
        self.logger.info("AssociationAnalyzerが初期化されました")
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
            # 明示的なメモリ解放
            self._resources.clear()
            gc.collect()
            self.logger.debug("リソースを解放しました")
        except Exception as e:
            self.logger.error(f"リソース解放中にエラーが発生しました: {str(e)}")

    @contextlib.contextmanager
    def _managed_dataframe(self, df: pd.DataFrame, copy: bool = False):
        """
        データフレームのリソース管理を行うコンテキストマネージャ

        Args:
            df (pd.DataFrame): 管理するデータフレーム
            copy (bool): データをコピーするかどうか

        Yields:
            pd.DataFrame: 管理対象のデータフレーム
        """
        try:
            if copy:
                df_copy = df.copy()
                yield df_copy
            else:
                yield df
        finally:
            # 明示的なクリーンアップ
            if copy:
                del df_copy

            # メモリ使用状況のログ（デバッグレベル）
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("データフレーム管理コンテキスト終了")

    def analyze(self,
                data: pd.DataFrame,
                min_support: float = 0.01,
                min_confidence: float = 0.5,
                min_lift: float = 1.0) -> Dict[str, Any]:
        """
        アソシエーション分析を実行します

        Args:
            data (pd.DataFrame): 分析対象データ
            min_support (float): 最小サポート値
            min_confidence (float): 最小確信度
            min_lift (float): 最小リフト値

        Returns:
            Dict[str, Any]: 分析結果

        Raises:
            ValueError: パラメータが無効な場合
            RuntimeError: 分析実行中のエラー
        """
        # パラメータの検証
        self._validate_parameters(data, min_support, min_confidence, min_lift)

        try:
            self.logger.info(f"アソシエーション分析を実行します: サンプル数={len(data)}")

            # 分析処理の実行
            with self._managed_dataframe(data) as df:
                # 頻出アイテムセットの抽出
                frequent_itemsets = self._find_frequent_itemsets(df, min_support)

                # アソシエーションルールの生成
                rules = self._generate_association_rules(frequent_itemsets, min_confidence, min_lift)

                # 結果の整形
                result = self._format_results(frequent_itemsets, rules, min_support, min_confidence, min_lift)

                self.logger.info(f"アソシエーション分析が完了しました: ルール数={len(rules)}")
                return result

        except Exception as e:
            self.logger.error(f"アソシエーション分析でエラーが発生しました: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise RuntimeError(f"アソシエーション分析の実行に失敗しました: {str(e)}")

    def _validate_parameters(self, data: pd.DataFrame, min_support: float,
                           min_confidence: float, min_lift: float) -> None:
        """
        分析パラメータを検証します

        Args:
            data (pd.DataFrame): 分析対象データ
            min_support (float): 最小サポート値
            min_confidence (float): 最小確信度
            min_lift (float): 最小リフト値

        Raises:
            ValueError: パラメータが無効な場合
        """
        if data is None or data.empty:
            raise ValueError("入力データが空です")

        if not 0 < min_support < 1:
            raise ValueError(f"最小サポート値は0より大きく1未満である必要があります（指定値: {min_support}）")

        if not 0 < min_confidence <= 1:
            raise ValueError(f"最小確信度は0より大きく1以下である必要があります（指定値: {min_confidence}）")

        if min_lift < 0:
            raise ValueError(f"最小リフト値は0以上である必要があります（指定値: {min_lift}）")

        # データ型のチェック - バイナリデータであることを確認
        if not ((data == 0) | (data == 1) | pd.isna(data)).all().all():
            self.logger.warning("データには0と1以外の値が含まれています。アソシエーション分析には通常バイナリデータが必要です。")

    def _find_frequent_itemsets(self, data: pd.DataFrame, min_support: float) -> pd.DataFrame:
        """
        頻出アイテムセットを抽出します

        Args:
            data (pd.DataFrame): 分析対象データ
            min_support (float): 最小サポート値

        Returns:
            pd.DataFrame: 頻出アイテムセット
        """
        self.logger.info(f"頻出アイテムセットを抽出しています（最小サポート={min_support}）...")
        frequent_itemsets = apriori(
            data,
            min_support=min_support,
            use_colnames=True
        )

        self.logger.info(f"頻出アイテムセット抽出完了: {len(frequent_itemsets)}件")
        return frequent_itemsets

    def _generate_association_rules(self, frequent_itemsets: pd.DataFrame,
                                  min_confidence: float, min_lift: float) -> pd.DataFrame:
        """
        アソシエーションルールを生成します

        Args:
            frequent_itemsets (pd.DataFrame): 頻出アイテムセット
            min_confidence (float): 最小確信度
            min_lift (float): 最小リフト値

        Returns:
            pd.DataFrame: アソシエーションルール
        """
        if len(frequent_itemsets) == 0:
            self.logger.warning("頻出アイテムセットが見つかりませんでした。ルールは生成されません。")
            return pd.DataFrame()

        self.logger.info(f"アソシエーションルールを生成しています（最小確信度={min_confidence}）...")
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )

        # リフトでフィルタリング
        if min_lift > 0:
            rules = rules[rules['lift'] >= min_lift]

        self.logger.info(f"アソシエーションルール生成完了: {len(rules)}件")
        return rules

    def _format_results(self, frequent_itemsets: pd.DataFrame, rules: pd.DataFrame,
                      min_support: float, min_confidence: float, min_lift: float) -> Dict[str, Any]:
        """
        分析結果を整形します

        Args:
            frequent_itemsets (pd.DataFrame): 頻出アイテムセット
            rules (pd.DataFrame): アソシエーションルール
            min_support (float): 最小サポート値
            min_confidence (float): 最小確信度
            min_lift (float): 最小リフト値

        Returns:
            Dict[str, Any]: 整形された結果
        """
        # DataFrameをレコードリストに変換（シリアライズ可能な形式）
        itemsets_records = [] if frequent_itemsets.empty else frequent_itemsets.to_dict('records')
        rules_records = [] if rules.empty else rules.to_dict('records')

        # 結果辞書の作成
        result = {
            'frequent_itemsets': itemsets_records,
            'rules': rules_records,
            'stats': {
                'itemset_count': len(frequent_itemsets),
                'rule_count': len(rules),
                'min_support': min_support,
                'min_confidence': min_confidence,
                'min_lift': min_lift,
                'timestamp': datetime.now().isoformat()
            }
        }

        return result


class DataQueries:
    """データ取得用のクエリビルダー"""

    @staticmethod
    def build_analysis_query(
        table_name: str,
        columns: List[str],
        conditions: Optional[str] = None
    ) -> str:
        """
        分析用のクエリを構築します

        Args:
            table_name (str): テーブル名
            columns (List[str]): 取得するカラム
            conditions (Optional[str]): WHERE句の条件

        Returns:
            str: 構築されたクエリ

        Raises:
            ValueError: パラメータが無効な場合
        """
        # パラメータの検証
        if not table_name:
            raise ValueError("テーブル名は空であってはなりません")
        if not columns:
            raise ValueError("カラムリストは空であってはなりません")

        # 識別子のサニタイズ
        safe_columns = [DataQueries._sanitize_identifier(col) for col in columns]
        safe_table = DataQueries._sanitize_identifier(table_name)

        # クエリの構築
        query = f"""
            SELECT {', '.join(safe_columns)}
            FROM {safe_table}
            WHERE 1=1
        """
        if conditions:
            query += f"\nAND {conditions}"
        return query

    @staticmethod
    def build_aggregation_query(
        config: AggregationConfig
    ) -> str:
        """
        集計クエリを構築します

        Args:
            config (AggregationConfig): 集計設定

        Returns:
            str: 構築されたクエリ

        Raises:
            ValueError: パラメータが無効な場合
        """
        # 設定の検証
        if not config.table_name:
            raise ValueError("テーブル名は空であってはなりません")
        if not config.group_by_columns:
            raise ValueError("グループ化カラムは空であってはなりません")
        if not config.agg_functions:
            raise ValueError("集計関数は空であってはなりません")

        # 識別子のサニタイズ
        safe_table = DataQueries._sanitize_identifier(config.table_name)
        safe_group_by = [
            DataQueries._sanitize_identifier(col)
            for col in config.group_by_columns
        ]

        # 集計カラムの構築
        agg_cols = [
            f"{func}({DataQueries._sanitize_identifier(col)}) as {DataQueries._sanitize_identifier(f'{col}_{func}')}"
            for col, func in config.agg_functions.items()
        ]

        # クエリの構築
        query = f"""
            SELECT
                {', '.join(safe_group_by)},
                {', '.join(agg_cols)}
            FROM {safe_table}
            WHERE 1=1
        """

        if config.conditions:
            query += f"\nAND {config.conditions}"

        query += f"\nGROUP BY {', '.join(safe_group_by)}"
        return query

    @staticmethod
    def _sanitize_identifier(identifier: str) -> str:
        """
        SQL識別子をサニタイズします

        Args:
            identifier (str): サニタイズする識別子

        Returns:
            str: サニタイズされた識別子
        """
        # 基本的なサニタイズ
        sanitized = identifier.replace('"', '').replace('`', '')
        return f"`{sanitized}`"