# -*- coding: utf-8 -*-
"""
データ匿名化サービス
------------------
連合学習およびデータ分析のためのデータ匿名化機能を提供します。
データプライバシーを保護しながら効果的なデータ共有と分析を実現します。

主な機能:
- IDハッシング (SHA-256)
- データ正規化 (Z-スコア)
- K匿名性の実装
- 差分プライバシー

このモジュールは連合学習システムと統合されており、クライアント間でプライバシーを
保護しながらモデルトレーニングを可能にします。
"""

import os
import hashlib
import logging
import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from functools import lru_cache
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

# ロギング設定
logger = logging.getLogger(__name__)

class AnonymizationService:
    """
    データ匿名化サービスクラス

    データプライバシーを保護するための様々な匿名化技術を提供します。
    """

    def __init__(self, salt: Optional[str] = None):
        """
        匿名化サービスの初期化

        Args:
            salt: IDハッシングに使用するソルト。指定されない場合は環境変数から取得または生成。
        """
        # ソルトの設定（セキュリティのため環境変数または安全に生成したものを使用）
        self.salt = salt or os.environ.get("HASH_SALT", str(uuid.uuid4()))

        # 正規化のためのスケーラー
        self.scaler = StandardScaler()

        logger.info("匿名化サービスが初期化されました")

    async def anonymize_id(self, id_value: str) -> str:
        """
        ID値をハッシュ化して匿名化する

        SHA-256ハッシュとソルトを使用して一貫性のある匿名IDを生成します。

        Args:
            id_value: 匿名化する元のID

        Returns:
            str: 匿名化されたID
        """
        if not id_value:
            raise ValueError("匿名化するIDが指定されていません")

        # ソルトを追加してハッシュ化
        hash_input = f"{id_value}{self.salt}"
        hashed_id = hashlib.sha256(hash_input.encode()).hexdigest()

        return hashed_id

    async def normalize_data(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        指定された列のデータをZ-スコア正規化する

        データの平均を0、標準偏差を1に変換します。

        Args:
            data: 正規化する元のデータフレーム
            columns: 正規化する列のリスト（Noneの場合は数値列全て）

        Returns:
            pd.DataFrame: 正規化されたデータフレーム
        """
        if data.empty:
            return data

        # 正規化する列が指定されていない場合は数値列を自動選択
        if not columns:
            columns = data.select_dtypes(include=np.number).columns.tolist()

        # 元のデータフレームをコピー
        normalized_df = data.copy()

        # 指定された列を正規化
        if columns:
            normalized_df[columns] = self.scaler.fit_transform(data[columns])

        return normalized_df

    async def apply_k_anonymity(
        self,
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        k: int = 5
    ) -> pd.DataFrame:
        """
        K匿名性を適用する

        準識別子の組み合わせが少なくとも k 個のレコードに出現するようにデータを変換します。

        Args:
            data: 匿名化するデータフレーム
            quasi_identifiers: 準識別子として扱う列のリスト
            k: 匿名性のパラメータ（最小グループサイズ）

        Returns:
            pd.DataFrame: K匿名性が適用されたデータフレーム
        """
        if data.empty:
            return data

        # 元のデータフレームをコピー
        anonymized_df = data.copy()

        # 準識別子のグループごとのカウントを集計
        group_counts = anonymized_df.groupby(quasi_identifiers).size().reset_index(name='count')

        # k未満のグループを特定
        small_groups = group_counts[group_counts['count'] < k]

        if small_groups.empty:
            # 既にK匿名性を満たしている場合はそのまま返す
            return anonymized_df

        # K匿名性を満たすように値を一般化
        for _, group in small_groups.iterrows():
            # 特定のグループに対応する行を取得
            mask = pd.Series(True, index=anonymized_df.index)
            for qi in quasi_identifiers:
                mask = mask & (anonymized_df[qi] == group[qi])

            # 一般化の実行（例：カテゴリの結合、数値の範囲化）
            for qi in quasi_identifiers:
                if pd.api.types.is_numeric_dtype(anonymized_df[qi]):
                    # 数値列の場合は範囲でマスク
                    col_min = anonymized_df.loc[mask, qi].min()
                    col_max = anonymized_df.loc[mask, qi].max()
                    # 範囲の表現に置き換え
                    anonymized_df.loc[mask, qi] = f"{col_min}-{col_max}"
                else:
                    # カテゴリ列の場合は一般化（例：部分的なマスキング）
                    anonymized_df.loc[mask, qi] = f"{str(group[qi])[0]}*"

        return anonymized_df

    async def apply_differential_privacy(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        epsilon: float = 0.1,
        delta: float = 0.00001,
        columns: Optional[List[str]] = None,
        mechanism: str = "gaussian"
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        差分プライバシーを適用する

        データに制御されたノイズを追加してプライバシーを保護します。

        Args:
            data: 匿名化するデータ
            epsilon: プライバシー予算（小さいほど保護が強い）
            delta: 失敗確率（小さいほど保護が強い）
            columns: ノイズを追加する列（データフレームの場合）
            mechanism: 使用するノイズメカニズム（"gaussian"または"laplace"）

        Returns:
            匿名化されたデータ
        """
        # データフレームの場合
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return data

            # 処理する列を特定
            if not columns:
                columns = data.select_dtypes(include=np.number).columns.tolist()

            # 元のデータフレームをコピー
            dp_df = data.copy()

            # 列ごとに感度（最大変化量）を計算
            for col in columns:
                # 感度を計算（例：列の範囲の1%）
                col_range = dp_df[col].max() - dp_df[col].min()
                sensitivity = max(0.01 * col_range, 1e-6)  # ゼロ除算を防ぐ

                # ノイズを生成
                if mechanism.lower() == "gaussian":
                    # ガウシアンメカニズム
                    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
                    noise = np.random.normal(0, sigma, size=len(dp_df))
                else:
                    # ラプラスメカニズム
                    scale = sensitivity / epsilon
                    noise = np.random.laplace(0, scale, size=len(dp_df))

                # ノイズをデータに追加
                dp_df[col] = dp_df[col] + noise

            return dp_df

        # NumPy配列の場合
        elif isinstance(data, np.ndarray):
            if data.size == 0:
                return data

            # 配列のコピーを作成
            dp_array = data.copy()

            # 感度を計算
            sensitivity = max(0.01 * (np.max(dp_array) - np.min(dp_array)), 1e-6)

            # ノイズを生成して追加
            if mechanism.lower() == "gaussian":
                sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
                noise = np.random.normal(0, sigma, size=dp_array.shape)
            else:
                scale = sensitivity / epsilon
                noise = np.random.laplace(0, scale, size=dp_array.shape)

            return dp_array + noise

        else:
            raise TypeError("サポートされていないデータ型です。pandas.DataFrameまたはnumpy.ndarrayを使用してください。")

    async def create_anonymous_view(
        self,
        data: pd.DataFrame,
        id_column: str,
        quasi_identifiers: List[str],
        sensitive_columns: List[str],
        anonymization_level: str = "medium"
    ) -> pd.DataFrame:
        """
        データフレームの匿名化ビューを作成する

        複数の匿名化技術を組み合わせて包括的な匿名化を行います。

        Args:
            data: 匿名化するデータフレーム
            id_column: 匿名化するID列
            quasi_identifiers: 準識別子列
            sensitive_columns: センシティブデータの列
            anonymization_level: 匿名化レベル（"low", "medium", "high"）

        Returns:
            pd.DataFrame: 匿名化されたデータフレーム
        """
        if data.empty:
            return data

        # 匿名化レベルに応じたパラメータ設定
        if anonymization_level == "low":
            k = 3
            epsilon = 0.5
            delta = 0.0001
        elif anonymization_level == "medium":
            k = 5
            epsilon = 0.1
            delta = 0.00001
        elif anonymization_level == "high":
            k = 10
            epsilon = 0.05
            delta = 0.000001
        else:
            raise ValueError(f"無効な匿名化レベル: {anonymization_level}")

        # 元のデータフレームをコピー
        result_df = data.copy()

        # IDの匿名化
        if id_column in result_df.columns:
            result_df[id_column] = await self._anonymize_id_column(result_df[id_column])

        # 準識別子にK匿名性を適用
        if quasi_identifiers:
            result_df = await self.apply_k_anonymity(result_df, quasi_identifiers, k)

        # センシティブデータに差分プライバシーを適用
        if sensitive_columns:
            result_df = await self.apply_differential_privacy(
                result_df,
                epsilon=epsilon,
                delta=delta,
                columns=sensitive_columns
            )

        # 数値データを正規化
        numeric_columns = result_df.select_dtypes(include=np.number).columns.tolist()
        # ID列は正規化から除外
        if id_column in numeric_columns:
            numeric_columns.remove(id_column)

        if numeric_columns:
            result_df = await self.normalize_data(result_df, numeric_columns)

        return result_df

    async def _anonymize_id_column(self, id_series: pd.Series) -> pd.Series:
        """ID列の全ての値を匿名化する"""
        return id_series.apply(lambda x: self.anonymize_id(str(x)) if pd.notna(x) else x)

    @staticmethod
    async def generate_database_views(
        source_table: str,
        target_view: str,
        db_session,
        anonymization_config: Dict[str, Any]
    ) -> bool:
        """
        データベース内に匿名化ビューを生成する

        Args:
            source_table: 元のテーブル名
            target_view: 生成するビュー名
            db_session: データベースセッション
            anonymization_config: 匿名化設定

        Returns:
            bool: 成功した場合はTrue
        """
        try:
            # 設定からハッシュソルトを取得
            hash_salt = anonymization_config.get("salt", os.environ.get("HASH_SALT", "default_salt"))

            # SQLクエリを構築
            # 例: 財務データの匿名化ビュー
            if "financial" in source_table.lower():
                query = f"""
                CREATE OR REPLACE VIEW {target_view} AS
                SELECT
                  md5(company_id::text || '{hash_salt}') AS anonymous_id,
                  industry_type,
                  company_size_category,
                  funding_stage,
                  date_trunc('month', report_date) AS report_month,
                  revenue / (SELECT AVG(revenue) FROM {source_table} WHERE date_trunc('year', report_date) = date_trunc('year', fd.report_date)) AS normalized_revenue,
                  burn_rate / (SELECT AVG(burn_rate) FROM {source_table} WHERE date_trunc('year', report_date) = date_trunc('year', fd.report_date)) AS normalized_burn_rate,
                  gross_margin,
                  customer_acquisition_cost,
                  lifetime_value,
                  monthly_recurring_revenue,
                  churn_rate
                FROM {source_table} fd;
                """
            # VAS（経済的付加価値）データの匿名化ビュー
            elif "vas" in source_table.lower():
                query = f"""
                CREATE OR REPLACE VIEW {target_view} AS
                SELECT
                  md5(company_id::text || '{hash_salt}') AS anonymous_id,
                  industry_type,
                  role_category,
                  date_trunc('month', assessment_date) AS assessment_month,
                  COUNT(DISTINCT employee_id) AS employee_count,
                  AVG(performance_score) AS avg_performance_score,
                  STDDEV(performance_score) AS std_performance_score,
                  AVG(health_score) AS avg_health_score,
                  STDDEV(health_score) AS std_health_score
                FROM {source_table}
                GROUP BY
                  md5(company_id::text || '{hash_salt}'),
                  industry_type,
                  role_category,
                  date_trunc('month', assessment_date);
                """
            # その他のテーブルには一般的な匿名化を適用
            else:
                # 簡易的なクエリ例（実際にはテーブル構造に合わせる必要がある）
                query = f"""
                CREATE OR REPLACE VIEW {target_view} AS
                SELECT
                  md5(id::text || '{hash_salt}') AS anonymous_id,
                  -- 他の列は適切に一般化または集計
                  -- 実際には対象テーブルの構造に合わせて実装
                  *
                FROM {source_table};
                """

            # クエリを実行
            await db_session.execute(query)
            await db_session.commit()

            logger.info(f"匿名化ビュー {target_view} が正常に作成されました")
            return True

        except Exception as e:
            logger.error(f"匿名化ビューの作成に失敗しました: {str(e)}")
            await db_session.rollback()
            return False

    @staticmethod
    async def check_anonymization_effectiveness(
        original_data: pd.DataFrame,
        anonymized_data: pd.DataFrame,
        sensitive_columns: List[str]
    ) -> Dict[str, Any]:
        """
        匿名化の効果を評価する

        Args:
            original_data: 元のデータ
            anonymized_data: 匿名化されたデータ
            sensitive_columns: センシティブデータの列

        Returns:
            Dict: 評価結果
        """
        results = {}

        # 1. レコード一意性の評価 - k匿名性が満たされているか
        try:
            # 共通の列を取得
            common_columns = list(set(original_data.columns) & set(anonymized_data.columns))

            # 匿名化データの準識別子の一意性をチェック
            duplicated_count = anonymized_data.duplicated(subset=common_columns).sum()
            uniqueness_rate = 1 - (duplicated_count / len(anonymized_data))

            results["uniqueness_rate"] = uniqueness_rate
            results["k_anonymity_satisfied"] = uniqueness_rate <= 0.2  # 20%以下が一意であれば良好
        except Exception as e:
            logger.error(f"レコード一意性評価エラー: {str(e)}")
            results["uniqueness_rate"] = None

        # 2. 情報損失の評価
        try:
            info_loss = {}
            for col in sensitive_columns:
                if col in common_columns and pd.api.types.is_numeric_dtype(original_data[col]):
                    # 元のデータと匿名化データの分布の違いを測定
                    original_mean = original_data[col].mean()
                    anon_mean = anonymized_data[col].mean()

                    # 平均の相対差
                    if original_mean != 0:
                        mean_diff = abs(original_mean - anon_mean) / abs(original_mean)
                    else:
                        mean_diff = abs(original_mean - anon_mean)

                    # 分散の相対差
                    original_var = original_data[col].var()
                    anon_var = anonymized_data[col].var()

                    if original_var != 0:
                        var_diff = abs(original_var - anon_var) / abs(original_var)
                    else:
                        var_diff = abs(original_var - anon_var)

                    info_loss[col] = {
                        "mean_difference": mean_diff,
                        "variance_difference": var_diff,
                        "acceptable": mean_diff <= 0.2 and var_diff <= 0.3  # 閾値は調整可能
                    }

            results["information_loss"] = info_loss
        except Exception as e:
            logger.error(f"情報損失評価エラー: {str(e)}")
            results["information_loss"] = None

        # 3. 再識別リスクの評価
        try:
            # 簡易的な再識別リスク評価（実際のシステムではより高度な方法を使用）
            risk_level = "low"

            if uniqueness_rate > 0.5:
                risk_level = "high"
            elif uniqueness_rate > 0.2:
                risk_level = "medium"

            results["reidentification_risk"] = risk_level
        except Exception as e:
            logger.error(f"再識別リスク評価エラー: {str(e)}")
            results["reidentification_risk"] = None

        return results