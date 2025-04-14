from typing import List, Optional, Dict, Any, Union, Set, Tuple, Callable
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import gc
import weakref
from abc import ABC, abstractmethod

# Firestoreクライアントのインポート
try:
    from backend.database.firestore.client import get_firestore_client  # type: ignore
except ImportError:
    # インポートエラーが発生した場合はダミー実装を使用
    def get_firestore_client():
        logging.warning("Firestoreクライアントが見つかりません。モック実装を使用します。")
        class MockFirestoreClient:
            async def get_document(self, *args, **kwargs):
                return {}
            async def set_document(self, *args, **kwargs):
                return "mock-doc-id"
            async def query_documents(self, *args, **kwargs):
                return []
        return MockFirestoreClient()


class AnalysisError(Exception):
    """分析処理に関するエラー"""
    pass


class DataValidationError(AnalysisError):
    """データ検証に関するエラー"""
    def __init__(self, message: str, validation_errors: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.validation_errors = validation_errors or {}


class BaseAnalyzer:
    """分析モジュールの基底クラス"""

    def __init__(self, analysis_type: str, firestore_client=None):
        """
        初期化メソッド

        Args:
            analysis_type (str): 分析タイプ（例: 'correlation', 'pca', 'clustering'）
            firestore_client: Firestoreクライアントのインスタンス（テスト用）
        """
        self.analysis_type = analysis_type
        self.firestore_client = firestore_client if firestore_client is not None else get_firestore_client()
        self.logger = logging.getLogger(f"{__name__}.{analysis_type}")
        # 一時データの追跡用辞書
        self._temp_data_registry = {}
        # 分析オブジェクトが破棄される際に自動クリーンアップするためのfinalizer
        self._finalizer = weakref.finalize(self, self._cleanup_resources)

    def __del__(self):
        """オブジェクト破棄時のクリーンアップ処理"""
        self.release_resources()

    def register_temp_data(self, key: str, data: Any) -> None:
        """
        一時データをレジストリに登録

        Args:
            key: データを識別するキー
            data: 登録するデータ
        """
        self._temp_data_registry[key] = data

    def get_temp_data(self, key: str) -> Any:
        """
        レジストリから一時データを取得

        Args:
            key: データを識別するキー

        Returns:
            登録されたデータ

        Raises:
            KeyError: 指定されたキーが見つからない場合
        """
        return self._temp_data_registry.get(key)

    def release_resource(self, key: str) -> bool:
        """
        特定の一時リソースを解放

        Args:
            key: 解放するリソースのキー

        Returns:
            bool: リソースが正常に解放された場合はTrue
        """
        if key in self._temp_data_registry:
            del self._temp_data_registry[key]
            gc.collect()
            return True
        return False

    def release_resources(self) -> None:
        """すべての一時リソースを解放"""
        self._temp_data_registry.clear()
        gc.collect()

    def _cleanup_resources(self) -> None:
        """オブジェクト破棄時の内部クリーンアップ処理"""
        try:
            self.release_resources()
        except Exception as e:
            # すでにロガーが利用できない可能性があるため標準エラーに出力
            import sys
            print(f"リソース解放中にエラーが発生しました: {e}", file=sys.stderr)

    async def fetch_data(
        self,
        collection: str,
        filters: Optional[List[Dict[str, Any]]] = None,
        order_by: Optional[tuple] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Firestoreからデータを取得してDataFrameに変換

        Args:
            collection (str): コレクション名
            filters (Optional[List[Dict[str, Any]]]): フィルター条件
            order_by (Optional[tuple]): ソート条件
            limit (Optional[int]): 取得件数

        Returns:
            pd.DataFrame: 取得したデータ
        """
        try:
            docs = await self.firestore_client.query_documents(
                collection=collection,
                filters=filters,
                order_by=order_by,
                limit=limit
            )
            df = pd.DataFrame(docs)
            # メモリ最適化：データの型を最適化
            df = self._optimize_dataframe_dtypes(df)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            raise AnalysisError(f"データの取得に失敗しました: {str(e)}")

    def _optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrameのメモリ使用量を最適化する

        Args:
            df: 最適化するDataFrame

        Returns:
            最適化されたDataFrame
        """
        if df.empty:
            return df

        # 数値型カラムの最適化
        for col in df.select_dtypes(include=['int']).columns:
            # 値の範囲に応じて最適な整数型を選択
            col_min, col_max = df[col].min(), df[col].max()

            # 符号なし整数が使用可能か確認
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:
                # 符号付き整数
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype(np.int8)
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype(np.int16)
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype(np.int32)

        # float型の最適化
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = df[col].astype(np.float32)

        # カテゴリ型に変換できるカラムの最適化
        for col in df.select_dtypes(include=['object']).columns:
            # ユニーク値の数が少ない場合にカテゴリ型に変換
            if df[col].nunique() / len(df) < 0.5:  # 50%未満のユニーク比率
                df[col] = df[col].astype('category')

        return df

    async def save_results(
        self,
        results: Dict[str, Any],
        collection: str = 'analysis_results'
    ) -> str:
        """
        分析結果を保存

        Args:
            results (Dict[str, Any]): 分析結果
            collection (str): 保存先コレクション名

        Returns:
            str: 保存したドキュメントのID
        """
        try:
            # メタデータを追加
            results.update({
                'analysis_type': self.analysis_type,
                'created_at': datetime.utcnow(),
                'status': 'completed'
            })

            # ドキュメントIDを生成して保存
            doc_id = f"{self.analysis_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            await self.firestore_client.set_document(collection, doc_id, results)
            return doc_id

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise AnalysisError(f"結果の保存に失敗しました: {str(e)}")

    def validate_data(self, data: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
        """
        データの基本バリデーション

        Args:
            data (pd.DataFrame): 検証対象のデータ
            required_columns (Optional[List[str]]): 必須カラムのリスト

        Returns:
            bool: バリデーション結果

        Raises:
            DataValidationError: バリデーションエラーが発生した場合
        """
        validation_errors = {}

        # 空チェック
        if data.empty:
            validation_errors['empty_data'] = "データが空です"
            raise DataValidationError("データが空です", validation_errors)

        # 必須カラムの存在チェック
        if required_columns:
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                validation_errors['missing_columns'] = missing_columns
                raise DataValidationError(
                    f"必須カラムがありません: {', '.join(missing_columns)}",
                    validation_errors
                )

        return True

    def empty_check(self, data: pd.DataFrame) -> bool:
        """
        データフレームが空でないかを検証

        Args:
            data: 検証対象のDataFrame

        Returns:
            bool: データが有効な場合はTrue

        Raises:
            DataValidationError: データが空の場合
        """
        if data.empty:
            raise DataValidationError("データが空です", {"empty_data": True})
        return True

    def column_existence(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        必須カラムの存在を検証

        Args:
            data: 検証対象のDataFrame
            required_columns: 必須カラムのリスト

        Returns:
            bool: 検証が成功した場合はTrue

        Raises:
            DataValidationError: 必須カラムが存在しない場合
        """
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataValidationError(
                f"必須カラムがありません: {', '.join(missing_columns)}",
                {"missing_columns": missing_columns}
            )
        return True

    def data_type_check(
        self,
        data: pd.DataFrame,
        column_types: Dict[str, Union[type, List[type]]]
    ) -> bool:
        """
        指定カラムのデータ型を検証

        Args:
            data: 検証対象のDataFrame
            column_types: カラム名をキー、期待するデータ型（または型のリスト）を値とする辞書

        Returns:
            bool: 検証が成功した場合はTrue

        Raises:
            DataValidationError: データ型が一致しない場合
        """
        type_errors = {}

        for col, expected_types in column_types.items():
            if col not in data.columns:
                continue

            # 期待する型のリストに変換
            if not isinstance(expected_types, list):
                expected_types = [expected_types]

            # 型チェック
            col_dtype = data[col].dtype
            valid_type = False

            for expected_type in expected_types:
                # NumPyの型と組み込み型の互換性を考慮
                if expected_type == int and np.issubdtype(col_dtype, np.integer):
                    valid_type = True
                    break
                elif expected_type == float and np.issubdtype(col_dtype, np.floating):
                    valid_type = True
                    break
                elif expected_type == str and col_dtype == 'object':
                    valid_type = True
                    break
                elif expected_type == bool and col_dtype == 'bool':
                    valid_type = True
                    break

            if not valid_type:
                type_errors[col] = f"カラム '{col}' の型が不正です: 期待={expected_types}, 実際={col_dtype}"

        if type_errors:
            raise DataValidationError("データ型の検証に失敗しました", {"type_errors": type_errors})

        return True

    def missing_value_check(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: float = 0.0,
        handle_missing: Optional[str] = None
    ) -> pd.DataFrame:
        """
        欠損値の検出と処理オプション提供

        Args:
            data: 検証対象のDataFrame
            columns: 欠損値をチェックするカラムのリスト（Noneの場合は全カラム）
            threshold: 許容される欠損値の割合（0.0〜1.0）
            handle_missing: 欠損値の処理方法（'drop', 'fill_mean', 'fill_median', 'fill_mode', 'fill_zero'）

        Returns:
            pd.DataFrame: 欠損値処理後のDataFrame

        Raises:
            DataValidationError: 許容閾値を超える欠損値が存在する場合
        """
        # 対象カラムが指定されていない場合は全カラムを対象とする
        target_cols = columns if columns else data.columns

        # カラムごとの欠損値割合を計算
        missing_ratio = data[target_cols].isna().mean()
        problematic_cols = missing_ratio[missing_ratio > threshold].to_dict()

        # 閾値を超える欠損値がある場合
        if problematic_cols and handle_missing is None:
            raise DataValidationError(
                f"許容閾値（{threshold}）を超える欠損値が存在します",
                {"missing_values": problematic_cols}
            )

        # 欠損値の処理
        result_df = data.copy()
        if handle_missing and problematic_cols:
            if handle_missing == 'drop':
                # 欠損値を含む行を削除
                result_df = result_df.dropna(subset=list(problematic_cols.keys()))
            else:
                for col in problematic_cols:
                    if handle_missing == 'fill_mean' and np.issubdtype(result_df[col].dtype, np.number):
                        result_df[col] = result_df[col].fillna(result_df[col].mean())
                    elif handle_missing == 'fill_median' and np.issubdtype(result_df[col].dtype, np.number):
                        result_df[col] = result_df[col].fillna(result_df[col].median())
                    elif handle_missing == 'fill_mode':
                        result_df[col] = result_df[col].fillna(result_df[col].mode()[0] if not result_df[col].mode().empty else None)
                    elif handle_missing == 'fill_zero' and np.issubdtype(result_df[col].dtype, np.number):
                        result_df[col] = result_df[col].fillna(0)

        return result_df

    def sample_size_check(self, data: pd.DataFrame, min_samples: int) -> bool:
        """
        分析に必要な最小データサイズを検証

        Args:
            data: 検証対象のDataFrame
            min_samples: 必要な最小サンプル数

        Returns:
            bool: 検証が成功した場合はTrue

        Raises:
            DataValidationError: サンプルサイズが不足している場合
        """
        if len(data) < min_samples:
            raise DataValidationError(
                f"サンプルサイズが不足しています: 必要={min_samples}, 実際={len(data)}",
                {"insufficient_samples": {"required": min_samples, "actual": len(data)}}
            )
        return True

    def outlier_detection(
        self,
        data: pd.DataFrame,
        columns: List[str],
        method: str = 'iqr',
        threshold: float = 1.5,
        handle_outliers: Optional[str] = None
    ) -> pd.DataFrame:
        """
        外れ値の検出と処理オプション

        Args:
            data: 検証対象のDataFrame
            columns: 外れ値を検出するカラムのリスト
            method: 検出方法（'iqr'または'zscore'）
            threshold: 外れ値と判定する閾値（IQR法の倍率またはZ-Scoreの標準偏差倍率）
            handle_outliers: 外れ値の処理方法（'drop', 'clip', 'fill_mean', 'fill_median'）

        Returns:
            pd.DataFrame: 外れ値処理後のDataFrame

        Raises:
            DataValidationError: サポートされていない検出方法が指定された場合
        """
        result_df = data.copy()
        outliers_info = {}

        for col in columns:
            if col not in data.columns or not np.issubdtype(data[col].dtype, np.number):
                continue

            # 外れ値の検出
            outlier_mask = pd.Series(False, index=data.index)

            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                outliers_info[col] = {
                    'method': 'iqr',
                    'threshold': threshold,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'count': outlier_mask.sum()
                }
            elif method == 'zscore':
                mean = data[col].mean()
                std = data[col].std()
                z_scores = (data[col] - mean) / std
                outlier_mask = (z_scores.abs() > threshold)
                outliers_info[col] = {
                    'method': 'zscore',
                    'threshold': threshold,
                    'count': outlier_mask.sum()
                }
            else:
                raise DataValidationError(
                    f"サポートされていない外れ値検出方法です: {method}",
                    {"unsupported_method": method}
                )

            # 外れ値の処理
            if handle_outliers and outlier_mask.any():
                if handle_outliers == 'drop':
                    # 外れ値を含む行を削除
                    result_df = result_df[~outlier_mask]
                elif handle_outliers == 'clip':
                    if method == 'iqr':
                        result_df.loc[outlier_mask, col] = result_df.loc[outlier_mask, col].clip(lower_bound, upper_bound)
                    else:  # zscore
                        z_scores = (result_df[col] - mean) / std
                        result_df.loc[z_scores > threshold, col] = mean + threshold * std
                        result_df.loc[z_scores < -threshold, col] = mean - threshold * std
                elif handle_outliers == 'fill_mean':
                    result_df.loc[outlier_mask, col] = result_df[col].mean()
                elif handle_outliers == 'fill_median':
                    result_df.loc[outlier_mask, col] = result_df[col].median()

        # 外れ値情報を登録
        self.register_temp_data('outliers_info', outliers_info)

        return result_df

    def data_consistency(
        self,
        data: pd.DataFrame,
        consistency_rules: List[Dict[str, Any]]
    ) -> bool:
        """
        複数カラム間の整合性を検証

        Args:
            data: 検証対象のDataFrame
            consistency_rules: 検証ルールのリスト。各ルールは辞書形式。
                例: [{'columns': ['start_date', 'end_date'], 'check': 'start_before_end'}]

        Returns:
            bool: 検証が成功した場合はTrue

        Raises:
            DataValidationError: 整合性エラーが発生した場合
        """
        consistency_errors = {}

        for rule in consistency_rules:
            columns = rule.get('columns', [])
            check_type = rule.get('check', '')

            # カラムの存在確認
            missing_columns = [col for col in columns if col not in data.columns]
            if missing_columns:
                continue

            # 整合性チェックの種類に応じた処理
            if check_type == 'start_before_end' and len(columns) >= 2:
                # 開始日が終了日より前であることを確認
                start_col, end_col = columns[0], columns[1]
                invalid_records = data[data[start_col] > data[end_col]]
                if not invalid_records.empty:
                    consistency_errors[f"{start_col}_{end_col}"] = {
                        'type': 'start_before_end',
                        'description': f"{start_col}が{end_col}より後になっているレコードがあります",
                        'count': len(invalid_records)
                    }
            elif check_type == 'sum_equals' and len(columns) >= 3:
                # 複数列の合計が別の列と一致することを確認
                sum_cols = columns[:-1]
                total_col = columns[-1]
                data['calculated_sum'] = data[sum_cols].sum(axis=1)
                invalid_records = data[~np.isclose(data['calculated_sum'], data[total_col])]
                if not invalid_records.empty:
                    consistency_errors[f"sum_{total_col}"] = {
                        'type': 'sum_equals',
                        'description': f"{sum_cols}の合計が{total_col}と一致しないレコードがあります",
                        'count': len(invalid_records)
                    }
                data = data.drop('calculated_sum', axis=1)
            elif check_type == 'custom' and 'custom_func' in rule:
                # カスタム検証関数の実行
                custom_func = rule['custom_func']
                if callable(custom_func):
                    try:
                        invalid_mask = ~custom_func(data)
                        invalid_count = invalid_mask.sum()
                        if invalid_count > 0:
                            rule_name = rule.get('name', 'custom_rule')
                            consistency_errors[rule_name] = {
                                'type': 'custom',
                                'description': rule.get('description', 'カスタム整合性ルールに違反しています'),
                                'count': invalid_count
                            }
                    except Exception as e:
                        self.logger.warning(f"カスタム検証関数の実行中にエラーが発生しました: {str(e)}")

        if consistency_errors:
            raise DataValidationError("データの整合性検証に失敗しました", {"consistency_errors": consistency_errors})

        return True

    def time_series_validation(
        self,
        data: pd.DataFrame,
        datetime_column: str,
        frequency: Optional[str] = None,
        min_periods: Optional[int] = None,
        max_gap: Optional[str] = None
    ) -> bool:
        """
        時系列データの特殊検証

        Args:
            data: 検証対象のDataFrame
            datetime_column: 日時カラム名
            frequency: 期待される頻度（例: 'D'=日次, 'H'=時間次, 'T'=分次）
            min_periods: 必要な最小期間数
            max_gap: 許容される最大ギャップ（例: '1D', '4H'）

        Returns:
            bool: 検証が成功した場合はTrue

        Raises:
            DataValidationError: 時系列データの検証に失敗した場合
        """
        time_series_errors = {}

        # カラムの存在確認
        if datetime_column not in data.columns:
            raise DataValidationError(f"時系列カラム '{datetime_column}' が存在しません",
                                    {"missing_column": datetime_column})

        # 日時型への変換
        try:
            data = data.copy()
            data[datetime_column] = pd.to_datetime(data[datetime_column])
        except Exception as e:
            raise DataValidationError(f"時系列カラムのデータ型変換に失敗しました: {str(e)}",
                                    {"type_conversion_error": str(e)})

        # ソート
        data = data.sort_values(by=datetime_column)

        # 最小期間数の検証
        if min_periods is not None and len(data) < min_periods:
            time_series_errors['insufficient_periods'] = {
                'description': f"期間数が不足しています: 必要={min_periods}, 実際={len(data)}",
                'required': min_periods,
                'actual': len(data)
            }

        # 頻度の検証
        if frequency is not None:
            try:
                # 理想的な時系列インデックスを生成
                ideal_index = pd.date_range(
                    start=data[datetime_column].min(),
                    end=data[datetime_column].max(),
                    freq=frequency
                )

                # 実際のデータポイントが理想的なインデックスに存在するかチェック
                missing_dates = ideal_index.difference(data[datetime_column])
                if len(missing_dates) > 0:
                    time_series_errors['irregular_frequency'] = {
                        'description': f"不規則な頻度：{frequency}の頻度で欠損があります",
                        'expected_points': len(ideal_index),
                        'actual_points': len(data),
                        'missing_count': len(missing_dates)
                    }
            except Exception as e:
                self.logger.warning(f"頻度検証中にエラーが発生しました: {str(e)}")

        # 最大ギャップの検証
        if max_gap is not None:
            try:
                date_diffs = data[datetime_column].diff()
                max_diff = date_diffs.max()
                max_allowed = pd.Timedelta(max_gap)

                if max_diff > max_allowed:
                    time_series_errors['excessive_gap'] = {
                        'description': f"許容を超えるギャップが存在します",
                        'max_allowed': max_gap,
                        'actual_max_gap': str(max_diff)
                    }
            except Exception as e:
                self.logger.warning(f"ギャップ検証中にエラーが発生しました: {str(e)}")

        if time_series_errors:
            raise DataValidationError("時系列データの検証に失敗しました", {"time_series_errors": time_series_errors})

        return True

    async def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        分析を実行（サブクラスで実装）

        Args:
            data (pd.DataFrame): 分析対象データ
            **kwargs: 追加のパラメータ

        Returns:
            Dict[str, Any]: 分析結果
        """
        raise NotImplementedError("このメソッドはサブクラスで実装する必要があります")

    async def analyze_and_save(
        self,
        data: pd.DataFrame,
        save_results: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        分析を実行して結果を保存

        Args:
            data (pd.DataFrame): 分析対象データ
            save_results (bool): 結果を保存するかどうか
            **kwargs: 追加のパラメータ

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            # データのバリデーション
            self.validate_data(data)

            # 分析の実行
            results = await self.analyze(data, **kwargs)

            # 結果の保存
            if save_results:
                doc_id = await self.save_results(results)
                results['document_id'] = doc_id

            # 不要な一時データの解放
            self.release_resources()

            return results

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise AnalysisError(f"分析に失敗しました: {str(e)}")


class DataRepository(ABC):
    """データリポジトリの抽象基底クラス"""

    @abstractmethod
    async def get_document(self, collection: str, document_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def query_documents(self, collection: str, filters=None, order_by=None, limit=None) -> List[Dict[str, Any]]:
        pass


class PostgresRepository(DataRepository):
    """PostgreSQL用のリポジトリ実装"""
    # PostgreSQL特有の実装
    async def get_document(self, collection: str, document_id: str) -> Dict[str, Any]:
        """PostgreSQLからドキュメントを取得する"""
        # 実際にはSQLクエリを実行するコードが入る
        logger = logging.getLogger(__name__)
        logger.warning("PostgresRepository.get_documentはまだ実装されていません")
        return {}

    async def query_documents(self, collection: str, filters=None, order_by=None, limit=None) -> List[Dict[str, Any]]:
        """PostgreSQLからクエリに基づいてドキュメントを取得する"""
        # 実際にはSQLクエリを実行するコードが入る
        logger = logging.getLogger(__name__)
        logger.warning("PostgresRepository.query_documentsはまだ実装されていません")
        return []


class FirestoreRepository(DataRepository):
    """Firestore用のリポジトリ実装"""
    def __init__(self, firestore_client=None):
        """初期化メソッド"""
        self.client = firestore_client if firestore_client is not None else get_firestore_client()

    async def get_document(self, collection: str, document_id: str) -> Dict[str, Any]:
        """Firestoreからドキュメントを取得する"""
        return await self.client.get_document(collection, document_id)

    async def query_documents(self, collection: str, filters=None, order_by=None, limit=None) -> List[Dict[str, Any]]:
        """Firestoreからクエリに基づいてドキュメントを取得する"""
        return await self.client.query_documents(collection, filters, order_by, limit)

    async def set_document(self, collection: str, document_id: str, data: Dict[str, Any]) -> str:
        """Firestoreにドキュメントを保存する"""
        return await self.client.set_document(collection, document_id, data)