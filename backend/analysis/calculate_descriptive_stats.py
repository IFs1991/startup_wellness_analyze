# -*- coding: utf-8 -*-

"""
記述統計量の計算

VAS データと損益計算書データの記述統計量を計算します。
BigQueryService を利用した非同期処理に対応しています。
"""

from typing import Optional, Tuple, Any, Dict, List
import pandas as pd
import numpy as np
import statsmodels.api as sm
from dataclasses import dataclass
from backend.src.database.bigquery.client import BigQueryService
from backend.src.database.firestore.client import FirestoreClient
from scipy import stats
from . import BaseAnalyzer

@dataclass
class DescriptiveStatsConfig:
    """記述統計量の計算設定を保持するデータクラス"""
    query: str
    target_variable: str
    arima_order: Tuple[int, int, int] = (5, 1, 0)
    columns: Optional[List[str]] = None
    save_results: bool = True
    dataset_id: Optional[str] = None
    table_id: Optional[str] = None

class DescriptiveStatsCalculator(BaseAnalyzer):
    """記述統計分析を行うクラス"""

    def __init__(self, include_percentiles: bool = True):
        """
        初期化メソッド

        Args:
            include_percentiles (bool): パーセンタイル値を含めるかどうか
        """
        super().__init__("descriptive_stats")
        self.include_percentiles = include_percentiles
        self.firestore_client = FirestoreClient()

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        記述統計分析を実行

        Args:
            data (pd.DataFrame): 分析対象データ

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            # 数値型のカラムのみを抽出
            numeric_data = data.select_dtypes(include=[np.number])

            # 基本統計量の計算
            basic_stats = {
                'count': numeric_data.count().to_dict(),
                'mean': numeric_data.mean().to_dict(),
                'std': numeric_data.std().to_dict(),
                'min': numeric_data.min().to_dict(),
                'max': numeric_data.max().to_dict(),
                'median': numeric_data.median().to_dict(),
                'skewness': numeric_data.skew().to_dict(),
                'kurtosis': numeric_data.kurtosis().to_dict()
            }

            # パーセンタイル値の計算（オプション）
            if self.include_percentiles:
                percentiles = [0.1, 0.25, 0.75, 0.9]
                percentile_stats = {
                    f'percentile_{int(p*100)}': numeric_data.quantile(p).to_dict()
                    for p in percentiles
                }
                basic_stats.update(percentile_stats)

            # 正規性検定
            normality_tests = {}
            for column in numeric_data.columns:
                if len(numeric_data[column].dropna()) >= 3:  # 最小サンプルサイズ
                    _, p_value = stats.normaltest(numeric_data[column].dropna())
                    normality_tests[column] = {
                        'is_normal': p_value > 0.05,
                        'p_value': p_value
                    }

            # 結果の整形
            result = {
                'basic_statistics': basic_stats,
                'normality_tests': normality_tests,
                'summary': {
                    'total_variables': len(numeric_data.columns),
                    'total_samples': len(numeric_data),
                    'variables_analyzed': list(numeric_data.columns)
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in descriptive statistics analysis: {str(e)}")
            raise

    def _validate_data(self, data: pd.DataFrame | pd.Series, target_variable: str) -> Tuple[bool, Optional[str]]:
        """
        データのバリデーションを行います。

        Args:
            data (pd.DataFrame | pd.Series): 検証対象のデータ
            target_variable (str): 分析対象の変数名

        Returns:
            Tuple[bool, Optional[str]]: (検証結果, エラーメッセージ)
        """
        if data.empty:
            return False, "データが空です"

        if target_variable not in data.columns:
            return False, f"指定された変数 '{target_variable}' がデータに存在しません"

        if not pd.api.types.is_numeric_dtype(data[target_variable]):
            return False, f"変数 '{target_variable}' が数値型ではありません"

        if data[target_variable].isnull().any():
            return False, f"変数 '{target_variable}' に欠損値が含まれています"

        if len(data) < 10:  # 時系列分析に最低限必要なデータ数
            return False, "データ数が不足しています（10以上必要）"

        return True, None

    def _calculate_arima_metrics(self, data: pd.DataFrame | pd.Series, target_variable: str, arima_order: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        ARIMA モデルの評価指標を計算します。

        Args:
            data (pd.DataFrame | pd.Series): 分析対象のデータ
            target_variable (str): 分析対象の変数名
            arima_order (Tuple[int, int, int]): ARIMA モデルのオーダー

        Returns:
            Dict[str, Any]: ARIMA モデルの評価指標
        """
        model = sm.tsa.ARIMA(data[target_variable], order=arima_order)
        results = model.fit()

        return {
            "aic": float(results.aic),
            "bic": float(results.bic),
            "hqic": float(results.hqic),
            "mse": float(results.mse),
            "mae": float(np.mean(np.abs(results.resid))),
            "rmse": float(np.sqrt(results.mse)),
            "residual_std": float(results.resid.std())
        }

    async def calculate(self, config: DescriptiveStatsConfig) -> Dict[str, Any]:
        """
        記述統計量の計算を実行します。

        Args:
            config: 記述統計量の計算設定

        Returns:
            Dict[str, Any]: 計算結果

        Raises:
            ValueError: データのバリデーションエラー
            RuntimeError: 計算実行中のエラー
        """
        try:
            # データ取得
            data = await self.bq_service.fetch_data(config.query)

            # 指定されたカラムのみを抽出
            if config.columns:
                available_columns = [col for col in config.columns if col in data.columns]
                if not available_columns:
                    raise ValueError("指定されたカラムが存在しません")
                data = data[available_columns]

            # データバリデーション
            is_valid, error_message = self._validate_data(data, config.target_variable)
            if not is_valid:
                raise ValueError(error_message)

            # 記述統計量の計算
            analysis_results = self.analyze(data)

            # ARIMA モデルの評価指標の計算
            arima_metrics = self._calculate_arima_metrics(data, config.target_variable, config.arima_order)

            # 結果の整形
            analysis_results["arima_metrics"] = arima_metrics
            analysis_results["metadata"] = {
                "target_variable": config.target_variable,
                "arima_order": config.arima_order,
                "column_list": data.columns.tolist(),
                "analysis_timestamp": pd.Timestamp.now().isoformat()
            }

            # 結果の保存
            if config.save_results and config.dataset_id and config.table_id:
                results_df = pd.DataFrame({
                    "metric": list(analysis_results["arima_metrics"].keys()),
                    "value": list(analysis_results["arima_metrics"].values()),
                    "target_variable": config.target_variable,
                    "timestamp": pd.Timestamp.now()
                })

                await self.bq_service.save_results(
                    results_df,
                    dataset_id=config.dataset_id,
                    table_id=config.table_id
                )

            return analysis_results

        except ValueError as e:
            raise ValueError(f"データの検証に失敗しました: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"記述統計量の計算中にエラーが発生しました: {str(e)}") from e

async def calculate_descriptive_stats(request: Any) -> Tuple[Dict[str, Any], int]:
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
            return {'error': 'リクエストデータがありません'}, 400

        # 必須パラメータのバリデーション
        required_params = ['query', 'target_variable']
        for param in required_params:
            if param not in request_json:
                return {'error': f"必須パラメータ '{param}' が指定されていません"}, 400

        # 設定オブジェクトの作成
        config = DescriptiveStatsConfig(
            query=request_json['query'],
            target_variable=request_json['target_variable'],
            arima_order=tuple(request_json.get('arima_order', (5, 1, 0))),
            columns=request_json.get('columns'),
            dataset_id=request_json.get('dataset_id'),
            table_id=request_json.get('table_id'),
            save_results=request_json.get('save_results', True)
        )

        # サービスの初期化
        bq_service = BigQueryService()
        calculator = DescriptiveStatsCalculator()

        # 記述統計量の計算
        results = await calculator.calculate(config)

        return {
            'status': 'success',
            'results': results
        }, 200

    except ValueError as e:
        return {
            'status': 'error',
            'type': 'validation_error',
            'message': str(e)
        }, 400
    except Exception as e:
        return {
            'status': 'error',
            'type': 'internal_error',
            'message': str(e)
        }, 500