# -*- coding: utf-8 -*-

"""
時系列分析モジュール
時系列データのトレンド分析と予測を行います。
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from . import BaseAnalyzer, AnalysisError

class TimeSeriesAnalyzer(BaseAnalyzer):
    """時系列分析を行うクラス"""

    def __init__(
        self,
        time_column: str,
        value_column: str,
        frequency: str = 'D',
        decomposition_model: str = 'additive',
        forecast_periods: int = 30
    ):
        """
        初期化メソッド

        Args:
            time_column (str): 時間列の名前
            value_column (str): 値列の名前
            frequency (str): データの頻度（'D': 日次, 'W': 週次, 'M': 月次, etc.）
            decomposition_model (str): 分解モデル（'additive' or 'multiplicative'）
            forecast_periods (int): 予測期間
        """
        super().__init__("time_series_analysis")
        self.time_column = time_column
        self.value_column = value_column
        self.frequency = frequency
        self.decomposition_model = decomposition_model
        self.forecast_periods = forecast_periods

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        データのバリデーション

        Args:
            data (pd.DataFrame): 検証対象のデータ

        Returns:
            bool: バリデーション結果
        """
        if data.empty:
            raise AnalysisError("データが空です")

        if self.time_column not in data.columns:
            raise AnalysisError(f"時間列 {self.time_column} が存在しません")

        if self.value_column not in data.columns:
            raise AnalysisError(f"値列 {self.value_column} が存在しません")

        if not pd.api.types.is_datetime64_any_dtype(data[self.time_column]):
            raise AnalysisError("時間列が日時型ではありません")

        if data[self.value_column].isnull().any():
            raise AnalysisError("値列に欠損値が含まれています")

        return True

    def _prepare_data(self, data: pd.DataFrame) -> pd.Series:
        """
        データの前処理

        Args:
            data (pd.DataFrame): 前処理対象のデータ

        Returns:
            pd.Series: 前処理済みの時系列データ
        """
        # 時系列インデックスの設定
        ts_data = data.set_index(self.time_column)[self.value_column]

        # 時系列の並べ替えとリサンプリング
        ts_data = ts_data.sort_index().resample(self.frequency).mean()

        # 欠損値の補間
        if ts_data.isnull().any():
            ts_data = ts_data.interpolate(method='time')

        return ts_data

    def _perform_decomposition(self, ts_data: pd.Series) -> Dict[str, Any]:
        """
        時系列分解を実行

        Args:
            ts_data (pd.Series): 時系列データ

        Returns:
            Dict[str, Any]: 分解結果
        """
        decomposition = seasonal_decompose(
            ts_data,
            model=self.decomposition_model,
            period=self._determine_seasonality_period(ts_data)
        )

        return {
            'trend': decomposition.trend.fillna(method='bfill').fillna(method='ffill').tolist(),
            'seasonal': decomposition.seasonal.fillna(method='bfill').fillna(method='ffill').tolist(),
            'residual': decomposition.resid.fillna(method='bfill').fillna(method='ffill').tolist()
        }

    def _determine_seasonality_period(self, ts_data: pd.Series) -> int:
        """
        季節性の周期を決定

        Args:
            ts_data (pd.Series): 時系列データ

        Returns:
            int: 季節性の周期
        """
        if self.frequency == 'D':
            return 7  # 週次周期
        elif self.frequency == 'M':
            return 12  # 年次周期
        elif self.frequency == 'Q':
            return 4  # 四半期周期
        else:
            return max(2, len(ts_data) // 10)  # デフォルトの周期

    def _check_stationarity(self, ts_data: pd.Series) -> Dict[str, Any]:
        """
        定常性の検定

        Args:
            ts_data (pd.Series): 時系列データ

        Returns:
            Dict[str, Any]: 検定結果
        """
        result = adfuller(ts_data)
        return {
            'test_statistic': float(result[0]),
            'p_value': float(result[1]),
            'critical_values': {
                f'{key}%': float(value)
                for key, value in result[4].items()
            },
            'is_stationary': result[1] < 0.05
        }

    def _fit_arima_model(
        self,
        ts_data: pd.Series
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        ARIMAモデルによる予測

        Args:
            ts_data (pd.Series): 時系列データ

        Returns:
            Tuple[List[float], List[float], List[float]]: 予測値、信頼区間下限、信頼区間上限
        """
        # ARIMAモデルのフィッティング
        model = ARIMA(ts_data, order=(1, 1, 1))
        results = model.fit()

        # 予測の実行
        forecast = results.forecast(steps=self.forecast_periods, alpha=0.05)

        # 予測値と信頼区間の取得
        predictions = forecast.predicted_mean.tolist()
        conf_int = forecast.conf_int()
        lower_bound = conf_int.iloc[:, 0].tolist()
        upper_bound = conf_int.iloc[:, 1].tolist()

        return predictions, lower_bound, upper_bound

    async def analyze(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        時系列分析を実行

        Args:
            data (pd.DataFrame): 分析対象データ
            target_columns (Optional[List[str]]): 分析対象の列名リスト
            **kwargs: 追加のパラメータ

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            # データの前処理
            ts_data = self._prepare_data(data)

            # 時系列分解
            decomposition = self._perform_decomposition(ts_data)

            # 定常性の検定
            stationarity = self._check_stationarity(ts_data)

            # 予測の実行
            predictions, lower_bound, upper_bound = self._fit_arima_model(ts_data)

            # 基本統計量の計算
            basic_stats = {
                'mean': float(ts_data.mean()),
                'std': float(ts_data.std()),
                'min': float(ts_data.min()),
                'max': float(ts_data.max()),
                'median': float(ts_data.median())
            }

            # 結果の整形
            result = {
                'time_series_data': {
                    'timestamps': [str(idx) for idx in ts_data.index],
                    'values': ts_data.tolist()
                },
                'decomposition': decomposition,
                'stationarity_test': stationarity,
                'forecast': {
                    'predictions': predictions,
                    'confidence_interval': {
                        'lower': lower_bound,
                        'upper': upper_bound
                    },
                    'forecast_periods': self.forecast_periods
                },
                'basic_statistics': basic_stats,
                'summary': {
                    'total_observations': len(ts_data),
                    'frequency': self.frequency,
                    'start_date': str(ts_data.index[0]),
                    'end_date': str(ts_data.index[-1]),
                    'parameters': {
                        'time_column': self.time_column,
                        'value_column': self.value_column,
                        'decomposition_model': self.decomposition_model
                    }
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in time series analysis: {str(e)}")
            raise AnalysisError(f"時系列分析に失敗しました: {str(e)}")

async def analyze_time_series(
    collection: str,
    time_column: str,
    value_column: str,
    frequency: str = 'D',
    decomposition_model: str = 'additive',
    forecast_periods: int = 30,
    filters: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    時系列分析を実行するヘルパー関数

    Args:
        collection (str): 分析対象のコレクション名
        time_column (str): 時間列の名前
        value_column (str): 値列の名前
        frequency (str): データの頻度
        decomposition_model (str): 分解モデル
        forecast_periods (int): 予測期間
        filters (Optional[List[Dict[str, Any]]]): データ取得時のフィルター条件

    Returns:
        Dict[str, Any]: 分析結果
    """
    try:
        analyzer = TimeSeriesAnalyzer(
            time_column=time_column,
            value_column=value_column,
            frequency=frequency,
            decomposition_model=decomposition_model,
            forecast_periods=forecast_periods
        )

        # データの取得
        data = await analyzer.fetch_data(
            collection=collection,
            filters=filters
        )

        # 分析の実行と結果の保存
        results = await analyzer.analyze_and_save(
            data=data
        )

        return results

    except Exception as e:
        raise AnalysisError(f"時系列分析の実行に失敗しました: {str(e)}")

async def analyze_time_series_request(request: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    時系列分析のリクエストを処理

    Args:
        request (Dict[str, Any]): リクエストデータ

    Returns:
        Tuple[Dict[str, Any], int]: (レスポンス, ステータスコード)
    """
    try:
        # 必須パラメータの確認
        required_params = ['collection', 'time_column', 'value_column']
        for param in required_params:
            if param not in request:
                return {
                    'status': 'error',
                    'message': f'必須パラメータ {param} が指定されていません'
                }, 400

        # オプションパラメータの取得
        frequency = request.get('frequency', 'D')
        decomposition_model = request.get('decomposition_model', 'additive')
        forecast_periods = int(request.get('forecast_periods', 30))
        filters = request.get('filters')

        # 分析の実行
        results = await analyze_time_series(
            collection=request['collection'],
            time_column=request['time_column'],
            value_column=request['value_column'],
            frequency=frequency,
            decomposition_model=decomposition_model,
            forecast_periods=forecast_periods,
            filters=filters
        )

        return {
            'status': 'success',
            'results': results
        }, 200

    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }, 500