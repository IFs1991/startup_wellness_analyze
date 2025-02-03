# -*- coding: utf-8 -*-

"""
生存時間分析

Startup Wellness プログラム導入前後における、従業員の離職までの時間を比較分析します。
BigQueryServiceを利用した非同期処理に対応しています。
"""

from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from backend.src.database.bigquery.client import BigQueryService
from backend.src.database.firestore.client import FirestoreClient
from . import BaseAnalyzer

class SurvivalAnalyzer(BaseAnalyzer):
    """生存分析を行うクラス"""

    def __init__(self, time_column: str, event_column: str, covariates: list = None):
        """
        初期化メソッド

        Args:
            time_column (str): 時間を表す列名
            event_column (str): イベント発生を表す列名
            covariates (list): 共変量の列名リスト
        """
        super().__init__("survival_analysis")
        self.time_column = time_column
        self.event_column = event_column
        self.covariates = covariates or []

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        生存分析を実行

        Args:
            data (pd.DataFrame): 分析対象データ

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            # Kaplan-Meier推定
            kmf = KaplanMeierFitter()
            kmf.fit(
                data[self.time_column],
                data[self.event_column],
                label='Overall'
            )

            # 基本的な生存分析結果
            survival_results = {
                'survival_function': kmf.survival_function_.to_dict(),
                'median_survival_time': kmf.median_survival_time_,
                'mean_survival_time': kmf.mean_survival_time_,
                'survival_stats': {
                    'timeline': kmf.timeline.tolist(),
                    'survival_probability': kmf.survival_function_.values.flatten().tolist(),
                    'confidence_intervals': {
                        'lower': kmf.confidence_interval_['KM_estimate_lower_0.95'].tolist(),
                        'upper': kmf.confidence_interval_['KM_estimate_upper_0.95'].tolist()
                    }
                }
            }

            # Cox比例ハザードモデル（共変量がある場合）
            if self.covariates:
                cph = CoxPHFitter()
                cox_data = data[[self.time_column, self.event_column] + self.covariates]
                cph.fit(
                    cox_data,
                    duration_col=self.time_column,
                    event_col=self.event_column
                )

                # Cox回帰の結果
                cox_results = {
                    'model_summary': {
                        'log_likelihood': float(cph.log_likelihood_),
                        'concordance_index': float(cph.concordance_index_),
                        'aic': float(cph.AIC_partial_)
                    },
                    'coefficients': cph.print_summary().to_dict(),
                    'hazard_ratios': cph.hazard_ratios_.to_dict()
                }
            else:
                cox_results = None

            # 結果の整形
            result = {
                'kaplan_meier_analysis': survival_results,
                'cox_regression': cox_results,
                'summary': {
                    'total_samples': len(data),
                    'event_count': data[self.event_column].sum(),
                    'censored_count': len(data) - data[self.event_column].sum(),
                    'covariates_analyzed': self.covariates if self.covariates else []
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in survival analysis: {str(e)}")
            raise

async def analyze_survival(request: Any) -> Tuple[Dict, int]:
    """
    Cloud Functions用のエントリーポイント関数

    Args:
        request: Cloud Functionsのリクエストオブジェクト

    Returns:
        Tuple[Dict, int]: (レスポンス, ステータスコード)
    """
    try:
        request_json = request.get_json()

        if not request_json:
            return {'error': 'リクエストデータがありません'}, 400

        # 必須パラメータの確認
        required_params = ['query', 'duration_col', 'event_col']
        missing_params = [param for param in required_params if param not in request_json]
        if missing_params:
            return {
                'error': f'必須パラメータが不足しています: {", ".join(missing_params)}'
            }, 400

        # サービスの初期化
        bq_service = BigQueryService()
        firestore_client = FirestoreClient()
        analyzer = SurvivalAnalyzer(
            time_column=request_json['duration_col'],
            event_column=request_json['event_col'],
            covariates=request_json.get('covariates', [])
        )

        # パラメータの取得
        query = request_json['query']
        duration_col = request_json['duration_col']
        event_col = request_json['event_col']
        dataset_id = request_json.get('dataset_id')
        table_id = request_json.get('table_id')

        # 分析実行
        results = analyzer.analyze(await bq_service.fetch_data(query))

        return {
            'status': 'success',
            'results': results,
            'metadata': {
                'row_count': len(results['summary']['total_samples']),
                'event_count': results['summary']['event_count'],
                'censored_count': results['summary']['censored_count'],
                'max_duration': results['kaplan_meier_analysis']['survival_stats']['timeline'][-1],
                'min_duration': results['kaplan_meier_analysis']['survival_stats']['timeline'][0]
            }
        }, 200

    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }, 500