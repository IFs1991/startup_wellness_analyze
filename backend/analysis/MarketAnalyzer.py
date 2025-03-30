import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from gensim import corpora, models
import networkx as nx
import json
from datetime import datetime, timedelta
import re
from .base import BaseAnalyzer

class MarketAnalyzer(BaseAnalyzer):
    """
    市場・競合分析モジュール

    投資先企業の市場ポジションと競争環境を分析するクラス
    """

    def __init__(self):
        """
        MarketAnalyzerの初期化
        """
        super().__init__(analysis_type='market')
        self.logger.info("MarketAnalyzer initialized")

    def estimate_market_size(self,
                           market_data: Dict[str, Any],
                           growth_factors: Optional[Dict[str, float]] = None,
                           projection_years: int = 5) -> Dict[str, Any]:
        """
        TAM/SAM/SOMの推定と予測を行う

        Parameters
        ----------
        market_data : Dict[str, Any]
            市場データ {'tam': 値, 'sam': 値, 'som': 値, 'year': 年}
        growth_factors : Dict[str, float], optional
            成長係数 {'tam_growth': 率, 'sam_growth': 率, 'som_growth': 率}
        projection_years : int, optional
            予測年数（デフォルト: 5）

        Returns
        -------
        Dict[str, Any]
            TAM/SAM/SOMの現在値と将来予測
        """
        try:
            # デフォルトの成長係数
            if growth_factors is None:
                growth_factors = {
                    'tam_growth': 0.1,  # 10%
                    'sam_growth': 0.15,  # 15%
                    'som_growth': 0.2   # 20%
                }

            # 基準値の取得
            base_year = market_data.get('year', datetime.now().year)
            base_tam = market_data.get('tam', 0)
            base_sam = market_data.get('sam', 0)
            base_som = market_data.get('som', 0)

            # 予測値の計算
            projections = []

            for year in range(base_year, base_year + projection_years + 1):
                year_offset = year - base_year

                # 複利成長で計算
                tam = base_tam * (1 + growth_factors['tam_growth']) ** year_offset
                sam = base_sam * (1 + growth_factors['sam_growth']) ** year_offset
                som = base_som * (1 + growth_factors['som_growth']) ** year_offset

                # 整合性の確認（SOM ≤ SAM ≤ TAM）
                som = min(som, sam)
                sam = min(sam, tam)

                projection = {
                    'year': year,
                    'tam': tam,
                    'sam': sam,
                    'som': som,
                    'som_sam_ratio': som / sam if sam > 0 else 0,
                    'sam_tam_ratio': sam / tam if tam > 0 else 0
                }

                projections.append(projection)

            # 現在の比率
            current_som_sam_ratio = base_som / base_sam if base_sam > 0 else 0
            current_sam_tam_ratio = base_sam / base_tam if base_tam > 0 else 0

            result = {
                'current': {
                    'tam': base_tam,
                    'sam': base_sam,
                    'som': base_som,
                    'som_sam_ratio': current_som_sam_ratio,
                    'sam_tam_ratio': current_sam_tam_ratio,
                    'year': base_year
                },
                'projections': projections,
                'cagr': {
                    'tam': growth_factors['tam_growth'],
                    'sam': growth_factors['sam_growth'],
                    'som': growth_factors['som_growth']
                }
            }

            self.logger.info(f"Market size estimation completed. Base TAM: {base_tam}, SAM: {base_sam}, SOM: {base_som}")
            return result
        except Exception as e:
            self.logger.error(f"Error estimating market size: {str(e)}")
            raise

    def create_competitive_map(self,
                             competitor_data: pd.DataFrame,
                             dimensions: List[str],
                             focal_company: Optional[str] = None) -> Dict[str, Any]:
        """
        競合マッピングの生成

        Parameters
        ----------
        competitor_data : pd.DataFrame
            競合企業データ (各行が1社、列が各指標)
        dimensions : List[str]
            マッピングに使用する2つの次元（カラム名）
            または5つ以上の次元（次元削減に使用）
        focal_company : str, optional
            焦点企業（ハイライト表示する企業）

        Returns
        -------
        Dict[str, Any]
            競合マップのデータとメタデータ
        """
        try:
            if not competitor_data.index.name:
                self.logger.warning("competitor_data index is not set, using default")
                competitor_data = competitor_data.copy()
                competitor_data.index.name = 'company'

            # 2次元マッピングか多次元マッピングかを判断
            direct_mapping = len(dimensions) == 2

            if direct_mapping:
                # 2次元の直接マッピング
                x_dimension, y_dimension = dimensions

                if x_dimension not in competitor_data.columns or y_dimension not in competitor_data.columns:
                    raise ValueError(f"Dimensions {dimensions} not found in competitor data")

                # マッピングデータの作成
                map_data = competitor_data[[x_dimension, y_dimension]].copy()
                dimension_names = {
                    'x': x_dimension,
                    'y': y_dimension
                }

            else:
                # 多次元から2次元への削減（PCA）
                if any(dim not in competitor_data.columns for dim in dimensions):
                    missing_dims = [dim for dim in dimensions if dim not in competitor_data.columns]
                    raise ValueError(f"Dimensions {missing_dims} not found in competitor data")

                # 次元削減用のデータ
                reduction_data = competitor_data[dimensions].copy()

                # 欠損値処理
                if reduction_data.isnull().any().any():
                    reduction_data = reduction_data.fillna(reduction_data.mean())

                # スケーリング
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(reduction_data)

                # PCAの実行
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(scaled_data)

                # 結果をDataFrameに変換
                map_data = pd.DataFrame(
                    pca_result,
                    index=competitor_data.index,
                    columns=['PC1', 'PC2']
                )

                # 次元の解釈（各主成分における元の次元の寄与度）
                loadings = pd.DataFrame(
                    pca.components_.T,
                    index=dimensions,
                    columns=['PC1', 'PC2']
                )

                dimension_names = {
                    'x': 'PC1',
                    'y': 'PC2',
                    'loadings': loadings.to_dict(),
                    'explained_variance': pca.explained_variance_ratio_.tolist(),
                    'original_dimensions': dimensions
                }

            # 象限の識別（各社の位置）
            map_data['quadrant'] = 0
            map_data.loc[(map_data.iloc[:, 0] > 0) & (map_data.iloc[:, 1] > 0), 'quadrant'] = 1  # 右上
            map_data.loc[(map_data.iloc[:, 0] < 0) & (map_data.iloc[:, 1] > 0), 'quadrant'] = 2  # 左上
            map_data.loc[(map_data.iloc[:, 0] < 0) & (map_data.iloc[:, 1] < 0), 'quadrant'] = 3  # 左下
            map_data.loc[(map_data.iloc[:, 0] > 0) & (map_data.iloc[:, 1] < 0), 'quadrant'] = 4  # 右下

            # 企業間の相対的な距離
            if len(competitor_data) > 1:
                companies = map_data.index.tolist()
                distances = {}

                for i, company1 in enumerate(companies):
                    distances[company1] = {}
                    for company2 in companies[i+1:]:
                        pos1 = map_data.loc[company1, [map_data.columns[0], map_data.columns[1]]].values
                        pos2 = map_data.loc[company2, [map_data.columns[0], map_data.columns[1]]].values
                        distance = np.linalg.norm(pos1 - pos2)
                        distances[company1][company2] = distance

                        # 両方向の距離を記録
                        if company2 not in distances:
                            distances[company2] = {}
                        distances[company2][company1] = distance

                # 各企業の最も近い競合とその距離
                nearest_competitors = {}
                for company in companies:
                    company_distances = {other: dist for other, dist in distances[company].items()}
                    if company_distances:
                        nearest = min(company_distances.items(), key=lambda x: x[1])
                        nearest_competitors[company] = {
                            'nearest_competitor': nearest[0],
                            'distance': nearest[1]
                        }
            else:
                nearest_competitors = {}

            # 焦点企業のハイライト
            if focal_company and focal_company in map_data.index:
                map_data.loc[focal_company, 'is_focal'] = True
            else:
                map_data['is_focal'] = False

            # 結果の整形
            result = {
                'mapping_type': 'direct' if direct_mapping else 'pca',
                'dimensions': dimension_names,
                'positions': map_data.reset_index().to_dict('records'),
                'distances': distances if len(competitor_data) > 1 else {},
                'nearest_competitors': nearest_competitors,
                'focal_company': focal_company if focal_company in map_data.index else None,
                'quadrant_distribution': map_data['quadrant'].value_counts().to_dict(),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            self.logger.info(f"Competitive mapping created with {len(map_data)} companies")
            return result
        except Exception as e:
            self.logger.error(f"Error creating competitive map: {str(e)}")
            raise

    def analyze_competitive_positioning(self,
                                      company_data: Dict[str, Dict[str, float]],
                                      dimensions: List[str],
                                      focal_company: str) -> Dict[str, Any]:
        """
        多次元での競合ポジショニング分析（レーダーチャート用）

        Parameters
        ----------
        company_data : Dict[str, Dict[str, float]]
            各社の各次元でのスコア {'企業名': {'次元1': 値, '次元2': 値, ...}}
        dimensions : List[str]
            分析に使用する次元リスト
        focal_company : str
            焦点企業

        Returns
        -------
        Dict[str, Any]
            競合ポジショニング分析結果
        """
        try:
            # 各企業のデータを検証
            all_companies = list(company_data.keys())

            if focal_company not in all_companies:
                raise ValueError(f"Focal company '{focal_company}' not found in company data")

            # 各次元でのランキング作成
            rankings = {dim: {} for dim in dimensions}

            for dimension in dimensions:
                # その次元でのスコアを全企業で取得
                dimension_scores = [(company, data.get(dimension, 0)) for company, data in company_data.items()]

                # スコアでソート（降順）
                dimension_scores.sort(key=lambda x: x[1], reverse=True)

                # ランキング付け
                for rank, (company, score) in enumerate(dimension_scores, 1):
                    rankings[dimension][company] = {
                        'rank': rank,
                        'score': score,
                        'percentile': 100 * (len(all_companies) - rank + 1) / len(all_companies)
                    }

            # 焦点企業と競合企業の比較
            focal_company_data = company_data[focal_company]

            # 各次元の業界平均を計算
            industry_averages = {}
            for dimension in dimensions:
                scores = [data.get(dimension, 0) for data in company_data.values()]
                industry_averages[dimension] = np.mean(scores)

            # 差別化スコアの計算（焦点企業の各次元のスコアと業界平均との差）
            differentiation_scores = {}
            for dimension in dimensions:
                focal_score = focal_company_data.get(dimension, 0)
                avg_score = industry_averages[dimension]

                if avg_score != 0:
                    diff_score = (focal_score - avg_score) / avg_score
                else:
                    diff_score = 0 if focal_score == 0 else float('inf')

                differentiation_scores[dimension] = diff_score

            # 強み・弱みの特定
            strengths = []
            weaknesses = []

            for dimension in dimensions:
                diff_score = differentiation_scores[dimension]

                if diff_score > 0.1:  # 10%以上優れている
                    strengths.append({
                        'dimension': dimension,
                        'score': focal_company_data.get(dimension, 0),
                        'industry_avg': industry_averages[dimension],
                        'difference': diff_score
                    })
                elif diff_score < -0.1:  # 10%以上劣っている
                    weaknesses.append({
                        'dimension': dimension,
                        'score': focal_company_data.get(dimension, 0),
                        'industry_avg': industry_averages[dimension],
                        'difference': diff_score
                    })

            # 強みを差の降順でソート
            strengths.sort(key=lambda x: x['difference'], reverse=True)
            # 弱みを差の昇順でソート（最も弱いものが先頭）
            weaknesses.sort(key=lambda x: x['difference'])

            # 機能パリティの次元（業界平均と同等）
            parity_dimensions = [
                dimension for dimension in dimensions
                if abs(differentiation_scores[dimension]) <= 0.1
            ]

            # 結果の整形
            result = {
                'focal_company': focal_company,
                'rankings': rankings,
                'industry_averages': industry_averages,
                'differentiation_scores': differentiation_scores,
                'strengths': strengths,
                'weaknesses': weaknesses,
                'parity_dimensions': parity_dimensions,
                'radar_chart_data': {
                    'dimensions': dimensions,
                    'focal_company': [focal_company_data.get(dim, 0) for dim in dimensions],
                    'industry_avg': [industry_averages[dim] for dim in dimensions]
                },
                'competitors_count': len(all_companies),
                'analyzed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            self.logger.info(f"Competitive positioning analysis completed for {focal_company}")
            return result
        except Exception as e:
            self.logger.error(f"Error analyzing competitive positioning: {str(e)}")
            raise

    def track_competitors(self,
                        competitor_data: Dict[str, Dict],
                        time_period: str = '1y',
                        metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        競合企業の追跡情報を分析

        Parameters
        ----------
        competitor_data : Dict[str, Dict]
            各競合企業の時系列データ
            {'企業名': {'funding': [...], 'product_updates': [...], 'hiring': [...]}}
        time_period : str, optional
            分析対象期間（'1m', '3m', '6m', '1y', 'all'）
        metrics : List[str], optional
            分析対象指標（指定がなければすべて）

        Returns
        -------
        Dict[str, Any]
            競合追跡分析結果
        """
        try:
            # 分析対象期間の設定
            now = datetime.now()

            if time_period == '1m':
                start_date = now - timedelta(days=30)
            elif time_period == '3m':
                start_date = now - timedelta(days=90)
            elif time_period == '6m':
                start_date = now - timedelta(days=180)
            elif time_period == '1y':
                start_date = now - timedelta(days=365)
            else:  # 'all'
                start_date = datetime(1970, 1, 1)  # 十分昔の日付

            # 分析対象指標の設定
            if metrics is None:
                metrics = ['funding', 'product_updates', 'hiring', 'news']

            # 各競合企業のアクティビティ集計
            competitor_activities = {}

            for competitor, data in competitor_data.items():
                activities = {}

                for metric in metrics:
                    if metric in data:
                        # そのメトリクスの期間内データのみ抽出
                        period_data = [
                            item for item in data[metric]
                            if 'date' in item and datetime.fromisoformat(item['date']) >= start_date
                        ]

                        # カウントと統計情報
                        activities[metric] = {
                            'count': len(period_data),
                            'latest': period_data[-1] if period_data else None,
                            'data': period_data
                        }

                        # メトリクス固有の集計
                        if metric == 'funding' and period_data:
                            total_amount = sum(item.get('amount', 0) for item in period_data)
                            activities[metric]['total_amount'] = total_amount
                            activities[metric]['avg_amount'] = total_amount / len(period_data) if period_data else 0

                        elif metric == 'hiring' and period_data:
                            roles = [item.get('role', '') for item in period_data]
                            roles_count = {}
                            for role in roles:
                                if role:
                                    roles_count[role] = roles_count.get(role, 0) + 1
                            activities[metric]['roles'] = roles_count

                            senior_count = sum(1 for item in period_data if 'senior' in item.get('role', '').lower())
                            activities[metric]['senior_ratio'] = senior_count / len(period_data) if period_data else 0

                competitor_activities[competitor] = activities

            # アクティビティのランキング
            rankings = {}

            for metric in metrics:
                metric_scores = []

                for competitor, activities in competitor_activities.items():
                    if metric in activities:
                        # メトリクスごとのスコア計算（単純なカウントまたは合計額など）
                        if metric == 'funding':
                            score = activities[metric].get('total_amount', 0)
                        else:
                            score = activities[metric]['count']

                        metric_scores.append((competitor, score))

                # スコアでランキング（降順）
                metric_scores.sort(key=lambda x: x[1], reverse=True)

                # ランキング結果を格納
                rankings[metric] = {
                    competitor: {'rank': rank, 'score': score}
                    for rank, (competitor, score) in enumerate(metric_scores, 1)
                }

            # 総合アクティビティスコア
            total_scores = {}

            for competitor in competitor_data.keys():
                score = 0
                for metric in metrics:
                    if metric in rankings and competitor in rankings[metric]:
                        # ランキングの逆数でスコア付け（1位は最高スコア）
                        rank = rankings[metric][competitor]['rank']
                        score += 1 / rank if rank > 0 else 0

                total_scores[competitor] = score

            # 総合ランキング
            total_ranking = sorted(
                [(competitor, score) for competitor, score in total_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )

            result = {
                'time_period': time_period,
                'metrics': metrics,
                'competitor_activities': competitor_activities,
                'rankings': rankings,
                'total_scores': total_scores,
                'total_ranking': [
                    {'rank': rank, 'competitor': competitor, 'score': score}
                    for rank, (competitor, score) in enumerate(total_ranking, 1)
                ],
                'analysis_date': now.strftime('%Y-%m-%d %H:%M:%S')
            }

            self.logger.info(f"Competitor tracking analysis completed for {len(competitor_data)} competitors")
            return result
        except Exception as e:
            self.logger.error(f"Error tracking competitors: {str(e)}")
            raise

    def analyze_market_trends(self,
                            keyword_data: Dict[str, List],
                            date_range: Optional[Tuple[str, str]] = None,
                            industry: Optional[str] = None) -> Dict[str, Any]:
        """
        市場トレンドを分析

        Parameters
        ----------
        keyword_data : Dict[str, List]
            キーワード検索データ {'keyword': [{'date': 日付, 'value': 値}, ...]}
        date_range : Tuple[str, str], optional
            分析対象期間 ('YYYY-MM-DD', 'YYYY-MM-DD')
        industry : str, optional
            業界カテゴリ

        Returns
        -------
        Dict[str, Any]
            市場トレンド分析結果
        """
        try:
            # キーワードリスト
            keywords = list(keyword_data.keys())

            # 日付範囲の処理
            if date_range:
                start_date = datetime.fromisoformat(date_range[0])
                end_date = datetime.fromisoformat(date_range[1])
            else:
                # デフォルトは全期間
                all_dates = []
                for keyword, data_points in keyword_data.items():
                    all_dates.extend([datetime.fromisoformat(dp['date']) for dp in data_points])

                if all_dates:
                    start_date = min(all_dates)
                    end_date = max(all_dates)
                else:
                    raise ValueError("No date information found in keyword data")

            # 各キーワードの時系列データをDataFrameに変換
            trend_data = pd.DataFrame()

            for keyword in keywords:
                # 日付とスコアのリストを抽出
                data_points = keyword_data[keyword]
                dates = [datetime.fromisoformat(dp['date']) for dp in data_points]
                values = [dp['value'] for dp in data_points]

                # キーワードのDataFrameを作成
                keyword_df = pd.DataFrame({
                    'date': dates,
                    keyword: values
                })

                # 日付でフィルタリング
                keyword_df = keyword_df[
                    (keyword_df['date'] >= start_date) &
                    (keyword_df['date'] <= end_date)
                ]

                if trend_data.empty:
                    trend_data = keyword_df
                else:
                    # 既存のDataFrameとマージ
                    trend_data = pd.merge(
                        trend_data, keyword_df,
                        on='date',
                        how='outer'
                    )

            # 欠損値の処理
            trend_data = trend_data.ffill().bfill()

            # 各キーワードの傾向分析
            keyword_trends = {}

            for keyword in keywords:
                if keyword in trend_data.columns:
                    # トレンドラインの計算（線形回帰）
                    x = np.arange(len(trend_data))
                    y = trend_data[keyword].values

                    # 単純な線形トレンド（最小二乗法）
                    if len(x) > 1:  # 少なくとも2点が必要
                        slope, intercept = np.polyfit(x, y, 1)
                    else:
                        slope, intercept = 0, y[0] if len(y) > 0 else 0

                    # トレンドの方向判定
                    if slope > 0.05:
                        trend_direction = 'increasing'
                    elif slope < -0.05:
                        trend_direction = 'decreasing'
                    else:
                        trend_direction = 'stable'

                    # 成長率の計算
                    first_value = trend_data[keyword].iloc[0] if not trend_data.empty else 0
                    last_value = trend_data[keyword].iloc[-1] if not trend_data.empty else 0

                    if first_value > 0:
                        growth_rate = (last_value - first_value) / first_value
                    else:
                        growth_rate = float('inf') if last_value > 0 else 0

                    # 変動性（標準偏差 / 平均）
                    mean_value = trend_data[keyword].mean()
                    std_value = trend_data[keyword].std()
                    volatility = std_value / mean_value if mean_value > 0 else 0

                    # 結果をまとめる
                    keyword_trends[keyword] = {
                        'slope': slope,
                        'trend_direction': trend_direction,
                        'growth_rate': growth_rate,
                        'volatility': volatility,
                        'mean': mean_value,
                        'latest_value': last_value,
                        'peak_value': trend_data[keyword].max(),
                        'peak_date': trend_data.loc[trend_data[keyword].idxmax(), 'date'].strftime('%Y-%m-%d') if not trend_data.empty else None
                    }

            # キーワード間の相関分析
            correlation_matrix = {}

            if len(keywords) > 1:
                keyword_columns = [k for k in keywords if k in trend_data.columns]
                corr_df = trend_data[keyword_columns].corr()

                for k1 in keyword_columns:
                    correlation_matrix[k1] = {}
                    for k2 in keyword_columns:
                        if k1 != k2:
                            correlation_matrix[k1][k2] = corr_df.loc[k1, k2]

            # 全体的な市場トレンドの判定
            overall_growth_rates = [kt['growth_rate'] for kt in keyword_trends.values()]
            overall_growth_rate = np.mean(overall_growth_rates) if overall_growth_rates else 0

            if overall_growth_rate > 0.1:
                overall_trend = 'strongly_growing'
            elif overall_growth_rate > 0:
                overall_trend = 'moderately_growing'
            elif overall_growth_rate > -0.1:
                overall_trend = 'stable'
            else:
                overall_trend = 'declining'

            # 結果の整形
            result = {
                'date_range': {
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d')
                },
                'keywords': keywords,
                'keyword_trends': keyword_trends,
                'correlation_matrix': correlation_matrix,
                'overall_trend': overall_trend,
                'overall_growth_rate': overall_growth_rate,
                'industry': industry,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            self.logger.info(f"Market trends analysis completed for {len(keywords)} keywords")
            return result
        except Exception as e:
            self.logger.error(f"Error analyzing market trends: {str(e)}")
            raise