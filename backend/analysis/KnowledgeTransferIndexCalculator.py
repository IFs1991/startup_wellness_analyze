import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from .base import BaseAnalyzer

class KnowledgeTransferIndexCalculator(BaseAnalyzer):
    """
    知識移転指数計算クラス

    企業間の健康施策に関する知識共有度を計測するためのクラス
    """

    def __init__(self):
        """
        KnowledgeTransferIndexCalculatorの初期化
        """
        super().__init__()
        self.logger.info("KnowledgeTransferIndexCalculator initialized")

    def calculate_kti(self,
                     network_data: pd.DataFrame,
                     adoption_time_data: pd.DataFrame = None,
                     weighting_factors: Dict[str, float] = None) -> Dict[str, float]:
        """
        知識移転指数（KTI）の基本計算

        Parameters
        ----------
        network_data : pd.DataFrame
            企業間のネットワーク関係を表すデータフレーム
            (source, target, weight)の形式
        adoption_time_data : pd.DataFrame, optional
            健康施策の採用タイミングデータ (デフォルト: None)
            (company, initiative, adoption_time)の形式
        weighting_factors : Dict[str, float], optional
            各要素の重み付け (デフォルト: None)

        Returns
        -------
        Dict[str, float]
            各企業の知識移転指数
        """
        try:
            # デフォルトの重み設定
            if weighting_factors is None:
                weighting_factors = {
                    'network_centrality': 0.4,  # ネットワーク中心性の重み
                    'adoption_speed': 0.3,      # 施策採用スピードの重み
                    'initiative_count': 0.3     # 採用施策数の重み
                }

            # ネットワークグラフ構築
            G = nx.DiGraph()

            # ネットワークデータからエッジを追加
            for _, row in network_data.iterrows():
                source = row['source']
                target = row['target']
                weight = row.get('weight', 1.0)
                G.add_edge(source, target, weight=weight)

            # 各企業のネットワーク中心性を計算
            centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)

            # 結果格納用の辞書
            kti_values = {}

            # 各企業のKTI計算
            for company in G.nodes():
                # 中心性スコア (0-1スケール)
                centrality_score = centrality.get(company, 0)

                # 採用スピードと施策数のスコアを初期化
                adoption_speed_score = 0.5  # デフォルト値
                initiative_count_score = 0.5  # デフォルト値

                # 施策採用データが提供されている場合
                if adoption_time_data is not None:
                    company_data = adoption_time_data[adoption_time_data['company'] == company]

                    if not company_data.empty:
                        # 施策数のスコア計算
                        initiative_count = len(company_data)
                        max_count = adoption_time_data['company'].value_counts().max()
                        initiative_count_score = min(initiative_count / max_count, 1.0) if max_count > 0 else 0.5

                        # 採用スピードのスコア計算
                        if 'adoption_time' in company_data.columns:
                            # 各施策ごとに採用タイミングの相対ランクを計算
                            speed_scores = []

                            for initiative in company_data['initiative'].unique():
                                initiative_data = adoption_time_data[adoption_time_data['initiative'] == initiative]
                                if not initiative_data.empty:
                                    # 小さい値（早期採用）ほど高スコア
                                    min_time = initiative_data['adoption_time'].min()
                                    max_time = initiative_data['adoption_time'].max()
                                    time_range = max_time - min_time

                                    if time_range > 0:
                                        company_time = company_data[company_data['initiative'] == initiative]['adoption_time'].iloc[0]
                                        relative_speed = 1 - ((company_time - min_time) / time_range)
                                        speed_scores.append(relative_speed)

                            if speed_scores:
                                adoption_speed_score = np.mean(speed_scores)

                # 重み付けしたKTI計算
                kti = (
                    weighting_factors['network_centrality'] * centrality_score +
                    weighting_factors['adoption_speed'] * adoption_speed_score +
                    weighting_factors['initiative_count'] * initiative_count_score
                )

                kti_values[company] = kti

            self.logger.info(f"KTI calculation completed for {len(kti_values)} companies")
            return kti_values
        except Exception as e:
            self.logger.error(f"Error calculating KTI: {str(e)}")
            raise

    def calculate_industry_kti(self,
                             company_kti: Dict[str, float],
                             industry_mapping: Dict[str, str]) -> Dict[str, float]:
        """
        業界ごとの知識移転指数を計算

        Parameters
        ----------
        company_kti : Dict[str, float]
            各企業の知識移転指数
        industry_mapping : Dict[str, str]
            企業と業界のマッピング

        Returns
        -------
        Dict[str, float]
            業界ごとの平均知識移転指数
        """
        try:
            # 業界ごとのKTI値をグループ化
            industry_kti = {}

            for company, kti in company_kti.items():
                if company in industry_mapping:
                    industry = industry_mapping[company]

                    if industry not in industry_kti:
                        industry_kti[industry] = []

                    industry_kti[industry].append(kti)

            # 業界ごとの平均KTI計算
            industry_avg_kti = {
                industry: np.mean(kti_list) if kti_list else 0
                for industry, kti_list in industry_kti.items()
            }

            self.logger.info(f"Industry KTI calculation completed for {len(industry_avg_kti)} industries")
            return industry_avg_kti
        except Exception as e:
            self.logger.error(f"Error calculating industry KTI: {str(e)}")
            raise

    def calculate_best_practice_sharing_score(self,
                                           company_kti: Dict[str, float],
                                           practice_adoption_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        ベストプラクティス共有スコアを計算

        Parameters
        ----------
        company_kti : Dict[str, float]
            各企業の知識移転指数
        practice_adoption_data : pd.DataFrame
            ベストプラクティス採用データ
            (company, practice, success_rate, adoption_time)の形式

        Returns
        -------
        Dict[str, Dict]
            各企業のベストプラクティス共有スコアと詳細
        """
        try:
            # 結果格納用の辞書
            sharing_scores = {}

            # 各企業のスコア計算
            for company in company_kti.keys():
                company_data = practice_adoption_data[practice_adoption_data['company'] == company]

                if company_data.empty:
                    sharing_scores[company] = {
                        'overall_score': 0,
                        'practice_details': {},
                        'kti_factor': company_kti[company]
                    }
                    continue

                # 各プラクティスのスコア計算
                practice_scores = {}

                for practice in company_data['practice'].unique():
                    practice_rows = company_data[company_data['practice'] == practice]

                    # 成功率が高いほど高スコア
                    success_rate = practice_rows['success_rate'].mean() if 'success_rate' in practice_rows.columns else 0.5

                    # 早期採用ほど高スコア
                    adoption_time = practice_rows['adoption_time'].iloc[0] if 'adoption_time' in practice_rows.columns else 0
                    all_practice_data = practice_adoption_data[practice_adoption_data['practice'] == practice]

                    adoption_speed_score = 0.5  # デフォルト値
                    if not all_practice_data.empty and 'adoption_time' in all_practice_data.columns:
                        min_time = all_practice_data['adoption_time'].min()
                        max_time = all_practice_data['adoption_time'].max()
                        time_range = max_time - min_time

                        if time_range > 0:
                            adoption_speed_score = 1 - ((adoption_time - min_time) / time_range)

                    # プラクティススコア計算（成功率と採用スピードの平均）
                    practice_score = (success_rate + adoption_speed_score) / 2
                    practice_scores[practice] = practice_score

                # 全体スコア計算（KTIを考慮）
                avg_practice_score = np.mean(list(practice_scores.values())) if practice_scores else 0
                overall_score = avg_practice_score * (0.5 + (company_kti[company] * 0.5))  # KTIが高いほど高スコア

                sharing_scores[company] = {
                    'overall_score': overall_score,
                    'practice_details': practice_scores,
                    'kti_factor': company_kti[company]
                }

            self.logger.info(f"Best practice sharing score calculation completed for {len(sharing_scores)} companies")
            return sharing_scores
        except Exception as e:
            self.logger.error(f"Error calculating best practice sharing score: {str(e)}")
            raise

    def identify_knowledge_hubs(self,
                              company_kti: Dict[str, float],
                              network_data: pd.DataFrame,
                              top_n: int = 5) -> List[Dict]:
        """
        ナレッジハブ（知識共有の中心企業）を特定

        Parameters
        ----------
        company_kti : Dict[str, float]
            各企業の知識移転指数
        network_data : pd.DataFrame
            企業間のネットワーク関係を表すデータフレーム
        top_n : int, optional
            上位何社を返すか (デフォルト: 5)

        Returns
        -------
        List[Dict]
            ナレッジハブ企業の情報
        """
        try:
            # ネットワークグラフ構築
            G = nx.DiGraph()

            # ネットワークデータからエッジを追加
            for _, row in network_data.iterrows():
                G.add_edge(row['source'], row['target'], weight=row.get('weight', 1.0))

            # 各種中心性指標の計算
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
            closeness_centrality = nx.closeness_centrality(G, distance='weight')

            # 結果格納用のリスト
            hub_scores = []

            for company, kti in company_kti.items():
                if company in G.nodes():
                    # 各種中心性スコアの取得
                    degree = degree_centrality.get(company, 0)
                    betweenness = betweenness_centrality.get(company, 0)
                    closeness = closeness_centrality.get(company, 0)

                    # 総合スコア計算（KTIと中心性指標の組み合わせ）
                    hub_score = (kti * 0.4) + (degree * 0.3) + (betweenness * 0.2) + (closeness * 0.1)

                    hub_scores.append({
                        'company': company,
                        'hub_score': hub_score,
                        'kti': kti,
                        'degree_centrality': degree,
                        'betweenness_centrality': betweenness,
                        'closeness_centrality': closeness
                    })

            # スコアの降順でソート
            hub_scores.sort(key=lambda x: x['hub_score'], reverse=True)

            # 上位N社を返す
            top_hubs = hub_scores[:top_n]

            self.logger.info(f"Knowledge hub identification completed, found {len(top_hubs)} top hubs")
            return top_hubs
        except Exception as e:
            self.logger.error(f"Error identifying knowledge hubs: {str(e)}")
            raise