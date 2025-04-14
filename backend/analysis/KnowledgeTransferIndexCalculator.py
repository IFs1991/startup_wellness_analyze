import pandas as pd
import numpy as np
import networkx as nx
import gc
import weakref
from typing import Dict, List, Tuple, Optional, Union, Any, Iterator, ContextManager
from contextlib import contextmanager
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
        self._temp_graphs = weakref.WeakValueDictionary()  # グラフオブジェクトを弱参照で管理
        self._data_cache = {}  # 計算結果のキャッシュ

    @contextmanager
    def _managed_graph(self, name: str = "default") -> Iterator[nx.DiGraph]:
        """
        グラフオブジェクトのライフサイクルを管理するコンテキストマネージャー

        Parameters
        ----------
        name : str, optional
            グラフの識別名 (デフォルト: "default")

        Yields
        ------
        nx.DiGraph
            管理されたグラフオブジェクト
        """
        try:
            # 新しいグラフを作成して弱参照辞書に登録
            graph = nx.DiGraph()
            self._temp_graphs[name] = graph
            yield graph
        finally:
            # コンテキスト終了時に明示的にグラフ参照を削除
            if name in self._temp_graphs:
                del self._temp_graphs[name]
            # ガベージコレクションを促進
            gc.collect()

    @contextmanager
    def _managed_dataframe(self, df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """
        データフレームのライフサイクルを管理するコンテキストマネージャー

        Parameters
        ----------
        df : pd.DataFrame
            管理対象のデータフレーム

        Yields
        ------
        pd.DataFrame
            管理されたデータフレーム
        """
        try:
            yield df
        finally:
            # 明示的にデータフレームを削除（参照解除）
            del df
            gc.collect()

    def calculate_kti(self,
                     network_data: pd.DataFrame,
                     adoption_time_data: pd.DataFrame = None,
                     weighting_factors: Dict[str, float] = None,
                     batch_size: int = 1000) -> Dict[str, float]:
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
        batch_size : int, optional
            大規模ネットワークを処理する際のバッチサイズ (デフォルト: 1000)

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

            # グラフオブジェクトをコンテキストマネージャで管理
            with self._managed_graph("kti_calculation") as G:
                # バッチ処理でネットワークのエッジを追加
                nodes = set()

                # 大規模データ処理のため、バッチごとに追加
                for i in range(0, len(network_data), batch_size):
                    batch = network_data.iloc[i:i+batch_size]

                    for _, row in batch.iterrows():
                        source = row['source']
                        target = row['target']
                        weight = row.get('weight', 1.0)
                        G.add_edge(source, target, weight=weight)
                        nodes.update([source, target])

                    # バッチごとにメモリ解放を促進
                    del batch
                    gc.collect()

                # ノード数が多い場合、適切なアルゴリズムを選択
                if len(nodes) > 5000:
                    # 大規模ネットワーク用のアルゴリズム選択
                    centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
                else:
                    # 通常の中心性計算
                    centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000, tol=1e-06)

                # 結果格納用の辞書
                kti_values = {}

                if adoption_time_data is not None:
                    # 採用データのキャッシュ計算 - 複数回使用する情報を事前計算
                    company_data_groups = {}
                    max_count = 0

                    for company in G.nodes():
                        company_data = adoption_time_data[adoption_time_data['company'] == company]
                        company_data_groups[company] = company_data
                        count = len(company_data)
                        max_count = max(max_count, count)

                # 各企業のKTI計算
                for company in G.nodes():
                    # 中心性スコア (0-1スケール)
                    centrality_score = centrality.get(company, 0)

                    # 採用スピードと施策数のスコアを初期化
                    adoption_speed_score = 0.5  # デフォルト値
                    initiative_count_score = 0.5  # デフォルト値

                    # 施策採用データが提供されている場合
                    if adoption_time_data is not None:
                        company_data = company_data_groups.get(company, pd.DataFrame())

                        if not company_data.empty:
                            # 施策数のスコア計算
                            initiative_count = len(company_data)
                            initiative_count_score = min(initiative_count / max_count, 1.0) if max_count > 0 else 0.5

                            # 採用スピードのスコア計算
                            if 'adoption_time' in company_data.columns:
                                # 各施策ごとに採用タイミングの相対ランクを計算
                                speed_scores = self._calculate_speed_scores(company_data, adoption_time_data)
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
            # エラーコンテキストを保存してリソースを解放
            self._cleanup_on_error()
            raise

    def _calculate_speed_scores(self, company_data: pd.DataFrame, adoption_time_data: pd.DataFrame) -> List[float]:
        """
        採用スピードスコアを計算する内部メソッド

        Parameters
        ----------
        company_data : pd.DataFrame
            特定企業の採用データ
        adoption_time_data : pd.DataFrame
            全企業の採用データ

        Returns
        -------
        List[float]
            各施策の採用スピードスコアのリスト
        """
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

        return speed_scores

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
            self._cleanup_on_error()
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
            # データフレームをコンテキストマネージャで管理
            with self._managed_dataframe(practice_adoption_data) as managed_data:
                # 結果格納用の辞書
                sharing_scores = {}

                # 前処理：各プラクティスの採用時間範囲を事前計算
                practice_time_ranges = {}
                for practice in managed_data['practice'].unique():
                    practice_data = managed_data[managed_data['practice'] == practice]
                    if 'adoption_time' in practice_data.columns and not practice_data.empty:
                        min_time = practice_data['adoption_time'].min()
                        max_time = practice_data['adoption_time'].max()
                        time_range = max_time - min_time
                        practice_time_ranges[practice] = (min_time, time_range)

                # 各企業のスコア計算
                for company in company_kti.keys():
                    company_data = managed_data[managed_data['company'] == company]

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

                        # 早期採用ほど高スコア - 事前計算した範囲を使用
                        adoption_speed_score = 0.5  # デフォルト値
                        if practice in practice_time_ranges and 'adoption_time' in practice_rows.columns:
                            min_time, time_range = practice_time_ranges[practice]
                            if time_range > 0:
                                adoption_time = practice_rows['adoption_time'].iloc[0]
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
            self._cleanup_on_error()
            raise

    def identify_knowledge_hubs(self,
                              company_kti: Dict[str, float],
                              network_data: pd.DataFrame,
                              top_n: int = 5,
                              max_nodes: int = 10000) -> List[Dict]:
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
        max_nodes : int, optional
            処理する最大ノード数 (デフォルト: 10000)

        Returns
        -------
        List[Dict]
            ナレッジハブ企業の情報
        """
        try:
            # グラフオブジェクトをコンテキストマネージャで管理
            with self._managed_graph("hub_identification") as G:
                # ネットワークデータからエッジを追加
                for _, row in network_data.iterrows():
                    G.add_edge(row['source'], row['target'], weight=row.get('weight', 1.0))

                # 大規模ネットワークの場合はサンプリング
                if len(G.nodes()) > max_nodes:
                    self.logger.info(f"Sampling network from {len(G.nodes())} to {max_nodes} nodes")
                    # KTI値が高いノードを優先的に保持
                    important_nodes = sorted(
                        [(node, company_kti.get(node, 0)) for node in G.nodes()],
                        key=lambda x: x[1], reverse=True
                    )[:max_nodes]
                    important_node_ids = [node for node, _ in important_nodes]
                    G = G.subgraph(important_node_ids).copy()

                # 各種中心性指標の計算 - メモリ効率を考慮
                try:
                    degree_centrality = nx.degree_centrality(G)
                    betweenness_centrality = nx.betweenness_centrality(G, weight='weight', k=min(100, len(G.nodes)))
                    closeness_centrality = nx.closeness_centrality(G, distance='weight')
                except Exception as e:
                    self.logger.warning(f"Error in centrality calculation: {str(e)}. Using fallback method.")
                    # フォールバック: より単純な中心性計算
                    degree_centrality = {node: G.degree(node) / max(1, len(G) - 1) for node in G.nodes()}
                    betweenness_centrality = {node: 0 for node in G.nodes()}
                    closeness_centrality = {node: 0 for node in G.nodes()}

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
            self._cleanup_on_error()
            raise

    def release_resources(self) -> None:
        """
        メモリリソースを解放する
        """
        try:
            # 一時グラフの削除
            self._temp_graphs.clear()

            # キャッシュの削除
            self._data_cache.clear()

            # 明示的なガベージコレクション
            gc.collect()
            self.logger.info("Resources released successfully")
        except Exception as e:
            self.logger.error(f"Error releasing resources: {str(e)}")

    def _cleanup_on_error(self) -> None:
        """
        エラー発生時のクリーンアップ処理
        """
        try:
            self.release_resources()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """
        デストラクタ - リソース解放を保証
        """
        self.release_resources()

    def __enter__(self):
        """
        コンテキストマネージャのエントリーポイント
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        コンテキストマネージャの終了処理
        """
        self.release_resources()
        return False  # 例外を伝播させる