from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import logging
import warnings
import gc
import weakref
from contextlib import contextmanager
from .base import BaseAnalyzer, AnalysisError

# pgmpy関連のインポート
from pgmpy.estimators import PC, HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# DoWhyのインポート
import dowhy
from dowhy import CausalModel

class PortfolioNetworkAnalyzer(BaseAnalyzer):
    """
    ポートフォリオ企業間のネットワーク関係を分析するクラス
    NetworkXとpgmpyを使用した因果ネットワーク分析機能を提供
    """

    def __init__(self, firestore_client=None):
        """
        コンストラクタ

        Args:
            firestore_client: Firestoreクライアント
        """
        super().__init__(analysis_type="portfolio_network", firestore_client=firestore_client)
        self.logger = logging.getLogger(__name__)
        self.causal_graph = None
        self.bayes_model = None
        self._temp_graphs = weakref.WeakValueDictionary()  # 一時グラフを弱参照で管理
        self._plot_buffers = []  # プロットバッファの追跡

    def __del__(self):
        """デストラクタによるリソース解放"""
        self.release_resources()

    @contextmanager
    def _managed_graph(self, graph_id=None):
        """グラフオブジェクトを管理するコンテキストマネージャー

        Args:
            graph_id: グラフの識別子（省略時は自動生成）

        Yields:
            nx.Graph: 管理されたグラフオブジェクト
        """
        if graph_id is None:
            graph_id = f"graph_{datetime.now().timestamp()}"

        graph = nx.Graph()
        self._temp_graphs[graph_id] = graph

        try:
            yield graph
        finally:
            # コンテキスト終了時に弱参照から削除（必要に応じて）
            if graph_id in self._temp_graphs and len(self._temp_graphs) > 10:
                del self._temp_graphs[graph_id]

    def release_resources(self):
        """全てのリソースを明示的に解放する"""
        # グラフオブジェクトの解放
        self._temp_graphs.clear()

        # ベイジアンネットワークモデルの解放
        self.bayes_model = None
        self.causal_graph = None

        # プロットバッファのクリーンアップ
        for buf in self._plot_buffers:
            try:
                if hasattr(buf, 'close'):
                    buf.close()
            except:
                pass
        self._plot_buffers.clear()

        # 明示的にガベージコレクションを実行
        gc.collect()

    async def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        ポートフォリオネットワーク分析を実行する

        Args:
            data: 分析対象のデータフレーム
            **kwargs: 追加パラメータ
                - min_edge_weight: エッジの最小ウェイト (デフォルト: 0.3)
                - include_causal: 因果分析を含めるか (デフォルト: False)
                - causal_method: 因果構造学習手法 ('pc', 'hill_climb')
                - portfolio_id: ポートフォリオID (結果保存用)
                - max_nodes: 分析対象の最大ノード数 (デフォルト: 1000)
                - batch_processing: 大規模グラフをバッチ処理するか (デフォルト: False)

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            # データ検証
            self._validate_data(data)

            # パラメータ取得
            min_edge_weight = kwargs.get('min_edge_weight', 0.3)
            include_causal = kwargs.get('include_causal', False)
            max_nodes = kwargs.get('max_nodes', 1000)
            batch_processing = kwargs.get('batch_processing', False)

            # データサイズが大きい場合はサンプリング
            if len(data) > max_nodes:
                self.logger.warning(f"データサイズが大きいため{max_nodes}ノードにサンプリングします")
                data = data.sample(max_nodes, random_state=42)

            # グラフ構築をコンテキストマネージャーで管理
            with self._managed_graph('main_analysis') as G:
                # 基本ネットワークグラフの構築
                self._build_network_graph(G, data, min_edge_weight, batch_processing)

                # 基本ネットワークメトリクスの計算
                network_metrics = self._calculate_network_metrics(G)

                # エコシステム係数の計算
                ecosystem_coefficient = self._calculate_ecosystem_coefficient(G, data)

                # 知識移転指数の計算
                knowledge_transfer_index = self._calculate_knowledge_transfer_index(G, data)

                # コミュニティ検出
                communities = self._detect_communities(G)

                # 中心ノードの特定
                central_nodes = self._identify_central_nodes(G)

                # ネットワーク可視化
                network_plot = self._generate_network_plot(G, data, communities)

                # 基本結果の構築
                results = {
                    "network_metrics": network_metrics,
                    "ecosystem_coefficient": ecosystem_coefficient,
                    "knowledge_transfer_index": knowledge_transfer_index,
                    "communities": communities,
                    "central_nodes": central_nodes,
                    "network_plot": network_plot,
                    "timestamp": datetime.now().isoformat()
                }

            # 因果ネットワーク分析（オプション）
            if include_causal:
                causal_method = kwargs.get('causal_method', 'pc')
                causal_results = self._analyze_causal_structure(data, method=causal_method)
                results["causal_analysis"] = causal_results

            # 結果の保存
            if self.firestore_client and kwargs.get('portfolio_id'):
                await self._save_results(results, kwargs.get('portfolio_id'))

            # 明示的なリソース解放
            gc.collect()

            return results

        except Exception as e:
            self.logger.error(f"ネットワーク分析中にエラーが発生しました: {str(e)}")
            self.release_resources()  # エラー時にリソースを解放
            raise AnalysisError(f"ネットワーク分析エラー: {str(e)}")

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        データの妥当性を検証する

        Args:
            data: 検証対象のデータフレーム

        Raises:
            AnalysisError: データが無効な場合
        """
        if data is None or data.empty:
            raise AnalysisError("データが空です")

        required_columns = ['company_id', 'company_name']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise AnalysisError(f"必須カラムがありません: {', '.join(missing_columns)}")

    def _build_network_graph(self, G: nx.Graph, data: pd.DataFrame, min_edge_weight: float, batch_processing: bool = False) -> None:
        """
        ネットワークグラフを構築する

        Args:
            G: 構築対象のネットワークグラフ
            data: データフレーム
            min_edge_weight: エッジの最小ウェイト
            batch_processing: バッチ処理を使用するか

        Note:
            グラフオブジェクトを返さず、引数のグラフに直接構築
        """
        # ノードの追加
        for _, row in data.iterrows():
            G.add_node(
                row['company_id'],
                name=row['company_name'],
                industry=row.get('industry', 'Unknown'),
                size=row.get('company_size', 10),
                funding=row.get('funding_amount', 0),
                growth=row.get('growth_rate', 0)
            )

        # エッジの追加（大規模データの場合はバッチ処理）
        companies = data['company_id'].unique()

        if batch_processing and len(companies) > 100:
            # バッチサイズの決定
            batch_size = min(50, max(10, len(companies) // 20))

            for i in range(0, len(companies), batch_size):
                batch_companies = companies[i:i+batch_size]
                self._add_edges_batch(G, data, batch_companies, companies, min_edge_weight)
                # バッチ処理後にガベージコレクション
                if i % (batch_size * 5) == 0:
                    gc.collect()
        else:
            # 通常の処理
            for i, company1 in enumerate(companies):
                for company2 in companies[i+1:]:
                    company1_data = data[data['company_id'] == company1].iloc[0]
                    company2_data = data[data['company_id'] == company2].iloc[0]

                    synergy = self._calculate_synergy(company1_data, company2_data)

                    if synergy >= min_edge_weight:
                        G.add_edge(company1, company2, weight=synergy)

    def _add_edges_batch(self, G: nx.Graph, data: pd.DataFrame, batch_companies: np.ndarray,
                         all_companies: np.ndarray, min_edge_weight: float) -> None:
        """
        バッチ処理でエッジを追加する

        Args:
            G: ネットワークグラフ
            data: データフレーム
            batch_companies: 処理対象の企業バッチ
            all_companies: 全企業リスト
            min_edge_weight: エッジの最小ウェイト
        """
        # データをディクショナリに変換してルックアップを高速化
        company_data = {}
        for company_id in batch_companies:
            company_data[company_id] = data[data['company_id'] == company_id].iloc[0]

        for company1 in batch_companies:
            company1_data = company_data[company1]

            # company1より後の企業に対してのみ処理
            start_idx = np.where(all_companies == company1)[0][0] + 1
            for company2 in all_companies[start_idx:]:
                # company2のデータがキャッシュになければ取得
                if company2 not in company_data:
                    try:
                        company_data[company2] = data[data['company_id'] == company2].iloc[0]
                    except IndexError:
                        continue

                company2_data = company_data[company2]
                synergy = self._calculate_synergy(company1_data, company2_data)

                if synergy >= min_edge_weight:
                    G.add_edge(company1, company2, weight=synergy)

    def _calculate_synergy(self, company1: pd.Series, company2: pd.Series) -> float:
        """
        2社間のシナジースコアを計算する

        Args:
            company1: 1社目のデータ
            company2: 2社目のデータ

        Returns:
            float: シナジースコア (0-1)
        """
        # シナジーの計算ロジック
        # 例: 業界の近さ、成長率の相関などを考慮
        industry_synergy = 1.0 if company1.get('industry') == company2.get('industry') else 0.5

        # 成長率の相関
        growth_diff = abs(company1.get('growth_rate', 0) - company2.get('growth_rate', 0))
        growth_synergy = max(0, 1 - growth_diff / 100) if growth_diff < 100 else 0

        # 規模の補完性
        size_diff = abs(company1.get('company_size', 0) - company2.get('company_size', 0))
        size_synergy = min(1, size_diff / 100) if size_diff < 100 else 1

        # 総合スコア
        synergy_score = (industry_synergy + growth_synergy + size_synergy) / 3

        return min(1, max(0, synergy_score))

    def _calculate_network_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """
        ネットワークの基本メトリクスを計算する

        Args:
            G: ネットワークグラフ

        Returns:
            Dict[str, float]: ネットワークメトリクス
        """
        metrics = {}

        # ノード数とエッジ数
        metrics['node_count'] = G.number_of_nodes()
        metrics['edge_count'] = G.number_of_edges()

        # 平均次数
        degrees = [d for _, d in G.degree()]
        metrics['avg_degree'] = np.mean(degrees) if degrees else 0

        # 密度
        metrics['density'] = nx.density(G)

        # クラスタリング係数
        try:
            metrics['clustering_coefficient'] = nx.average_clustering(G)
        except ZeroDivisionError:
            metrics['clustering_coefficient'] = 0

        # 平均パス長
        if nx.is_connected(G):
            try:
                metrics['avg_path_length'] = nx.average_shortest_path_length(G)
            except (nx.NetworkXError, ZeroDivisionError):
                metrics['avg_path_length'] = 0
        else:
            # 非連結グラフの場合は最大連結成分で計算
            try:
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                metrics['avg_path_length'] = nx.average_shortest_path_length(subgraph)
                metrics['largest_component_size'] = len(largest_cc)
            except (ValueError, nx.NetworkXError, ZeroDivisionError):
                metrics['avg_path_length'] = 0
                metrics['largest_component_size'] = 0

        return metrics

    def _calculate_ecosystem_coefficient(self, G: nx.Graph, data: pd.DataFrame) -> float:
        """
        エコシステム係数を計算する

        Args:
            G: ネットワークグラフ
            data: データフレーム

        Returns:
            float: エコシステム係数
        """
        if G.number_of_nodes() < 2:
            return 0.0

        try:
            # コミュニティ構造の強さ
            communities = list(nx.community.greedy_modularity_communities(G))
            modularity = nx.community.modularity(G, communities)
        except (ZeroDivisionError, ValueError):
            modularity = 0.0

        # 連結性
        if nx.is_connected(G):
            connectivity = 1.0
        else:
            components = list(nx.connected_components(G))
            if components:
                connectivity = len(max(components, key=len)) / G.number_of_nodes()
            else:
                connectivity = 0.0

        # ハブノードの存在
        try:
            centrality = nx.betweenness_centrality(G)
            max_centrality = max(centrality.values()) if centrality else 0
        except (ZeroDivisionError, ValueError):
            max_centrality = 0.0

        # エコシステム係数の計算
        ecosystem_coef = (modularity + connectivity + max_centrality) / 3

        return min(1, max(0, ecosystem_coef))

    def _calculate_knowledge_transfer_index(self, G: nx.Graph, data: pd.DataFrame) -> float:
        """
        知識移転指数を計算する

        Args:
            G: ネットワークグラフ
            data: データフレーム

        Returns:
            float: 知識移転指数
        """
        if G.number_of_nodes() < 2:
            return 0.0

        try:
            # 中心性と次数の積
            centrality = nx.eigenvector_centrality_numpy(G)
            degree_dict = dict(G.degree())

            knowledge_flow = 0
            for node, cent in centrality.items():
                knowledge_flow += cent * degree_dict[node]

            # グラフの密度を考慮
            density_factor = nx.density(G)

            # クラスタリング係数を考慮
            clustering = nx.average_clustering(G)

            # 知識移転指数の計算
            kti = (knowledge_flow / G.number_of_nodes()) * density_factor * (1 + clustering)
        except (ZeroDivisionError, ValueError, nx.NetworkXError):
            return 0.0

        # 正規化
        max_possible = G.number_of_nodes() - 1  # 最大可能な次数
        normalized_kti = kti / max_possible if max_possible > 0 else 0

        return min(1, max(0, normalized_kti))

    def _detect_communities(self, G: nx.Graph) -> List[Dict[str, Any]]:
        """
        コミュニティを検出する

        Args:
            G: ネットワークグラフ

        Returns:
            List[Dict[str, Any]]: 検出されたコミュニティ
        """
        result = []

        # グラフが小さい場合は空リストを返す
        if G.number_of_nodes() < 3:
            return result

        try:
            # Louvainアルゴリズムでコミュニティ検出
            communities = list(nx.community.greedy_modularity_communities(G))
        except (ZeroDivisionError, ValueError, nx.NetworkXError):
            return result

        for i, community in enumerate(communities):
            # コミュニティのメンバー
            members = list(community)

            # コミュニティ内の企業名を取得
            member_names = []
            for node in members:
                name = G.nodes[node].get('name', str(node))
                member_names.append(name)

            # コミュニティのサイズ
            size = len(members)

            # コミュニティの密度
            subgraph = G.subgraph(members)
            density = nx.density(subgraph)

            # コミュニティの中心企業
            if members:
                try:
                    centrality = nx.betweenness_centrality(subgraph)
                    central_node = max(centrality, key=centrality.get) if centrality else members[0]
                    central_name = G.nodes[central_node].get('name', str(central_node))
                except (ZeroDivisionError, ValueError):
                    central_name = G.nodes[members[0]].get('name', str(members[0])) if members else ""
            else:
                central_name = ""

            result.append({
                "id": i,
                "size": size,
                "density": density,
                "central_company": central_name,
                "members": member_names
            })

        return result

    def _identify_central_nodes(self, G: nx.Graph) -> List[Dict[str, Any]]:
        """
        中心的なノードを特定する

        Args:
            G: ネットワークグラフ

        Returns:
            List[Dict[str, Any]]: 中心ノード情報
        """
        if G.number_of_nodes() < 2:
            return []

        try:
            # 各種中心性指標の計算
            betweenness = nx.betweenness_centrality(G)
            eigenvector = nx.eigenvector_centrality_numpy(G)
            closeness = nx.closeness_centrality(G)
            degree = dict(G.degree())
        except (ZeroDivisionError, ValueError, nx.NetworkXError):
            return []

        # 中心性スコアの合計
        combined_centrality = {}
        for node in G.nodes():
            combined_centrality[node] = (
                betweenness.get(node, 0) +
                eigenvector.get(node, 0) +
                closeness.get(node, 0) +
                degree.get(node, 0) / max(max(degree.values()), 1)
            ) / 4

        # スコア上位ノードを取得
        top_count = min(10, G.number_of_nodes())
        top_nodes = sorted(combined_centrality.keys(),
                          key=lambda x: combined_centrality[x],
                          reverse=True)[:top_count]

        result = []
        for node in top_nodes:
            name = G.nodes[node].get('name', str(node))
            industry = G.nodes[node].get('industry', "Unknown")

            result.append({
                "id": node,
                "name": name,
                "industry": industry,
                "betweenness": betweenness.get(node, 0),
                "eigenvector": eigenvector.get(node, 0),
                "closeness": closeness.get(node, 0),
                "degree": degree.get(node, 0),
                "combined_score": combined_centrality[node]
            })

        return result

    def _generate_network_plot(
        self,
        G: nx.Graph,
        data: pd.DataFrame,
        communities: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        ネットワークのプロット画像を生成する

        Args:
            G: ネットワークグラフ
            data: データフレーム
            communities: コミュニティ情報

        Returns:
            str: Base64エンコードされた画像
        """
        if G.number_of_nodes() < 2:
            # 描画可能なグラフが存在しない場合は空の画像を返す
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "描画可能なネットワークがありません",
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            return img_str

        # 大規模グラフの場合はサンプリング
        if G.number_of_nodes() > 100:
            self.logger.info(f"大規模グラフのため描画用に縮小します: {G.number_of_nodes()} → 100")
            # 中心性の高いノードを優先的に選択
            try:
                degree_centrality = nx.degree_centrality(G)
                top_nodes = sorted(degree_centrality.keys(),
                                  key=lambda x: degree_centrality[x],
                                  reverse=True)[:100]
                G = G.subgraph(top_nodes).copy()
            except:
                # エラー時はランダムサンプリング
                import random
                nodes = list(G.nodes())
                sample_nodes = random.sample(nodes, min(100, len(nodes)))
                G = G.subgraph(sample_nodes).copy()

        # メモリ効率のためにプロットを管理
        plt.figure(figsize=(12, 10))

        # ノードの位置をspring layoutで計算
        try:
            pos = nx.spring_layout(G, seed=42)
        except:
            # フォールバックとしてcircular layoutを使用
            pos = nx.circular_layout(G)

        # ノードの色を決定（コミュニティごと）
        if communities:
            colors = []
            for node in G.nodes():
                for i, comm in enumerate(communities):
                    if any(node_id == node for node_id in [G.nodes[n].get('id', n) for n in comm.get('members', [])]):
                        colors.append(i)
                        break
                else:
                    colors.append(len(communities))
        else:
            # コミュニティがない場合はデフォルト色
            colors = 'skyblue'

        # ノードのサイズを企業規模に応じて調整
        node_sizes = []
        for node in G.nodes():
            size = G.nodes[node].get('size', 10)
            funding = G.nodes[node].get('funding', 0)
            # サイズとファンディングを考慮
            node_size = 100 + (size * 10) + (funding / 1000000)
            node_sizes.append(node_size)

        # エッジの幅を重みに応じて調整
        edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]

        # グラフ描画
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=True,
            node_color=colors,
            node_size=node_sizes,
            font_size=8,
            width=edge_widths,
            edge_color='gray',
            alpha=0.8,
            cmap=plt.cm.tab20
        )

        plt.title("ポートフォリオネットワークマップ")
        plt.axis('off')

        # 画像をBase64エンコード
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')

        # バッファをリストに追加してトラッキング
        self._plot_buffers.append(buf)

        # リソース解放
        plt.close()

        # 定期的にバッファをクリーンアップ
        if len(self._plot_buffers) > 10:
            for old_buf in self._plot_buffers[:-10]:
                try:
                    old_buf.close()
                except:
                    pass
            self._plot_buffers = self._plot_buffers[-10:]

        return img_str

    def _analyze_causal_structure(self, data: pd.DataFrame, method: str = 'pc') -> Dict[str, Any]:
        """
        因果構造分析を実行する

        Args:
            data: データフレーム
            method: 構造学習手法 ('pc', 'hill_climb')

        Returns:
            Dict[str, Any]: 因果構造分析結果
        """
        # 数値列の抽出
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

        # カラム数が多すぎる場合は削減
        if len(numeric_cols) > 15:
            self.logger.warning(f"変数が多すぎるため上位15変数に制限します: {len(numeric_cols)} → 15")
            # 分散の高い上位カラムを選択
            variances = data[numeric_cols].var().sort_values(ascending=False)
            numeric_cols = variances.index[:15]

        numeric_data = data[numeric_cols].copy()

        # 欠損値の処理
        numeric_data = numeric_data.fillna(numeric_data.mean())

        try:
            # 構造学習の実行
            if method == 'pc':
                # PCアルゴリズムによる構造学習
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pc = PC(data=numeric_data)
                    causal_graph = pc.estimate(significance_level=0.05)
            elif method == 'hill_climb':
                # Hill-Climbing探索による構造学習
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hc = HillClimbSearch(data=numeric_data)
                    score = BicScore(data=numeric_data)
                    causal_graph = hc.estimate(scoring_method=score)
            else:
                raise AnalysisError(f"未対応の構造学習手法です: {method}")

            # BayesianNetworkモデルの作成
            try:
                self.bayes_model = BayesianNetwork(causal_graph.edges())
                self.bayes_model.fit(numeric_data)
                self.causal_graph = causal_graph
            except Exception as e:
                self.logger.warning(f"ベイジアンネットワークモデルの作成に失敗しました: {str(e)}")
                self.bayes_model = None

            # 中心性の計算
            with self._managed_graph('causal_graph') as G:
                G.add_nodes_from(causal_graph.nodes())
                G.add_edges_from(causal_graph.edges())

                try:
                    centrality = nx.betweenness_centrality(G)
                    top_causes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                except:
                    top_causes = []

                # 因果グラフの可視化
                causal_plot = self._visualize_causal_graph(G)

            return {
                "method": method,
                "nodes": list(causal_graph.nodes()),
                "edges": [{"from": e[0], "to": e[1]} for e in causal_graph.edges()],
                "top_causes": [{"variable": v, "centrality": c} for v, c in top_causes],
                "model_valid": self.bayes_model is not None,
                "causal_plot": causal_plot
            }
        except Exception as e:
            self.logger.error(f"因果構造分析中にエラーが発生しました: {str(e)}")
            return {
                "method": method,
                "error": str(e),
                "model_valid": False
            }
        finally:
            # 明示的なメモリ解放
            gc.collect()

    def _visualize_causal_graph(self, G: nx.DiGraph) -> str:
        """
        因果グラフを可視化する

        Args:
            G: 有向グラフ

        Returns:
            str: Base64エンコードされた画像
        """
        plt.figure(figsize=(10, 8))

        try:
            pos = nx.spring_layout(G, seed=42)
        except:
            pos = nx.circular_layout(G)

        try:
            # ノードの中心性に基づいてサイズを調整
            centrality = nx.betweenness_centrality(G)
            node_sizes = [centrality[n] * 3000 + 300 for n in G.nodes()]
        except:
            # 中心性計算に失敗した場合は均一サイズ
            node_sizes = 500

        # グラフ描画
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=True,
            node_color='lightblue',
            node_size=node_sizes,
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            arrows=True,
            arrowsize=15,
            arrowstyle='->'
        )

        plt.title("因果構造グラフ")
        plt.axis('off')

        # 画像をBase64エンコード
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')

        # バッファをリストに追加してトラッキング
        self._plot_buffers.append(buf)

        # リソース解放
        plt.close()

        return img_str

    async def estimate_causal_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        因果効果を推定する

        Args:
            data: データフレーム
            treatment: 介入変数
            outcome: 結果変数
            **kwargs: 追加パラメータ
                - confounders: 交絡変数のリスト
                - method: 効果推定手法

        Returns:
            Dict[str, Any]: 因果効果推定結果
        """
        try:
            # DoWhyでの因果効果推定
            confounders = kwargs.get('confounders', [])
            method = kwargs.get('method', 'backdoor')

            # 因果グラフの構築
            if self.causal_graph is None:
                # グラフがなければ新たに学習
                causal_results = self._analyze_causal_structure(data, method='pc')

            # DoWhyモデルの初期化
            if confounders:
                # 交絡変数が指定されている場合
                model = CausalModel(
                    data=data,
                    treatment=treatment,
                    outcome=outcome,
                    common_causes=confounders
                )
            else:
                # グラフから自動検出
                model = CausalModel(
                    data=data,
                    treatment=treatment,
                    outcome=outcome,
                    graph=self.causal_graph
                )

            # 識別
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

            # 推定
            estimate = model.estimate_effect(identified_estimand, method_name=method)

            # 結果の整形
            result = {
                "treatment": treatment,
                "outcome": outcome,
                "effect": float(estimate.value),
                "confidence_interval": [float(ci) for ci in estimate.get_confidence_intervals()]
                                      if hasattr(estimate, 'get_confidence_intervals') else None,
                "p_value": float(estimate.p_value) if hasattr(estimate, 'p_value') else None,
                "method": method,
                "estimand": str(identified_estimand)
            }

            # メモリ解放
            del model
            gc.collect()

            return result

        except Exception as e:
            self.logger.error(f"因果効果推定中にエラーが発生しました: {str(e)}")
            return {
                "treatment": treatment,
                "outcome": outcome,
                "error": str(e),
                "method": method
            }
        finally:
            # 念のためリソース解放
            gc.collect()