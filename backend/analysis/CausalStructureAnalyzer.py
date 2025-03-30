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
from sklearn.preprocessing import StandardScaler
from .base import BaseAnalyzer, AnalysisError

# pgmpy関連のインポート
from pgmpy.estimators import PC, HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

# DoWhyのインポート（因果効果の推定）
import dowhy
from dowhy import CausalModel

class CausalStructureAnalyzer(BaseAnalyzer):
    """
    pgmpyとDoWhyを使用した因果構造分析を行うアナライザー

    CausalNexの代替として機能し、Python 3.12との互換性を持つ。
    1. pgmpy: ベイジアンネットワーク構造学習
    2. DoWhy: 因果効果の推定
    3. networkx: グラフ可視化
    """

    def __init__(self, firestore_client=None):
        """
        コンストラクタ

        Args:
            firestore_client: Firestoreクライアントのインスタンス
        """
        super().__init__(analysis_type="causal_structure", firestore_client=firestore_client)
        self.logger = logging.getLogger(__name__)

    async def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        因果構造分析を実行する

        Args:
            data (pd.DataFrame): 分析対象のデータ
            **kwargs: 追加のパラメータ
                - method (str): 構造学習手法 ('pc', 'hill_climb', 'expert')
                - significance_level (float): PC算出の有意水準（デフォルト0.05）
                - max_cond_vars (int): 条件付き独立性テストの最大変数数
                - prior_edges (List[Tuple]): 事前知識によるエッジリスト（expertモード）
                - forbidden_edges (List[Tuple]): 禁止エッジリスト

        Returns:
            Dict[str, Any]: 分析結果
        """
        # パラメータの取得
        method = kwargs.get('method', 'pc')

        # データの前処理
        processed_data = self._preprocess_data(data)

        # 構造学習の実行
        if method == 'pc':
            model, graph = self._learn_structure_pc(processed_data, **kwargs)
        elif method == 'hill_climb':
            model, graph = self._learn_structure_hill_climb(processed_data, **kwargs)
        elif method == 'expert':
            model, graph = self._learn_structure_expert(processed_data, **kwargs)
        else:
            raise AnalysisError(f"未対応の構造学習手法です: {method}")

        # パラメータ学習
        if model is not None:
            model = self._learn_parameters(model, processed_data)

        # 結果を整形
        results = self._format_results(model, graph, processed_data.columns)

        # 可視化
        results["visualization"] = self._visualize_graph(graph, processed_data.columns)

        # 結果の保存
        if self.firestore_client and kwargs.get('portfolio_id'):
            await self._save_results(results, kwargs.get('portfolio_id'))

        return results

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        データの前処理を行う

        Args:
            data (pd.DataFrame): 元のデータフレーム

        Returns:
            pd.DataFrame: 前処理済みデータフレーム
        """
        # 欠損値の処理
        processed_data = data.copy()
        processed_data = processed_data.dropna()

        # 特徴量の標準化
        numeric_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            processed_data[numeric_cols] = scaler.fit_transform(processed_data[numeric_cols])

        return processed_data

    def _learn_structure_pc(self, data: pd.DataFrame, **kwargs) -> Tuple[Optional[BayesianNetwork], nx.DiGraph]:
        """
        PCアルゴリズムを使用して因果構造を学習する

        Args:
            data (pd.DataFrame): 前処理済みデータ
            **kwargs: 追加のパラメータ
                - significance_level (float): 有意水準
                - max_cond_vars (int): 条件付き独立性テストの最大変数数

        Returns:
            Tuple[Optional[BayesianNetwork], nx.DiGraph]: 学習済みモデルとグラフ
        """
        significance_level = kwargs.get('significance_level', 0.05)
        max_cond_vars = kwargs.get('max_cond_vars', 3)

        # PCアルゴリズムの初期化と実行
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pc = PC(data=data)
            graph = pc.estimate(significance_level=significance_level, max_cond_vars=max_cond_vars)

        # BayesianNetworkモデルの作成
        try:
            model = BayesianNetwork(graph.edges())
            return model, graph
        except Exception as e:
            self.logger.error(f"BayesianNetworkモデルの作成中にエラーが発生しました: {str(e)}")
            return None, graph

    def _learn_structure_hill_climb(self, data: pd.DataFrame, **kwargs) -> Tuple[Optional[BayesianNetwork], nx.DiGraph]:
        """
        Hill-Climbアルゴリズムを使用して因果構造を学習する

        Args:
            data (pd.DataFrame): 前処理済みデータ
            **kwargs: 追加のパラメータ
                - max_indegree (int): 各ノードの最大入次数

        Returns:
            Tuple[Optional[BayesianNetwork], nx.DiGraph]: 学習済みモデルとグラフ
        """
        max_indegree = kwargs.get('max_indegree', 3)

        # BICスコアとHill-Climbの初期化と実行
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hc = HillClimbSearch(data=data)
            model_score = BicScore(data=data)
            best_model = hc.estimate(scoring_method=model_score, max_indegree=max_indegree)

        # グラフへの変換
        graph = nx.DiGraph()
        graph.add_nodes_from(best_model.nodes())
        graph.add_edges_from(best_model.edges())

        # BayesianNetworkモデルの作成
        try:
            model = BayesianNetwork(best_model.edges())
            return model, graph
        except Exception as e:
            self.logger.error(f"BayesianNetworkモデルの作成中にエラーが発生しました: {str(e)}")
            return None, graph

    def _learn_structure_expert(self, data: pd.DataFrame, **kwargs) -> Tuple[Optional[BayesianNetwork], nx.DiGraph]:
        """
        専門家の知識に基づいて因果構造を定義する

        Args:
            data (pd.DataFrame): 前処理済みデータ
            **kwargs: 追加のパラメータ
                - prior_edges (List[Tuple]): 事前知識によるエッジリスト

        Returns:
            Tuple[Optional[BayesianNetwork], nx.DiGraph]: 学習済みモデルとグラフ
        """
        prior_edges = kwargs.get('prior_edges', [])

        if not prior_edges:
            raise AnalysisError("事前エッジリストが指定されていません。expertモードにはprior_edgesが必要です。")

        # グラフの構築
        graph = nx.DiGraph()
        graph.add_nodes_from(data.columns)
        graph.add_edges_from(prior_edges)

        # BayesianNetworkモデルの作成
        try:
            model = BayesianNetwork(graph.edges())
            return model, graph
        except Exception as e:
            self.logger.error(f"BayesianNetworkモデルの作成中にエラーが発生しました: {str(e)}")
            return None, graph

    def _learn_parameters(self, model: BayesianNetwork, data: pd.DataFrame) -> BayesianNetwork:
        """
        ベイジアンネットワークのパラメータを学習する

        Args:
            model (BayesianNetwork): 構造が学習されたモデル
            data (pd.DataFrame): トレーニングデータ

        Returns:
            BayesianNetwork: パラメータが学習されたモデル
        """
        try:
            # 変数の状態スペースを確認
            discrete_data = data.copy()

            # 連続変数を離散化（必要な場合）
            for col in data.columns:
                if data[col].dtype in ['float64', 'float32']:
                    discrete_data[col] = pd.qcut(data[col], q=4, labels=False, duplicates='drop')

            # モデルに構造を適合
            model.fit(discrete_data, estimator=MaximumLikelihoodEstimator)

            return model
        except Exception as e:
            self.logger.error(f"パラメータ学習中にエラーが発生しました: {str(e)}")
            # エラーが発生しても元のモデルを返す
            return model

    def _format_results(self, model: Optional[BayesianNetwork], graph: nx.DiGraph, columns: pd.Index) -> Dict[str, Any]:
        """
        分析結果を整形する

        Args:
            model (Optional[BayesianNetwork]): 学習済みモデル
            graph (nx.DiGraph): 因果グラフ
            columns (pd.Index): データの列名

        Returns:
            Dict[str, Any]: 整形された結果
        """
        nodes = list(columns)
        edges = list(graph.edges())

        # グラフの基本的な指標を計算
        centrality = nx.betweenness_centrality(graph)

        # 結果の整形
        results = {
            "nodes": nodes,
            "edges": [{"from": e[0], "to": e[1]} for e in edges],
            "centrality": {node: float(score) for node, score in centrality.items()},
            "timestamp": datetime.now().isoformat(),
        }

        # モデルからCPTを抽出（モデルが存在する場合）
        if model is not None:
            try:
                cpd_info = []
                for node in model.nodes():
                    cpd = model.get_cpds(node)
                    if cpd:
                        cpd_info.append({
                            "node": node,
                            "parents": list(cpd.variables[1:]),
                            "values": cpd.values.tolist() if hasattr(cpd.values, 'tolist') else cpd.values,
                        })
                results["cpd_info"] = cpd_info
            except Exception as e:
                self.logger.error(f"CPD情報の抽出中にエラーが発生しました: {str(e)}")

        return results

    def _visualize_graph(self, graph: nx.DiGraph, columns: pd.Index) -> str:
        """
        因果グラフを可視化してBase64エンコードされた画像として返す

        Args:
            graph (nx.DiGraph): 因果グラフ
            columns (pd.Index): データの列名

        Returns:
            str: Base64エンコードされたプロット画像
        """
        plt.figure(figsize=(12, 8))

        # グラフのレイアウト設定
        pos = nx.spring_layout(graph, seed=42)

        # ノードの中心性に基づいてサイズを調整
        centrality = nx.betweenness_centrality(graph)
        node_size = [centrality[n] * 5000 + 300 for n in graph.nodes()]

        # グラフの描画
        nx.draw_networkx(
            graph,
            pos=pos,
            with_labels=True,
            node_color='lightblue',
            node_size=node_size,
            edge_color='gray',
            arrows=True,
            font_size=10,
            font_weight='bold'
        )

        plt.title("因果構造グラフ")
        plt.axis('off')
        plt.tight_layout()

        # 画像をBase64エンコード
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return img_str

    async def _save_results(self, results: Dict[str, Any], portfolio_id: str) -> None:
        """
        分析結果をFirestoreに保存する

        Args:
            results (Dict[str, Any]): 分析結果
            portfolio_id (str): ポートフォリオID
        """
        if not self.firestore_client:
            self.logger.warning("Firestoreクライアントが設定されていないため、結果を保存できません。")
            return

        try:
            # 分析結果のドキュメントを作成
            document_data = {
                "portfolio_id": portfolio_id,
                "analysis_type": "causal_structure",
                "result": results,
                "created_at": datetime.now().isoformat(),
            }

            # Firestoreに保存
            await self.firestore_client.set_document(
                collection="analysis_results",
                document_id=None,  # 自動生成
                data=document_data
            )

            self.logger.info(f"因果構造分析結果をFirestoreに保存しました: {portfolio_id}")
        except Exception as e:
            self.logger.error(f"因果構造分析結果の保存中にエラーが発生しました: {str(e)}")
            raise AnalysisError(f"結果の保存に失敗しました: {str(e)}")

    async def estimate_causal_effect(self, data: pd.DataFrame, treatment: str,
                                  outcome: str, **kwargs) -> Dict[str, Any]:
        """
        DoWhyを使用して因果効果を推定する

        Args:
            data (pd.DataFrame): 分析対象のデータ
            treatment (str): 介入変数
            outcome (str): 結果変数
            **kwargs: 追加のパラメータ
                - confounders (List[str]): 交絡変数のリスト
                - graph (nx.DiGraph): 事前に学習した因果グラフ（オプション）
                - method (str): 効果推定方法 ('backdoor', 'frontdoor', 'iv', etc.)

        Returns:
            Dict[str, Any]: 因果効果推定結果
        """
        # データの前処理
        processed_data = self._preprocess_data(data)

        # パラメータの取得
        confounders = kwargs.get('confounders', [])
        graph = kwargs.get('graph', None)
        method = kwargs.get('method', 'backdoor')

        # グラフが提供されていない場合、学習する
        if graph is None:
            _, graph = self._learn_structure_pc(processed_data)

        # DoWhyのCausalModelを初期化
        if confounders:
            # 事前知識がある場合のDoWhyモデル構築
            causal_graph = """
                graph [
            """
            # ノードの追加
            for col in processed_data.columns:
                causal_graph += f'        node[label="{col}"];\n'

            # エッジの追加
            for edge in graph.edges():
                causal_graph += f'        {edge[0]} -> {edge[1]};\n'

            causal_graph += "    ]"

            model = CausalModel(
                data=processed_data,
                treatment=treatment,
                outcome=outcome,
                graph=causal_graph,
                common_causes=confounders
            )
        else:
            # 事前知識がない場合のDoWhyモデル構築
            model = CausalModel(
                data=processed_data,
                treatment=treatment,
                outcome=outcome,
                graph=graph
            )

        # 識別
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # 推定
        estimate = model.estimate_effect(identified_estimand, method_name=method)

        # リフューテーションテスト
        refutation_results = {}
        try:
            refute_random = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
            refutation_results["random_common_cause"] = {
                "refutation_result": refute_random.refutation_result,
                "p_value": float(refute_random.p_value) if hasattr(refute_random, 'p_value') else None
            }
        except Exception as e:
            self.logger.warning(f"リフューテーションテスト中にエラーが発生しました: {str(e)}")

        # 結果を整形
        results = {
            "treatment": treatment,
            "outcome": outcome,
            "confounders": confounders,
            "causal_effect": float(estimate.value),
            "confidence_interval": [float(ci) for ci in estimate.get_confidence_intervals()]
                                  if hasattr(estimate, 'get_confidence_intervals') else None,
            "p_value": float(estimate.p_value) if hasattr(estimate, 'p_value') else None,
            "method": method,
            "refutation_results": refutation_results,
            "timestamp": datetime.now().isoformat(),
        }

        return results