from typing import Dict, Any, List, Optional, Union, Tuple, ContextManager
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
import contextlib
import gc
import weakref
import tempfile
import os
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

    def __init__(self, firestore_client=None, storage_mode: str = 'memory', max_nodes: int = 1000):
        """
        コンストラクタ

        Args:
            firestore_client: Firestoreクライアントのインスタンス
            storage_mode: ストレージモード ('memory', 'disk', 'hybrid')
            max_nodes: 処理する最大ノード数 (これを超える場合は自動サンプリング)
        """
        super().__init__(analysis_type="causal_structure", firestore_client=firestore_client)
        self.logger = logging.getLogger(__name__)
        self.storage_mode = storage_mode
        self.max_nodes = max_nodes
        self._temp_files = []
        self._plot_resources = weakref.WeakValueDictionary()
        self._graph_objects = weakref.WeakValueDictionary()

    def __del__(self):
        """デストラクタ - リソースの解放"""
        self.release_resources()

    def release_resources(self) -> None:
        """すべてのリソースを解放"""
        super().release_resources()
        self._clean_plot_resources()
        self._clean_temp_files()
        self._clean_graph_objects()
        gc.collect()

    def _clean_plot_resources(self) -> None:
        """プロットリソースをクリーンアップ"""
        for fig_id in list(self._plot_resources.keys()):
            plt.close(self._plot_resources[fig_id])
        self._plot_resources.clear()

    def _clean_temp_files(self) -> None:
        """一時ファイルをクリーンアップ"""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                self.logger.warning(f"一時ファイルの削除中にエラーが発生しました: {str(e)}")
        self._temp_files = []

    def _clean_graph_objects(self) -> None:
        """グラフオブジェクトをクリーンアップ"""
        self._graph_objects.clear()

    @contextlib.contextmanager
    def _plot_context(self, figsize=(12, 8)) -> ContextManager:
        """
        プロット作成用のコンテキストマネージャ

        Args:
            figsize: フィギュアサイズ

        Yields:
            tuple: (fig, ax) matplotlib図とaxesオブジェクト
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig_id = id(fig)
        self._plot_resources[fig_id] = fig
        try:
            yield fig, ax
        finally:
            if fig_id in self._plot_resources:
                plt.close(fig)
                del self._plot_resources[fig_id]

    @contextlib.contextmanager
    def _managed_graph(self, graph=None) -> ContextManager[nx.DiGraph]:
        """
        グラフオブジェクト管理用のコンテキストマネージャ

        Args:
            graph: 既存のグラフ（Noneの場合は新規作成）

        Yields:
            nx.DiGraph: 管理されるグラフオブジェクト
        """
        if graph is None:
            graph = nx.DiGraph()

        graph_id = id(graph)
        self._graph_objects[graph_id] = graph

        try:
            yield graph
        finally:
            if graph_id in self._graph_objects:
                # グラフはWeakValueDictionaryで管理されているため
                # 参照を削除するだけでガベージコレクション対象になる
                del self._graph_objects[graph_id]

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
                - max_nodes (int): 処理する最大ノード数（オプション）
                - callback (callable): 進捗コールバック関数（オプション）

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            # パラメータの取得
            method = kwargs.get('method', 'pc')
            max_nodes = kwargs.get('max_nodes', self.max_nodes)
            callback = kwargs.get('callback', None)

            # 進捗更新
            if callback:
                callback({"status": "preprocessing", "progress": 0.1})

            # データの前処理
            processed_data = self._preprocess_data(data, max_nodes)

            # 進捗更新
            if callback:
                callback({"status": "learning_structure", "progress": 0.3})

            # 構造学習の実行
            if method == 'pc':
                model, graph = self._learn_structure_pc(processed_data, **kwargs)
            elif method == 'hill_climb':
                model, graph = self._learn_structure_hill_climb(processed_data, **kwargs)
            elif method == 'expert':
                model, graph = self._learn_structure_expert(processed_data, **kwargs)
            else:
                raise AnalysisError(f"未対応の構造学習手法です: {method}")

            # 一時メモリを解放
            if processed_data is not data:
                del processed_data
                gc.collect()

            # 進捗更新
            if callback:
                callback({"status": "learning_parameters", "progress": 0.6})

            # パラメータ学習
            if model is not None:
                model = self._learn_parameters(model, data)

            # 進捗更新
            if callback:
                callback({"status": "formatting_results", "progress": 0.8})

            # グラフオブジェクトを管理
            with self._managed_graph(graph) as managed_graph:
                # 結果を整形
                results = self._format_results(model, managed_graph, data.columns)

                # 可視化
                results["visualization"] = self._visualize_graph(managed_graph, data.columns)

            # 進捗更新
            if callback:
                callback({"status": "saving_results", "progress": 0.9})

            # 結果の保存
            if self.firestore_client and kwargs.get('portfolio_id'):
                await self._save_results(results, kwargs.get('portfolio_id'))

            # 進捗更新
            if callback:
                callback({"status": "completed", "progress": 1.0})

            return results

        except Exception as e:
            self.logger.error(f"因果構造分析中にエラーが発生しました: {str(e)}")
            self.release_resources()  # エラー発生時も確実にリソースを解放
            raise AnalysisError(f"因果構造分析に失敗しました: {str(e)}")

    def _preprocess_data(self, data: pd.DataFrame, max_nodes: int = None) -> pd.DataFrame:
        """
        データの前処理を行う

        Args:
            data (pd.DataFrame): 元のデータフレーム
            max_nodes (int): 処理する最大変数数

        Returns:
            pd.DataFrame: 前処理済みデータフレーム
        """
        # 欠損値の処理
        processed_data = data.copy()
        processed_data = processed_data.dropna()

        # 変数数が多すぎる場合はサンプリング
        max_nodes = max_nodes or self.max_nodes
        if len(processed_data.columns) > max_nodes:
            self.logger.warning(f"変数数が多すぎます（{len(processed_data.columns)}）。先頭{max_nodes}変数のみを使用します。")

            # メモリ効率のため新しいDataFrameを作成せず、既存のものを更新
            columns_to_keep = list(processed_data.columns[:max_nodes])
            processed_data = processed_data[columns_to_keep]

        # 特徴量の標準化
        numeric_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            processed_data[numeric_cols] = scaler.fit_transform(processed_data[numeric_cols])

            # スケーラーをクリーンアップ
            del scaler
            gc.collect()

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
        try:
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

        except Exception as e:
            self.logger.error(f"PC構造学習中にエラーが発生しました: {str(e)}")
            # 空のグラフを返す
            return None, nx.DiGraph()

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
        try:
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

        except Exception as e:
            self.logger.error(f"Hill-Climb構造学習中にエラーが発生しました: {str(e)}")
            # 空のグラフを返す
            return None, nx.DiGraph()

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
        try:
            prior_edges = kwargs.get('prior_edges', [])

            if not prior_edges:
                raise AnalysisError("事前エッジリストが指定されていません。expertモードにはprior_edgesが必要です。")

            # グラフを管理付きで構築
            with self._managed_graph() as graph:
                graph.add_nodes_from(data.columns)
                graph.add_edges_from(prior_edges)

                # BayesianNetworkモデルの作成
                try:
                    model = BayesianNetwork(graph.edges())
                    return model, graph
                except Exception as e:
                    self.logger.error(f"BayesianNetworkモデルの作成中にエラーが発生しました: {str(e)}")
                    return None, graph

        except Exception as e:
            self.logger.error(f"専門家モデル構築中にエラーが発生しました: {str(e)}")
            # 空のグラフを返す
            return None, nx.DiGraph()

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

            # 不要なデータをクリーンアップ
            del discrete_data
            gc.collect()

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
        try:
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
                "node_count": len(nodes),
                "edge_count": len(edges)
            }

            # モデルからCPTを抽出（モデルが存在する場合）
            if model is not None:
                try:
                    cpd_info = []
                    for node in model.nodes():
                        cpd = model.get_cpds(node)
                        if cpd:
                            # メモリ効率のためcpd情報を厳選
                            cpd_info.append({
                                "node": node,
                                "parents": list(cpd.variables[1:]),
                                "values_shape": list(cpd.values.shape) if hasattr(cpd.values, 'shape') else None,
                                # 大きな配列はストレージモードによって処理を変える
                                "values": cpd.values.tolist() if hasattr(cpd.values, 'tolist') and
                                                          (self.storage_mode == 'memory' or cpd.values.size < 1000)
                                                       else "large_array"
                            })
                    results["cpd_info"] = cpd_info
                except Exception as e:
                    self.logger.error(f"CPD情報の抽出中にエラーが発生しました: {str(e)}")
                    results["cpd_error"] = str(e)

            return results

        except Exception as e:
            self.logger.error(f"結果整形中にエラーが発生しました: {str(e)}")
            # 最低限の結果を返す
            return {
                "nodes": list(columns),
                "edges": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _visualize_graph(self, graph: nx.DiGraph, columns: pd.Index) -> str:
        """
        因果グラフを可視化してBase64エンコードされた画像として返す

        Args:
            graph (nx.DiGraph): 因果グラフ
            columns (pd.Index): データの列名

        Returns:
            str: Base64エンコードされたプロット画像
        """
        try:
            # プロット作成用コンテキストマネージャ使用
            with self._plot_context(figsize=(12, 8)) as (fig, _):
                # グラフのノード数に基づいたサンプリング
                display_graph = graph
                if len(graph.nodes()) > 30:
                    # 中心性に基づいて重要なノードを選択
                    centrality = nx.betweenness_centrality(graph)
                    important_nodes = sorted(centrality, key=centrality.get, reverse=True)[:30]
                    display_graph = graph.subgraph(important_nodes)
                    self.logger.info(f"グラフが大きすぎるため、重要な30ノードのみを表示します。")

                # グラフのレイアウト設定
                try:
                    pos = nx.spring_layout(display_graph, seed=42)
                except Exception:
                    # レイアウト計算エラー時のフォールバック
                    pos = {node: (i % 5, i // 5) for i, node in enumerate(display_graph.nodes())}

                # ノードの中心性に基づいてサイズを調整
                centrality = nx.betweenness_centrality(display_graph)
                node_size = [centrality[n] * 5000 + 300 for n in display_graph.nodes()]

                # グラフの描画
                nx.draw_networkx(
                    display_graph,
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

                return img_str

        except Exception as e:
            self.logger.error(f"グラフ可視化中にエラーが発生しました: {str(e)}")
            # エラー時は空の画像を返す
            return ""

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

    @contextlib.contextmanager
    def _managed_causal_model(self, data: pd.DataFrame, treatment: str, outcome: str,
                             graph=None, confounders=None) -> ContextManager:
        """
        DoWhyのCausalModelを管理するコンテキストマネージャ

        Args:
            data: 分析データ
            treatment: 介入変数
            outcome: 結果変数
            graph: 因果グラフ
            confounders: 交絡変数のリスト

        Yields:
            CausalModel: 因果モデル
        """
        try:
            # グラフが提供されていない場合、最小限の学習を実行
            if graph is None:
                _, graph = self._learn_structure_pc(data, max_cond_vars=2)

            # グラフを管理
            with self._managed_graph(graph) as managed_graph:
                if confounders:
                    # 事前知識がある場合のDoWhyモデル構築
                    causal_graph = """
                        graph [
                    """
                    # ノードの追加
                    for col in data.columns:
                        causal_graph += f'        node[label="{col}"];\n'

                    # エッジの追加
                    for edge in managed_graph.edges():
                        causal_graph += f'        {edge[0]} -> {edge[1]};\n'

                    causal_graph += "    ]"

                    model = CausalModel(
                        data=data,
                        treatment=treatment,
                        outcome=outcome,
                        graph=causal_graph,
                        common_causes=confounders
                    )
                else:
                    # 事前知識がない場合のDoWhyモデル構築
                    model = CausalModel(
                        data=data,
                        treatment=treatment,
                        outcome=outcome,
                        graph=managed_graph
                    )

                yield model

        except Exception as e:
            self.logger.error(f"因果モデル作成中にエラーが発生しました: {str(e)}")
            raise

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
                - max_nodes (int): 処理する最大変数数
                - callback (callable): 進捗コールバック関数

        Returns:
            Dict[str, Any]: 因果効果推定結果
        """
        try:
            # データの前処理
            max_nodes = kwargs.get('max_nodes', self.max_nodes)
            callback = kwargs.get('callback', None)

            # 進捗更新
            if callback:
                callback({"status": "preprocessing", "progress": 0.1})

            processed_data = self._preprocess_data(data, max_nodes)

            # パラメータの取得
            confounders = kwargs.get('confounders', [])
            graph = kwargs.get('graph', None)
            method = kwargs.get('method', 'backdoor')

            # 進捗更新
            if callback:
                callback({"status": "building_model", "progress": 0.3})

            # DoWhyのCausalModelを初期化
            with self._managed_causal_model(processed_data, treatment, outcome, graph, confounders) as model:
                # 進捗更新
                if callback:
                    callback({"status": "identifying_effect", "progress": 0.5})

                # 識別
                identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

                # 進捗更新
                if callback:
                    callback({"status": "estimating_effect", "progress": 0.7})

                # 推定
                estimate = model.estimate_effect(identified_estimand, method_name=method)

                # 進捗更新
                if callback:
                    callback({"status": "refuting_estimates", "progress": 0.8})

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
                    refutation_results["error"] = str(e)

                # 進捗更新
                if callback:
                    callback({"status": "completed", "progress": 1.0})

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

        except Exception as e:
            self.logger.error(f"因果効果推定中にエラーが発生しました: {str(e)}")
            self.release_resources()  # エラー発生時も確実にリソースを解放
            raise AnalysisError(f"因果効果推定に失敗しました: {str(e)}")

    def estimate_memory_usage(self, node_count: int, edge_density: float = 0.2) -> Dict[str, Any]:
        """
        予想メモリ使用量を計算し、適切なパラメータを推奨する

        Args:
            node_count: ノード数（変数の数）
            edge_density: エッジ密度（0.0〜1.0）

        Returns:
            Dict[str, Any]: メモリ使用量予測
        """
        # 簡易的なメモリ使用量予測
        base_memory = 50  # MB

        # PCアルゴリズムの計算量は変数数の二乗オーダー
        pc_memory = 0.1 * (node_count ** 2)  # MB

        # グラフ保存のメモリ
        estimated_edge_count = int(node_count * (node_count - 1) * edge_density / 2)
        graph_memory = 0.01 * node_count + 0.005 * estimated_edge_count  # MB

        # 可視化用メモリ
        viz_memory = 5 + 0.05 * node_count  # MB

        total_estimated_memory = base_memory + pc_memory + graph_memory + viz_memory

        # 推奨パラメータ
        recommended_max_nodes = node_count
        if total_estimated_memory > 1000:  # 1GB以上
            recommended_max_nodes = int(node_count * (1000 / total_estimated_memory) ** 0.5)
            recommended_max_nodes = max(20, recommended_max_nodes)  # 最低20ノード保証

        recommended_max_cond_vars = min(3, node_count // 10)  # ノード数の1/10か3のうち小さい方

        return {
            'estimated_memory_mb': total_estimated_memory,
            'recommended_max_nodes': recommended_max_nodes,
            'recommended_max_cond_vars': recommended_max_cond_vars,
            'node_count': node_count,
            'estimated_edge_count': estimated_edge_count,
            'base_memory_mb': base_memory,
            'pc_algorithm_memory_mb': pc_memory,
            'graph_memory_mb': graph_memory,
            'visualization_memory_mb': viz_memory
        }