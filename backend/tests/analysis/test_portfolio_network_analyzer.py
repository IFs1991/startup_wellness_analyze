import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
import networkx as nx
import io
import base64
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from backend.analysis.PortfolioNetworkAnalyzer import PortfolioNetworkAnalyzer

@pytest.mark.asyncio
async def test_analyze_with_valid_data(sample_company_data, sample_network_data):
    """有効なデータでポートフォリオネットワーク分析を実行するテスト"""
    # テスト対象インスタンスを作成
    analyzer = PortfolioNetworkAnalyzer()

    # 内部メソッドをモック
    with patch.object(analyzer, '_build_network_graph') as mock_build, \
         patch.object(analyzer, '_calculate_network_metrics') as mock_metrics, \
         patch.object(analyzer, '_detect_communities') as mock_communities, \
         patch.object(analyzer, '_identify_central_nodes') as mock_central, \
         patch.object(analyzer, '_calculate_ecosystem_coefficient') as mock_ecosystem, \
         patch.object(analyzer, '_calculate_knowledge_transfer_index') as mock_kti, \
         patch.object(analyzer, '_generate_network_plot') as mock_plot, \
         patch.object(analyzer, '_analyze_causal_structure') as mock_causal:

        # モックの戻り値を設定
        mock_build.return_value = sample_network_data
        mock_metrics.return_value = {
            'density': 0.3,
            'clustering_coefficient': 0.5,
            'average_path_length': 2.1
        }
        mock_communities.return_value = [
            {'community_id': 1, 'nodes': ['comp1', 'comp2']},
            {'community_id': 2, 'nodes': ['comp3', 'comp4', 'comp5']}
        ]
        mock_central.return_value = [
            {'node': 'comp3', 'centrality': 0.8},
            {'node': 'comp1', 'centrality': 0.6}
        ]
        mock_ecosystem.return_value = 0.75
        mock_kti.return_value = 0.65
        mock_plot.return_value = "base64_encoded_image"
        mock_causal.return_value = {
            'causal_graph': nx.DiGraph(),
            'directed_edges': [('comp1', 'comp2'), ('comp2', 'comp3')],
            'causal_strength': {'comp1->comp2': 0.7, 'comp2->comp3': 0.5}
        }

        # 分析を実行
        result = await analyzer.analyze(data=sample_company_data)

        # 戻り値の検証
        assert isinstance(result, dict)
        assert 'network_metrics' in result
        assert 'communities' in result
        assert 'central_nodes' in result
        assert 'ecosystem_coefficient' in result
        assert 'knowledge_transfer_index' in result
        assert 'network_visualization' in result
        assert 'causal_structure' in result
        assert 'timestamp' in result

        # 各メソッドが正しく呼び出されたかを確認
        mock_build.assert_called_once()
        mock_metrics.assert_called_once()
        mock_communities.assert_called_once()
        mock_central.assert_called_once()
        mock_ecosystem.assert_called_once()
        mock_kti.assert_called_once()
        mock_plot.assert_called_once()
        mock_causal.assert_called_once()

def test_build_network_graph(sample_company_data):
    """ネットワークグラフ構築機能のテスト"""
    # テスト対象インスタンスを作成
    analyzer = PortfolioNetworkAnalyzer()

    # 内部メソッドをモック
    with patch.object(analyzer, '_calculate_synergy', return_value=0.7):
        # グラフを構築
        G = analyzer._build_network_graph(data=sample_company_data, min_edge_weight=0.5)

        # 戻り値の検証
        assert isinstance(G, nx.Graph)
        assert len(G.nodes) == len(sample_company_data)
        # エッジが存在することを確認（正確な数はシナジー計算結果に依存）
        assert len(G.edges) > 0

        # 各ノードに適切な属性が設定されていることを確認
        for node in G.nodes:
            assert 'industry' in G.nodes[node]
            assert 'stage' in G.nodes[node]
            assert 'size' in G.nodes[node]

def test_calculate_synergy():
    """企業間シナジー計算のテスト"""
    # テスト用の会社データを作成
    company1 = pd.Series({
        'industry': 'Tech',
        'stage': 'Series A',
        'growth_rate': 0.3,
        'employee_count': 25,
        'wellness_score': 75
    })

    company2 = pd.Series({
        'industry': 'Tech',
        'stage': 'Series B',
        'growth_rate': 0.2,
        'employee_count': 50,
        'wellness_score': 80
    })

    # 異なる業界の会社
    company3 = pd.Series({
        'industry': 'Healthcare',
        'stage': 'Series A',
        'growth_rate': 0.25,
        'employee_count': 30,
        'wellness_score': 70
    })

    # テスト対象インスタンスを作成
    analyzer = PortfolioNetworkAnalyzer()

    # シナジーを計算
    synergy1_2 = analyzer._calculate_synergy(company1, company2)
    synergy1_3 = analyzer._calculate_synergy(company1, company3)

    # 戻り値の検証
    assert isinstance(synergy1_2, float)
    assert isinstance(synergy1_3, float)
    assert 0 <= synergy1_2 <= 1
    assert 0 <= synergy1_3 <= 1
    # 同じ業界の会社間シナジーが異なる業界の会社間より高いことを確認
    assert synergy1_2 > synergy1_3

def test_calculate_network_metrics():
    """ネットワークメトリクス計算のテスト"""
    # テスト用のグラフを作成
    G = nx.Graph()
    G.add_nodes_from(['comp1', 'comp2', 'comp3', 'comp4', 'comp5'])
    G.add_edges_from([
        ('comp1', 'comp2', {'weight': 0.7}),
        ('comp1', 'comp3', {'weight': 0.6}),
        ('comp2', 'comp3', {'weight': 0.8}),
        ('comp3', 'comp4', {'weight': 0.5}),
        ('comp4', 'comp5', {'weight': 0.9})
    ])

    # テスト対象インスタンスを作成
    analyzer = PortfolioNetworkAnalyzer()

    # ネットワークメトリクスを計算
    metrics = analyzer._calculate_network_metrics(G)

    # 戻り値の検証
    assert isinstance(metrics, dict)
    assert 'density' in metrics
    assert 'clustering_coefficient' in metrics
    assert 'average_path_length' in metrics
    assert 'diameter' in metrics
    assert 'average_degree' in metrics

    # 値の範囲を検証
    assert 0 <= metrics['density'] <= 1
    assert 0 <= metrics['clustering_coefficient'] <= 1
    assert metrics['average_path_length'] > 0
    assert metrics['diameter'] >= 0
    assert metrics['average_degree'] > 0

def test_detect_communities():
    """コミュニティ検出機能のテスト"""
    # テスト用のグラフを作成
    G = nx.Graph()
    G.add_nodes_from(['comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6'])
    G.add_edges_from([
        ('comp1', 'comp2', {'weight': 0.9}),
        ('comp2', 'comp3', {'weight': 0.8}),
        ('comp1', 'comp3', {'weight': 0.7}),
        ('comp4', 'comp5', {'weight': 0.9}),
        ('comp5', 'comp6', {'weight': 0.8}),
        ('comp4', 'comp6', {'weight': 0.7}),
        ('comp3', 'comp4', {'weight': 0.3})  # 弱い結合
    ])

    # ノード属性を設定
    for node in G.nodes:
        G.nodes[node]['industry'] = 'Tech' if node in ['comp1', 'comp2', 'comp3'] else 'Healthcare'

    # テスト対象インスタンスを作成
    analyzer = PortfolioNetworkAnalyzer()

    # コミュニティを検出
    communities = analyzer._detect_communities(G)

    # 戻り値の検証
    assert isinstance(communities, list)
    assert len(communities) > 0

    # 各コミュニティが期待される形式か確認
    for community in communities:
        assert 'community_id' in community
        assert 'nodes' in community
        assert 'size' in community
        assert 'dominant_industry' in community
        assert isinstance(community['nodes'], list)
        assert len(community['nodes']) > 0

def test_identify_central_nodes():
    """中心ノード特定機能のテスト"""
    # テスト用のグラフを作成
    G = nx.Graph()
    G.add_nodes_from(['comp1', 'comp2', 'comp3', 'comp4', 'comp5'])
    G.add_edges_from([
        ('comp1', 'comp2'),
        ('comp1', 'comp3'),
        ('comp1', 'comp4'),
        ('comp2', 'comp3'),
        ('comp3', 'comp5')
    ])

    # テスト対象インスタンスを作成
    analyzer = PortfolioNetworkAnalyzer()

    # 中心ノードを特定
    central_nodes = analyzer._identify_central_nodes(G)

    # 戻り値の検証
    assert isinstance(central_nodes, list)
    assert len(central_nodes) > 0

    # 各ノード情報が期待される形式か確認
    for node_info in central_nodes:
        assert 'node' in node_info
        assert 'centrality' in node_info
        assert 'centrality_type' in node_info
        assert isinstance(node_info['centrality'], float)

    # comp1とcomp3が上位に含まれているか確認（ネットワーク構造から予想される）
    top_nodes = [node_info['node'] for node_info in central_nodes[:2]]
    assert 'comp1' in top_nodes or 'comp3' in top_nodes

@pytest.mark.asyncio
async def test_generate_network_plot():
    """ネットワーク可視化機能のテスト"""
    # テスト用のグラフを作成
    G = nx.Graph()
    G.add_nodes_from(['comp1', 'comp2', 'comp3', 'comp4', 'comp5'])

    # ノード属性を設定
    for i, node in enumerate(G.nodes):
        G.nodes[node]['industry'] = ['Tech', 'Healthcare', 'Finance', 'Retail', 'Manufacturing'][i]
        G.nodes[node]['stage'] = ['Seed', 'Series A', 'Series B', 'Series C', 'Growth'][i]
        G.nodes[node]['size'] = [10, 25, 50, 100, 200][i]

    G.add_edges_from([
        ('comp1', 'comp2', {'weight': 0.7}),
        ('comp1', 'comp3', {'weight': 0.6}),
        ('comp2', 'comp3', {'weight': 0.8}),
        ('comp3', 'comp4', {'weight': 0.5}),
        ('comp4', 'comp5', {'weight': 0.9})
    ])

    # コミュニティ情報
    communities = [
        {'community_id': 1, 'nodes': ['comp1', 'comp2', 'comp3']},
        {'community_id': 2, 'nodes': ['comp4', 'comp5']}
    ]

    # テスト対象インスタンスを作成
    analyzer = PortfolioNetworkAnalyzer()

    # matplotlibのフィギュアとaxesをモック
    with patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('io.BytesIO') as mock_bytesio, \
         patch('base64.b64encode') as mock_b64encode:

        # モックの設定
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_figure.return_value = mock_fig

        mock_buffer = MagicMock()
        mock_bytesio.return_value = mock_buffer
        mock_buffer.getvalue.return_value = b'image_bytes'

        mock_b64encode.return_value = b'base64_encoded_bytes'

        # ネットワーク可視化を実行
        img_data = analyzer._generate_network_plot(
            G=G,
            data=pd.DataFrame({'company_id': G.nodes, 'wellness_score': [70, 75, 80, 85, 90]}),
            communities=communities
        )

        # 戻り値の検証
        assert isinstance(img_data, str)
        assert img_data == 'base64_encoded_bytes'

        # 適切なメソッドが呼ばれたか確認
        mock_figure.assert_called_once()
        mock_fig.add_subplot.assert_called_once()
        mock_savefig.assert_called_once()
        mock_b64encode.assert_called_once_with(b'image_bytes')

@pytest.mark.asyncio
async def test_analyze_causal_structure():
    """因果構造分析機能のテスト"""
    # テスト用のデータを作成
    data = pd.DataFrame({
        'company_id': ['comp1', 'comp2', 'comp3', 'comp4', 'comp5'] * 10,
        'revenue': np.random.normal(1000000, 100000, 50),
        'wellness_score': np.random.normal(75, 5, 50),
        'growth_rate': np.random.normal(0.2, 0.05, 50),
        'employee_count': np.random.normal(50, 10, 50),
        'customer_count': np.random.normal(1000, 200, 50)
    })

    # テスト対象インスタンスを作成
    analyzer = PortfolioNetworkAnalyzer()

    # pgmpyのエスティメータをモック
    with patch('pgmpy.estimators.PC') as mock_pc, \
         patch('pgmpy.estimators.HillClimbSearch') as mock_hc:

        # PCアルゴリズムのモック
        mock_pc_instance = MagicMock()
        mock_pc.return_value = mock_pc_instance
        mock_pc_instance.estimate.return_value = nx.DiGraph()

        # HillClimbSearchのモック
        mock_hc_instance = MagicMock()
        mock_hc.return_value = mock_hc_instance
        mock_hc_instance.estimate.return_value = nx.DiGraph()

        # 因果構造分析を実行
        result = analyzer._analyze_causal_structure(
            data=data,
            method='pc'  # PCアルゴリズムを使用
        )

        # 戻り値の検証
        assert isinstance(result, dict)
        assert 'causal_graph' in result
        assert 'directed_edges' in result
        assert 'causal_strength' in result
        assert isinstance(result['causal_graph'], nx.DiGraph)

        # PCメソッドが呼ばれたことを確認
        mock_pc.assert_called_once()
        mock_pc_instance.estimate.assert_called_once()

@pytest.mark.asyncio
async def test_estimate_causal_effect(sample_company_data):
    """因果効果推定機能のテスト"""
    # DoWhyのCausalModelをモック
    with patch('dowhy.CausalModel') as mock_causal_model_cls:
        # モックのCausalModelインスタンスを作成
        mock_model = MagicMock()
        mock_causal_model_cls.return_value = mock_model

        # 必要なメソッドのモック
        mock_model.identify_effect.return_value = MagicMock()
        mock_model.estimate_effect.return_value = MagicMock()
        mock_model.estimate_effect.return_value.value = 0.3
        mock_model.estimate_effect.return_value.get_confidence_intervals.return_value = (0.2, 0.4)
        mock_model.refute_estimate.return_value = [
            MagicMock(refutation_result=0.29)  # 近い値を返す
        ]

        # テスト対象インスタンスを作成
        analyzer = PortfolioNetworkAnalyzer()

        # 因果効果推定を実行
        result = await analyzer.estimate_causal_effect(
            data=sample_company_data,
            treatment='wellness_score',
            outcome='revenue'
        )

        # 戻り値の検証
        assert isinstance(result, dict)
        assert 'causal_effect' in result
        assert 'confidence_interval' in result
        assert 'refutation_results' in result
        assert result['causal_effect'] == 0.3
        assert result['confidence_interval'] == (0.2, 0.4)

        # 適切なメソッドが呼ばれたか確認
        mock_causal_model_cls.assert_called_once()
        mock_model.identify_effect.assert_called_once()
        mock_model.estimate_effect.assert_called_once()
        mock_model.refute_estimate.assert_called()