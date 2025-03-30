import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
import networkx as nx
import io
import base64
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

from backend.analysis.ClusterAnalyzer import DataAnalyzer

@pytest.fixture
def mock_bq_service():
    """BigQueryServiceのモックを提供します"""
    service = MagicMock()
    service.query = AsyncMock()
    service.save_dataframe = AsyncMock()
    return service

@pytest.fixture
def sample_cluster_data():
    """クラスタ分析用のサンプルデータを作成します"""
    np.random.seed(42)

    # 4つのクラスタを持つデータを生成
    n_samples = 100

    # クラスタ中心
    centers = [
        [0, 0],  # クラスタ1
        [10, 10],  # クラスタ2
        [0, 10],  # クラスタ3
        [10, 0]   # クラスタ4
    ]

    # 各クラスタからサンプルを生成
    data = []
    labels = []

    for i, center in enumerate(centers):
        cluster_samples = np.random.randn(n_samples, 2) + center
        data.append(cluster_samples)
        labels.extend([i] * n_samples)

    # データを結合
    X = np.vstack(data)
    y = np.array(labels)

    # DataFrameを作成
    df = pd.DataFrame({
        'feature1': X[:, 0],
        'feature2': X[:, 1],
        'true_label': y,
        'revenue': np.random.normal(1000000, 200000, len(y)),
        'profit': np.random.normal(200000, 50000, len(y)),
        'growth_rate': np.random.normal(0.2, 0.05, len(y))
    })

    return df

@pytest.fixture
def mock_causal_structure_analyzer():
    """CausalStructureAnalyzerのモックを提供します"""
    analyzer = MagicMock()

    # PC algorithmメソッドのモック
    analyzer.run_pc_algorithm = AsyncMock()
    analyzer.run_pc_algorithm.return_value = nx.DiGraph()

    # 他のメソッドのモック
    analyzer.calculate_causal_effects = MagicMock()
    analyzer.calculate_causal_effects.return_value = {
        'feature1->feature2': 0.7,
        'feature2->revenue': 0.5
    }

    return analyzer

@pytest.mark.asyncio
async def test_analyze_with_kmeans(mock_bq_service, sample_cluster_data):
    """K-meansクラスタリング分析のテスト"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_cluster_data

    # PCAとKMeansをモック
    with patch('sklearn.preprocessing.StandardScaler') as mock_scaler, \
         patch('sklearn.decomposition.PCA') as mock_pca, \
         patch('sklearn.cluster.KMeans') as mock_kmeans:

        # スケーラーのモック
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.randn(400, 2)
        mock_scaler.return_value = mock_scaler_instance

        # PCAのモック
        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.return_value = np.random.randn(400, 2)
        mock_pca.return_value = mock_pca_instance

        # KMeansのモック
        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.fit_predict.return_value = np.random.randint(0, 4, 400)
        mock_kmeans_instance.cluster_centers_ = np.random.randn(4, 2)
        mock_kmeans.return_value = mock_kmeans_instance

        # 可視化関数をモック
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('io.BytesIO') as mock_bytesio, \
             patch('base64.b64encode') as mock_b64encode:

            # 画像データのモック
            mock_bytesio_instance = MagicMock()
            mock_bytesio.return_value = mock_bytesio_instance
            mock_bytesio_instance.getvalue.return_value = b'image_data'
            mock_b64encode.return_value = b'base64_encoded_image'

            # DataAnalyzerのインスタンスを作成
            analyzer = DataAnalyzer(bq_service=mock_bq_service)

            # クラスタリング分析を実行
            df_result, results = await analyzer.analyze(
                query="SELECT * FROM dataset.table",
                save_results=False,
                algorithm='kmeans',
                n_clusters=4
            )

            # 結果の検証
            assert isinstance(df_result, pd.DataFrame)
            assert 'cluster' in df_result.columns

            assert isinstance(results, dict)
            assert 'cluster_centers' in results
            assert 'cluster_sizes' in results
            assert 'visualization' in results
            assert 'metrics' in results
            assert 'timestamp' in results

            # BigQueryのqueryメソッドが呼び出されたことを確認
            mock_bq_service.query.assert_called_once()

            # 保存メソッドが呼び出されていないことを確認
            mock_bq_service.save_dataframe.assert_not_called()

@pytest.mark.asyncio
async def test_analyze_with_dbscan(mock_bq_service, sample_cluster_data):
    """DBSCANクラスタリング分析のテスト"""
    # BigQueryのクエリ結果をモック
    mock_bq_service.query.return_value = sample_cluster_data

    # モジュールをモック
    with patch('sklearn.preprocessing.StandardScaler') as mock_scaler, \
         patch('sklearn.decomposition.PCA') as mock_pca, \
         patch('sklearn.cluster.DBSCAN') as mock_dbscan:

        # スケーラーのモック
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.randn(400, 2)
        mock_scaler.return_value = mock_scaler_instance

        # PCAのモック
        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.return_value = np.random.randn(400, 2)
        mock_pca.return_value = mock_pca_instance

        # DBSCANのモック
        mock_dbscan_instance = MagicMock()
        mock_dbscan_instance.fit_predict.return_value = np.random.randint(-1, 3, 400)
        mock_dbscan.return_value = mock_dbscan_instance

        # 可視化関数をモック
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('io.BytesIO') as mock_bytesio, \
             patch('base64.b64encode') as mock_b64encode:

            # 画像データのモック
            mock_bytesio_instance = MagicMock()
            mock_bytesio.return_value = mock_bytesio_instance
            mock_bytesio_instance.getvalue.return_value = b'image_data'
            mock_b64encode.return_value = b'base64_encoded_image'

            # DataAnalyzerのインスタンスを作成
            analyzer = DataAnalyzer(bq_service=mock_bq_service)

            # クラスタリング分析を実行
            df_result, results = await analyzer.analyze(
                query="SELECT * FROM dataset.table",
                save_results=True,
                dataset_id="output_dataset",
                table_id="output_table",
                algorithm='dbscan',
                eps=0.5,
                min_samples=5
            )

            # 結果の検証
            assert isinstance(df_result, pd.DataFrame)
            assert 'cluster' in df_result.columns

            assert isinstance(results, dict)
            assert 'cluster_sizes' in results
            assert 'visualization' in results
            assert 'metrics' in results
            assert 'timestamp' in results

            # BigQueryのクエリと保存メソッドが呼び出されたことを確認
            mock_bq_service.query.assert_called_once()
            mock_bq_service.save_dataframe.assert_called_once()

@pytest.mark.asyncio
async def test_causal_cluster_analysis(mock_bq_service, sample_cluster_data, mock_causal_structure_analyzer):
    """因果クラスタ分析のテスト"""
    # DataAnalyzerのインスタンスを作成してCausalStructureAnalyzerをモックに置き換え
    analyzer = DataAnalyzer(bq_service=mock_bq_service)
    analyzer.causal_analyzer = mock_causal_structure_analyzer

    # Kmeansのモック
    with patch('sklearn.preprocessing.StandardScaler') as mock_scaler, \
         patch('sklearn.cluster.KMeans') as mock_kmeans:

        # スケーラーのモック
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.randn(400, 2)
        mock_scaler.return_value = mock_scaler_instance

        # KMeansのモック
        mock_kmeans_instance = MagicMock()
        cluster_labels = np.array([0, 1, 2, 3] * 100)
        mock_kmeans_instance.fit_predict.return_value = cluster_labels
        mock_kmeans.return_value = mock_kmeans_instance

        # networkxのメソッドをモック
        with patch('networkx.Graph') as mock_graph, \
             patch('networkx.is_isomorphic') as mock_is_isomorphic, \
             patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('io.BytesIO') as mock_bytesio, \
             patch('base64.b64encode') as mock_b64encode:

            # モックの設定
            mock_is_isomorphic.return_value = False
            mock_bytesio_instance = MagicMock()
            mock_bytesio.return_value = mock_bytesio_instance
            mock_bytesio_instance.getvalue.return_value = b'image_data'
            mock_b64encode.return_value = b'base64_encoded_image'

            # 因果クラスタ分析を実行
            result = await analyzer.causal_cluster_analysis(
                data=sample_cluster_data,
                n_clusters=4,
                method='pc'
            )

            # 結果の検証
            assert isinstance(result, dict)
            assert 'clusters' in result
            assert 'causal_structures' in result
            assert 'common_structures' in result
            assert 'visualization' in result
            assert 'metrics' in result

            # 個別のクラスタが正しく処理されているか確認
            assert len(result['clusters']) == 4

            # 因果分析が各クラスタに対して呼び出されたことを確認
            assert mock_causal_structure_analyzer.run_pc_algorithm.call_count == 4
            assert mock_causal_structure_analyzer.calculate_causal_effects.call_count >= 4

def test_validate_data(mock_bq_service):
    """データ検証機能のテスト"""
    # 有効なデータ
    valid_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1]
    })

    # NaNを含むデータ
    invalid_data_nan = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [5, 4, 3, 2, 1]
    })

    # 文字列を含むデータ
    invalid_data_str = pd.DataFrame({
        'feature1': [1, 2, '3', 4, 5],
        'feature2': [5, 4, 3, 2, 1]
    })

    # インスタンスを作成
    analyzer = DataAnalyzer(bq_service=mock_bq_service)

    # 有効なデータの検証
    is_valid, error_msg = analyzer._validate_data(valid_data)
    assert is_valid is True
    assert error_msg is None

    # NaNを含むデータの検証
    is_valid, error_msg = analyzer._validate_data(invalid_data_nan)
    assert is_valid is False
    assert error_msg is not None
    assert 'missing values' in error_msg.lower()

    # 文字列を含むデータの検証
    is_valid, error_msg = analyzer._validate_data(invalid_data_str)
    assert is_valid is False
    assert error_msg is not None
    assert 'numeric' in error_msg.lower()

def test_analyze_common_structures(mock_bq_service):
    """クラスタ間の共通構造分析機能のテスト"""
    # 複数のクラスタのグラフを作成
    G1 = nx.DiGraph()
    G1.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])

    G2 = nx.DiGraph()
    G2.add_edges_from([('A', 'B'), ('B', 'C'), ('X', 'Y')])

    G3 = nx.DiGraph()
    G3.add_edges_from([('A', 'B'), ('X', 'Y'), ('Z', 'W')])

    cluster_graphs = {0: G1, 1: G2, 2: G3}

    # インスタンスを作成
    analyzer = DataAnalyzer(bq_service=mock_bq_service)

    # networkxのメソッドをモック
    with patch('networkx.is_isomorphic', return_value=False), \
         patch('networkx.DiGraph') as mock_digraph:

        # 共通構造を分析
        result = analyzer._analyze_common_structures(cluster_graphs)

        # 結果の検証
        assert isinstance(result, dict)
        assert 'common_edges' in result
        assert 'common_subgraphs' in result
        assert 'edge_frequency' in result

        # エッジ頻度に'A->B'が含まれていることを確認
        assert ('A', 'B') in result['edge_frequency']
        assert result['edge_frequency'][('A', 'B')] == 3  # 3つのグラフすべてに存在

        # エッジ頻度に'B->C'が含まれていることを確認
        assert ('B', 'C') in result['edge_frequency']
        assert result['edge_frequency'][('B', 'C')] == 2  # 2つのグラフに存在