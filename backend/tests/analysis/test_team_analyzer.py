import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import io
import base64
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from sklearn.cluster import KMeans

from backend.analysis.Team_Analyzer import TeamAnalyzer

@pytest.fixture
def sample_founder_profiles():
    """創業メンバーのプロフィールサンプルデータを提供します"""
    return [
        {
            "id": "founder1",
            "role": "CEO",
            "years_experience": 12,
            "previous_startups": 2,
            "previous_exits": 1,
            "education": "MBA",
            "domain_expertise": 8,
            "technical_expertise": 5,
            "leadership_score": 9,
            "network_score": 8
        },
        {
            "id": "founder2",
            "role": "CTO",
            "years_experience": 15,
            "previous_startups": 1,
            "previous_exits": 0,
            "education": "PhD",
            "domain_expertise": 7,
            "technical_expertise": 9,
            "leadership_score": 7,
            "network_score": 6
        },
        {
            "id": "founder3",
            "role": "CFO",
            "years_experience": 10,
            "previous_startups": 0,
            "previous_exits": 0,
            "education": "MBA",
            "domain_expertise": 6,
            "technical_expertise": 4,
            "leadership_score": 7,
            "network_score": 7
        }
    ]

@pytest.fixture
def sample_employee_data():
    """従業員データのサンプルを提供します"""
    # 日付の範囲を作成
    dates = pd.date_range(start='2021-01-01', end='2022-12-31', freq='M')

    # 毎月の従業員数（成長を模倣）
    employee_counts = [5, 5, 6, 7, 8, 10, 12, 14, 15, 16, 18, 20,
                      22, 25, 28, 30, 33, 36, 38, 40, 43, 45, 48, 50]

    # 部門別の人数の割合を定義
    dept_ratios = {
        'Engineering': 0.5,
        'Product': 0.2,
        'Sales': 0.15,
        'Marketing': 0.1,
        'Operations': 0.05
    }

    # データフレームを作成するためのデータを格納するリスト
    data = []

    # 各月のデータを生成
    for i, date in enumerate(dates):
        count = employee_counts[i]

        # 部門ごとのデータを追加
        for dept, ratio in dept_ratios.items():
            dept_count = int(count * ratio)
            if dept_count > 0:  # 少なくとも1人の従業員がいる場合
                data.append({
                    'date': date,
                    'department': dept,
                    'employee_count': dept_count,
                    'turnover_rate': np.random.uniform(0.01, 0.05),
                    'tenure_months': np.random.normal(i/2 + 3, 2),  # 時間とともに平均在職期間が伸びる
                    'avg_experience_years': np.random.normal(5, 1.5)
                })

    return pd.DataFrame(data)

@pytest.fixture
def sample_engagement_data():
    """エンゲージメントデータのサンプルを提供します"""
    # 日付の範囲を作成
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='W')

    # データフレームを作成するためのデータを格納するリスト
    data = []

    # 部門リスト
    departments = ['Engineering', 'Product', 'Sales', 'Marketing', 'Operations']

    # 各週のデータを生成
    for date in dates:
        for dept in departments:
            # ランダムなエンゲージメントスコアを生成
            data.append({
                'date': date,
                'department': dept,
                'engagement_score': np.random.normal(3.8, 0.3),  # 5点満点
                'satisfaction_score': np.random.normal(3.7, 0.4),
                'communication_score': np.random.normal(3.6, 0.5),
                'work_life_balance': np.random.normal(3.5, 0.6),
                'manager_rating': np.random.normal(3.9, 0.3),
                'participation_rate': np.random.uniform(0.7, 0.95)
            })

    return pd.DataFrame(data)

@pytest.fixture
def sample_hiring_data():
    """採用データのサンプルを提供します"""
    # 採用データの期間
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='W')

    # 役職リスト
    positions = ['Software Engineer', 'Product Manager', 'Sales Representative', 'Marketing Specialist', 'Operations Analyst']

    # データを格納するためのリスト
    data = []

    # 各週のデータを生成
    for date in dates:
        for position in positions:
            # このポジションの採用プロセスが行われるかどうかをランダムに決定
            if np.random.random() > 0.7:  # 30%の確率で採用プロセスが行われる
                data.append({
                    'date': date,
                    'position': position,
                    'applications': np.random.randint(10, 100),
                    'interviews': np.random.randint(3, 15),
                    'offers': np.random.randint(1, 5),
                    'acceptances': np.random.randint(0, 3),
                    'time_to_fill_days': np.random.randint(14, 60),
                    'cost_per_hire': np.random.uniform(5000, 15000),
                    'source_quality': np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.5, 0.2])
                })

    return pd.DataFrame(data)

@pytest.fixture
def sample_interaction_data():
    """チーム内交流データのサンプルを提供します"""
    # 社員ID
    employee_ids = [f'emp{i}' for i in range(1, 21)]

    # 部門情報
    departments = {
        'Engineering': employee_ids[:10],
        'Product': employee_ids[10:13],
        'Sales': employee_ids[13:16],
        'Marketing': employee_ids[16:18],
        'Operations': employee_ids[18:20]
    }

    # データを格納するためのリスト
    data = []

    # 交流データを生成
    for i, emp1 in enumerate(employee_ids):
        for j, emp2 in enumerate(employee_ids):
            if i != j:  # 自分自身との交流は除外
                # 同じ部門内の交流は頻度が高い
                same_dept = False
                for dept, members in departments.items():
                    if emp1 in members and emp2 in members:
                        same_dept = True
                        break

                # 交流頻度を決定
                if same_dept:
                    interaction_freq = np.random.randint(10, 50)
                else:
                    interaction_freq = np.random.randint(1, 15)

                if interaction_freq > 0:  # 交流がある場合のみデータに追加
                    data.append({
                        'employee1_id': emp1,
                        'employee2_id': emp2,
                        'department1': next(dept for dept, members in departments.items() if emp1 in members),
                        'department2': next(dept for dept, members in departments.items() if emp2 in members),
                        'interaction_count': interaction_freq,
                        'collaboration_score': np.random.normal(3.5, 0.8),
                        'communication_channel': np.random.choice(['Email', 'Slack', 'Meeting', 'Call'], p=[0.4, 0.3, 0.2, 0.1])
                    })

    return pd.DataFrame(data)

@pytest.fixture
def mock_firestore_client():
    """Firestoreクライアントのモックを提供します"""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_doc = MagicMock()

    mock_db.collection.return_value = mock_collection
    mock_collection.document.return_value = mock_doc
    mock_doc.get.return_value = MagicMock(exists=True, to_dict=lambda: {})
    mock_doc.set.return_value = None

    return mock_db

def test_evaluate_founding_team(sample_founder_profiles, mock_firestore_client):
    """創業チーム評価機能のテスト"""
    # Team Analyzerのインスタンスを作成
    analyzer = TeamAnalyzer(db=mock_firestore_client)

    # 創業チームを評価
    result = analyzer.evaluate_founding_team(
        founder_profiles=sample_founder_profiles,
        company_stage="series_a",
        industry="software"
    )

    # 戻り値の検証
    assert isinstance(result, dict)
    assert 'overall_score' in result
    assert 'team_composition' in result
    assert 'strengths' in result
    assert 'weaknesses' in result
    assert 'recommendations' in result

    # スコアの範囲を検証
    assert 0 <= result['overall_score'] <= 100

    # チーム構成の内容を検証
    assert 'roles_coverage' in result['team_composition']
    assert 'experience' in result['team_composition']
    assert 'domain_expertise' in result['team_composition']

    # 強みと弱みがリストであることを確認
    assert isinstance(result['strengths'], list)
    assert isinstance(result['weaknesses'], list)
    assert isinstance(result['recommendations'], list)

def test_analyze_org_growth(sample_employee_data, mock_firestore_client):
    """組織成長分析機能のテスト"""
    # Team Analyzerのインスタンスを作成
    analyzer = TeamAnalyzer(db=mock_firestore_client)

    # 組織成長を分析
    result = analyzer.analyze_org_growth(
        employee_data=sample_employee_data,
        timeline="1y",
        company_stage="series_a",
        industry="software"
    )

    # 戻り値の検証
    assert isinstance(result, dict)
    assert 'growth_metrics' in result
    assert 'department_breakdown' in result
    assert 'structural_health' in result
    assert 'risks' in result
    assert 'recommendations' in result

    # 成長メトリクスの内容を検証
    assert 'headcount_growth_rate' in result['growth_metrics']
    assert 'burn_rate_per_employee' in result['growth_metrics']
    assert 'avg_tenure' in result['growth_metrics']

    # 部門別内訳が辞書であることを確認
    assert isinstance(result['department_breakdown'], dict)

    # 組織構造の健全性スコアの範囲を検証
    assert 0 <= result['structural_health']['overall_score'] <= 100

    # リスクと推奨事項がリストであることを確認
    assert isinstance(result['risks'], list)
    assert isinstance(result['recommendations'], list)

def test_measure_culture_strength(sample_engagement_data, mock_firestore_client):
    """文化強度測定機能のテスト"""
    # Team Analyzerのインスタンスを作成
    analyzer = TeamAnalyzer(db=mock_firestore_client)

    # nltk.sentiment.SentimentIntensityAnalyzerをモック
    with patch('nltk.sentiment.SentimentIntensityAnalyzer') as mock_sia:
        # センチメント分析のモック結果
        mock_sia_instance = MagicMock()
        mock_sia_instance.polarity_scores.return_value = {'compound': 0.6, 'pos': 0.7, 'neu': 0.2, 'neg': 0.1}
        mock_sia.return_value = mock_sia_instance

        # 文化強度を測定
        result = analyzer.measure_culture_strength(
            engagement_data=sample_engagement_data
        )

        # 戻り値の検証
        assert isinstance(result, dict)
        assert 'culture_score' in result
        assert 'engagement_metrics' in result
        assert 'department_scores' in result
        assert 'trends' in result
        assert 'strengths' in result
        assert 'areas_for_improvement' in result
        assert 'recommendations' in result

        # スコアの範囲を検証
        assert 0 <= result['culture_score'] <= 100

        # エンゲージメントメトリクスの内容を検証
        assert 'avg_engagement' in result['engagement_metrics']
        assert 'avg_satisfaction' in result['engagement_metrics']
        assert 'participation_rate' in result['engagement_metrics']

        # 部門別スコアが辞書であることを確認
        assert isinstance(result['department_scores'], dict)

        # 強みと改善点がリストであることを確認
        assert isinstance(result['strengths'], list)
        assert isinstance(result['areas_for_improvement'], list)
        assert isinstance(result['recommendations'], list)

def test_analyze_hiring_effectiveness(sample_hiring_data, mock_firestore_client):
    """採用効果分析機能のテスト"""
    # Team Analyzerのインスタンスを作成
    analyzer = TeamAnalyzer(db=mock_firestore_client)

    # KMeansをモック
    with patch('sklearn.cluster.KMeans') as mock_kmeans:
        # クラスタリングのモック結果
        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_kmeans_instance.cluster_centers_ = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
        mock_kmeans.return_value = mock_kmeans_instance

        # 採用効果を分析
        result = analyzer.analyze_hiring_effectiveness(
            hiring_data=sample_hiring_data
        )

        # 戻り値の検証
        assert isinstance(result, dict)
        assert 'overall_effectiveness' in result
        assert 'key_metrics' in result
        assert 'position_analysis' in result
        assert 'source_quality' in result
        assert 'bottlenecks' in result
        assert 'recommendations' in result

        # 全体的な効果スコアの範囲を検証
        assert 0 <= result['overall_effectiveness'] <= 100

        # 主要メトリクスの内容を検証
        assert 'conversion_rate' in result['key_metrics']
        assert 'time_to_fill' in result['key_metrics']
        assert 'cost_per_hire' in result['key_metrics']

        # 役職別分析が辞書であることを確認
        assert isinstance(result['position_analysis'], dict)

        # ボトルネックと推奨事項がリストであることを確認
        assert isinstance(result['bottlenecks'], list)
        assert isinstance(result['recommendations'], list)

def test_generate_org_network_graph(sample_interaction_data, mock_firestore_client):
    """組織ネットワークグラフ生成機能のテスト"""
    # Team Analyzerのインスタンスを作成
    analyzer = TeamAnalyzer(db=mock_firestore_client)

    # NetworkX関連の機能をモック
    with patch('networkx.Graph') as mock_graph, \
         patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('io.BytesIO') as mock_bytesio, \
         patch('base64.b64encode') as mock_b64encode:

        # モックの設定
        mock_graph_instance = MagicMock()
        mock_graph.return_value = mock_graph_instance

        mock_graph_instance.nodes.return_value = ['emp1', 'emp2', 'emp3']
        mock_graph_instance.edges.return_value = [('emp1', 'emp2'), ('emp2', 'emp3')]

        # 中心性のモック
        mock_graph_instance.degree.return_value = {'emp1': 2, 'emp2': 3, 'emp3': 1}.items()
        mock_betweenness = {'emp1': 0.5, 'emp2': 0.8, 'emp3': 0.2}
        mock_closeness = {'emp1': 0.6, 'emp2': 0.9, 'emp3': 0.3}

        with patch('networkx.betweenness_centrality', return_value=mock_betweenness), \
             patch('networkx.closeness_centrality', return_value=mock_closeness), \
             patch('networkx.density', return_value=0.6), \
             patch('networkx.average_clustering', return_value=0.7):

            # 画像データのモック
            mock_bytesio_instance = MagicMock()
            mock_bytesio.return_value = mock_bytesio_instance
            mock_bytesio_instance.getvalue.return_value = b'image_data'
            mock_b64encode.return_value = b'base64_encoded_image'

            # ネットワークグラフを生成
            result = analyzer.generate_org_network_graph(
                interaction_data=sample_interaction_data
            )

            # 戻り値の検証
            assert isinstance(result, dict)
            assert 'network_visualization' in result
            assert 'network_metrics' in result
            assert 'central_employees' in result
            assert 'team_connectivity' in result
            assert 'silos' in result
            assert 'recommendations' in result

            # ネットワーク可視化が文字列（Base64エンコード画像）であることを確認
            assert isinstance(result['network_visualization'], str)

            # ネットワークメトリクスの内容を検証
            assert 'density' in result['network_metrics']
            assert 'clustering_coefficient' in result['network_metrics']
            assert 'avg_connections' in result['network_metrics']

            # 中心社員がリストであることを確認
            assert isinstance(result['central_employees'], list)

            # チーム接続性とサイロがそれぞれ辞書とリストであることを確認
            assert isinstance(result['team_connectivity'], dict)
            assert isinstance(result['silos'], list)
            assert isinstance(result['recommendations'], list)

def test_generate_team_recommendations(mock_firestore_client):
    """チーム推奨事項生成機能のテスト"""
    # Team Analyzerのインスタンスを作成
    analyzer = TeamAnalyzer(db=mock_firestore_client)

    # 弱みリストを作成
    weaknesses = [
        "technical_expertise_gap",
        "limited_startup_experience",
        "missing_key_role"
    ]

    # 推奨事項を生成
    recommendations = analyzer._generate_team_recommendations(
        weaknesses=weaknesses,
        company_stage="seed"
    )

    # 戻り値の検証
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert all(isinstance(rec, str) for rec in recommendations)

    # 推奨事項が弱みに対応していることを確認（キーワードチェック）
    rec_text = ' '.join(recommendations).lower()
    assert 'technical' in rec_text or 'expertise' in rec_text
    assert 'startup' in rec_text or 'experience' in rec_text
    assert 'role' in rec_text or 'hiring' in rec_text

def test_save_analysis_to_firestore(mock_firestore_client):
    """Firestoreへの分析結果保存機能のテスト"""
    # Team Analyzerのインスタンスを作成
    analyzer = TeamAnalyzer(db=mock_firestore_client)

    # 分析結果サンプル
    analysis_result = {
        'overall_score': 85,
        'strengths': ['Strong leadership', 'Technical expertise'],
        'weaknesses': ['Limited market experience'],
        'recommendations': ['Hire experienced marketing executive']
    }

    # Firestoreに保存
    doc_id = analyzer.save_analysis_to_firestore(
        company_id="company123",
        analysis_type="founding_team",
        analysis_result=analysis_result
    )

    # 戻り値の検証
    assert isinstance(doc_id, str)

    # Firestoreメソッドが正しく呼び出されたか確認
    mock_firestore_client.collection.assert_called_once()
    mock_firestore_client.collection().document.assert_called_once()
    mock_firestore_client.collection().document().set.assert_called_once()

    # setに渡されたデータにタイムスタンプが含まれていることを確認
    set_call_args = mock_firestore_client.collection().document().set.call_args[0][0]
    assert 'timestamp' in set_call_args
    assert 'company_id' in set_call_args
    assert 'analysis_type' in set_call_args
    assert 'result' in set_call_args