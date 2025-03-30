import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
from datetime import datetime

from core.auth_manager import User, UserRole

# 前提条件を確認するテスト
def test_router_exists():
    """分析ルーターが正しくインポートされていることを確認"""
    from api.routers import analysis
    assert analysis.router is not None
    assert "analysis" in analysis.router.tags

# 相関分析テスト
@patch("api.routers.analysis.get_current_analyst_user")
@patch("api.routers.analysis.CorrelationAnalyzer")
@patch("api.routers.analysis._check_analysis_access")
def test_perform_correlation_analysis(mock_check_access, mock_analyzer, mock_get_user, client, mock_analyst_user, token_header):
    """相関分析エンドポイントが正しく動作することを確認"""
    # モックの設定
    mock_get_user.return_value = mock_analyst_user
    mock_check_access.return_value = None

    analyzer_instance = MagicMock()
    mock_analyzer.return_value = analyzer_instance
    analyzer_instance.analyze.return_value = {
        "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
        "features": ["feature1", "feature2"]
    }

    # テストデータ
    analysis_data = {
        "collection_name": "financial_data",
        "conditions": [{"field": "company_id", "operator": "==", "value": "test_company_id"}],
        "target_column": "revenue"
    }

    # リクエスト実行
    response = client.post("/analysis/correlation", json=analysis_data, headers=token_header)

    # 結果確認
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "correlation_matrix" in data["data"]
    assert "features" in data["data"]
    mock_check_access.assert_called_once()
    analyzer_instance.analyze.assert_called_once()

# クラスタリング分析テスト
@patch("api.routers.analysis.get_current_analyst_user")
@patch("api.routers.analysis.ClusterAnalyzer")
@patch("api.routers.analysis._check_analysis_access")
def test_perform_cluster_analysis(mock_check_access, mock_analyzer, mock_get_user, client, mock_analyst_user, token_header):
    """クラスタリング分析エンドポイントが正しく動作することを確認"""
    # モックの設定
    mock_get_user.return_value = mock_analyst_user
    mock_check_access.return_value = None

    analyzer_instance = MagicMock()
    mock_analyzer.return_value = analyzer_instance
    analyzer_instance.analyze.return_value = {
        "clusters": [0, 1, 0, 2],
        "centroids": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "feature_importance": {"feature1": 0.7, "feature2": 0.3}
    }

    # テストデータ
    analysis_data = {
        "collection_name": "financial_data",
        "conditions": [{"field": "company_id", "operator": "==", "value": "test_company_id"}],
        "n_clusters": 3
    }

    # リクエスト実行
    response = client.post("/analysis/clustering", json=analysis_data, headers=token_header)

    # 結果確認
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "clusters" in data["data"]
    assert "centroids" in data["data"]
    assert "feature_importance" in data["data"]
    mock_check_access.assert_called_once()
    analyzer_instance.analyze.assert_called_once()

# PCA分析テスト
@patch("api.routers.analysis.get_current_analyst_user")
@patch("api.routers.analysis.PCAAnalyzer")
@patch("api.routers.analysis._check_analysis_access")
def test_perform_pca_analysis(mock_check_access, mock_analyzer, mock_get_user, client, mock_analyst_user, token_header):
    """PCA分析エンドポイントが正しく動作することを確認"""
    # モックの設定
    mock_get_user.return_value = mock_analyst_user
    mock_check_access.return_value = None

    analyzer_instance = MagicMock()
    mock_analyzer.return_value = analyzer_instance
    analyzer_instance.analyze.return_value = {
        "components": [[0.8, 0.2], [0.3, 0.7]],
        "explained_variance": [0.65, 0.25],
        "transformed_data": [[1.0, 2.0], [3.0, 4.0]]
    }

    # テストデータ
    analysis_data = {
        "collection_name": "financial_data",
        "conditions": [{"field": "company_id", "operator": "==", "value": "test_company_id"}],
        "n_components": 2
    }

    # リクエスト実行
    response = client.post("/analysis/pca", json=analysis_data, headers=token_header)

    # 結果確認
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "components" in data["data"]
    assert "explained_variance" in data["data"]
    assert "transformed_data" in data["data"]
    mock_check_access.assert_called_once()
    analyzer_instance.analyze.assert_called_once()

# アクセス権限エラーテスト
@patch("api.routers.analysis.get_current_analyst_user")
@patch("api.routers.analysis._check_analysis_access")
def test_analysis_access_denied(mock_check_access, mock_get_user, client, mock_analyst_user, token_header):
    """アクセス権限が無い場合のエラーハンドリングを確認"""
    # モックの設定
    mock_get_user.return_value = mock_analyst_user
    mock_check_access.side_effect = HTTPException(status_code=403, detail="Access denied")

    # テストデータ
    analysis_data = {
        "collection_name": "financial_data",
        "conditions": [{"field": "company_id", "operator": "==", "value": "other_company_id"}],
        "target_column": "revenue"
    }

    # リクエスト実行
    response = client.post("/analysis/correlation", json=analysis_data, headers=token_header)

    # 結果確認
    assert response.status_code == 403
    data = response.json()
    assert "detail" in data
    assert "Access denied" in data["detail"]