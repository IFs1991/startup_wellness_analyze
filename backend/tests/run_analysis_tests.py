#!/usr/bin/env python
"""
分析モジュールテスト実行スクリプト

CI/CD環境でのテスト実行、カバレッジレポート生成のためのスクリプトです。
"""
import pytest
import sys
import os
from pathlib import Path

def run_tests():
    """テストを実行し、カバレッジレポートを生成します"""
    # ルートディレクトリの設定
    root_dir = Path(__file__).parent.parent.parent

    # テストディレクトリの設定
    test_dir = root_dir / "backend" / "tests" / "analysis"

    # カバレッジレポートの設定
    cov_path = root_dir / "backend" / "analysis"

    # テスト対象の指定
    test_files = [
        "test_causal_inference_analyzer.py",
        "test_cluster_analyzer.py",
        "test_portfolio_network_analyzer.py",
        "test_bayesian_inference_analyzer.py",
        "test_team_analyzer.py",
        "test_financial_analyzer.py",
        "test_startup_survivability_analyzer.py",
        "test_predictive_model_analyzer.py",
        "test_monte_carlo_simulator.py",
        "test_knowledge_transfer_index_calculator.py"
    ]

    # テストファイルのパスを生成
    test_paths = [str(test_dir / file) for file in test_files]

    # テスト実行
    args = [
        *test_paths,  # 明示的にテストファイルを指定
        f"--cov={cov_path}",
        "--cov-report=term",
        "--cov-report=html:coverage_html_analysis",
        "-v"
    ]

    # 環境変数の設定（テスト環境であることを示す）
    os.environ["TESTING"] = "True"

    # テスト実行
    exit_code = pytest.main(args)

    # 終了コードの設定
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests()