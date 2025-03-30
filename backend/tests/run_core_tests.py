#!/usr/bin/env python
"""
コアモジュールテスト実行スクリプト

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
    test_dir = root_dir / "backend" / "tests" / "core"

    # カバレッジレポートの設定
    cov_path = root_dir / "backend" / "core"

    # テスト対象の指定
    test_files = [
        "test_auth_manager.py",
        "test_correlation_analyzer.py",
        "test_data_preprocessor.py",
        "test_pca_analyzer.py",
        "test_pdf_report_generator.py",
        "test_security.py",
        "test_text_miner.py",
        "test_wellness_score_calculator.py"
    ]

    # テストファイルのパスを生成
    test_paths = [str(test_dir / file) for file in test_files]

    # テスト実行
    args = [
        *test_paths,  # 明示的にテストファイルを指定
        f"--cov={cov_path}",
        "--cov-report=term",
        "--cov-report=html:coverage_html_core",
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