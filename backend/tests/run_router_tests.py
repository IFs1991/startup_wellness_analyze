#!/usr/bin/env python
"""
ルーターテスト実行スクリプト

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
    test_dir = root_dir / "backend" / "tests" / "routers"

    # カバレッジレポートの設定
    cov_path = root_dir / "backend" / "api" / "routers"

    # テスト実行
    args = [
        str(test_dir),
        f"--cov={cov_path}",
        "--cov-report=term",
        "--cov-report=html:coverage_html",
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