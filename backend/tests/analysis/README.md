# 分析モジュールテスト

このディレクトリには、バックエンドの分析モジュールに対するテストが含まれています。

## テスト構成

テストは以下のファイルで構成されています：

- `test_causal_inference_analyzer.py` - 因果推論分析機能のテスト
- `test_cluster_analyzer.py` - クラスタ分析機能のテスト
- `test_portfolio_network_analyzer.py` - ポートフォリオネットワーク分析機能のテスト
- `test_bayesian_inference_analyzer.py` - ベイジアン推論分析機能のテスト
- `test_team_analyzer.py` - チーム分析機能のテスト
- `test_financial_analyzer.py` - 財務分析機能のテスト
- `test_startup_survivability_analyzer.py` - スタートアップ生存分析機能のテスト
- `test_predictive_model_analyzer.py` - 予測モデル分析機能のテスト
- `test_monte_carlo_simulator.py` - モンテカルロシミュレーション機能のテスト
- `test_knowledge_transfer_index_calculator.py` - 知識移転指数計算機能のテスト

## テスト実行方法

### 全てのテストを実行

プロジェクトのルートディレクトリから以下のコマンドを実行します：

```bash
python -m backend.tests.run_analysis_tests
```

### 特定のテストファイルを実行

特定のテストファイルのみを実行する場合は、以下のコマンドを使用します：

```bash
python -m backend.tests.run_analysis_tests --files test_financial_analyzer.py test_monte_carlo_simulator.py
```

### カバレッジレポートの生成

カバレッジレポートを生成するには、以下のオプションを追加します：

```bash
python -m backend.tests.run_analysis_tests --coverage --html-report
```

## テスト環境の設定

テストを実行する前に、以下の環境変数を設定することをお勧めします：

```bash
export TESTING=True
export PYTHONPATH=/path/to/project/root
```

Windows環境では：

```powershell
$env:TESTING = "True"
$env:PYTHONPATH = "C:\path\to\project\root"
```

## モックの使用

テストでは、外部依存関係（BigQuery、Firestore、機械学習モデルなど）をモック化しています。モックの設定は `conftest.py` ファイルで行われています。

## テスト作成のガイドライン

新しいテストを作成する際は、以下のガイドラインに従ってください：

1. 各テストは単一の機能や関数をテストすること
2. 適切なモックを使用して外部依存関係を分離すること
3. 非同期関数のテストには `@pytest.mark.asyncio` デコレータを使用すること
4. テストデータは `conftest.py` のフィクスチャを活用すること
5. 各テストケースには明確な説明コメントを追加すること

## 注意事項

- テスト実行時にインポートエラーが発生する場合は、`__init__.py` ファイルでモジュールのモック化が適切に行われているか確認してください。
- 非同期テストの実行には `pytest-asyncio` プラグインが必要です。
- カバレッジレポートの生成には `pytest-cov` プラグインが必要です。