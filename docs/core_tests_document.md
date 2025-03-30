# コアモジュールテスト仕様書

## 概要

このドキュメントでは、スタートアップウェルネス分析システムのバックエンドコアモジュールに対するテスト仕様と、CI/CD環境でのテスト実行方法について説明します。

## テスト対象

以下のコアモジュールが主要なテスト対象です：

1. 認証マネージャー (`auth_manager.py`)
2. 相関分析ツール (`correlation_analyzer.py`)
3. 主成分分析ツール (`pca_analyzer.py`)
4. テキストマイニングツール (`text_miner.py`)
5. クラスタ分析ツール (`cluster_analyzer.py`)
6. 時系列分析ツール (`time_series_analyzer.py`)
7. 生成AI管理ツール (`generative_ai_manager.py`)
8. ウェルネススコア計算ツール (`wellness_score_calculator.py`)

## テスト方針

テストは以下の方針で設計されています：

1. **単体テスト**: 各コアモジュールのメソッドが正しく動作することを確認します。
2. **モック使用**: 外部依存（Firestore、Firebase Authentication、生成AI API）はモックを使用してテストします。
3. **カバレッジ**: 全コアモジュールの主要な機能に対してテストカバレッジを確保します。
4. **非同期テスト**: 非同期メソッドに対しては、`pytest.mark.asyncio`を使用してテストします。
5. **エラーケース**: 正常系だけでなく、異常系（エラーケース）のテストも含めます。

## テスト項目一覧

### 認証マネージャー (`test_auth_manager.py`)

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_auth_manager_initialization` | 認証マネージャー初期化 | AuthManagerが正しく初期化されること |
| `test_password_hashing_verification` | パスワードハッシュ化 | パスワードが正しくハッシュ化・検証されること |
| `test_register_user` | ユーザー登録 | ユーザーが正しく登録されること |
| `test_create_access_token` | アクセストークン作成 | JWTトークンが正しく生成されること |
| `test_get_current_user` | 現在のユーザー取得 | トークンからユーザー情報が取得できること |
| `test_get_current_user_invalid_token` | 無効なトークン処理 | 無効なトークンでエラーが発生すること |
| `test_get_current_active_user` | アクティブユーザー取得 | アクティブユーザーが取得できること |
| `test_get_current_active_user_inactive` | 非アクティブユーザー処理 | 非アクティブユーザーでエラーが発生すること |

### 相関分析ツール (`test_correlation_analyzer.py`)

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_analyze` | 相関分析実行 | 相関行列が正しく計算されること |
| `test_get_analysis_history` | 分析履歴取得 | 過去の分析結果が取得できること |
| `test_get_analysis_by_id` | ID指定分析取得 | 特定の分析結果が取得できること |
| `test_update_analysis_metadata` | メタデータ更新 | 分析のメタデータが更新されること |

### 主成分分析ツール (`test_pca_analyzer.py`)

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_analyze_and_save` | PCA分析実行と保存 | 主成分分析が正しく計算・保存されること |
| `test_get_analysis_history` | 分析履歴取得 | 過去のPCA分析結果が取得できること |
| `test_error_handling` | エラー処理 | エラー発生時に正しく処理されること |
| `test_close` | リソース解放 | リソースが正しく解放されること |

### テキストマイニングツール (`test_text_miner.py`)

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_analyze_text` | テキスト分析 | テキストから正しく情報が抽出されること |
| `test_analyze_text_error_handling` | 分析エラー処理 | エラー発生時に正しく処理されること |
| `test_get_analysis_history` | 分析履歴取得 | 過去のテキスト分析結果が取得できること |
| `test_result_conversion` | 結果変換 | 分析結果が正しい形式に変換されること |

## CI/CD環境でのテスト実行

### テスト実行スクリプト

プロジェクトには専用のテスト実行スクリプト `backend/tests/run_core_tests.py` が用意されています。このスクリプトは、全てのコアモジュールテストを実行し、カバレッジレポートを生成します。

```bash
python backend/tests/run_core_tests.py
```

### テスト環境の準備

1. 必要なパッケージのインストール：

```bash
pip install -r backend/requirements.txt
pip install pytest pytest-cov pytest-asyncio
```

2. テスト用の環境変数の設定：

```bash
export TESTING=True
export FIRESTORE_EMULATOR_HOST=localhost:8080  # Firestoreエミュレータを使用する場合
```

### GitHub Actionsによる自動テスト

プロジェクトには GitHub Actions のワークフロー設定が含まれています。この設定により、以下のタイミングで自動的にテストが実行されます：

- `main` ブランチへのプッシュ
- `develop` ブランチへのプッシュ
- `main` または `develop` ブランチに対するプルリクエスト
- 手動実行（workflow_dispatch）

## テストカバレッジ目標

- コアモジュールの主要機能に対するテスト: 100%
- コード行カバレッジ: 80%以上

## モックの利用方法

テストでは以下のモックを利用しています：

1. Firebase Authentication (`firebase_admin.auth`)
2. Firestore クライアント (`google.cloud.firestore.Client`)
3. 生成AI API (`google.generativeai`)
4. scikit-learn モジュール（`sklearn.decomposition.PCA`など）

各テストファイルでは、`unittest.mock.patch`デコレータを使用して、外部依存をモック化しています。これにより、外部サービスに依存せずにテストを実行できます。

## 非同期テスト

多くのコアモジュールは非同期メソッド（`async def`）を使用しています。非同期メソッドのテストには、`pytest-asyncio`プラグインを使用し、テスト関数に`@pytest.mark.asyncio`デコレータを付けています。

同期的な関数から非同期メソッドを呼び出す必要がある場合は、補助関数`run_async`を使用してイベントループで実行します：

```python
def run_async(coroutine):
    return asyncio.get_event_loop().run_until_complete(coroutine)
```

## フィクスチャの利用方法

テストでは、以下のようなフィクスチャを使用して、テストデータやモックオブジェクトを提供しています：

1. `sample_dataframe`: テスト用の pandas DataFrame
2. `sample_text_data`: テキスト分析用のサンプルテキスト
3. `sample_time_series_data`: 時系列分析用のサンプルデータ
4. `mock_firebase_app`: Firebase アプリケーションのモック
5. `mock_firestore_client`: Firestore クライアントのモック
6. `mock_auth_client`: Firebase 認証クライアントのモック

これらのフィクスチャは`conftest.py`で定義されており、全てのテストで共有されます。

## テスト追加ガイドライン

新しいコアモジュールを追加した場合は、以下のガイドラインに従ってテストを追加してください：

1. 対応するテストファイルを作成または更新する（例: `module_name.py` → `test_module_name.py`）
2. 正常系と異常系の両方のテストケースを作成する
3. 必要なモックを適切に設定する
4. 非同期メソッドには`@pytest.mark.asyncio`デコレータを使用する
5. テストカバレッジが目標を達成していることを確認する

## 参考資料

- [pytest ドキュメント](https://docs.pytest.org/)
- [pytest-asyncio ドキュメント](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock ドキュメント](https://docs.python.org/3/library/unittest.mock.html)
- [Firestore エミュレータ ドキュメント](https://firebase.google.com/docs/firestore/security/test-rules-emulator)