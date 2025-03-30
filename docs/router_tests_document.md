# ルーターテスト仕様書

## 概要

このドキュメントでは、スタートアップウェルネス分析システムのバックエンドAPIルーターに対するテスト仕様と、CI/CD環境でのテスト実行方法について説明します。

## テスト対象

以下のルーターモジュールが主要なテスト対象です：

1. 認証ルーター (`auth.py`)
2. 分析ルーター (`analysis.py`)
3. データ入力ルーター (`data_input.py`)
4. データ処理ルーター (`data_processing.py`)
5. レポート生成ルーター (`report_generation.py`)
6. 視覚化ルーター (`visualization.py`)
7. 予測ルーター (`prediction.py`)

## テスト方針

テストは以下の方針で設計されています：

1. **単体テスト**: 各ルーターのエンドポイントが正しく動作することを確認します。
2. **モック使用**: 外部依存（Firestore、認証サービスなど）はモックを使用してテストします。
3. **カバレッジ**: 全ルーターのエンドポイントに対してテストカバレッジを確保します。
4. **エラーケース**: 正常系だけでなく、異常系（エラーケース）のテストも含めます。

## テスト項目一覧

### 認証ルーター (`test_auth.py`)

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_router_exists` | ルーターの存在確認 | ルーターが正しく初期化されていること |
| `test_register_user` | ユーザー登録機能 | 正しいデータでユーザーが登録できること |
| `test_login_for_access_token` | ログイン機能 | 正しい認証情報でトークンが発行されること |
| `test_login_invalid_credentials` | 不正ログイン | 不正な認証情報でエラーが発生すること |
| `test_read_users_me` | ユーザー情報取得 | 現在のユーザー情報が取得できること |
| `test_reset_password` | パスワードリセット | パスワードリセットメールが送信されること |
| `test_get_users` | ユーザー一覧取得 | 管理者権限でユーザー一覧が取得できること |

### 分析ルーター (`test_analysis.py`)

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_router_exists` | ルーターの存在確認 | ルーターが正しく初期化されていること |
| `test_perform_correlation_analysis` | 相関分析機能 | 相関分析が正しく実行されること |
| `test_perform_cluster_analysis` | クラスタリング分析機能 | クラスタリング分析が正しく実行されること |
| `test_perform_pca_analysis` | 主成分分析機能 | 主成分分析が正しく実行されること |
| `test_analysis_access_denied` | アクセス権限エラー | 権限がない場合にエラーが発生すること |

### データ入力ルーター (`test_data_input.py`)

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_router_exists` | ルーターの存在確認 | ルーターが正しく初期化されていること |
| `test_upload_data` | データアップロード機能 | データが正しくアップロードされること |
| `test_upload_csv` | CSVアップロード機能 | CSVファイルが正しくアップロードされること |
| `test_delete_data` | データ削除機能 | データが正しく削除されること |
| `test_get_data` | データ取得機能 | データが正しく取得されること |

## CI/CD環境でのテスト実行

### テスト実行スクリプト

プロジェクトには専用のテスト実行スクリプト `backend/tests/run_router_tests.py` が用意されています。このスクリプトは、全てのルーターテストを実行し、カバレッジレポートを生成します。

```bash
python backend/tests/run_router_tests.py
```

### テスト環境の準備

1. 必要なパッケージのインストール：

```bash
pip install -r backend/requirements.txt
pip install pytest pytest-cov
```

2. テスト用の環境変数の設定：

```bash
export TESTING=True
```

### GitHub Actionsによる自動テスト

プロジェクトには GitHub Actions のワークフロー設定が含まれています。この設定により、以下のタイミングで自動的にテストが実行されます：

- `main` ブランチへのプッシュ
- `develop` ブランチへのプッシュ
- `main` または `develop` ブランチに対するプルリクエスト
- 手動実行（workflow_dispatch）

## テストカバレッジ目標

- ルーターの全エンドポイントに対する機能テスト: 100%
- コード行カバレッジ: 80%以上

## モックの利用方法

テストでは以下のモックを利用しています：

1. 認証マネージャー (`AuthManager`)
2. データストレージ (`DataStorage`)
3. データプロセッサー (`CSVProcessor` など)
4. 分析モジュール (`CorrelationAnalyzer` など)

各テストファイルでは、`unittest.mock.patch` デコレータを使用して、依存オブジェクトをモック化しています。

## テスト追加ガイドライン

新しいルーターやエンドポイントを追加した場合は、以下のガイドラインに従ってテストを追加してください：

1. 対応するテストファイルを作成または更新する
2. 正常系と異常系の両方のテストケースを作成する
3. 必要なモックを適切に設定する
4. テストカバレッジが目標を達成していることを確認する

## 参考資料

- [FastAPI テストガイド](https://fastapi.tiangolo.com/tutorial/testing/)
- [Pytest ドキュメント](https://docs.pytest.org/)
- [unittest.mock ドキュメント](https://docs.python.org/3/library/unittest.mock.html)