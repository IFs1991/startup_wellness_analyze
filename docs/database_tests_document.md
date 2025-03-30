# データベーステスト仕様書

## 概要

このドキュメントでは、スタートアップウェルネス分析システムのデータベースモジュールに対するテスト仕様と、CI/CD環境でのテスト実行方法について説明します。

## テスト対象

以下のデータベースモジュールが主要なテスト対象です：

1. データベースマネージャー (`database.py`)
2. Firestoreモデル (`models.py`)
3. SQLモデル (`models_sql.py`)
4. Firestore CRUD操作 (`crud.py`)
5. SQL CRUD操作 (`crud_sql.py`)
6. PostgreSQL接続 (`postgres.py`)
7. マイグレーション (`migration.py`)

## テスト方針

テストは以下の方針で設計されています：

1. **単体テスト**: 各データベースモジュールの機能が正しく動作することを確認します。
2. **インメモリデータベース**: テストではSQLiteインメモリデータベースを使用して高速に実行します。
3. **モック使用**: Firestoreなどの外部サービスはモックを使用してテストします。
4. **リレーションシップ**: データベースモデル間のリレーションシップが正しく動作することを確認します。
5. **カバレッジ**: 全データベースモジュールの主要な機能に対してテストカバレッジを確保します。
6. **非同期テスト**: 非同期メソッドに対しては、`pytest.mark.asyncio`を使用してテストします。
7. **エラーケース**: 正常系だけでなく、異常系（エラーケース）のテストも含めます。

## テスト項目一覧

### データベースマネージャー (`test_database.py`)

#### DatabaseManagerのテスト

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_singleton_instance` | シングルトンパターン | DatabaseManagerが単一インスタンスとして動作すること |
| `test_initialize_databases` | データベース初期化 | Firestoreと SQLエンジンが正しく初期化されること |
| `test_get_firestore_client` | Firestoreクライアント取得 | Firestoreクライアントが正しく取得できること |
| `test_get_sql_session` | SQLセッション取得 | SQLセッションが正しく取得できること |
| `test_get_db_by_type` | タイプ指定でのDB取得 | 指定したタイプのDBが正しく取得できること |
| `test_get_db_for_data_category` | カテゴリ指定でのDB取得 | データカテゴリに適したDBが選択されること |
| `test_get_collection_name` | コレクション名取得 | データカテゴリから正しいコレクション名が取得できること |

#### ユーティリティ関数のテスト

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_get_firestore_client` | グローバル関数のテスト | 関数が正しいFirestoreクライアントを返すこと |
| `test_get_db_session` | DBセッション取得関数 | 関数が正しいDBセッションを返すこと |
| `test_get_db_for_category` | カテゴリでのDB取得関数 | 関数が正しいDBを返すこと |
| `test_init_db` | DB初期化関数 | DBの初期化が正しく行われること |

### Firestoreモデル (`test_models.py`)

#### FirestoreModelのテスト

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_base_model_initialization` | モデル初期化 | FirestoreModelが正しく初期化されること |
| `test_to_dict` | 辞書変換メソッド | モデルが正しく辞書形式に変換されること |
| `test_from_dict` | 辞書からのモデル作成 | 辞書からモデルが正しく生成されること |

#### FirestoreServiceのテスト

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_init` | サービス初期化 | FirestoreServiceが正しく初期化されること |
| `test_create_document` | ドキュメント作成 | ドキュメントが正しく作成されること |
| `test_get_document` | ドキュメント取得 | ドキュメントが正しく取得されること |
| `test_get_document_not_exists` | 存在しないドキュメント | 存在しないドキュメントの取得時にNoneが返ること |
| `test_query_documents` | ドキュメントクエリ | クエリ条件に基づいてドキュメントが取得できること |

### SQLモデル (`test_models_sql.py`)

#### Userモデルのテスト

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_user_creation` | ユーザー作成 | ユーザーが正しく作成・保存されること |
| `test_user_relationships` | リレーションシップ | ユーザーとスタートアップ/ノートのリレーションが正しく動作すること |

#### Startupモデルのテスト

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_startup_creation` | スタートアップ作成 | スタートアップが正しく作成・保存されること |
| `test_startup_relationships` | リレーションシップ | スタートアップと関連モデルのリレーションが正しく動作すること |

#### VASDataモデルのテスト

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_vas_data_creation` | VASデータ作成 | VASデータが正しく作成・保存されること |
| `test_vas_data_relationships` | リレーションシップ | VASデータとスタートアップのリレーションが正しく動作すること |

### Firestore CRUD操作 (`test_crud.py`)

#### UserCrudのテスト

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_get_user` | ユーザー取得 | ユーザーIDによる取得が正しく動作すること |
| `test_get_user_not_found` | 存在しないユーザー | 存在しないユーザーの取得時にNoneが返ること |
| `test_get_user_by_username` | ユーザー名によるユーザー取得 | ユーザー名による検索が正しく動作すること |
| `test_get_users` | ユーザー一覧取得 | ユーザー一覧が正しく取得できること |
| `test_create_user` | ユーザー作成 | ユーザーが正しく作成されること |

#### StartupCrudのテスト

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_get_startup` | スタートアップ取得 | IDによるスタートアップ取得が正しく動作すること |
| `test_get_startups` | スタートアップ一覧取得 | スタートアップ一覧が正しく取得できること |
| `test_create_startup` | スタートアップ作成 | スタートアップが正しく作成されること |

#### VASCrudのテスト

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_get_vas_data` | VASデータ取得 | IDによるVASデータ取得が正しく動作すること |
| `test_get_vas_datas` | スタートアップごとのVASデータ取得 | スタートアップに紐づくVASデータが取得できること |
| `test_create_vas_data` | VASデータ作成 | VASデータが正しく作成されること |

### SQL CRUD操作 (`test_crud_sql.py`)

#### UserCrudSQLのテスト

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_get_user` | ユーザー取得 | ユーザーIDによる取得が正しく動作すること |
| `test_get_user_not_found` | 存在しないユーザー | 存在しないユーザーの取得時にNoneが返ること |
| `test_get_user_by_username` | ユーザー名によるユーザー取得 | ユーザー名による検索が正しく動作すること |
| `test_get_user_by_email` | メールアドレスによるユーザー取得 | メールアドレスによる検索が正しく動作すること |
| `test_get_users` | ユーザー一覧取得 | ユーザー一覧が正しく取得できること |
| `test_create_user` | ユーザー作成 | ユーザーが正しく作成されること |
| `test_update_user` | ユーザー更新 | ユーザー情報が正しく更新されること |
| `test_delete_user` | ユーザー削除 | ユーザーが正しく削除されること |

#### StartupCrudSQLのテスト

| テスト名 | 概要 | 検証項目 |
|---------|------|---------|
| `test_get_startup` | スタートアップ取得 | IDによるスタートアップ取得が正しく動作すること |
| `test_get_startups` | スタートアップ一覧取得 | スタートアップ一覧が正しく取得できること |
| `test_get_startups_by_owner` | オーナーごとのスタートアップ取得 | オーナーに紐づくスタートアップが取得できること |
| `test_create_startup` | スタートアップ作成 | スタートアップが正しく作成されること |

## CI/CD環境でのテスト実行

### テスト実行スクリプト

プロジェクトには専用のテスト実行スクリプト `backend/tests/run_database_tests.py` が用意されています。このスクリプトは、全てのデータベースモジュールテストを実行し、カバレッジレポートを生成します。

```bash
python backend/tests/run_database_tests.py
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
```

### GitHub Actionsによる自動テスト

プロジェクトには GitHub Actions のワークフロー設定が含まれています。この設定により、以下のタイミングで自動的にテストが実行されます：

- `main` ブランチへのプッシュ
- `develop` ブランチへのプッシュ
- `main` または `develop` ブランチに対するプルリクエスト
- 手動実行（workflow_dispatch）

## テストカバレッジ目標

- データベースモジュールの全機能に対するテスト: 100%
- コード行カバレッジ: 80%以上

## テストデータとフィクスチャ

テストでは以下のフィクスチャを使用しています：

1. `mock_firestore_client`: Firestoreクライアントのモック
2. `test_db`: テスト用のSQLiteインメモリデータベース
3. `patched_db_session`: パッチ適用済みのDBセッション
4. `patched_firestore`: パッチ適用済みのFirestoreクライアント
5. `sample_user_data`: テスト用ユーザーデータ
6. `sample_startup_data`: テスト用スタートアップデータ
7. `sample_vas_data`: テスト用VASデータ
8. `sample_financial_data`: テスト用財務データ
9. `sample_note_data`: テスト用メモデータ

これらのフィクスチャは `conftest.py` で定義されており、全てのテストで共有されます。

## モックの利用方法

テストでは、Firestoreクライアントや複雑なオブジェクトのモックに `unittest.mock` モジュールを使用しています。モックの主な使用方法は以下の通りです：

1. パッチデコレータの使用:
```python
@patch('backend.database.models.firestore.Client')
def test_function(mock_client):
    # テストコード
```

2. オブジェクトメソッドのモック:
```python
@patch.object(DatabaseManager, 'get_firestore_client')
def test_method(mock_get_firestore):
    # テストコード
```

3. 戻り値の設定:
```python
mock_client.return_value = MagicMock()
```

4. 例外の発生:
```python
mock_function.side_effect = Exception("Test error")
```

## 非同期テスト

非同期メソッドのテストには `pytest-asyncio` プラグインを使用し、テスト関数に `@pytest.mark.asyncio` デコレータを付けています。

```python
@pytest.mark.asyncio
async def test_async_function():
    # 非同期テストコード
```

## テスト追加ガイドライン

新しいデータベースモジュールを追加した場合は、以下のガイドラインに従ってテストを追加してください：

1. 対応するテストファイルを作成または更新する
2. `conftest.py` に必要なフィクスチャを追加する
3. 正常系と異常系の両方のテストケースを作成する
4. `run_database_tests.py` でカバレッジが測定されることを確認する
5. テストカバレッジが目標を達成していることを確認する

## 参考資料

- [pytest ドキュメント](https://docs.pytest.org/)
- [pytest-asyncio ドキュメント](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock ドキュメント](https://docs.python.org/3/library/unittest.mock.html)
- [SQLAlchemy テストガイド](https://docs.sqlalchemy.org/en/14/orm/session_basics.html#session-frequently-asked-questions)
- [Firebase Emulator Suite](https://firebase.google.com/docs/emulator-suite)