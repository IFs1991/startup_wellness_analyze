# 依存性注入（DI）システム

## 概要

このドキュメントでは、スタートアップウェルネス分析プラットフォームで実装されている依存性注入（DI）システムについて説明します。DIシステムは、コンポーネント間の結合度を低減し、テスト容易性を向上させるために実装されています。

## 基本原則

依存性注入は以下の原則に基づいています：

1. **依存関係の明示化**: クラスやモジュールが必要とする依存関係を明示的に宣言する
2. **外部からの依存性の提供**: 依存関係を外部から注入することで、モジュールの結合度を低減する
3. **インターフェースへの依存**: 具体的な実装ではなく抽象インターフェースに依存する

## DIコンテナ

システムのDIコンテナは `core/di.py` モジュールで実装されており、以下の機能を提供しています：

### 1. 依存関係の登録

```python
# インターフェースと実装の登録
container.register(
    FirebaseClientInterface,  # インターフェース
    FirebaseClient,          # 実装
    singleton=True           # シングルトンとして管理
)

# ファクトリ関数による登録
container.register_factory(
    WellnessRepositoryInterface,
    lambda: FirebaseWellnessRepository(get_firebase_client()),
    singleton=True
)

# インスタンスの直接登録
redis_client = RedisClient()
container.register_instance(RedisClientInterface, redis_client)
```

### 2. 依存関係の解決

```python
# インターフェースからインスタンスを取得
firebase_client = container.get(FirebaseClientInterface)
wellness_repository = container.get(WellnessRepositoryInterface)
```

### 3. デコレータによる依存性の注入

```python
# 関数への注入
@inject(WellnessRepositoryInterface)
def process_wellness_data(wellness_repository, user_id: str):
    # wellness_repositoryが自動的に注入される
    return wellness_repository.get_data(user_id)

# クラスメソッドへの注入
class WellnessService:
    @inject(WellnessRepositoryInterface)
    def calculate_score(self, wellness_repository, user_data):
        # wellness_repositoryが自動的に注入される
        return wellness_repository.calculate_score(user_data)
```

## DIコンテナの設定

システム起動時に `core/di_config.py` モジュールによってDIコンテナが設定されます：

```python
def setup_di_container(use_mock: bool = False) -> None:
    """DIコンテナのセットアップ"""
    container = get_container()

    # 環境変数に基づいて設定
    environment = os.environ.get("ENVIRONMENT", "development")

    # Firebaseクライアントの登録
    container.register_factory(
        FirebaseClientInterface,
        lambda: get_firebase_client(use_mock),
        singleton=True
    )

    # その他のコンポーネント登録...
```

## レガシーコードとの互換性

レガシーコードをサポートするために、一連のヘルパー関数が提供されています：

```python
def get_wellness_repository() -> WellnessRepositoryInterface:
    """
    レガシーコードとの互換性のためのヘルパー関数
    """
    container = get_container()
    return container.get(WellnessRepositoryInterface)

def get_auth_manager_from_di() -> AuthManagerInterface:
    """
    レガシーコードとの互換性のためのヘルパー関数
    """
    container = get_container()
    return container.get(AuthManagerInterface)
```

## テスト時の依存関係のモック化

テスト実行時には、モックオブジェクトを簡単に注入できます：

```python
def test_wellness_calculation():
    # DIコンテナの取得
    container = get_container()

    # モックリポジトリの作成
    mock_repository = MockWellnessRepository()

    # モックをDIコンテナに登録
    container.register_instance(WellnessRepositoryInterface, mock_repository)

    # テスト対象オブジェクトの作成
    wellness_service = WellnessService()

    # テスト実行
    result = wellness_service.calculate_score(test_data)

    # 検証
    assert result == expected_result
    mock_repository.calculate_score.assert_called_once_with(test_data)
```

## ベストプラクティス

1. **インターフェースの定義**: クラスに依存する前に、まずインターフェース（プロトコル）を定義する
2. **単一責任の原則**: 各クラスに明確な責任を持たせ、依存関係を最小限に保つ
3. **コンストラクタ注入**: メソッド注入よりもコンストラクタ注入を優先する
4. **DIコンテナの一元管理**: DIコンテナの設定は一箇所で管理する
5. **テスト容易性**: テスト時に依存関係を簡単にモック化できるように設計する

## DIシステムの恩恵

1. **結合度の低減**: モジュール間の結合度が低減され、変更の影響範囲が限定される
2. **テスト容易性の向上**: 依存関係をモックに置き換えることで単体テストが容易になる
3. **柔軟性の向上**: 実装の詳細を変更せずにコンポーネントを置き換えることが可能
4. **並行開発の促進**: インターフェースが定義されていれば、複数の開発者が同時に異なるコンポーネントを開発できる
5. **コード品質の向上**: 明示的な依存関係により、責任の所在が明確になる
