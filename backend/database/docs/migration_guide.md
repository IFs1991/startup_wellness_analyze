# リポジトリパターンへの移行ガイド

このガイドは、既存のコードを古いCRUDモジュール（`crud.py`、`crud_sql.py`）から新しいリポジトリパターンへ移行する方法を説明します。

## 移行の基本方針

1. **段階的移行**: 一度にすべてを変更するのではなく、コンポーネントごとに段階的に移行
2. **下位互換性の維持**: 移行期間中は古いAPIと新しいAPIを共存させる
3. **テスト駆動**: 移行作業は常にテストで検証する

## 移行手順

### 1. 依存関係の確認

まず、古いCRUDモジュールへの依存関係を確認します。

```bash
# grepなどを使って依存関係を検索
grep -r "import .*crud" --include="*.py" backend/
grep -r "from .*crud import" --include="*.py" backend/
```

### 2. 移行ヘルパーの使用

移行の第一段階として、移行ヘルパーを使用して古い呼び出しを新しいパターンにブリッジします。

```python
# 移行前（古いCRUDモジュール）
from backend.database import crud

user = crud.get_user("user123")
```

```python
# 移行中（移行ヘルパーを使用）
from backend.database import MigrationHelper

user = MigrationHelper.get_user("user123")
```

### 3. リポジトリパターンへの完全移行

最終的には、移行ヘルパーではなく直接リポジトリパターンを使用するように変更します。

```python
# 移行後（リポジトリパターン）
from backend.database import repository_factory
from backend.database.models import UserEntity

user_repo = repository_factory.get_repository(UserEntity)
user = user_repo.find_by_id("user123")
```

## 具体的な変換例

### エンティティの取得（find_by_id）

```python
# 移行前：
user = crud.get_user("user123")

# 移行後：
user_repo = repository_factory.get_repository(UserEntity)
user = user_repo.find_by_id("user123")
```

### 条件によるエンティティの検索（find_by_criteria）

```python
# 移行前：
users = crud.get_users_by_criteria({"is_active": True})

# 移行後：
user_repo = repository_factory.get_repository(UserEntity)
users = user_repo.find_by_criteria({"is_active": True})
```

### エンティティの作成（save）

```python
# 移行前：
user_data = {
    "email": "user@example.com",
    "display_name": "テストユーザー",
    "is_active": True
}
created_user = crud.create_user(user_data)

# 移行後：
user = UserEntity(
    email="user@example.com",
    display_name="テストユーザー",
    is_active=True
)
user_repo = repository_factory.get_repository(UserEntity)
created_user = user_repo.save(user)
```

### エンティティの更新（update）

```python
# 移行前：
updated_user = crud.update_user("user123", {"display_name": "更新済み"})

# 移行後：
user_repo = repository_factory.get_repository(UserEntity)
updated_user = user_repo.update("user123", {"display_name": "更新済み"})
```

### エンティティの削除（delete）

```python
# 移行前：
success = crud.delete_user("user123")

# 移行後：
user_repo = repository_factory.get_repository(UserEntity)
success = user_repo.delete("user123")
```

## データモデルの変換

古いデータモデル（辞書型）から新しいエンティティモデルへの変換：

```python
# 古いスタイル（辞書型）
user_data = {
    "id": "user123",
    "email": "user@example.com",
    "display_name": "テストユーザー",
    "is_active": True
}

# 新しいスタイル（エンティティオブジェクト）
user = UserEntity(
    id="user123",
    email="user@example.com",
    display_name="テストユーザー",
    is_active=True
)

# 辞書からエンティティへの変換
user = UserEntity(**user_data)

# エンティティから辞書への変換
user_dict = user.dict()
```

## 共通の移行パターン

### サービスレイヤーでの移行

サービスクラス内での移行例：

```python
# 移行前
class UserService:
    def get_user_profile(self, user_id: str):
        user = crud.get_user(user_id)
        return user

# 移行中（コンストラクタでリポジトリを取得）
class UserService:
    def __init__(self):
        self.user_repo = repository_factory.get_repository(UserEntity)

    def get_user_profile(self, user_id: str):
        user = self.user_repo.find_by_id(user_id)
        return user
```

### 依存性注入を活用した移行

テスト容易性を高めるために、依存性注入パターンを使用した移行：

```python
# 依存性注入を活用した実装
class UserService:
    def __init__(self, user_repo=None):
        # リポジトリが外部から注入されない場合はデフォルトを使用
        self.user_repo = user_repo or repository_factory.get_repository(UserEntity)

    def get_user_profile(self, user_id: str):
        user = self.user_repo.find_by_id(user_id)
        return user
```

## 移行後のテスト

移行後のコードは、必ずテストで検証します：

```python
def test_migrated_service():
    # モックリポジトリの作成
    mock_repo = MagicMock()
    mock_user = UserEntity(id="test123", email="test@example.com")
    mock_repo.find_by_id.return_value = mock_user

    # モックを注入したサービスの作成
    service = UserService(user_repo=mock_repo)

    # メソッド呼び出し
    profile = service.get_user_profile("test123")

    # 検証
    mock_repo.find_by_id.assert_called_once_with("test123")
    assert profile.email == "test@example.com"
```

## 特殊ケースの移行

### 複雑なクエリ

複雑なクエリは、リポジトリの拡張メソッドとして実装します：

```python
# カスタムリポジトリの実装
class CustomUserRepository(SQLRepository):
    def __init__(self, session, orm_model_class, entity_class):
        super().__init__(session, orm_model_class, entity_class)

    def find_users_with_recent_login(self, days=30):
        """最近ログインしたユーザーを検索する特殊なクエリ"""
        # リポジトリ固有の実装...
        pass

# カスタムリポジトリの使用
custom_repo = CustomUserRepository(db_session, UserORMModel, UserEntity)
recent_users = custom_repo.find_users_with_recent_login(days=7)
```

### 複数データベースをまたがるトランザクション

複数のデータベースをまたがる操作は、サービスレイヤーでの調整が必要です：

```python
def create_user_with_related_data(user_data, startup_data):
    # 各リポジトリを取得
    user_repo = repository_factory.get_repository(UserEntity, DataCategory.USER_MASTER)
    startup_repo = repository_factory.get_repository(StartupEntity, DataCategory.REALTIME)

    try:
        # ユーザー作成（PostgreSQL）
        user = UserEntity(**user_data)
        saved_user = user_repo.save(user)

        try:
            # スタートアップ作成（Firestore）
            startup_data["owner_id"] = saved_user.id
            startup = StartupEntity(**startup_data)
            saved_startup = startup_repo.save(startup)

            return saved_user, saved_startup
        except Exception as e:
            # Firestoreでのエラー発生時、PostgreSQLのデータも削除（補償トランザクション）
            user_repo.delete(saved_user.id)
            raise e
    except Exception as e:
        # 最初のステップでのエラー
        raise e
```

## 移行計画の例

以下は、段階的な移行計画の例です：

1. **準備段階**:
   - 依存関係の確認とドキュメント化
   - テスト計画の策定

2. **第1段階（低リスクコンポーネント）**:
   - 内部的な使用のみのモジュールの移行
   - 単体テストの実装

3. **第2段階（中リスクコンポーネント）**:
   - 一部のエンドポイントで使用されているモジュールの移行
   - 統合テストの実装

4. **第3段階（高リスクコンポーネント）**:
   - 主要なアプリケーションロジックの移行
   - エンドツーエンドテストの実施

5. **最終段階**:
   - 旧モジュールの削除
   - パフォーマンステストと最終確認