# データベースモジュール使用ガイド

このガイドでは、リファクタリング後のデータベースモジュールの使用方法を説明します。

## 基本的な使用方法

### 1. リポジトリの取得

```python
from backend.database import repository_factory
from backend.database.models import UserEntity
from backend.database.repository import DataCategory

# ユーザーリポジトリの取得（カテゴリ指定）
user_repo = repository_factory.get_repository(UserEntity, DataCategory.USER_MASTER)

# または、エンティティクラスのみ指定（自動的に適切なカテゴリが選択される）
user_repo = repository_factory.get_repository(UserEntity)
```

### 2. 基本的なCRUD操作

#### データの取得

```python
# IDによる取得
user = user_repo.find_by_id("user123")

# 全件取得（ページネーション付き）
users = user_repo.find_all(skip=0, limit=10)

# 条件による検索
active_users = user_repo.find_by_criteria({"is_active": True})
```

#### データの作成

```python
from backend.database.models import UserEntity

# 新規エンティティの作成
new_user = UserEntity(
    email="user@example.com",
    display_name="テストユーザー",
    is_active=True
)

# 保存
saved_user = user_repo.save(new_user)
```

#### データの更新

```python
# 部分更新
updated_user = user_repo.update("user123", {"display_name": "更新済みユーザー"})

# または、エンティティを取得して変更してから保存
user = user_repo.find_by_id("user123")
if user:
    user.is_active = False
    user_repo.save(user)
```

#### データの削除

```python
# ID指定での削除
success = user_repo.delete("user123")
```

## 高度な使用方法

### 1. 複雑な検索条件

```python
# AND条件
users = user_repo.find_by_criteria({
    "is_active": True,
    "email_domain": "example.com"
})

# 比較演算子
financial_data = financial_repo.find_by_criteria({
    "year": 2023,
    "revenue__gt": 1000000  # 収益が100万円超
})
```

### 2. リレーションシップの扱い

```python
# ユーザーに関連するスタートアップを取得
user = user_repo.find_by_id("user123")
if user:
    startup_repo = repository_factory.get_repository(StartupEntity)
    startups = startup_repo.find_by_criteria({"owner_id": user.id})
```

### 3. トランザクション

PostgreSQLのトランザクション：

```python
from backend.database import get_db
from backend.database.repositories import SQLRepository

def create_user_and_startup(user_data, startup_data):
    with get_db() as db:
        # トランザクション開始
        user_repo = SQLRepository(db, UserORMModel, UserEntity)
        startup_repo = SQLRepository(db, StartupORMModel, StartupEntity)

        try:
            # ユーザー作成
            user = UserEntity(**user_data)
            saved_user = user_repo.save(user)

            # スタートアップ作成
            startup_data["owner_id"] = saved_user.id
            startup = StartupEntity(**startup_data)
            saved_startup = startup_repo.save(startup)

            # コミット（明示的に行う必要はない - with句の終了時に自動的にコミットされる）
            return saved_user, saved_startup
        except Exception as e:
            # エラー発生時はロールバック（with句の終了時に自動的にロールバックされる）
            raise e
```

### 4. 複数データベースをまたがる操作

```python
def register_user_with_startup(user_data, startup_data):
    # ユーザー作成（PostgreSQL）
    user_repo = repository_factory.get_repository(UserEntity, DataCategory.USER_MASTER)
    user = UserEntity(**user_data)
    saved_user = user_repo.save(user)

    # スタートアップ作成（Firestore）
    startup_repo = repository_factory.get_repository(StartupEntity, DataCategory.REALTIME)
    startup_data["owner_id"] = saved_user.id
    startup = StartupEntity(**startup_data)
    saved_startup = startup_repo.save(startup)

    # ユーザーとスタートアップの関係を登録（Neo4j）
    relationship_repo = repository_factory.get_repository(RelationshipEntity, DataCategory.RELATIONSHIP)
    relationship = RelationshipEntity(
        from_id=saved_user.id,
        to_id=saved_startup.id,
        type="OWNS"
    )
    saved_relationship = relationship_repo.save(relationship)

    return saved_user, saved_startup, saved_relationship
```

## エラー処理

リポジトリ操作における例外処理の例：

```python
from backend.database.repository import EntityNotFoundException, ValidationException, RepositoryException

try:
    user = user_repo.find_by_id("non_existent_id")
    if user is None:
        # 結果がNoneの場合の処理
        pass
except EntityNotFoundException:
    # エンティティが見つからない場合の処理
    pass
except ValidationException as e:
    # バリデーションエラーの処理
    print(f"バリデーションエラー: {str(e)}")
except RepositoryException as e:
    # その他のリポジトリエラーの処理
    print(f"リポジトリエラー: {str(e)}")
```

## 移行に関する注意点

古いCRUDモジュールを使用しているコードは、以下のように新しいリポジトリパターンに移行できます：

### 移行前（非推奨）
```python
from backend.database import crud

# ユーザー取得
user = crud.get_user("user123")

# ユーザー作成
user_data = {"email": "new@example.com", "display_name": "新規ユーザー"}
created_user = crud.create_user(user_data)
```

### 移行後（推奨）
```python
from backend.database import repository_factory
from backend.database.models import UserEntity

# リポジトリ取得
user_repo = repository_factory.get_repository(UserEntity)

# ユーザー取得
user = user_repo.find_by_id("user123")

# ユーザー作成
new_user = UserEntity(email="new@example.com", display_name="新規ユーザー")
created_user = user_repo.save(new_user)
```

### 移行ヘルパーの使用（一時的な対応）
```python
from backend.database import MigrationHelper

# 古いAPIと同じ形式で呼び出せる移行ヘルパー
user = MigrationHelper.get_user("user123")
```

## テスト

リポジトリのモックを使用したテストの例：

```python
from unittest.mock import MagicMock
import pytest

def test_user_service(mocker):
    # リポジトリのモック
    mock_user_repo = MagicMock()
    mocker.patch(
        'backend.database.repositories.factory.ConcreteRepositoryFactory.get_repository',
        return_value=mock_user_repo
    )

    # モックの振る舞いを定義
    mock_user = UserEntity(id="test123", email="test@example.com")
    mock_user_repo.find_by_id.return_value = mock_user

    # テスト対象のサービス
    from backend.services.user_service import get_user_profile

    # サービスメソッドを呼び出し
    profile = get_user_profile("test123")

    # アサーション
    mock_user_repo.find_by_id.assert_called_once_with("test123")
    assert profile.email == "test@example.com"
```

## パフォーマンスに関する注意点

1. **N+1問題の回避**: 関連データを取得する際に、ループ内でリポジトリメソッドを呼び出すのは避け、一括で条件を指定して取得する

2. **大量データの処理**: 大量のレコードを扱う場合は、ページネーションを活用し、一度に処理するデータ量を制限する

3. **Firestoreの特性理解**: Firestoreはクエリ能力に制限があるため、複雑な検索条件は適切に設計されたインデックスが必要

4. **キャッシュの活用**: 頻繁にアクセスされるデータはキャッシュを検討する