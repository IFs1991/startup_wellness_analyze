# データベースモジュールアーキテクチャ

## 概要

Startup Wellness Analyzeプロジェクトのデータベースモジュールは、複数のデータベースエンジン（Firestore、PostgreSQL、Neo4j）を使用したハイブリッドデータベース構成を採用しています。これにより、各データ種別に最適なデータベースを選択しつつ、アプリケーション側からは統一されたAPIでアクセスすることが可能になっています。

## アーキテクチャの構成要素

データベースモジュールは次の主要な構成要素から成り立っています：

```
backend/database/
  ├── __init__.py       - モジュール定義
  ├── config.py         - データベース設定
  ├── connection.py     - 接続管理
  ├── repository.py     - リポジトリ抽象インターフェース
  ├── repositories/     - リポジトリパターン実装
  │   ├── __init__.py
  │   ├── factory.py    - リポジトリファクトリ
  │   ├── firestore.py  - Firestore実装
  │   ├── sql.py        - PostgreSQL実装
  │   └── neo4j.py      - Neo4j実装
  ├── models/           - エンティティモデル
  │   ├── __init__.py
  │   ├── base.py       - 基底モデル
  │   ├── entities.py   - エンティティ定義
  │   └── adapters.py   - アダプター実装
  └── docs/             - ドキュメント
```

## リポジトリパターン

データベースアクセスには「リポジトリパターン」を採用しています。このパターンの主な特徴：

1. **データアクセスの抽象化**: 具体的なデータベース実装の詳細をアプリケーションコードから隠蔽
2. **共通インターフェース**: 異なるデータベースでも同じインターフェースでアクセス
3. **依存性の逆転**: ビジネスロジックがデータベース実装に依存しない構造
4. **テスト容易性**: モックやスタブによるテストが容易

## データモデル

エンティティモデルは`BaseEntity`を基底クラスとして定義され、各データベース固有の実装詳細はアダプターによって抽象化されます。

```python
class BaseEntity:
    # 共通フィールドと機能
    id: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
```

## データカテゴリと適切なデータベースの選択

データは以下のカテゴリに分類され、それぞれに最適なデータベースが割り当てられます：

- **構造化データ（PostgreSQL）**:
  - USER_MASTER, COMPANY_MASTER, FINANCIAL, BILLING, AUDIT_LOG

- **スケーラブルデータ（Firestore）**:
  - REALTIME, CHAT, USER_SESSION, SURVEY, TREATMENT, REPORT

- **グラフデータ（Neo4j）**:
  - GRAPH, RELATIONSHIP, NETWORK, PATH

## 実装詳細

### リポジトリインターフェース

`Repository`抽象クラスは次のメソッドを定義します：

```python
class Repository(Generic[T, ID], ABC):
    @abstractmethod
    def find_by_id(self, id: ID) -> Optional[T]: ...

    @abstractmethod
    def find_all(self, skip: int = 0, limit: int = 100) -> List[T]: ...

    @abstractmethod
    def find_by_criteria(self, criteria: Dict[str, Any]) -> List[T]: ...

    @abstractmethod
    def save(self, entity: T) -> T: ...

    @abstractmethod
    def update(self, id: ID, data: Dict[str, Any]) -> T: ...

    @abstractmethod
    def delete(self, id: ID) -> bool: ...

    @abstractmethod
    def count(self, criteria: Optional[Dict[str, Any]] = None) -> int: ...
```

### リポジトリファクトリ

`RepositoryFactory`はエンティティタイプに応じた適切なリポジトリインスタンスを提供します：

```python
class RepositoryFactory:
    @staticmethod
    def get_repository(entity_type: Type[T], data_category: Optional[DataCategory] = None) -> Repository[T, Any]: ...
```

これにより、アプリケーションコードは具体的なリポジトリ実装を意識せずに適切なリポジトリを取得できます。

## エラー処理

リポジトリ操作の例外は階層的に定義されています：

```
RepositoryException
  ├── EntityNotFoundException
  ├── EntityDuplicateException
  └── ValidationException
```

## トランザクション管理

PostgreSQLなどRDBMSでは、SQLAlchemyのセッションを利用してトランザクション管理を行います。分散トランザクションについては、一貫性を保証するためのパターンを適用しています。

## パフォーマンスと最適化

各データベース実装では、効率的なクエリ実行とキャッシュ戦略を採用しています。特にFirestoreでは、過度な読み書きを避けるためのバッチ処理とクエリ設計を注意深く行っています。