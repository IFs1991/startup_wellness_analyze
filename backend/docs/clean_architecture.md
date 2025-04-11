# クリーンアーキテクチャ実装ドキュメント

## 概要

このドキュメントでは、スタートアップウェルネス分析プラットフォームに実装されたクリーンアーキテクチャについて説明します。このアーキテクチャ改善は、コードの保守性、テスト容易性、拡張性を向上させるために実施されました。

## アーキテクチャの層

クリーンアーキテクチャでは、アプリケーションを以下の4つの層に分割しています：

1. **ドメイン層** (Domain)
   - ビジネスロジックの中心となるエンティティと値オブジェクト
   - 外部依存のないビジネスルール
   - 場所: `backend/domain/models/`

2. **ユースケース層** (Use Cases)
   - アプリケーション固有のビジネスルール
   - ドメインオブジェクトを操作するビジネスロジック
   - 場所: `backend/usecases/`

3. **インターフェースアダプター層** (Interface Adapters)
   - 外部のフレームワークやデータベースとの連携
   - リポジトリの実装、APIコントローラーなど
   - 場所: `backend/infrastructure/`, `backend/api/`

4. **フレームワーク・ドライバー層** (Frameworks & Drivers)
   - Webフレームワーク、データベース、UI、外部APIなど
   - サードパーティフレームワークとの統合
   - 場所: 外部ライブラリ、`backend/infrastructure/`の一部

## 依存関係の方向

クリーンアーキテクチャの重要な点は、依存関係の方向です：

```
ドメイン層 ← ユースケース層 ← インターフェースアダプター層 ← フレームワーク・ドライバー層
```

- 内側の層は外側の層に依存してはいけません
- 外側の層は内側の層に依存します
- 依存性逆転の原則（Dependency Inversion Principle）を適用することで、内側の層が外側の層のインターフェースを定義し、外側の層がそれを実装します

## 主要コンポーネント

### ドメインモデル

ドメインモデルはビジネスロジックの中心となるエンティティと値オブジェクトを定義します。

例：

```python
@dataclass
class User:
    """ユーザーエンティティ"""
    id: str
    email: str
    display_name: Optional[str] = None
    role: UserRole = UserRole.USER
    # ...
```

### リポジトリインターフェース

データアクセスを抽象化するためのインターフェースです。ドメイン層で定義され、インフラストラクチャ層で実装されます。

例：

```python
class UserRepositoryInterface(ABC):
    """
    ユーザーリポジトリインターフェース
    """
    @abstractmethod
    async def get_by_id(self, user_id: str) -> Optional[User]:
        pass
    # ...
```

### ユースケース

アプリケーション固有のビジネスロジックを実装します。

例：

```python
class AuthUseCase:
    """
    認証ユースケース
    """
    def __init__(self, user_repository: UserRepositoryInterface, ...):
        self.user_repository = user_repository
        # ...

    async def authenticate(self, email: str, password: str) -> Tuple[User, List[str]]:
        # 認証ロジックの実装
        # ...
```

### リポジトリ実装

リポジトリインターフェースの具体的な実装です。インフラストラクチャ層に属します。

例：

```python
class FirebaseWellnessRepository(WellnessRepositoryInterface):
    """
    Firebase/Firestoreを使用したウェルネスリポジトリの実装
    """
    def __init__(self, firebase_client: FirebaseClientInterface):
        self.firebase_client = firebase_client
        # ...

    async def get_score_by_id(self, score_id: str) -> Optional[WellnessScore]:
        # Firestoreからデータを取得する実装
        # ...
```

## 依存性注入（DI）

依存性注入を使用して、コンポーネント間の結合度を低減し、テスト容易性を向上させています。

例：

```python
# DIコンテナの設定
container = get_container()
container.register(FirebaseClientInterface, FirebaseClient, singleton=True)
container.register(WellnessRepositoryInterface, FirebaseWellnessRepository, singleton=True)

# 依存性の解決
firebase_client = container.get(FirebaseClientInterface)
wellness_repository = container.get(WellnessRepositoryInterface)

# デコレータを使用した依存性の注入
@inject(WellnessRepositoryInterface)
def get_wellness_score_usecase(wellness_repository: WellnessRepositoryInterface) -> WellnessScoreUseCase:
    return WellnessScoreUseCase(wellness_repository)
```

## テスト戦略

クリーンアーキテクチャを採用することで、効果的なテストが可能になります：

1. **ユニットテスト**
   - ドメインモデルとビジネスロジックの単体テスト
   - 外部依存なしでテスト可能

2. **インテグレーションテスト**
   - リポジトリの実装と外部システムの連携テスト
   - モックの使用で依存を制御

3. **エンドツーエンドテスト**
   - APIエンドポイントなど、全体フローのテスト

## 利点

クリーンアーキテクチャの主な利点：

1. **関心の分離**：各層が明確な責務を持ち、コードが整理されます
2. **テスト容易性**：外部依存をモックに置き換えて、ビジネスロジックを単体でテストできます
3. **拡張性**：新機能の追加や既存機能の変更が容易になります
4. **技術的柔軟性**：外部フレームワークやデータベースを変更しても、コアビジネスロジックは影響を受けません
5. **長期的保守性**：アーキテクチャが明確なため、長期的な保守が容易になります

## 今後の展開

1. ユーザーリポジトリの実装
2. APIコントローラーのクリーンアーキテクチャへの統合
3. テスト自動化の強化
4. ドキュメントの充実