# データベースアーキテクチャ - PostgreSQLとFirestoreの役割分担

## 概要

スタートアップウェルネス分析プラットフォームでは、PostgreSQLとFirestoreのハイブリッドデータベースアーキテクチャを採用しています。この文書では、それぞれのデータベースが担当する機能と適したデータ種別について説明します。

## PostgreSQLに適した機能とデータ

PostgreSQLは主に**構造化された時系列データ**と**分析処理に必要なデータ**の保存に使用します。

### 1. コアモジュールの対応機能

#### VASデータ管理 (`google_forms_vas_collector.py`)
- **対象データ**: VASスケールの回答データ
- **理由**:
  - 高頻度で収集される時系列データ
  - 複雑なクエリと集計が必要
  - 大量のデータ行が発生する

#### 財務データ管理
- **対象データ**: 財務諸表、経営指標
- **理由**:
  - 構造化された数値データ
  - 時系列での推移分析
  - 複雑な計算と集計処理

#### ウェルネススコア計算 (`wellness_score_calculator.py`)
- **対象データ**:
  - スコア計算履歴
  - カテゴリ別スコア推移
  - 調整係数の履歴
- **理由**:
  - 時系列分析の必要性
  - 大量の数値データの処理
  - 集計クエリのパフォーマンス要件

#### 相関分析 (`correlation_analyzer.py`)
- **対象データ**:
  - 分析結果データ
  - 相関係数
  - p値など統計指標
- **理由**:
  - 複雑な計算処理
  - 大量のデータセットの処理

#### 時系列分析 (`time_series_analyzer.py`)
- **対象データ**:
  - トレンド分析結果
  - 予測モデルデータ
- **理由**:
  - 時系列特化の処理
  - 大量のデータポイント

### 2. PostgreSQLスキーマ設計

```sql
-- VASデータテーブル
CREATE TABLE vas_data (
  id SERIAL PRIMARY KEY,
  company_id VARCHAR(100) NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  work_environment NUMERIC(5,2),
  engagement NUMERIC(5,2),
  health NUMERIC(5,2),
  leadership NUMERIC(5,2),
  communication NUMERIC(5,2),
  overall_score NUMERIC(5,2),
  response_id VARCHAR(255),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 財務データテーブル
CREATE TABLE financial_data (
  id SERIAL PRIMARY KEY,
  company_id VARCHAR(100) NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  revenue NUMERIC(15,2),
  profit NUMERIC(15,2),
  employee_count INTEGER,
  growth_rate NUMERIC(6,2),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ウェルネススコア履歴テーブル
CREATE TABLE wellness_scores (
  id SERIAL PRIMARY KEY,
  company_id VARCHAR(100) NOT NULL,
  score NUMERIC(5,2) NOT NULL,
  category_scores JSONB NOT NULL,
  adjustments JSONB NOT NULL,
  metadata JSONB NOT NULL,
  calculated_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Firestoreに適した機能とデータ

Firestoreは主に**認証関連のデータ**と**更新頻度の低いマスタデータ**の保存に使用します。

### 1. コアモジュールの対応機能

#### 認証管理 (`auth_manager.py`)
- **対象データ**:
  - ユーザー認証情報
  - アクセス権限
  - セッション管理
- **理由**:
  - リアルタイム性が求められる
  - Firebase Authenticationとの連携
  - セキュリティ要件

#### 企業基本情報管理
- **対象データ**:
  - 企業プロファイル
  - 企業設定
  - 最新のウェルネススコア（サマリーのみ）
- **理由**:
  - 更新頻度が低い
  - リアルタイム表示が必要

#### システム設定 (`utils.py`)
- **対象データ**:
  - アプリケーション設定
  - 分析パラメータ設定
  - UI設定
- **理由**:
  - 更新頻度が非常に低い
  - すばやいアクセスが必要

#### 通知管理
- **対象データ**:
  - ユーザー通知
  - アラート
  - システムメッセージ
- **理由**:
  - リアルタイム配信が必要
  - Firebase Cloud Messagingとの連携

### 2. Firestoreデータモデル

```javascript
// companies/{companyId}
{
  "id": "company-123",
  "name": "テックイノベート株式会社",
  "industry": "IT",
  "stage": "Series B",
  "createdAt": Timestamp,
  "wellnessScore": 82.5,       // 最新スコアのみ
  "categoryScores": {          // 最新カテゴリスコアのみ
    "work_environment": 80.2,
    "engagement": 85.1,
    "health": 78.9,
    "leadership": 82.3,
    "communication": 75.8
  },
  "lastScoreUpdate": Timestamp
}

// users/{userId}
{
  "email": "user@example.com",
  "displayName": "山田太郎",
  "role": "admin",
  "companyId": "company-123",
  "lastLogin": Timestamp,
  "settings": {
    "theme": "dark",
    "notifications": true
  }
}

// system_settings/{settingId}
{
  "category": "analysis",
  "parameters": {
    "defaultCorrelationThreshold": 0.7,
    "timeSeriesDefaultPeriod": 90
  },
  "updatedAt": Timestamp
}
```

## リポジトリパターンの実装

データアクセスは抽象化されたリポジトリパターンを通じて行います：

```python
from abc import ABC, abstractmethod

class DataRepository(ABC):
    """データリポジトリの抽象基底クラス"""

    @abstractmethod
    async def get_document(self, collection: str, document_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def query_documents(self, collection: str, filters=None, order_by=None, limit=None) -> List[Dict[str, Any]]:
        pass

class PostgresRepository(DataRepository):
    """PostgreSQL用のリポジトリ実装"""
    # PostgreSQL特有の実装

class FirestoreRepository(DataRepository):
    """Firestore用のリポジトリ実装"""
    # Firestore特有の実装
```

各機能ごとに専用のリポジトリクラスを実装：

```python
class VASDataRepository(PostgresRepository):
    """VASデータ専用リポジトリ"""

    async def get_recent_vas_data(self, company_id: str, limit: int = 100) -> pd.DataFrame:
        """直近のVASデータを取得"""
        # 実装...

class CompanyRepository(FirestoreRepository):
    """企業情報専用リポジトリ"""

    async def get_company(self, company_id: str) -> Dict[str, Any]:
        """企業情報を取得"""
        # 実装...
```

## まとめ

### PostgreSQL
- **適したデータ**: 時系列データ、大量のデータ、複雑な分析が必要なデータ
- **主要モジュール**: VASデータ管理、財務データ管理、ウェルネススコア計算、分析機能
- **特徴**: 高性能な時系列クエリ、大量データの効率的な処理、複雑な集計処理

### Firestore
- **適したデータ**: 認証情報、マスタデータ、リアルタイム更新が必要なデータ
- **主要モジュール**: 認証管理、企業基本情報、システム設定、通知管理
- **特徴**: リアルタイムデータ同期、Firebase連携、スケーラビリティ

このハイブリッドアプローチにより、各データベースの強みを活かしながら、パフォーマンスとコスト効率を最適化しています。