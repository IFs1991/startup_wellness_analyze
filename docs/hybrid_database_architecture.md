# Startup Wellness ハイブリッドデータベースアーキテクチャ

## 概要

Startup Wellness データ分析システムでは、FirestoreとPostgreSQLを組み合わせたハイブリッドデータベース構成を採用しています。この設計により、各データベースの強みを活かした効率的なデータ管理を実現しています。

- **Firestore**: スケーラブルなNoSQLデータベースとして、柔軟なスキーマ要件や高頻度の読み書きが必要なデータに最適
- **PostgreSQL**: 厳格なスキーマと強力なトランザクション機能を持つリレーショナルデータベースとして、構造化データや関連性の高いデータに最適

## システムアーキテクチャ

### レイヤー構造

```
アプリケーション層 (API/ビジネスロジック)
    ↓
サービス層 (backend/service/)
    ↓
データベース層 (backend/database/)
    ↓ ↓
Firestore  PostgreSQL
```

### モジュール構成

```
backend/
├── database/                 # データベース層
│   ├── __init__.py           # 初期化と公開インターフェース
│   ├── database.py           # データベース接続管理
│   ├── crud.py               # Firestore CRUD操作
│   ├── crud_sql.py           # PostgreSQL CRUD操作
│   ├── models.py             # Firestoreモデル定義
│   ├── models_sql.py         # SQLモデル定義
│   ├── postgres.py           # PostgreSQL接続設定
│   └── migration.py          # データベースマイグレーション
│
└── service/                  # サービス層
    ├── __init__.py           # サービス層の初期化
    ├── firestore/            # Firestoreサービス
    │   ├── __init__.py       # 初期化
    │   └── client.py         # 高レベルなFirestore操作
    └── bigquery/             # BigQueryサービス
```

## データの種類と最適なデータベース

システムで扱うデータは、その性質に基づいて適切なデータベースに振り分けられます。

### PostgreSQL向きデータ (構造化データ)

| データカテゴリ | 説明 | テーブル例 |
|--------------|------|----------|
| USER_MASTER | ユーザーマスタ | users |
| COMPANY_MASTER | 企業マスタ | companies |
| EMPLOYEE_MASTER | 従業員マスタ | employees |
| FINANCIAL | 損益計算書データ | financial_statements |
| BILLING | 請求情報 | invoices, payments |
| AUDIT_LOG | 監査ログ | audit_logs |

### Firestore向きデータ (スケーラブルデータ)

| データカテゴリ | 説明 | コレクション例 |
|--------------|------|-------------|
| CHAT | チャットメッセージ | chats |
| ANALYTICS_CACHE | 分析結果のキャッシュ | analytics |
| USER_SESSION | ユーザーセッション情報 | sessions |
| SURVEY | アンケート回答データ | consultations |
| TREATMENT | 施術記録 | treatments |
| REPORT | 分析レポート | reports |

## データベース管理クラス

`DatabaseManager`クラスは、データの種類に基づいて適切なデータベースを自動的に選択する中核機能を担います。

### 主な機能

- シングルトンパターンによるデータベース接続の一元管理
- データカテゴリに基づく最適なデータベースの自動選択
- コレクション名の一貫した管理
- トランザクション管理とコネクションプール

### 例: データカテゴリに基づくDB選択

```python
# データカテゴリからデータベースを選択
db = get_db_for_category(DataCategory.FINANCIAL)  # PostgreSQLが返される
db = get_db_for_category(DataCategory.SURVEY)     # Firestoreが返される

# 文字列からも選択可能
db = get_db_for_category("financial")  # PostgreSQLが返される
```

## データの流れ

### 1. データ入力フロー

```
Google Forms/ファイルアップロード
        ↓
データ入力サービス (FastAPI)
        ↓
データカテゴリ判定 (DatabaseManager)
        ↓ ↓
Firestore    PostgreSQL
  ↓            ↓
Pub/Sub       イベント通知
  ↓            ↓
処理キュー     処理キュー
```

### 2. データ分析フロー

```
Firestore + PostgreSQL
        ↓
データ処理サービス
        ↓
分析エンジン (pandas/scikit-learn)
        ↓
分析結果
        ↓
Firestore (ANALYTICS_CACHE)
```

### 3. レポート生成フロー

```
分析結果 (Firestore)
        ↓
レポート生成サービス
        ↓
PDF/Excel生成
        ↓
Cloud Storage
        ↓
ダウンロードリンク
```

## トランザクション管理

### PostgreSQLトランザクション

```python
with get_db_session() as session:
    try:
        # トランザクション内の操作
        session.add(new_user)
        session.add(user_profile)
        session.commit()
    except Exception:
        session.rollback()
        raise
```

### Firestoreトランザクション

```python
db = get_firestore_client()
transaction = db.transaction()

@firestore.transactional
def update_in_transaction(transaction, doc_ref):
    doc = doc_ref.get(transaction=transaction)
    # ドキュメント更新
    transaction.update(doc_ref, {'counter': doc.get('counter') + 1})

doc_ref = db.collection('counters').document('counter1')
update_in_transaction(transaction, doc_ref)
```

## サービスレイヤーとの連携

サービスレイヤー（`backend/service/`）は、データベースレイヤーが提供する基本的なデータアクセス機能を活用し、高度なビジネスロジックを実装します。

### データベースレイヤーの責務
- 基本的なCRUD操作
- データベース接続管理
- トランザクション管理
- データモデル定義

### サービスレイヤーの責務
- 複合的なデータ操作
- 外部サービス連携（Cloud Storage等）
- ビジネスルールの適用
- データ変換・加工

例:
```python
# FirestoreServiceでのデータ取得と処理
async def get_user_analytics(user_id: str):
    # データベースレイヤーを利用してデータ取得
    user = crud.get_user(user_id)
    surveys = await firestore_service.fetch_documents(
        collection_name=DatabaseManager.get_collection_name(DataCategory.SURVEY),
        conditions=[{'field': 'user_id', 'operator': '==', 'value': user_id}]
    )

    # ビジネスロジックを適用（サービスレイヤーの責務）
    result = process_analytics_data(user, surveys)
    return result
```

## データベースのマイグレーションと初期化

システムの起動時に、`init_db()`関数が呼び出され、必要なデータベース構造が自動的に初期化されます。

- PostgreSQL: SQLAlchemyモデルからテーブル作成
- Firestore: 必要に応じてインデックスや初期データ作成

## パフォーマンスと最適化

- FirestoreのIndex設計による読み取り効率向上
- PostgreSQLのインデックス最適化
- データアクセスパターンに基づくキャッシュ戦略
- 大量データ処理のためのバッチ処理とクエリ最適化

## 今後の拡張性

- MongoDB、DynamoDBなどの他のNoSQLデータベースへの対応
- シャーディング戦略の実装
- 長期保存用の低コストストレージ層の追加
- イベント駆動アーキテクチャへの移行

## まとめ

ハイブリッドデータベースアーキテクチャを採用することで、システムは以下の利点を享受しています：

1. データの性質に合わせた最適なストレージ選択
2. スケーラビリティとパフォーマンスの向上
3. 開発効率と保守性の向上
4. 将来の拡張に対する柔軟性

このアーキテクチャにより、Startup Wellness データ分析システムは、多様なデータタイプを効率的に処理し、ユーザーに価値ある分析結果を提供することができます。