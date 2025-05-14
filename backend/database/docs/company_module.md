# 企業情報モジュール

## 概要

このモジュールは、Startup Wellness Analyze システムの企業情報管理機能を提供します。
企業の基本情報を保存、取得、更新、削除するための機能を実装しています。

## アーキテクチャ

企業情報モジュールは、以下のコンポーネントで構成されています：

1. **モデル層**
   - `CompanyEntity`: 企業情報のエンティティモデル
   - `Company`: SQLAlchemy ORM モデル

2. **リポジトリ層**
   - `CompanyRepository`: 企業情報のデータアクセス実装

3. **サービス層**
   - `CompanyService`: 企業情報管理のビジネスロジック

4. **API層**
   - `companies.py`: 企業情報 API エンドポイント

## 使用方法

### リポジトリの使用

```python
from backend.database.repositories.company_repository import CompanyRepository
from backend.database.connection import get_db

# セッションの取得
with get_db() as session:
    # リポジトリの作成
    repo = CompanyRepository(session)

    # 企業情報の取得
    company = repo.find_by_id("company_id")

    # 企業情報の検索
    companies = repo.find_by_name_contains("テスト企業")
```

### サービスの使用

```python
from backend.services.company_service import CompanyService
from backend.database.connection import get_db

# セッションの取得（必要な場合）
session = next(get_db())

# サービスの作成
service = CompanyService(session)

# 企業情報の取得
company = service.get_company_by_id("company_id")

# 企業情報の検索
companies = service.get_companies(search="テスト企業")

# 条件による検索
filtered_companies = service.get_companies(filters={"industry": "テクノロジー"})
```

### API の使用

```bash
# 企業一覧の取得
curl -X GET "http://localhost:8000/companies/" -H "accept: application/json"

# 検索条件付きで企業一覧を取得
curl -X GET "http://localhost:8000/companies/?search=テスト" -H "accept: application/json"

# フィルタ条件付きで企業一覧を取得
curl -X GET "http://localhost:8000/companies/?filters=industry=SaaS,location=東京都" -H "accept: application/json"

# 企業情報の取得
curl -X GET "http://localhost:8000/companies/company_id" -H "accept: application/json"

# 企業情報の作成
curl -X POST "http://localhost:8000/companies/" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"name": "新規企業", "industry": "テクノロジー", "employee_count": 50}'

# 企業情報の更新
curl -X PUT "http://localhost:8000/companies/company_id" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"name": "更新企業", "employee_count": 100}'

# 企業情報の削除
curl -X DELETE "http://localhost:8000/companies/company_id" -H "accept: application/json"
```

## 拡張ガイド

このモジュールを拡張する場合は、以下のガイドラインに従ってください：

1. **新しいフィールドの追加**
   - `CompanyEntity` と `Company` モデルの両方に新しいフィールドを追加します
   - マイグレーションスクリプトを作成して DB スキーマを更新します

2. **新しい検索メソッドの追加**
   - `CompanyRepository` に新しいメソッドを追加します
   - `CompanyService` にそのメソッドを呼び出すメソッドを追加します
   - 必要に応じて API ルーターを更新します

3. **パフォーマンスの最適化**
   - よく使われる検索条件にはインデックスを追加します
   - 大量データ取得時にはページネーションを活用します