# バックエンドAPIコーディング標準

## 概要

本ドキュメントでは、バックエンドAPIのコード品質と一貫性を確保するためのコーディング標準を定義します。
リファクタリングおよび今後の開発において、これらの標準に従うことで保守性を高め、技術的負債を削減します。

## ディレクトリ構造と命名規則

### ディレクトリ構造
```
backend/
├── api/
│   ├── routers/          # すべてのAPIルーターを格納
│   ├── models.py         # Pydanticモデル定義
│   ├── dependencies.py   # 依存性注入の定義
│   └── __init__.py       # APIのエントリーポイント
├── core/                 # アプリケーションコア機能
├── service/              # ビジネスロジックを含むサービス
├── utils/                # ユーティリティ機能
└── tests/                # テストコード
```

### 命名規則
- **ファイル名**: スネークケースを使用（例: `user_management.py`）
- **クラス名**: パスカルケースを使用（例: `UserManager`）
- **関数/メソッド名**: スネークケースを使用（例: `get_user_by_id`）
- **変数名**: スネークケースを使用（例: `user_data`）
- **定数名**: 大文字のスネークケースを使用（例: `MAX_CONNECTIONS`）

## APIエンドポイント設計

### URLパス構造
- 一貫したプレフィックスパターンを使用: `/api/{resource}`
- リソース名は複数形を使用: `/api/users` （`/api/user`ではない）
- バージョニングが必要な場合は `/api/v{n}/{resource}` 形式を使用

### HTTPメソッドの使用
- `GET`: リソースの取得
- `POST`: 新しいリソースの作成
- `PUT`: リソースの完全な更新
- `PATCH`: リソースの部分的な更新
- `DELETE`: リソースの削除

### レスポンス形式
一貫したJSONレスポンス構造を使用する:

成功レスポンス:
```json
{
  "success": true,
  "data": { ... },  // 実際のデータ
  "message": "操作が成功しました"  // オプション
}
```

エラーレスポンス:
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "エラーメッセージ"
  }
}
```

## FastAPIコンポーネント

### ルーター定義
```python
from fastapi import APIRouter

router = APIRouter(
    prefix="/api/resource",
    tags=["resource_tag"],
    responses={404: {"description": "Not found"}},
)
```

### Pydanticモデル
- リクエストとレスポンスに明示的なPydanticモデルを使用
- モデルには適切な説明と例を含める
- ドキュメントのために`Field(..., description="説明")`を使用

```python
from pydantic import BaseModel, Field

class ItemRequest(BaseModel):
    name: str = Field(..., description="アイテム名")
    description: str = Field(None, description="アイテムの説明")

    class Config:
        schema_extra = {
            "example": {
                "name": "サンプルアイテム",
                "description": "これはサンプルです"
            }
        }
```

## 依存性注入

### 依存関数の定義
- 依存関数は`dependencies.py`に集約する
- 機能ごとにクラス化されたプロバイダーを使用

```python
from fastapi import Depends

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme)):
    return validate_token(token)
```

### サービスの注入
- ビジネスロジックはサービスクラスに抽出
- Dependsを使用してサービスを注入

```python
@router.get("/{item_id}")
async def get_item(
    item_id: int,
    item_service: ItemService = Depends(get_item_service)
):
    return await item_service.get_item(item_id)
```

## エラー処理

### 例外処理
- 適切なHTTPステータスコードを使用
- 具体的なエラーメッセージを提供
- カスタム例外クラスを定義

```python
from fastapi import HTTPException

if not item:
    raise HTTPException(
        status_code=404,
        detail="Item not found"
    )
```

### グローバルな例外ハンドラ
- 一貫した例外応答を確保するためにグローバルハンドラを使用

```python
@app.exception_handler(CustomException)
async def custom_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"success": false, "error": {"message": str(exc)}}
    )
```

## ロギング

### ロギング標準
- 構造化ロギングを使用
- ログレベルを適切に使用（DEBUG, INFO, WARNING, ERROR, CRITICAL）
- コンテキスト情報を含める（ユーザーID、リクエストID等）

```python
import logging

logger = logging.getLogger(__name__)

@router.post("/")
async def create_item(item: ItemRequest):
    logger.info(f"Creating new item: {item.name}", extra={"user_id": user.id})
    # 処理
    logger.debug("Item created with ID: {item_id}")
```

## ドキュメンテーション

### Docstring
- すべての関数、クラス、モジュールにdocstringを使用
- Googleスタイルのdocstringフォーマットに準拠

```python
def function_name(param1: type, param2: type) -> return_type:
    """関数の簡潔な説明

    より詳細な説明を記述できます。
    複数行に渡ることもあります。

    Args:
        param1: 最初のパラメータの説明
        param2: 2番目のパラメータの説明

    Returns:
        戻り値の説明

    Raises:
        ValueError: エラーが発生する条件の説明
    """
```

### APIドキュメント
- OpenAPIスキーマを最大限に活用
- エンドポイントには適切な説明とタグを含める

```python
@router.get(
    "/{item_id}",
    response_model=ItemResponse,
    summary="アイテムの取得",
    description="指定されたIDのアイテムを取得します。"
)
```

## テスト

### テスト構造
- 単体テスト、統合テスト、エンドツーエンドテストを分離
- テストファイル名は`test_`で始める
- テスト関数名も`test_`で始める

### テストカバレッジ
- 最低80%のコードカバレッジを目標とする
- クリティカルなパスには100%のカバレッジを確保
- テストにはモック/スタブを適切に使用

## セキュリティプラクティス

### 入力バリデーション
- すべてのユーザー入力に対してPydanticモデルによるバリデーションを実施
- 追加のバリデーションにはdependenciesを使用

### 認証と認可
- JWTベースの認証を一貫して使用
- ロールベースのアクセス制御を実装
- センシティブなエンドポイントには適切な保護を行う

## パフォーマンス考慮事項

### 非同期処理
- I/O集約的な処理には非同期関数を使用
- 長時間実行される処理にはバックグラウンドタスクを使用

### キャッシング
- 頻繁に使用されるデータや計算結果にはキャッシュを適用
- キャッシュ戦略（TTL、無効化ポリシー）を明示的に定義

## コードレビュー基準

コードレビューでは以下の点を確認:
1. コードが本標準ドキュメントに従っていること
2. 機能が仕様通りに実装されていること
3. テストが十分であること
4. パフォーマンスとセキュリティの考慮が行われていること
5. ドキュメントが適切に更新されていること

## 採用ツール

- コードフォーマッター: `black`
- リンター: `flake8`
- 型チェッカー: `mypy`
- ドキュメントジェネレーター: `mkdocs` + `mkdocstrings`