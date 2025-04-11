# API移行ガイド: routes -> routers

## 概要

この文書は、APIエンドポイントの移行に関するガイドです。従来の`routes/`ディレクトリからより構造化された`routers/`ディレクトリへの移行を説明しています。

**重要**: `routes/`ディレクトリは**2023年7月15日**に削除される予定です。それまでに新しいAPIエンドポイントへの移行を完了してください。

## 移行の背景と目的

このリファクタリングの主な目的:

1. APIエンドポイントの一貫性の向上
2. 依存関係の明確化とサービスレイヤーの強化
3. ドキュメント、エラーハンドリング、ロギングの標準化
4. 構造化されたレスポンス形式の採用
5. 保守性とテスト容易性の向上

## 非推奨になったAPIエンドポイント

以下のAPIエンドポイントは非推奨となり、新しいエンドポイントに移行されました:

| 旧エンドポイント | 新エンドポイント | 非推奨日 | 削除予定日 |
|-----------------|-----------------|----------|-----------|
| /api/v1/visualizations/* | /api/visualization/* | 2023-04-15 | 2023-07-15 |
| /api/v1/reports/* | /api/reports/* | 2023-04-15 | 2023-07-15 |

## 移行手順

### 1. インポートパスの更新

```python
# 変更前
from backend.api.routes.visualization import generate_chart

# 変更後
from backend.api.routers.visualization import generate_chart
```

### 2. APIパスの更新

```python
# 変更前
response = await fetch('/api/v1/visualizations/chart', {
  method: 'POST',
  body: JSON.stringify(data)
});

# 変更後
response = await fetch('/api/visualization/chart', {
  method: 'POST',
  body: JSON.stringify(data)
});
```

### 3. レスポンス形式の変更点

新しいAPIは一貫したレスポンス形式を返します:

```json
{
  "success": true,
  "data": { ... },
  "message": "操作が成功しました"
}
```

エラーの場合:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "エラーメッセージ",
    "details": { ... }
  }
}
```

### 4. 移行状況の確認

APIの使用状況を確認するには、管理者専用の以下のエンドポイントを使用できます:

- `/api/monitoring/deprecated-api-usage` - 非推奨APIの使用状況
- `/api/monitoring/deprecated-apis` - 非推奨APIのリスト

## 後方互換性レイヤー

当面の間、後方互換性のために旧APIパスは使用可能です。これらのリクエストは自動的に新しいAPIエンドポイントにリダイレクトされます。このリダイレクトは一時的なものであり、**2023年7月15日**以降は機能しなくなります。

非推奨APIを使用した場合、レスポンスヘッダーに以下の警告が含まれます:

```
Warning: 299 - 'このAPIパスは非推奨です。新しいパスを使用してください。'
X-Deprecated-API: true
X-New-API-Path: /api/visualization/chart
```

## 新APIの利点

1. **一貫したURLパス構造**: すべてのAPIエンドポイントが統一された命名規則に従っています
2. **改善されたエラーハンドリング**: より詳細なエラー情報とステータスコード
3. **強化されたドキュメント**: すべてのエンドポイントに詳細な説明と使用例
4. **型安全性の向上**: すべてのAPIモデルがより厳密に型定義されています
5. **標準化されたレスポンス形式**: すべてのAPIが一貫したレスポンス形式を使用
6. **パフォーマンスの向上**: 最適化されたビジネスロジックとキャッシング戦略

## 補足情報

- 移行に関する質問やサポートは `api-migration@example.com` までご連絡ください
- API変更の詳細なドキュメントは `/api/docs` で確認できます
- APIモニタリングダッシュボードは管理者向けに `/admin/api-monitoring` で提供されています

## 関連リソース

- [APIドキュメント](/api/docs)
- [コーディング標準](./coding_standards.md)
- [テストカバレッジ計画](./test_coverage_plan.md)
- [APIリファクタリング計画](./apiimpovemrnt.yaml)