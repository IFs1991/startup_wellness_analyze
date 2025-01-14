# VASスケール分析システム 技術仕様書

## 1. 概要

このドキュメントでは、VASスケールデータと財務データの相関分析システムの技術的な実装詳細について説明します。

## 2. データ構造

### 2.1 VASスケールデータ
- 痛みレベル（0-10）
- ストレスレベル（0-10）
- 睡眠の質（0-10）
- タイムスタンプ

### 2.2 財務データ
- 売上高
- 売上原価
- 粗利益
- 記録日時

## 3. 分析機能

### 3.1 相関分析
```python
async def analyze(
    query: str,
    vas_variables: List[str],
    financial_variables: List[str],
    save_results: bool = True,
    dataset_id: Optional[str] = None,
    table_id: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
```

#### 処理フロー
1. データ取得（BigQuery）
2. データバリデーション
3. 相関行列の計算
4. 結果の保存
5. メタデータの生成

#### バリデーション項目
- データの存在確認
- カラムの存在確認
- データ型の確認（数値型）
- 欠損値のチェック

### 3.2 記述統計量
```python
class DescriptiveStatsConfig:
    query: str
    target_variable: str
    arima_order: Tuple[int, int, int] = (5, 1, 0)
    columns: Optional[List[str]] = None
```

#### 計算される統計量
- 平均値
- 中央値
- 標準偏差
- サンプルサイズ
- 最小値/最大値

### 3.3 時系列分析
- ARIMA モデルによる時系列予測
- トレンド分析
- 季節性の検出

## 4. データ永続化

### 4.1 BigQuery
- 生データの保存
- 分析用クエリの実行
- 大規模データの効率的な処理

### 4.2 Firestore
- 分析結果の保存
- メタデータの管理
- ユーザー別の結果管理

## 5. フロントエンド表示

### 5.1 相関マトリックス
```typescript
export const CorrelationMatrix: React.FC<CorrelationMatrixProps> = ({ data }) => {
  // 相関係数の色分け表示
  // -1.0 から 1.0 の値を色の濃淡で表現
}
```

### 5.2 時系列チャート
- VASスケールの経時変化
- 財務指標との重ね合わせ表示
- インタラクティブな期間選択

## 6. エラーハンドリング

### 6.1 データバリデーションエラー
```python
def _validate_data(
    data: pd.DataFrame,
    vas_variables: List[str],
    financial_variables: List[str]
) -> Tuple[bool, Optional[str]]:
    # データの品質チェック
    # エラーメッセージの生成
```

### 6.2 分析エラー
- 相関計算エラー
- 統計量計算エラー
- データ保存エラー

## 7. API エンドポイント

### 7.1 相関分析
```
POST /api/analysis/correlation
```
#### リクエストパラメータ
- `query`: 分析対象データ取得クエリ
- `vas_variables`: VAS変数リスト
- `financial_variables`: 財務変数リスト

### 7.2 記述統計
```
POST /api/analysis/descriptive_stats
```
#### リクエストパラメータ
- `collection_name`: 対象コレクション名
- `conditions`: フィルタ条件

## 8. セキュリティ

### 8.1 認証
- ユーザー認証必須
- APIキーによるアクセス制御

### 8.2 データアクセス制御
- ユーザーごとのデータ分離
- 権限に基づくアクセス制限

## 9. パフォーマンス最適化

### 9.1 非同期処理
- 分析処理の非同期実行
- バッチ処理による効率化

### 9.2 キャッシュ戦略
- 分析結果のキャッシュ
- 頻繁なクエリの最適化

## 10. 拡張性

### 10.1 新規分析手法の追加
- 分析インターフェースの統一
- プラグイン形式での機能追加

### 10.2 データソースの追加
- 外部データソースの統合
- データ形式の標準化

## 11. 監視とログ

### 11.1 分析ログ
- 実行時間の記録
- エラー発生箇所の特定

### 11.2 パフォーマンスモニタリング
- リソース使用状況の追跡
- ボトルネックの特定

## 12. 今後の展開

### 12.1 機能拡張予定
- 機械学習モデルの統合
- リアルタイム分析の実装
- 予測モデルの精度向上

### 12.2 UI/UX改善
- インタラクティブ性の向上
- カスタマイズ可能なダッシュボード
- モバイル対応の強化