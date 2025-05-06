# 可視化コンポーネント共通化システム

## 概要
このモジュールは、複数の分析タイプにまたがる可視化機能を共通化し、コード重複を削減するプロジェクトです。
統一されたAPIインターフェース、共通データモデル、ファクトリーパターンによるプロセッサ管理を通じて、
一貫性のある高パフォーマンスな可視化機能を提供します。

## アーキテクチャ

### 主要コンポーネント

1. **統一可視化エンドポイント**
   `/api/visualizations/visualize` - 統一された可視化APIエンドポイント

2. **共通データモデル**
   - `UnifiedVisualizationRequest` - すべての可視化リクエストの基本モデル
   - `UnifiedVisualizationResponse` - 標準化された可視化レスポンス

3. **ファクトリーパターン**
   - `VisualizationProcessorFactory` - 分析タイプに応じたプロセッサの作成・管理
   - 遅延ロードと弱参照キャッシュによるメモリ最適化

4. **可視化プロセッサ**
   各分析タイプ（アソシエーション分析、相関分析、記述統計など）に特化したプロセッサ

5. **キャッシュシステム**
   - メモリ内キャッシュによる高速レスポンス
   - 定期的なキャッシュクリーンアップによるメモリ管理

6. **エラー処理**
   統一されたエラー処理メカニズム

### フロー図

```
クライアント
    │
    ▼
統一可視化エンドポイント (/api/visualizations/visualize)
    │
    ├── キャッシュ検索 ──► キャッシュヒット ──► 即時レスポンス
    │
    ├── キャッシュミス
    │       │
    │       ▼
    ├── VisualizationProcessorFactory
    │       │
    │       ▼
    ├── 適切なプロセッサの取得
    │       │
    │       ▼
    ├── チャートデータの準備（最適化）
    │       │
    │       ▼
    ├── チャート生成
    │       │
    │       ▼
    ├── レスポンス生成
    │       │
    │       ▼
    └── キャッシュ保存
```

## パフォーマンス最適化

本システムは以下のパフォーマンス最適化を実装しています：

1. **キャッシュシステム**
   - 同一リクエストに対する処理の重複を回避
   - デフォルトの有効期限：1時間（設定可能）
   - 最大キャッシュサイズ制限によるメモリ使用量制御
   - 定期的なキャッシュクリーンアップ（10分ごと）

2. **メモリ使用量の最適化**
   - 遅延ロードによるプロセッサクラスの必要時のみの初期化
   - 弱参照によるインスタンスキャッシュでメモリリーク防止
   - 大きなデータセットの検出と最適な処理（サンプリングなど）

3. **非同期処理**
   - 大規模データセットの非同期処理によるI/Oブロッキング回避
   - バックグラウンドジョブのサポート

4. **フォールバックメカニズム**
   - 未対応の分析タイプに対する汎用プロセッサの提供
   - エラー耐性の強化

## 使用方法

### 基本的な使用例

```python
# 1. リクエストの作成
request = UnifiedVisualizationRequest(
    analysis_type="correlation",  # 分析タイプ
    analysis_results=analysis_data,  # 分析結果データ
    visualization_type="heatmap",  # 可視化タイプ
    options={
        "title": "相関分析ヒートマップ",
        "color_scheme": "viridis",
        "width": 800,
        "height": 600
    }
)

# 2. APIエンドポイントの呼び出し
response = await client.post("/api/visualizations/visualize", json=request.dict())
```

### 新しいプロセッサの追加方法

新しい分析タイプのプロセッサを追加するには：

1. プロセッサクラスの作成：

```python
from api.visualization.factory import register_processor

@register_processor("my_analysis_type")
class MyVisualizationProcessor:
    def prepare_chart_data(self, analysis_results, visualization_type, options):
        # チャートデータの準備ロジック
        return {
            "config": {...},
            "data": {...}
        }

    def format_summary(self, analysis_results):
        # サマリー情報の整形
        return {...}
```

2. または、遅延ロードによる登録：

```python
from api.visualization.factory import VisualizationProcessorFactory

VisualizationProcessorFactory.register_lazy(
    "my_analysis_type",
    "app.analysis.my_module",
    "MyVisualizationProcessor"
)
```

## パフォーマンスモニタリング

パフォーマンスの監視には以下の指標を使用しています：

1. リクエスト処理時間
2. キャッシュヒット率
3. メモリ使用量
4. CPU使用率

定期的なキャッシュクリーンアップと最適化により、システムの安定性とパフォーマンスを維持しています。

## 今後の改善計画

1. データベースベースのキャッシュ拡張（Redis/Memcached）
2. コンテナ化環境での分散キャッシュ対応
3. リクエストベースの自動パラメータ最適化
4. 機械学習を活用した可視化の自動最適化