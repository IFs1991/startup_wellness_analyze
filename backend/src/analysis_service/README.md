# Analysis Service 機能概要

## 概要
このディレクトリは、ウェルネス分析システムの具体的な分析処理を実装するサービス層です。
Core層からの要求を受けて実際の計算処理を行い、結果をCore層に返却します。

## モジュール構成

### 1. 集計処理 (`aggregation.py`)
- データの集計処理を実行
- 主な機能：
  - グループごとの集計計算
  - 時系列データの集約
  - 統計的な集計処理
- Core連携：
  - `descriptive_stats_calculator.py`
  - `time_series_analyzer.py`

### 2. メトリクス計算 (`metrics.py`)
- 各種分析指標の計算処理
- 主な機能：
  - パフォーマンス指標の計算
  - KPIの算出
  - 統計的指標の計算
- Core連携：
  - `performance_predictor.py`
  - `model_evaluator.py`

### 3. 比較分析 (`comparison.py`)
- データセット間の比較分析
- 主な機能：
  - 時系列データの比較
  - グループ間の差異分析
  - パターン比較
- Core連携：
  - `correlation_analyzer.py`
  - `cluster_analyzer.py`

## 処理フロー

### 1. リクエ��トの受信
```
フロントエンド → Core層 → Analysis Service
```
- フロントエンドからのリクエストをCore層が受信
- Core層が必要なバリデーションを実施
- Analysis Serviceに処理を委譲

### 2. 分析処理の実行
```
Analysis Service
└── 各モジュールでの処理
    ├── aggregation.py: データ集計
    ├── metrics.py: 指標計算
    └── comparison.py: 比較分析
```
- 要求された分析タイプに応じて適切なモジュールが処理
- 必要に応じて複数モジュールを連携

### 3. 結果の返却
```
Analysis Service → Core層 → フロントエンド
```
- 分析結果をCore層に返却
- Core層で結果を整形
- フロントエンドに適切な形式で返却

## 設計上の特徴

### 1. 責務の分離
- Analysis Serviceは純粋な分析処理のみに注力
- データの入出力やバリデーションはCore層が担当

### 2. モジュール性
- 各分析処理が独立したモジュールとして実装
- 新しい分析手法の追加が容易

### 3. 再利用性
- 基本的な分析処理を共通化
- 複数のCore機能から同じ分析処理を利用可能

### 4. スケーラビリティ
- 処理を機能ごとに分割することで並列化が容易
- 負荷に応じて個別にスケールアウト可能

## Core層との連携例

### 集計処理の場合
1. Core層の`descriptive_stats_calculator.py`がリクエストを受信
2. `aggregation.py`に集計処理を依頼
3. 結果を受け取りフロントエンド用にフォーマット

### 比較分析の場合
1. Core層の`correlation_analyzer.py`がリクエストを受信
2. `comparison.py`に相関分析を依頼
3. 結果を受け取りフロントエンド用にフォーマット