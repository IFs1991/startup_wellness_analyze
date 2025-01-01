# Core機能概要

## 概要
このディレクトリには、スタートアップのウェルネス分析システムの主要な分析・処理機能が実装されています。

## 機能グループ

### 1. データ処理・分析系
- `data_preprocessor.py` - データの前処理機能
- `data_input.py` - データ入力処理
- `data_quality_checker.py` - データ品質チェック
- `descriptive_stats_calculator.py` - 記述統計の計算

### 2. 分析アルゴリズム系
- `survival_analyzer.py` - 生存分析
- `time_series_analyzer.py` - 時系列分析
- `cluster_analyzer.py` - クラスター分析
- `correlation_analyzer.py` - 相関分析
- `pca_analyzer.py` - 主成分分析
- `association_analyzer.py` - アソシエーション分析

### 3. 可視化・レポート系
- `dashboard_creator.py` - ダッシュボード作成
- `graph_generator.py` - グラフ生成
- `pdf_report_generator.py` - PDFレポート生成
- `custom_report_builder.py` - カスタムレポート作成
- `interactive_visualizer.py` - インタラクティブな可視化

### 4. 外部連携系
- `google_forms_connector.py` - Googleフォームとの連携
- `external_data_fetcher.py` - 外部データ取得
- `generative_ai_manager.py` - 生成AI管理

### 5. システム基盤系
- `auth_manager.py` - 認証管理
- `security.py` - セキュリティ機能
- `scalability.py` - スケーラビリティ対応
- `performance_predictor.py` - パフォーマンス予測
- `utils.py` - ユーティリティ機能

## フロントエンドとの連携

### データ可視化連携
- `dashboard_creator.py`と`interactive_visualizer.py`がフロントエンドのダッシュボード表示に必要なデータを提供
- `graph_generator.py`がフロントエンド側でのグラフ描画用のデータを生成

### レポート生成連携
- `pdf_report_generator.py`と`custom_report_builder.py`がフロントエンドからのレポート要求に応じてレポートを生成
- フロントエンドでダウンロードや表示が可能な形式でレポートを提供

### データ入出力連携
- `data_input.py`がフロントエンドからのデータ入力を受け付け
- `google_forms_connector.py`がフロントエンドのフォーム機能と連携

### 認証・セキュリティ連携
- `auth_manager.py`がフロントエンドのログイン・認証機能を支援
- `security.py`がフロントエンド-バックエンド間の通信セキュリティを確保

### 分析結果の提供
- 各種アナライザーがフロントエンドの分析結果表示に必要なデータを提供
- APIエンドポイントを通じてフロントエンドと通信