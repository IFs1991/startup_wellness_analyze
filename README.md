# スタートアップウェルネス分析プラットフォーム

## 概要

スタートアップの健康状態を分析し、財務パフォーマンスとの相関関係を解明するためのプラットフォームです。機械学習やAIを活用して、組織の健全性と業績の関連性を可視化します。

## 主な機能

- スタートアップの健康状態スコアリング
- 健康指標と財務パフォーマンスの相関分析
- Gemini APIを活用した高品質な可視化とレポート生成
- データに基づいた改善提案
- 連合学習を使用したプライバシー保護分析
- Google Formsからの健康データ自動収集
- PDF/CSVからの業績データ抽出・統合

## VAS・業績データ統合管理システム

本プロジェクトでは、Google Forms経由で収集したVAS（Value Assessment Score）データと、PDF/CSVから抽出した業績データを統合管理するためのシステムを実装しています。

### VASデータ収集機能

- **Google Forms連携**: Google Forms APIとSheets APIを使用して定期的にデータを同期
- **データ変換**: 収集したデータを最適なフォーマットに変換して保存
- **同期スケジューリング**: 定期的な自動同期処理
- **エラーハンドリング**: 同期エラーの検出と再試行メカニズムの実装

### 業績データ処理機能

- **ファイルアップロード**: PDF/CSVファイルのアップロードと保存
- **データ抽出**: ドキュメントからの構造化データの抽出
- **データマッピング**: 抽出データを業績指標にマッピング
- **検証処理**: データの整合性と妥当性の検証

### 統合分析機能

- **相関分析**: VASデータと業績データの相関関係の分析
- **トレンド分析**: 時系列データに基づく傾向分析
- **レポート生成**: 分析結果に基づく自動レポート生成
- **ダッシュボード**: 主要指標のリアルタイム可視化

## 技術スタック

- **バックエンド**: Python (FastAPI)
- **フロントエンド**: React, TypeScript
- **データベース**: PostgreSQL
- **機械学習**: TensorFlow, PyTorch, Scikit-learn
- **連合学習**: Flower
- **可視化**: Gemini API
- **レポート生成**: Gemini API, Puppeteer
- **コンテナ化**: Docker, docker-compose
- **外部API連携**: Google Forms API, Google Sheets API

## データベース設計

本システムでは、以下のエンティティを中心としたデータベース設計を実装しています。

### VASデータ関連テーブル

- **vas_health_performance**: VASによる健康・パフォーマンスデータ
  - 物理的健康、精神的健康、パフォーマンス、満足度のスコアを管理
  - ユーザーと企業の関連付け

- **google_forms_configurations**: Google Forms連携設定
  - フォームタイプ、フォームID、シートID
  - フィールドマッピング設定（JSONBで保存）

- **google_forms_sync_logs**: 同期ログ
  - 同期時間、処理レコード数、ステータス情報

### 業績データ関連テーブル

- **monthly_business_performance**: 月次業績データ
  - 売上、経費、利益率、従業員数、新規顧客獲得数などの業績指標

- **uploaded_documents**: アップロードされたドキュメント情報
  - ファイル情報、処理ステータス

- **document_extraction_results**: ドキュメント抽出結果
  - 抽出データ、信頼度スコア、レビュー情報

### 参照テーブル

- **position_levels**: 役職レベルマスター
- **industries**: 業種マスター
- **industry_weights**: 業種別重み係数
- **company_size_categories**: 企業規模分類

## コンポーネント構造

システムは以下のコンポーネントで構成されています：

### コネクタ

- **GoogleFormsConnector**: Google FormsとSheets APIへのアクセスを提供
  - キャッシュ機能と再試行メカニズムを実装
  - 非同期処理対応

### リポジトリ

- **VASRepository**: VASデータのCRUD操作と同期設定管理
- **BusinessPerformanceRepository**: 業績データとドキュメント管理

### サービス

- **FormsSyncService**: Google Formsデータの同期処理
- **DocumentProcessingService**: PDF/CSVからのデータ抽出・処理

## セットアップ

### 前提条件

- Python 3.10+
- Node.js 18+
- Docker & docker-compose
- PostgreSQL (ローカル開発の場合)
- Google Cloud Platform アカウント (Forms APIとSheets API利用のため)

### 環境構築

1. リポジトリのクローン:
   ```bash
   git clone https://github.com/yourusername/startup-wellness-analyze.git
   cd startup-wellness-analyze
   ```

2. 環境設定:
   ```bash
   cp .env.example .env
   # .envファイルを編集して必要な設定を行う
   ```

3. バックエンド依存関係のインストール:
   ```bash
   conda env create -f backend/environment.yml
   conda activate wellness
   pip install -r backend/requirements.txt
   ```

4. フロントエンド依存関係のインストール:
   ```bash
   npm install
   ```

5. データベースのセットアップ:
   ```bash
   cd backend
   python -m database.migration upgrade head
   ```

6. ローカル開発サーバーの起動:
   ```bash
   # バックエンド
   cd backend
   uvicorn main:app --reload

   # フロントエンド
   npm run dev
   ```

7. Dockerを使用した起動:
   ```bash
   docker-compose up --build
   ```

## Google Forms APIの設定

1. Google Cloud Platformでプロジェクトを作成
2. Forms APIとSheets APIを有効化
3. サービスアカウントを作成して鍵をダウンロード
4. 鍵ファイルを`backend/credentials`ディレクトリに配置
5. `.env`ファイルに以下の設定を追加:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=./credentials/your-key-file.json
   VAS_HEALTH_FORM_ID=your-form-id
   VAS_HEALTH_SHEET_ID=your-sheet-id
   ```

## Gemini API設定

本プロジェクトはGoogle Gemini APIを使用して高品質なデータ可視化とレポート生成を行います。以下の手順でAPIキーを設定してください。

1. [Google AI Studio](https://makersuite.google.com/)にアクセスし、APIキーを取得

2. `.env`ファイルにAPIキーを設定:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. 必要に応じてキャッシュ設定を調整:
   ```
   GEMINI_CACHE_ENABLED=true
   GEMINI_CACHE_TTL=86400
   ```

詳細は以下のドキュメントを参照してください:
- [Gemini可視化ガイド](docs/gemini_visualization_guide.md)
- [Geminiレポート生成ガイド](docs/gemini_report_generator.md)
- [Gemini活用概要](docs/gemini_usage_overview.md)

## ディレクトリ構造

```
startup-wellness-analyze/
├── backend/                     # バックエンドコード
│   ├── api/                     # API定義
│   ├── core/                    # コアロジック
│   ├── database/                # データベース関連
│   │   ├── connectors/          # 外部システム連携
│   │   ├── migrations/          # マイグレーションファイル
│   │   ├── repositories/        # データアクセスレイヤー
│   │   ├── services/            # ビジネスロジック
│   │   ├── schemas/             # スキーマ定義
│   │   └── seed/                # 初期データ
│   ├── federated_learning/      # 連合学習モジュール
│   ├── models/                  # データモデル
│   ├── schemas/                 # Pydanticスキーマ
│   ├── utils/                   # ユーティリティ
│   └── main.py                  # エントリーポイント
├── frontend/                    # フロントエンドコード
├── docs/                        # ドキュメント
├── tests/                       # テスト
└── docker-compose.yml           # Docker構成
```

## 開発ガイドライン

- コミット前に`pytest`と`npm test`でテストを実行
- PEP 8スタイルガイドに従うこと
- フロントエンドはESLintとPrettierの規約に従うこと
- 機能追加時は対応するテストも追加すること

## ライセンス

このプロジェクトは独自ライセンスで提供されています。詳細はLICENSEファイルを参照してください。

## 貢献

プロジェクトへの貢献を歓迎します。貢献方法については、CONTRIBUTING.mdをご覧ください。
