# スタートアップウェルネス分析プラットフォーム

## 概要

スタートアップの健康状態を分析し、財務パフォーマンスとの相関関係を解明するためのプラットフォームです。機械学習やAIを活用して、組織の健全性と業績の関連性を可視化します。

## 主な機能

- スタートアップの健康状態スコアリング
- 健康指標と財務パフォーマンスの相関分析
- Gemini APIを活用した高品質な可視化とレポート生成
- データに基づいた改善提案
- 連合学習を使用したプライバシー保護分析

## 技術スタック

- **バックエンド**: Python (FastAPI)
- **フロントエンド**: React, TypeScript
- **データベース**: PostgreSQL
- **機械学習**: TensorFlow, PyTorch, Scikit-learn
- **連合学習**: Flower
- **可視化**: Gemini API
- **レポート生成**: Gemini API, Puppeteer
- **コンテナ化**: Docker, docker-compose

## セットアップ

### 前提条件

- Python 3.10+
- Node.js 18+
- Docker & docker-compose
- PostgreSQL (ローカル開発の場合)

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

5. ローカル開発サーバーの起動:
   ```bash
   # バックエンド
   cd backend
   uvicorn main:app --reload

   # フロントエンド
   npm run dev
   ```

6. Dockerを使用した起動:
   ```bash
   docker-compose up --build
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
├── backend/             # バックエンドコード
│   ├── api/             # API定義
│   ├── core/            # コアロジック
│   ├── database/        # データベース関連
│   ├── federated_learning/ # 連合学習モジュール
│   ├── models/          # データモデル
│   ├── schemas/         # Pydanticスキーマ
│   ├── utils/           # ユーティリティ
│   └── main.py          # エントリーポイント
├── frontend/            # フロントエンドコード
├── docs/                # ドキュメント
├── tests/               # テスト
└── docker-compose.yml   # Docker構成
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
