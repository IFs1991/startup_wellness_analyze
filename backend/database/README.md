# VAS・業績データ統合管理システム

## プロジェクト概要

このプロジェクトは、Google Formsを介して収集したVAS（Visual Analog Scale）健康・パフォーマンスデータと、PDF/CSVから抽出した業績データを効率的に管理・分析するためのシステムを実装します。

## 目的

1. データベース構造の最適化とディレクトリ構造の体系化
2. Google FormsからのVASデータ自動収集システムの構築
3. PDF/CSVからの業績データ抽出・変換・保存プロセスの実装
4. 静的パラメータデータと動的データの効率的な管理体制の確立
5. データ分析基盤の整備

## 主要コンポーネント

### データベース設定

- `config.py`: フォームタイプ、データベース接続設定、Google Forms設定を管理
- スキーマファイル：VAS健康データ、業績データ、参照テーブル、分析ビューの定義

### SQLデータモデル

- `models_sql.py`: SQLAlchemyを使用したデータモデル定義
  - VASHealthPerformance: VAS健康・パフォーマンスデータ
  - GoogleFormsConfiguration: Google Forms連携設定
  - GoogleFormsSyncLog: 同期ログ
  - MonthlyBusinessPerformance: 月次業績データ
  - UploadedDocument: アップロードされたドキュメント
  - DocumentExtractionResult: ドキュメント抽出結果
  - 参照テーブル: PositionLevel, Industry, IndustryWeight, CompanySizeCategory

### データアクセスリポジトリ

- `repositories/vas_repository.py`: VASデータのCRUD操作と同期設定管理
- `repositories/business_performance_repository.py`: 業績データとドキュメント管理

### コネクタとサービス

- `connectors/google_forms_connector.py`: Google FormsとGoogle Sheetsの統合コネクタ
- `services/forms_sync_service.py`: VASデータの同期と検証
- `services/document_processing_service.py`: PDF/CSVからの業績データ抽出

### スキーマとマイグレーション

- `schemas/vas_health_performance.sql`: VASデータスキーマ
- `schemas/business_performance.sql`: 業績データスキーマ
- `schemas/reference_tables.sql`: 参照テーブルスキーマ
- `schemas/analysis_views.sql`: 分析・集計用ビュー

## 主な機能

1. **Google FormsデータのPostgreSQLへの同期**
   - フォーム回答の自動収集と変換
   - 重複検出とデータ検証
   - 同期ログの管理

2. **ドキュメントからの業績データ抽出**
   - PDFファイルからのテキスト抽出
   - CSVファイルの構造解析と変換
   - 抽出データの検証とレビュープロセス

3. **データ分析ビュー**
   - 月別・業種別VASデータ集計
   - VASデータと業績データの相関分析
   - 企業規模・業種別分析

## 主要ファイル

- `/backend/database/config.py`: データベース設定
- `/backend/database/models_sql.py`: SQLAlchemyモデル
- `/backend/database/repositories/`: データアクセス層
- `/backend/database/connectors/`: 外部サービス連携
- `/backend/database/services/`: ビジネスロジック
- `/backend/database/schemas/`: SQLスキーマ

## 技術スタック

- PostgreSQL 14.x: メインデータベース
- SQLAlchemy 1.4.x: ORM
- Alembic: マイグレーション管理
- Python 3.10+: バックエンド開発言語
- Google Sheets/Forms API: データ収集
- PDF処理ライブラリ: PDF解析

## 使用方法

### VASデータの同期

```python
from backend.database.services.forms_sync_service import FormsSyncService
from sqlalchemy.orm import Session

async def sync_vas_data(session: Session, company_id: str):
    service = FormsSyncService(session)
    await service.initialize()

    # 特定フォームの同期
    result = await service.sync_vas_form(
        company_id=company_id,
        form_id="your_form_id"
    )

    # または、すべてのアクティブなフォームを同期
    results = await service.synchronize_all_active_forms(company_id)

    return results
```

### 業績ドキュメントの処理

```python
from backend.database.services.document_processing_service import DocumentProcessingService
from sqlalchemy.orm import Session

async def process_document(session: Session, document_id: int):
    service = DocumentProcessingService(session)

    # ドキュメント処理と自動レポート生成
    result = await service.process_uploaded_document(
        document_id=document_id,
        save_extracted_data=True,
        auto_create_report=True
    )

    return result
```

## 連絡先

- プロジェクトオーナー: データベースチーム