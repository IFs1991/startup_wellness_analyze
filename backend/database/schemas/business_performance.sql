-- =============================================
-- 業績データ管理スキーマ
-- 作成日: 2025-06-01
-- 説明: PDF/CSVから抽出した業績データを格納するためのスキーマ
-- =============================================

-- 月次業績データテーブル
CREATE TABLE IF NOT EXISTS monthly_business_performance (
    report_id SERIAL PRIMARY KEY,
    company_id VARCHAR(100) NOT NULL,
    report_month DATE NOT NULL, -- 報告対象月（年月のみ）
    revenue NUMERIC(15, 2),
    expenses NUMERIC(15, 2),
    profit_margin NUMERIC(5, 2), -- パーセンテージ
    headcount INTEGER,
    new_clients INTEGER,
    turnover_rate NUMERIC(5, 2), -- パーセンテージ
    notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (company_id, report_month)
);

-- アップロードされたドキュメント情報テーブル
CREATE TABLE IF NOT EXISTS uploaded_documents (
    document_id SERIAL PRIMARY KEY,
    company_id VARCHAR(100) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    original_file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL, -- 'pdf', 'csv' など
    file_size BIGINT NOT NULL,
    upload_path VARCHAR(255) NOT NULL,
    content_type VARCHAR(100),
    processing_status VARCHAR(50) NOT NULL DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    processed_at TIMESTAMP,
    error_details TEXT,
    uploaded_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ドキュメント抽出結果テーブル
CREATE TABLE IF NOT EXISTS document_extraction_results (
    result_id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES uploaded_documents(document_id),
    report_id INTEGER REFERENCES monthly_business_performance(report_id),
    extracted_data JSONB NOT NULL DEFAULT '{}',
    confidence_score NUMERIC(5, 2), -- 抽出信頼度（パーセンテージ）
    review_status VARCHAR(50) NOT NULL DEFAULT 'pending', -- 'pending', 'reviewed', 'accepted', 'rejected'
    reviewed_by VARCHAR(100),
    reviewed_at TIMESTAMP,
    review_notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- インデックス作成
CREATE INDEX IF NOT EXISTS idx_performance_company_id ON monthly_business_performance(company_id);
CREATE INDEX IF NOT EXISTS idx_performance_report_month ON monthly_business_performance(report_month);

CREATE INDEX IF NOT EXISTS idx_documents_company_id ON uploaded_documents(company_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON uploaded_documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_file_type ON uploaded_documents(file_type);

CREATE INDEX IF NOT EXISTS idx_extraction_document_id ON document_extraction_results(document_id);
CREATE INDEX IF NOT EXISTS idx_extraction_report_id ON document_extraction_results(report_id);
CREATE INDEX IF NOT EXISTS idx_extraction_review_status ON document_extraction_results(review_status);

-- コメント追加
COMMENT ON TABLE monthly_business_performance IS '月次業績データテーブル';
COMMENT ON TABLE uploaded_documents IS 'アップロードされたドキュメント情報';
COMMENT ON TABLE document_extraction_results IS 'ドキュメント抽出結果';

-- 履歴用トリガー
-- monthly_business_performanceテーブルの更新日時を自動更新
DROP TRIGGER IF EXISTS update_monthly_business_performance_updated_at ON monthly_business_performance;
CREATE TRIGGER update_monthly_business_performance_updated_at
BEFORE UPDATE ON monthly_business_performance
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- uploaded_documentsテーブルの更新日時を自動更新
DROP TRIGGER IF EXISTS update_uploaded_documents_updated_at ON uploaded_documents;
CREATE TRIGGER update_uploaded_documents_updated_at
BEFORE UPDATE ON uploaded_documents
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- document_extraction_resultsテーブルの更新日時を自動更新
DROP TRIGGER IF EXISTS update_document_extraction_results_updated_at ON document_extraction_results;
CREATE TRIGGER update_document_extraction_results_updated_at
BEFORE UPDATE ON document_extraction_results
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();