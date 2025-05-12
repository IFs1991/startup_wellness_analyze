-- =============================================
-- VAS健康・パフォーマンスデータ管理スキーマ
-- 作成日: 2025-06-01
-- 説明: Google Formsから取得したVASデータを格納するためのスキーマ
-- =============================================

-- VASデータテーブル
CREATE TABLE IF NOT EXISTS vas_health_performance (
    record_id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    company_id VARCHAR(100) NOT NULL,
    record_date TIMESTAMP NOT NULL,
    physical_health INTEGER CHECK (physical_health BETWEEN 0 AND 100),
    mental_health INTEGER CHECK (mental_health BETWEEN 0 AND 100),
    work_performance INTEGER CHECK (work_performance BETWEEN 0 AND 100),
    work_satisfaction INTEGER CHECK (work_satisfaction BETWEEN 0 AND 100),
    additional_comments TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, company_id, record_date)
);

-- インデックス作成
CREATE INDEX IF NOT EXISTS idx_vas_company_id ON vas_health_performance(company_id);
CREATE INDEX IF NOT EXISTS idx_vas_user_id ON vas_health_performance(user_id);
CREATE INDEX IF NOT EXISTS idx_vas_record_date ON vas_health_performance(record_date);

-- Google Forms設定テーブル
CREATE TABLE IF NOT EXISTS google_forms_configurations (
    config_id SERIAL PRIMARY KEY,
    company_id VARCHAR(100) NOT NULL,
    form_type VARCHAR(50) NOT NULL,
    form_id VARCHAR(100) NOT NULL,
    sheet_id VARCHAR(100),
    field_mappings JSONB NOT NULL DEFAULT '{}',
    active BOOLEAN NOT NULL DEFAULT TRUE,
    sync_frequency INTEGER NOT NULL DEFAULT 3600, -- デフォルト1時間ごと（秒単位）
    last_sync_time TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (company_id, form_type)
);

-- インデックス作成
CREATE INDEX IF NOT EXISTS idx_forms_company_id ON google_forms_configurations(company_id);
CREATE INDEX IF NOT EXISTS idx_forms_form_type ON google_forms_configurations(form_type);

-- Google Forms同期ログテーブル
CREATE TABLE IF NOT EXISTS google_forms_sync_logs (
    log_id SERIAL PRIMARY KEY,
    config_id INTEGER NOT NULL REFERENCES google_forms_configurations(config_id),
    sync_start_time TIMESTAMP NOT NULL,
    sync_end_time TIMESTAMP NOT NULL,
    records_processed INTEGER NOT NULL DEFAULT 0,
    records_created INTEGER NOT NULL DEFAULT 0,
    records_updated INTEGER NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL,
    error_details TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- インデックス作成
CREATE INDEX IF NOT EXISTS idx_sync_logs_config_id ON google_forms_sync_logs(config_id);
CREATE INDEX IF NOT EXISTS idx_sync_logs_start_time ON google_forms_sync_logs(sync_start_time);

-- コメント追加
COMMENT ON TABLE vas_health_performance IS 'VASによる健康・パフォーマンスデータテーブル';
COMMENT ON TABLE google_forms_configurations IS 'Google Forms連携設定テーブル';
COMMENT ON TABLE google_forms_sync_logs IS 'Google Formsデータ同期ログテーブル';

-- 履歴用関数とトリガー
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- vas_health_performanceテーブルの更新日時を自動更新
DROP TRIGGER IF EXISTS update_vas_health_performance_updated_at ON vas_health_performance;
CREATE TRIGGER update_vas_health_performance_updated_at
BEFORE UPDATE ON vas_health_performance
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- google_forms_configurationsテーブルの更新日時を自動更新
DROP TRIGGER IF EXISTS update_google_forms_configurations_updated_at ON google_forms_configurations;
CREATE TRIGGER update_google_forms_configurations_updated_at
BEFORE UPDATE ON google_forms_configurations
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();