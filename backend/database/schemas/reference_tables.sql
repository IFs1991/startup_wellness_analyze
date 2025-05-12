-- =============================================
-- 参照テーブルスキーマ
-- 作成日: 2025-06-01
-- 説明: システムで使用される参照テーブル群
-- =============================================

-- 役職レベルマスター
CREATE TABLE IF NOT EXISTS position_levels (
    level_id SERIAL PRIMARY KEY,
    level_name VARCHAR(100) NOT NULL,
    position_title VARCHAR(100) NOT NULL,
    base_weight NUMERIC(5, 2) NOT NULL,
    theoretical_basis TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (position_title)
);

-- 業種マスター
CREATE TABLE IF NOT EXISTS industries (
    industry_id SERIAL PRIMARY KEY,
    industry_name VARCHAR(100) NOT NULL,
    industry_code VARCHAR(20) NOT NULL,
    industry_description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (industry_code)
);

-- 業種別重み係数
CREATE TABLE IF NOT EXISTS industry_weights (
    weight_id SERIAL PRIMARY KEY,
    industry_id INTEGER NOT NULL REFERENCES industries(industry_id),
    metric_name VARCHAR(100) NOT NULL,
    weight_value NUMERIC(5, 2) NOT NULL,
    weight_description TEXT,
    effective_from DATE NOT NULL,
    effective_to DATE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (industry_id, metric_name, effective_from)
);

-- 企業規模分類
CREATE TABLE IF NOT EXISTS company_size_categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL,
    min_employees INTEGER,
    max_employees INTEGER,
    min_revenue NUMERIC(15, 2),
    max_revenue NUMERIC(15, 2),
    adjustment_factor NUMERIC(5, 2) NOT NULL DEFAULT 1.0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (category_name)
);

-- コメント追加
COMMENT ON TABLE position_levels IS '役職レベルマスター';
COMMENT ON TABLE industries IS '業種マスター';
COMMENT ON TABLE industry_weights IS '業種別重み係数';
COMMENT ON TABLE company_size_categories IS '企業規模分類';

-- 履歴用トリガー
-- position_levelsテーブルの更新日時を自動更新
DROP TRIGGER IF EXISTS update_position_levels_updated_at ON position_levels;
CREATE TRIGGER update_position_levels_updated_at
BEFORE UPDATE ON position_levels
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- industriesテーブルの更新日時を自動更新
DROP TRIGGER IF EXISTS update_industries_updated_at ON industries;
CREATE TRIGGER update_industries_updated_at
BEFORE UPDATE ON industries
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- industry_weightsテーブルの更新日時を自動更新
DROP TRIGGER IF EXISTS update_industry_weights_updated_at ON industry_weights;
CREATE TRIGGER update_industry_weights_updated_at
BEFORE UPDATE ON industry_weights
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- company_size_categoriesテーブルの更新日時を自動更新
DROP TRIGGER IF EXISTS update_company_size_categories_updated_at ON company_size_categories;
CREATE TRIGGER update_company_size_categories_updated_at
BEFORE UPDATE ON company_size_categories
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();