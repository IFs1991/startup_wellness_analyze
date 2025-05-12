-- 業種・役職別健康影響度重み付け係数のテーブル構造
-- 作成日: 2025-03-08
-- 目的: 業種・役職別の健康影響度評価のためのデータスキーマ定義
-- 変更履歴:
-- 2025-03-08: 初期バージョン作成
-- 2025-05-12: 適切なディレクトリに移動

-- 役職レベルマスタテーブル
CREATE TABLE position_levels (
    level_id SERIAL PRIMARY KEY,
    level_name VARCHAR(50) NOT NULL,
    position_title VARCHAR(100) NOT NULL,
    base_weight DECIMAL(5,2) NOT NULL,
    theoretical_basis TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 業種マスタテーブル
CREATE TABLE industries (
    industry_id SERIAL PRIMARY KEY,
    industry_name VARCHAR(100) NOT NULL,
    industry_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 業種別調整係数テーブル
CREATE TABLE industry_adjustment_factors (
    adjustment_id SERIAL PRIMARY KEY,
    industry_id INTEGER REFERENCES industries(industry_id),
    level_id INTEGER REFERENCES position_levels(level_id),
    adjustment_factor DECIMAL(5,2) NOT NULL,
    explanation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 最終重み係数テーブル（計算結果の保存用）
CREATE TABLE final_weight_coefficients (
    coefficient_id SERIAL PRIMARY KEY,
    industry_id INTEGER REFERENCES industries(industry_id),
    level_id INTEGER REFERENCES position_levels(level_id),
    base_weight DECIMAL(5,2) NOT NULL,
    industry_adjustment DECIMAL(5,2) NOT NULL,
    final_weight DECIMAL(5,2) NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 役職レベルと業種の組み合わせに基づいて最終重み係数を取得するためのビュー
CREATE VIEW vw_health_impact_weights AS
SELECT
    i.industry_name,
    pl.level_name,
    pl.position_title,
    fwc.base_weight,
    fwc.industry_adjustment,
    fwc.final_weight
FROM final_weight_coefficients fwc
JOIN industries i ON fwc.industry_id = i.industry_id
JOIN position_levels pl ON fwc.level_id = pl.level_id;

-- 定期更新用のストアドプロシージャ
CREATE OR REPLACE PROCEDURE update_health_impact_coefficients()
LANGUAGE plpgsql
AS $$
BEGIN
    -- 重み係数の再計算
    TRUNCATE TABLE final_weight_coefficients;

    INSERT INTO final_weight_coefficients (industry_id, level_id, base_weight, industry_adjustment, final_weight)
    SELECT
        iaf.industry_id,
        iaf.level_id,
        pl.base_weight,
        iaf.adjustment_factor,
        pl.base_weight + iaf.adjustment_factor AS final_weight
    FROM industry_adjustment_factors iaf
    JOIN position_levels pl ON iaf.level_id = pl.level_id;

    -- 更新日時を記録
    UPDATE final_weight_coefficients
    SET last_updated = CURRENT_TIMESTAMP;

    RAISE NOTICE '健康影響度重み係数を更新しました。';
END;
$$;