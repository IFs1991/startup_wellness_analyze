-- 業種・役職別健康影響度重み付け係数の初期データ
-- 作成日: 2025-03-08
-- 目的: 業種・役職別の健康影響度評価のための初期データ
-- 変更履歴:
-- 2025-03-08: 初期バージョン作成
-- 2025-05-12: 適切なディレクトリに移動

-- 役職レベルのデータ投入
INSERT INTO position_levels (level_name, position_title, base_weight, theoretical_basis)
VALUES
    ('レベル1', 'C級役員/経営層', 0.40, 'Hambrick & Mason (1984)のUpper Echelons Theory、Teece (1997)のDynamic Capabilities、Podolny (2005)の象徴的価値研究'),
    ('レベル2', '上級管理職', 0.25, 'Van Fleet & Bedeian (1977)のスパンオブコントロール理論、Nonaka & Takeuchi (1995)の知識フロー理論'),
    ('レベル3', '中間管理職', 0.15, 'Simon (1977)の操作的意思決定理論、Burt (1992)の社会的ネットワーク理論'),
    ('レベル4', '専門職', 0.10, '専門的知識とスキルの活用と移転に関する研究 (Argote & Ingram, 2000)'),
    ('レベル5', '一般職員', 0.05, '集合的な組織市民行動研究 (Organ, 1988)、従業員エンゲージメント研究 (Harter et al., 2002)');

-- 業種のデータ投入
INSERT INTO industries (industry_name, industry_description)
VALUES
    ('SaaS・クラウドサービス', 'クラウドベースのソフトウェアサービス提供業界'),
    ('製薬・創薬', '医薬品開発・製造業界'),
    ('リテール・小売', '一般消費者向け小売業界'),
    ('フィンテック', '金融テクノロジー業界'),
    ('ヘルステック', '健康・医療テクノロジー業界'),
    ('エドテック', '教育テクノロジー業界'),
    ('AI・機械学習', '人工知能・機械学習関連業界'),
    ('IoT・ハードウェア', 'インターネット接続デバイス・ハードウェア業界'),
    ('ビジネスサービス', '企業向けサービス業界'),
    ('コンシューマーサービス', '一般消費者向けサービス業界');

-- 業種別調整係数の投入
-- SaaS・クラウドサービス業界の調整係数
INSERT INTO industry_adjustment_factors (industry_id, level_id, adjustment_factor)
VALUES
    (1, 1, 0.08), -- C級役員/経営層
    (1, 2, 0.06), -- 上級管理職
    (1, 3, 0.04), -- 中間管理職
    (1, 4, 0.03), -- 専門職
    (1, 5, 0.01); -- 一般職員

-- 製薬・創薬業界の調整係数
INSERT INTO industry_adjustment_factors (industry_id, level_id, adjustment_factor)
VALUES
    (2, 1, 0.10), -- C級役員/経営層
    (2, 2, 0.08), -- 上級管理職
    (2, 3, 0.05), -- 中間管理職
    (2, 4, 0.05), -- 専門職
    (2, 5, 0.02); -- 一般職員

-- リテール・小売業界の調整係数
INSERT INTO industry_adjustment_factors (industry_id, level_id, adjustment_factor)
VALUES
    (3, 1, 0.06), -- C級役員/経営層
    (3, 2, 0.05), -- 上級管理職
    (3, 3, 0.04), -- 中間管理職
    (3, 4, 0.03), -- 専門職
    (3, 5, 0.02); -- 一般職員

-- フィンテック業界の調整係数
INSERT INTO industry_adjustment_factors (industry_id, level_id, adjustment_factor)
VALUES
    (4, 1, 0.09), -- C級役員/経営層
    (4, 2, 0.07), -- 上級管理職
    (4, 3, 0.05), -- 中間管理職
    (4, 4, 0.04), -- 専門職
    (4, 5, 0.02); -- 一般職員

-- ヘルステック業界の調整係数
INSERT INTO industry_adjustment_factors (industry_id, level_id, adjustment_factor)
VALUES
    (5, 1, 0.10), -- C級役員/経営層
    (5, 2, 0.08), -- 上級管理職
    (5, 3, 0.06), -- 中間管理職
    (5, 4, 0.05), -- 専門職
    (5, 5, 0.03); -- 一般職員

-- エドテック業界の調整係数
INSERT INTO industry_adjustment_factors (industry_id, level_id, adjustment_factor)
VALUES
    (6, 1, 0.07), -- C級役員/経営層
    (6, 2, 0.06), -- 上級管理職
    (6, 3, 0.04), -- 中間管理職
    (6, 4, 0.03), -- 専門職
    (6, 5, 0.02); -- 一般職員

-- AI・機械学習業界の調整係数
INSERT INTO industry_adjustment_factors (industry_id, level_id, adjustment_factor)
VALUES
    (7, 1, 0.09), -- C級役員/経営層
    (7, 2, 0.07), -- 上級管理職
    (7, 3, 0.05), -- 中間管理職
    (7, 4, 0.05), -- 専門職
    (7, 5, 0.02); -- 一般職員

-- IoT・ハードウェア業界の調整係数
INSERT INTO industry_adjustment_factors (industry_id, level_id, adjustment_factor)
VALUES
    (8, 1, 0.08), -- C級役員/経営層
    (8, 2, 0.06), -- 上級管理職
    (8, 3, 0.04), -- 中間管理職
    (8, 4, 0.04), -- 専門職
    (8, 5, 0.02); -- 一般職員

-- ビジネスサービス業界の調整係数
INSERT INTO industry_adjustment_factors (industry_id, level_id, adjustment_factor)
VALUES
    (9, 1, 0.06), -- C級役員/経営層
    (9, 2, 0.05), -- 上級管理職
    (9, 3, 0.03), -- 中間管理職
    (9, 4, 0.02), -- 専門職
    (9, 5, 0.01); -- 一般職員

-- コンシューマーサービス業界の調整係数
INSERT INTO industry_adjustment_factors (industry_id, level_id, adjustment_factor)
VALUES
    (10, 1, 0.05), -- C級役員/経営層
    (10, 2, 0.04), -- 上級管理職
    (10, 3, 0.03), -- 中間管理職
    (10, 4, 0.02), -- 専門職
    (10, 5, 0.01); -- 一般職員

-- 最終重み係数の計算と挿入
-- 上記のINSERT実行後に以下を実行してください
CALL update_health_impact_coefficients();