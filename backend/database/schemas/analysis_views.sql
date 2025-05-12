-- =============================================
-- 分析・集計用ビュースキーマ
-- 作成日: 2025-06-01
-- 説明: VASデータと業績データの分析に使用するビュー群
-- =============================================

-- VASデータの月別集計ビュー
CREATE OR REPLACE VIEW vas_monthly_summary AS
SELECT
    company_id,
    DATE_TRUNC('month', record_date) AS month,
    COUNT(record_id) AS total_records,
    AVG(physical_health) AS avg_physical_health,
    AVG(mental_health) AS avg_mental_health,
    AVG(work_performance) AS avg_work_performance,
    AVG(work_satisfaction) AS avg_work_satisfaction,
    STDDEV(physical_health) AS stddev_physical_health,
    STDDEV(mental_health) AS stddev_mental_health,
    STDDEV(work_performance) AS stddev_work_performance,
    STDDEV(work_satisfaction) AS stddev_work_satisfaction
FROM
    vas_health_performance
GROUP BY
    company_id, DATE_TRUNC('month', record_date);

-- 業種別VASデータ集計ビュー
CREATE OR REPLACE VIEW vas_industry_summary AS
SELECT
    i.industry_id,
    i.industry_name,
    DATE_TRUNC('month', vhp.record_date) AS month,
    COUNT(vhp.record_id) AS total_records,
    AVG(vhp.physical_health) AS avg_physical_health,
    AVG(vhp.mental_health) AS avg_mental_health,
    AVG(vhp.work_performance) AS avg_work_performance,
    AVG(vhp.work_satisfaction) AS avg_work_satisfaction
FROM
    vas_health_performance vhp
JOIN
    companies c ON vhp.company_id = c.company_id
JOIN
    industries i ON c.industry_id = i.industry_id
GROUP BY
    i.industry_id, i.industry_name, DATE_TRUNC('month', vhp.record_date);

-- VASデータと業績データの相関分析ビュー
CREATE OR REPLACE VIEW vas_performance_correlation AS
SELECT
    vms.company_id,
    vms.month,
    vms.avg_physical_health,
    vms.avg_mental_health,
    vms.avg_work_performance,
    vms.avg_work_satisfaction,
    mbp.revenue,
    mbp.expenses,
    mbp.profit_margin,
    mbp.headcount,
    mbp.new_clients,
    mbp.turnover_rate,
    mbp.report_month AS performance_month
FROM
    vas_monthly_summary vms
JOIN
    monthly_business_performance mbp
    ON vms.company_id = mbp.company_id
    AND vms.month = DATE_TRUNC('month', mbp.report_month);

-- 企業規模別VASデータと業績の比較ビュー
CREATE OR REPLACE VIEW company_size_comparison AS
SELECT
    csc.category_id,
    csc.category_name,
    DATE_TRUNC('month', vhp.record_date) AS month,
    COUNT(DISTINCT vhp.company_id) AS company_count,
    AVG(vhp.physical_health) AS avg_physical_health,
    AVG(vhp.mental_health) AS avg_mental_health,
    AVG(vhp.work_performance) AS avg_work_performance,
    AVG(vhp.work_satisfaction) AS avg_work_satisfaction,
    AVG(mbp.revenue) AS avg_revenue,
    AVG(mbp.profit_margin) AS avg_profit_margin,
    AVG(mbp.turnover_rate) AS avg_turnover_rate
FROM
    vas_health_performance vhp
JOIN
    companies c ON vhp.company_id = c.company_id
JOIN
    company_size_categories csc ON
        (c.employee_count BETWEEN csc.min_employees AND csc.max_employees OR csc.max_employees IS NULL)
JOIN
    monthly_business_performance mbp
    ON vhp.company_id = mbp.company_id
    AND DATE_TRUNC('month', vhp.record_date) = DATE_TRUNC('month', mbp.report_month)
GROUP BY
    csc.category_id, csc.category_name, DATE_TRUNC('month', vhp.record_date);

-- VASデータの経時変化トレンドビュー
CREATE OR REPLACE VIEW vas_time_trends AS
SELECT
    company_id,
    DATE_TRUNC('month', record_date) AS month,
    AVG(physical_health) AS avg_physical_health,
    LAG(AVG(physical_health), 1) OVER (PARTITION BY company_id ORDER BY DATE_TRUNC('month', record_date)) AS prev_month_physical_health,
    AVG(mental_health) AS avg_mental_health,
    LAG(AVG(mental_health), 1) OVER (PARTITION BY company_id ORDER BY DATE_TRUNC('month', record_date)) AS prev_month_mental_health,
    AVG(work_performance) AS avg_work_performance,
    LAG(AVG(work_performance), 1) OVER (PARTITION BY company_id ORDER BY DATE_TRUNC('month', record_date)) AS prev_month_work_performance,
    AVG(work_satisfaction) AS avg_work_satisfaction,
    LAG(AVG(work_satisfaction), 1) OVER (PARTITION BY company_id ORDER BY DATE_TRUNC('month', record_date)) AS prev_month_work_satisfaction,
    (AVG(physical_health) - LAG(AVG(physical_health), 1) OVER (PARTITION BY company_id ORDER BY DATE_TRUNC('month', record_date))) AS physical_health_change,
    (AVG(mental_health) - LAG(AVG(mental_health), 1) OVER (PARTITION BY company_id ORDER BY DATE_TRUNC('month', record_date))) AS mental_health_change,
    (AVG(work_performance) - LAG(AVG(work_performance), 1) OVER (PARTITION BY company_id ORDER BY DATE_TRUNC('month', record_date))) AS work_performance_change,
    (AVG(work_satisfaction) - LAG(AVG(work_satisfaction), 1) OVER (PARTITION BY company_id ORDER BY DATE_TRUNC('month', record_date))) AS work_satisfaction_change
FROM
    vas_health_performance
GROUP BY
    company_id, DATE_TRUNC('month', record_date);

-- コメント追加
COMMENT ON VIEW vas_monthly_summary IS 'VASデータの月別集計ビュー';
COMMENT ON VIEW vas_industry_summary IS '業種別VASデータ集計ビュー';
COMMENT ON VIEW vas_performance_correlation IS 'VASデータと業績データの相関分析ビュー';
COMMENT ON VIEW company_size_comparison IS '企業規模別VASデータと業績の比較ビュー';
COMMENT ON VIEW vas_time_trends IS 'VASデータの経時変化トレンドビュー';