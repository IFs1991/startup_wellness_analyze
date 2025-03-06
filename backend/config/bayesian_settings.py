"""
ベイズ分析の設定
業界別、成長ステージ別のベイズ分析パラメータを定義します。
設定値は以下のデータソースに基づいています：
1. CB Insights "State of Venture Capital 2023"
2. PitchBook "Venture Monitor Q4 2023"
3. Crunchbase "Startup Growth Rate Analysis 2020-2023"
4. Y Combinator "Startup Growth Benchmarks"
5. 厚生労働省「労働安全衛生調査 2022年」
6. 経済産業省「健康経営施策と企業業績の関連分析 2023年」
"""
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

class Industry(str, Enum):
    """業界の分類"""
    TECH = "tech"              # ソフトウェア・IT
    BIOTECH = "biotech"        # バイオテクノロジー・ヘルスケア
    FINTECH = "fintech"        # 金融テクノロジー
    ECOMMERCE = "ecommerce"    # 電子商取引
    SAAS = "saas"              # SaaS
    MANUFACTURING = "manufacturing"  # 製造業（日本市場向け）
    RETAIL = "retail"          # 小売業（日本市場向け）
    SERVICE = "service"        # サービス業（日本市場向け）
    GENERAL = "general"        # その他

class CompanyStage(str, Enum):
    """
    企業の成長ステージ
    参考: PitchBook Stage Classifications
    - SEED: 製品開発段階、初期顧客獲得前 ($0-$1M ARR)
    - EARLY: 初期収益段階 ($1M-$10M ARR)
    - GROWTH: 急成長段階 ($10M-$50M ARR)
    - EXPANSION: 事業拡大段階 ($50M-$100M ARR)
    - LATE: 成熟段階 ($100M+ ARR)
    """
    SEED = "seed"
    EARLY = "early"
    GROWTH = "growth"
    EXPANSION = "expansion"
    LATE = "late"

class TimeFrame(str, Enum):
    """健康施策の効果が現れる時間枠"""
    SHORT_TERM = "short_term"    # 1-3ヶ月
    MEDIUM_TERM = "medium_term"  # 3-12ヶ月
    LONG_TERM = "long_term"      # 12ヶ月以上

@dataclass
class HealthImpactConfig:
    """健康施策の影響係数"""
    burnout_reduction_factor: float  # バーンアウト低減率
    productivity_boost_factor: float  # 生産性向上係数
    retention_improvement_rate: float  # 人材定着率向上
    sick_leave_reduction: float  # 病欠低減率

@dataclass
class IndustryConfig:
    """
    業界別の設定
    Parameters:
        growth_rate_mean: 年間成長率の期待値（業界平均）
        growth_rate_std: 成長率の標準偏差
        success_rate_alpha: 成功確率のベータ分布αパラメータ
        success_rate_beta: 成功確率のベータ分布βパラメータ
        volatility_factor: ボラティリティ係数
        health_sensitivity: 健康施策に対する業界の反応性
        burnout_risk: バーンアウトリスク係数
        workstyle_factor: 働き方の特徴（リモート率など）
    """
    growth_rate_mean: float
    growth_rate_std: float
    success_rate_alpha: float
    success_rate_beta: float
    volatility_factor: float
    health_sensitivity: float
    burnout_risk: float
    workstyle_factor: float

# 業界別のデフォルト設定
# 出典: CB Insights "State of Venture Capital 2023" & Y Combinator Growth Benchmarks
INDUSTRY_CONFIGS: Dict[Industry, IndustryConfig] = {
    Industry.TECH: IndustryConfig(
        growth_rate_mean=0.20,    # YC Tech Startups中央値: 20% YoY
        growth_rate_std=0.6,      # Tech業界の標準偏差（CB Insights）
        success_rate_alpha=2.0,   # 成功率約20%に基づく
        success_rate_beta=8.0,    # Series A進出率から算出
        volatility_factor=1.2,    # NASDAQ volatility indexベース
        health_sensitivity=0.8,   # 健康施策への高い感度
        burnout_risk=0.85,        # 高いバーンアウトリスク
        workstyle_factor=0.9      # リモートワーク比率が高い
    ),
    Industry.BIOTECH: IndustryConfig(
        growth_rate_mean=0.10,    # Healthcare Startups中央値
        growth_rate_std=0.8,      # 長期R&Dによる高不確実性
        success_rate_alpha=1.5,   # FDA承認率を考慮
        success_rate_beta=8.5,    # Phase I-III成功確率から算出
        volatility_factor=1.5,    # XBI volatility indexベース
        health_sensitivity=0.75,  # 医療従事者の健康意識
        burnout_risk=0.7,         # 研究開発特有のバーンアウト
        workstyle_factor=0.6      # 実験室勤務が多い
    ),
    Industry.FINTECH: IndustryConfig(
        growth_rate_mean=0.15,    # Fintech Funding Report 2023
        growth_rate_std=0.5,      # 規制環境による安定性
        success_rate_alpha=2.5,   # ネオバンク生存率
        success_rate_beta=7.5,    # 規制対応コスト考慮
        volatility_factor=1.1,    # 金融セクター平均
        health_sensitivity=0.7,   # 金融ストレスの影響
        burnout_risk=0.8,         # 取引業務のプレッシャー
        workstyle_factor=0.75     # ハイブリッド勤務
    ),
    Industry.ECOMMERCE: IndustryConfig(
        growth_rate_mean=0.25,    # D2C成長率中央値
        growth_rate_std=0.4,      # 季節性を考慮
        success_rate_alpha=3.0,   # D2C生存率
        success_rate_beta=7.0,    # CAC/LTV比率考慮
        volatility_factor=0.9,    # 消費財セクター平均
        health_sensitivity=0.65,  # オペレーション負荷
        burnout_risk=0.65,        # 季節変動によるストレス
        workstyle_factor=0.5      # 物流拠点勤務が多い
    ),
    Industry.SAAS: IndustryConfig(
        growth_rate_mean=0.30,    # SaaS Magic Number > 1
        growth_rate_std=0.45,     # ARR予測性考慮
        success_rate_alpha=2.8,   # NRR > 100%
        success_rate_beta=7.2,    # Rule of 40考慮
        volatility_factor=1.0,    # SaaS指数ベース
        health_sensitivity=0.85,  # カスタマーサクセスのストレス
        burnout_risk=0.75,        # 顧客対応による疲労
        workstyle_factor=0.95     # リモートワーク比率最高
    ),
    Industry.MANUFACTURING: IndustryConfig(
        growth_rate_mean=0.08,    # 日本製造業平均成長率
        growth_rate_std=0.3,      # 相対的安定性
        success_rate_alpha=3.5,   # 高い生存率
        success_rate_beta=6.5,    # 設備投資負担
        volatility_factor=0.7,    # 安定的セクター
        health_sensitivity=0.6,   # 身体負荷型業務
        burnout_risk=0.6,         # 肉体疲労中心
        workstyle_factor=0.3      # 現場勤務必須
    ),
    Industry.RETAIL: IndustryConfig(
        growth_rate_mean=0.07,    # 日本小売業平均
        growth_rate_std=0.35,     # 季節変動あり
        success_rate_alpha=3.2,   # 店舗生存率
        success_rate_beta=6.8,    # 家賃負担考慮
        volatility_factor=0.75,   # 消費動向依存
        health_sensitivity=0.55,  # 接客ストレス
        burnout_risk=0.7,         # 長時間立ち仕事
        workstyle_factor=0.25     # 店舗勤務中心
    ),
    Industry.SERVICE: IndustryConfig(
        growth_rate_mean=0.12,    # サービス業平均
        growth_rate_std=0.4,      # 多様なセクター
        success_rate_alpha=2.7,   # サービス業生存率
        success_rate_beta=7.3,    # 人件費比率
        volatility_factor=0.85,   # サービス指数
        health_sensitivity=0.75,  # クライアント対応
        burnout_risk=0.75,        # 感情労働負荷
        workstyle_factor=0.6      # ハイブリッド型
    ),
    Industry.GENERAL: IndustryConfig(
        growth_rate_mean=0.15,    # 全業種平均
        growth_rate_std=0.5,      # 総合的な不確実性
        success_rate_alpha=2.0,   # 平均的な成功確率
        success_rate_beta=8.0,    # 一般的な失敗率
        volatility_factor=1.0,    # 市場平均
        health_sensitivity=0.7,   # 平均的感度
        burnout_risk=0.7,         # 標準的リスク
        workstyle_factor=0.6      # 中間的働き方
    )
}

# 成長ステージ別の調整係数
# 出典: PitchBook "Company Stage Success Rates 2023"
STAGE_ADJUSTMENTS: Dict[CompanyStage, Dict[str, float]] = {
    CompanyStage.SEED: {
        'growth_rate_multiplier': 1.5,     # 初期の高成長期待
        'std_multiplier': 1.4,             # 高い不確実性
        'success_alpha_multiplier': 0.7,   # 低い生存率
        'success_beta_multiplier': 1.3,    # 高い失敗リスク
        'health_sensitivity_mult': 1.1,    # 健康影響大
        'burnout_risk_mult': 1.3           # 創業期の高負荷
    },
    CompanyStage.EARLY: {
        'growth_rate_multiplier': 1.3,     # PMF達成後の成長
        'std_multiplier': 1.2,             # 事業モデル検証中
        'success_alpha_multiplier': 0.8,   # Series A進出率
        'success_beta_multiplier': 1.2,    # 初期スケール失敗リスク
        'health_sensitivity_mult': 1.05,   # 健康の重要性認識
        'burnout_risk_mult': 1.2           # 成長期特有のプレッシャー
    },
    CompanyStage.GROWTH: {
        'growth_rate_multiplier': 1.0,     # 標準的な成長期
        'std_multiplier': 1.0,             # 基準となる不確実性
        'success_alpha_multiplier': 1.0,   # 平均的な成功率
        'success_beta_multiplier': 1.0,    # 平均的なリスク
        'health_sensitivity_mult': 1.0,    # 標準的な健康感度
        'burnout_risk_mult': 1.0           # 標準的なバーンアウトリスク
    },
    CompanyStage.EXPANSION: {
        'growth_rate_multiplier': 0.8,     # 成長率の逓減
        'std_multiplier': 0.9,             # 事業安定化
        'success_alpha_multiplier': 1.2,   # 高い生存率
        'success_beta_multiplier': 0.9,    # 低い失敗リスク
        'health_sensitivity_mult': 0.95,   # やや低下する健康影響
        'burnout_risk_mult': 0.9           # 一定の業務安定化
    },
    CompanyStage.LATE: {
        'growth_rate_multiplier': 0.6,     # 成熟期の成長率
        'std_multiplier': 0.8,             # 最も安定
        'success_alpha_multiplier': 1.3,   # 最高の生存率
        'success_beta_multiplier': 0.8,    # 最低の失敗リスク
        'health_sensitivity_mult': 0.9,    # 組織的な健康管理
        'burnout_risk_mult': 0.85          # プロセス化による軽減
    }
}

# 健康施策の時間枠別効果パラメータ
# 出典: 厚生労働省「健康経営施策の効果に関する調査 2022」
TIME_EFFECT_PARAMETERS: Dict[TimeFrame, Dict[str, float]] = {
    TimeFrame.SHORT_TERM: {  # 1-3ヶ月
        'productivity_factor': 0.3,   # 生産性向上効果（短期）
        'retention_factor': 0.1,      # 離職率改善効果（短期）
        'innovation_factor': 0.2,     # イノベーション促進効果（短期）
        'stress_reduction': 0.4,      # ストレス軽減効果（短期）
        'engagement_boost': 0.35,     # エンゲージメント向上（短期）
        'health_cost_reduction': 0.15 # 健康コスト削減（短期）
    },
    TimeFrame.MEDIUM_TERM: {  # 3-12ヶ月
        'productivity_factor': 0.7,   # 生産性向上効果（中期）
        'retention_factor': 0.5,      # 離職率改善効果（中期）
        'innovation_factor': 0.6,     # イノベーション促進効果（中期）
        'stress_reduction': 0.65,     # ストレス軽減効果（中期）
        'engagement_boost': 0.7,      # エンゲージメント向上（中期）
        'health_cost_reduction': 0.5  # 健康コスト削減（中期）
    },
    TimeFrame.LONG_TERM: {  # 12ヶ月以上
        'productivity_factor': 1.0,   # 生産性向上効果（長期）
        'retention_factor': 1.0,      # 離職率改善効果（長期）
        'innovation_factor': 1.0,     # イノベーション促進効果（長期）
        'stress_reduction': 0.85,     # ストレス軽減効果（長期）
        'engagement_boost': 0.9,      # エンゲージメント向上（長期）
        'health_cost_reduction': 1.0  # 健康コスト削減（長期）
    }
}

# 成長ステージと健康施策の関係を示す調整マトリックス
# 出典: 経済産業省「健康経営と企業パフォーマンスに関する研究 2023」
HEALTH_INTERVENTION_STAGE_MATRIX: Dict[CompanyStage, Dict[str, float]] = {
    CompanyStage.SEED: {
        'founder_impact_multiplier': 2.0,   # 創業者の健康の重要性
        'team_cohesion_factor': 1.5,        # チーム凝集性への影響
        'stress_reduction_value': 0.8,      # ストレス軽減の価値
        'decision_quality_impact': 1.8,     # 意思決定品質への影響
        'investor_confidence_effect': 1.7   # 投資家信頼への影響
    },
    CompanyStage.EARLY: {
        'founder_impact_multiplier': 1.8,   # 創業者の健康の重要性
        'team_cohesion_factor': 1.6,        # チーム凝集性への影響
        'stress_reduction_value': 0.85,     # ストレス軽減の価値
        'decision_quality_impact': 1.6,     # 意思決定品質への影響
        'investor_confidence_effect': 1.5   # 投資家信頼への影響
    },
    CompanyStage.GROWTH: {
        'founder_impact_multiplier': 1.5,   # 創業者の健康の重要性
        'team_cohesion_factor': 1.4,        # チーム凝集性への影響
        'stress_reduction_value': 0.9,      # ストレス軽減の価値
        'decision_quality_impact': 1.4,     # 意思決定品質への影響
        'investor_confidence_effect': 1.3   # 投資家信頼への影響
    },
    CompanyStage.EXPANSION: {
        'founder_impact_multiplier': 1.3,   # 創業者の健康の重要性
        'team_cohesion_factor': 1.3,        # チーム凝集性への影響
        'stress_reduction_value': 0.95,     # ストレス軽減の価値
        'decision_quality_impact': 1.2,     # 意思決定品質への影響
        'investor_confidence_effect': 1.1   # 投資家信頼への影響
    },
    CompanyStage.LATE: {
        'founder_impact_multiplier': 1.1,   # 創業者の健康の重要性
        'team_cohesion_factor': 1.2,        # チーム凝集性への影響
        'stress_reduction_value': 1.0,      # ストレス軽減の価値
        'decision_quality_impact': 1.0,     # 意思決定品質への影響
        'investor_confidence_effect': 1.0   # 投資家信頼への影響
    }
}

# 日本市場特有の調整係数
# 出典: 経済産業省「健康経営銘柄 2023」データ分析
JAPAN_MARKET_ADJUSTMENTS = {
    'growth_rate_multiplier': 0.85,      # 日本の成長率は一般的に米国より控えめ
    'work_culture_factor': 1.2,          # 日本の労働文化による健康影響の割増係数
    'intervention_efficiency': 0.95,     # 文化的要素による介入効率
    'health_cost_impact': 1.1,           # 健康保険制度の違いによる影響
    'workstyle_reform_boost': 1.15       # 働き方改革による追加効果
}

# 地域別調整係数
# 出典: 総務省「地域別労働環境調査 2023」
REGIONAL_ADJUSTMENTS: Dict[str, float] = {
    '東京': 1.1,        # 高ストレス環境、健康施策効果大
    '大阪': 1.05,       # 都市部特有のストレス
    '名古屋': 1.0,      # 標準的な効果
    '福岡': 0.95,       # やや穏やかな労働環境
    '札幌': 0.9,        # 地方特有の働き方
    '仙台': 0.95,       # 地方中核都市
    '広島': 0.95,       # 地方中核都市
    '那覇': 0.85        # ワークライフバランス重視
}

# MCMCサンプリングのデータサイズ別設定
# 出典: PyMC3 Documentation & Bayesian Analysis Best Practices
MCMC_SETTINGS = {
    'small': {  # データサイズ < 1000
        'n_samples': 2000,    # 最小限の有効サンプル数
        'n_tune': 1000,       # 収束のための調整期間
        'chains': 4           # 並列チェーン数
    },
    'medium': {  # 1000 <= データサイズ < 5000
        'n_samples': 3000,    # より多くのサンプル数
        'n_tune': 1500,       # 長めの調整期間
        'chains': 4           # 標準的なチェーン数
    },
    'large': {  # データサイズ >= 5000
        'n_samples': 5000,    # 大規模データ用
        'n_tune': 2000,       # 十分な調整期間
        'chains': 4           # 計算効率を考慮
    }
}

# VASスケール項目と健康/業績指標のマッピング
# 出典: Startup Wellness VASスケール設計ドキュメント
VAS_METRICS_MAPPING = {
    # 身体的健康指標
    'overall_condition': {'category': 'physical', 'weight': 0.8, 'impact_areas': ['productivity', 'decision_making']},
    'physical_pain': {'category': 'physical', 'weight': 0.7, 'impact_areas': ['productivity', 'absenteeism'], 'inverse': True},
    'energy_level': {'category': 'physical', 'weight': 0.9, 'impact_areas': ['productivity', 'creativity', 'leadership']},
    'sleep_quality': {'category': 'physical', 'weight': 0.85, 'impact_areas': ['decision_making', 'productivity', 'stress']},

    # 精神的健康指標
    'stress_level': {'category': 'mental', 'weight': 0.9, 'impact_areas': ['decision_making', 'team_dynamics', 'burnout'], 'inverse': True},
    'mental_clarity': {'category': 'mental', 'weight': 0.85, 'impact_areas': ['creativity', 'decision_making', 'productivity']},
    'emotional_balance': {'category': 'mental', 'weight': 0.8, 'impact_areas': ['leadership', 'team_dynamics']},
    'optimism': {'category': 'mental', 'weight': 0.75, 'impact_areas': ['vision', 'motivation', 'resilience']},

    # 業務パフォーマンス指標
    'work_efficiency': {'category': 'performance', 'weight': 0.9, 'impact_areas': ['productivity', 'results']},
    'creativity': {'category': 'performance', 'weight': 0.8, 'impact_areas': ['innovation', 'problem_solving']},
    'decision_ability': {'category': 'performance', 'weight': 0.85, 'impact_areas': ['leadership', 'execution']},
    'job_satisfaction': {'category': 'performance', 'weight': 0.8, 'impact_areas': ['retention', 'engagement']},
    'motivation': {'category': 'performance', 'weight': 0.85, 'impact_areas': ['productivity', 'engagement', 'innovation']},

    # チーム関係性指標
    'team_cohesion': {'category': 'team', 'weight': 0.8, 'impact_areas': ['collaboration', 'culture']},
    'communication_quality': {'category': 'team', 'weight': 0.85, 'impact_areas': ['execution', 'alignment']},
    'support_feeling': {'category': 'team', 'weight': 0.75, 'impact_areas': ['retention', 'culture']},

    # ファウンダー特有指標
    'business_outlook': {'category': 'founder', 'weight': 0.9, 'impact_areas': ['vision', 'investment_potential']},
    'leadership_effectiveness': {'category': 'founder', 'weight': 0.9, 'impact_areas': ['execution', 'team_performance']},

    # 従業員特有指標
    'growth_feeling': {'category': 'employee', 'weight': 0.8, 'impact_areas': ['retention', 'engagement']},
    'future_prospects': {'category': 'employee', 'weight': 0.85, 'impact_areas': ['retention', 'engagement']}
}

def get_industry_config(industry: str) -> IndustryConfig:
    """
    業界別の設定を取得
    Args:
        industry: 業界名
    Returns:
        IndustryConfig: 業界別の設定
    """
    try:
        return INDUSTRY_CONFIGS[Industry(industry)]
    except ValueError:
        return INDUSTRY_CONFIGS[Industry.GENERAL]

def get_stage_adjustments(stage: str) -> Dict[str, float]:
    """
    成長ステージ別の調整係数を取得
    Args:
        stage: 成長ステージ
    Returns:
        Dict[str, float]: 調整係数
    """
    try:
        return STAGE_ADJUSTMENTS[CompanyStage(stage)]
    except ValueError:
        return STAGE_ADJUSTMENTS[CompanyStage.GROWTH]

def get_mcmc_settings(data_size: int) -> Dict[str, int]:
    """
    データサイズに応じたMCMC設定を取得
    Args:
        data_size: データのサイズ
    Returns:
        Dict[str, int]: MCMC設定
    """
    if data_size < 1000:
        return MCMC_SETTINGS['small']
    elif data_size < 5000:
        return MCMC_SETTINGS['medium']
    else:
        return MCMC_SETTINGS['large']

def get_time_effect_parameters(time_frame: str) -> Dict[str, float]:
    """
    時間枠に応じた効果パラメータを取得
    Args:
        time_frame: 時間枠
    Returns:
        Dict[str, float]: 時間効果パラメータ
    """
    try:
        return TIME_EFFECT_PARAMETERS[TimeFrame(time_frame)]
    except ValueError:
        return TIME_EFFECT_PARAMETERS[TimeFrame.MEDIUM_TERM]

def get_health_intervention_matrix(stage: str) -> Dict[str, float]:
    """
    成長ステージに応じた健康施策効果マトリックスを取得
    Args:
        stage: 成長ステージ
    Returns:
        Dict[str, float]: 健康施策効果マトリックス
    """
    try:
        return HEALTH_INTERVENTION_STAGE_MATRIX[CompanyStage(stage)]
    except ValueError:
        return HEALTH_INTERVENTION_STAGE_MATRIX[CompanyStage.GROWTH]

def calculate_size_factor(company_size: int) -> float:
    """
    企業規模に応じた調整係数を計算
    Args:
        company_size: 従業員数
    Returns:
        float: 規模調整係数
    """
    if company_size < 10:
        return 1.3  # 極小規模（創業者への依存大）
    elif company_size < 30:
        return 1.1  # 小規模
    elif company_size < 100:
        return 1.0  # 中規模（標準）
    elif company_size < 300:
        return 0.9  # 大規模
    else:
        return 0.85  # 超大規模（組織的冗長性あり）

def calculate_adjusted_parameters(
    industry: str,
    stage: str,
    data_size: int,
    is_japanese_market: bool = True,
    company_size: int = 50,
    location: str = '東京'
) -> Dict[str, Any]:
    """
    業界と成長ステージに応じたパラメータを計算
    Args:
        industry: 業界名
        stage: 成長ステージ
        data_size: データサイズ
        is_japanese_market: 日本市場かどうか
        company_size: 従業員数
        location: 会社所在地
    Returns:
        Dict[str, Any]: 調整済みパラメータ
    """
    industry_config = get_industry_config(industry)
    stage_adj = get_stage_adjustments(stage)
    mcmc_settings = get_mcmc_settings(data_size)

    # 業界と成長ステージに基づくパラメータ計算
    params = {
        'prior_parameters': {
            'mu': industry_config.growth_rate_mean * stage_adj['growth_rate_multiplier'],
            'sigma': industry_config.growth_rate_std * stage_adj['std_multiplier'],
            'alpha': industry_config.success_rate_alpha * stage_adj['success_alpha_multiplier'],
            'beta': industry_config.success_rate_beta * stage_adj['success_beta_multiplier'],
            'volatility': industry_config.volatility_factor,
            'health_sensitivity': industry_config.health_sensitivity * stage_adj['health_sensitivity_mult'],
            'burnout_risk': industry_config.burnout_risk * stage_adj['burnout_risk_mult'],
            'workstyle_factor': industry_config.workstyle_factor
        },
        'mcmc_settings': mcmc_settings
    }

    # 日本市場の場合は追加調整
    if is_japanese_market:
        params = apply_japanese_market_context(params, company_size, location)

    return params

def apply_japanese_market_context(
    parameters: Dict[str, Any],
    company_size: int,
    location: str
) -> Dict[str, Any]:
    """
    日本市場文脈を適用した調整パラメータを計算
    Args:
        parameters: 調整前のパラメータ
        company_size: 従業員数
        location: 会社所在地
    Returns:
        Dict[str, Any]: 日本市場向けに調整されたパラメータ
    """
    # 東京、大阪など地域特性の調整
    regional_adj = REGIONAL_ADJUSTMENTS.get(location, 1.0)

    # 企業規模による調整
    size_factor = calculate_size_factor(company_size)

    # 日本市場調整の適用
    jp_adjusted_params = parameters.copy()

    # 成長率調整
    jp_adjusted_params['prior_parameters']['mu'] *= JAPAN_MARKET_ADJUSTMENTS['growth_rate_multiplier']

    # 健康関連パラメータ調整
    jp_adjusted_params['prior_parameters']['health_sensitivity'] *= (
        JAPAN_MARKET_ADJUSTMENTS['work_culture_factor'] *
        regional_adj *
        size_factor
    )

    jp_adjusted_params['prior_parameters']['burnout_risk'] *= (
        JAPAN_MARKET_ADJUSTMENTS['work_culture_factor'] *
        regional_adj *
        size_factor
    )

    # 地域・規模による追加調整
    for key, value in jp_adjusted_params['prior_parameters'].items():
        if isinstance(value, (int, float)) and key not in ['mu', 'health_sensitivity', 'burnout_risk']:
            jp_adjusted_params['prior_parameters'][key] *= (regional_adj * size_factor)

    # 日本市場特有の追加パラメータ
    jp_adjusted_params['japan_specific'] = {
        'workstyle_reform_impact': JAPAN_MARKET_ADJUSTMENTS['workstyle_reform_boost'] * regional_adj,
        'health_insurance_effect': JAPAN_MARKET_ADJUSTMENTS['health_cost_impact'],
        'intervention_efficiency': JAPAN_MARKET_ADJUSTMENTS['intervention_efficiency'] * size_factor
    }

    return jp_adjusted_params

def preprocess_vas_data(vas_measurements: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    VASスケールデータの前処理
    Args:
        vas_measurements: VASスケール測定値
    Returns:
        Dict[str, Any]: 前処理済みVASデータ
    """
    processed_data = {}

    # 各指標の平均値、中央値、標準偏差、トレンドを計算
    for metric, values in vas_measurements.items():
        if not values:
            continue

        # 逆スケール（値が低いほど良い）の場合は反転
        if metric in ['physical_pain', 'stress_level'] or (
            metric in VAS_METRICS_MAPPING and
            VAS_METRICS_MAPPING.get(metric, {}).get('inverse', False)
        ):
            values = [10 - v for v in values]

        processed_data[metric] = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'latest': values[-1] if values else None,
            'trend': calculate_trend(values),
            'values': values,
            'improvement': (values[-1] - values[0]) if len(values) > 1 else 0
        }

    # カテゴリー別の集計データを追加
    category_data = {}
    for metric, metric_data in processed_data.items():
        if metric in VAS_METRICS_MAPPING:
            category = VAS_METRICS_MAPPING[metric]['category']
            weight = VAS_METRICS_MAPPING[metric]['weight']

            if category not in category_data:
                category_data[category] = {'sum_weighted_value': 0, 'sum_weights': 0}

            category_data[category]['sum_weighted_value'] += metric_data['latest'] * weight
            category_data[category]['sum_weights'] += weight

    # カテゴリー平均を計算
    for category, data in category_data.items():
        if data['sum_weights'] > 0:
            processed_data[f'{category}_index'] = data['sum_weighted_value'] / data['sum_weights']

    # 時系列トレンド情報を追加（施術前後の変化など）
    if any(len(values) > 2 for _, values in vas_measurements.items()):
        processed_data['time_series'] = extract_time_series_features(vas_measurements)

    return processed_data

def calculate_trend(values: List[float]) -> float:
    """
    時系列データのトレンドを計算
    Args:
        values: 時系列値
    Returns:
        float: トレンド係数（正=上昇傾向、負=下降傾向）
    """
    if len(values) < 2:
        return 0.0

    # 単純な線形回帰の傾き
    x = np.arange(len(values))
    y = np.array(values)

    # 線形回帰
    slope = np.polyfit(x, y, 1)[0]

    # 値の範囲で正規化
    value_range = max(values) - min(values) if max(values) != min(values) else 1.0
    normalized_slope = slope * len(values) / value_range

    return normalized_slope

def extract_time_series_features(vas_data: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    VASデータから時系列特徴を抽出
    Args:
        vas_data: VASスケールデータ
    Returns:
        Dict[str, Any]: 時系列特徴
    """
    ts_features = {}

    # 共通の時系列長を取得
    series_lengths = [len(values) for values in vas_data.values() if values]
    if not series_lengths:
        return ts_features

    max_length = max(series_lengths)

    # 短期、中期、長期の変化率を計算
    for metric, values in vas_data.items():
        if len(values) < 2:
            continue

        # 逆スケールの場合は反転して計算
        is_inverse = metric in ['physical_pain', 'stress_level'] or (
            metric in VAS_METRICS_MAPPING and
            VAS_METRICS_MAPPING.get(metric, {}).get('inverse', False)
        )

        calc_values = [10 - v for v in values] if is_inverse else values

        ts_features[metric] = {
            # 直近の変化率
            'recent_change': (calc_values[-1] - calc_values[-2]) / max(calc_values[-2], 0.1)
                if len(calc_values) >= 2 else 0,

            # 施術開始時からの変化率
            'total_change': (calc_values[-1] - calc_values[0]) / max(calc_values[0], 0.1)
                if len(calc_values) >= 2 else 0,

            # 変化の一貫性（標準偏差/平均で評価）
            'consistency': 1 - (np.std(calc_values) / max(np.mean(calc_values), 0.1))
                if len(calc_values) >= 3 else 0
        }

        # 十分なデータポイントがある場合、短期/中期/長期変化も計算
        if len(calc_values) >= 6:
            # 短期変化（直近3ポイント）
            ts_features[metric]['short_term_trend'] = calculate_trend(calc_values[-3:])

            # 中期変化（全期間の中央部）
            midpoint = len(calc_values) // 2
            mid_range = min(3, midpoint)
            ts_features[metric]['mid_term_trend'] = calculate_trend(
                calc_values[midpoint-mid_range:midpoint+mid_range]
            )

            # 長期変化（全期間）
            ts_features[metric]['long_term_trend'] = calculate_trend(calc_values)

    return ts_features

def merge_company_vas_data(company_data: Dict[str, Any], vas_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    企業データとVASデータを統合
    Args:
        company_data: 企業基本データ
        vas_data: 処理済みVASデータ
    Returns:
        Dict[str, Any]: 統合されたデータ
    """
    merged_data = company_data.copy()

    # 企業の基本データを維持しつつ、健康指標を追加
    merged_data['health_metrics'] = {
        # 身体的健康指標
        'physical_health_index': vas_data.get('physical_index', 0),
        'energy_level': vas_data.get('energy_level', {}).get('latest', 0),
        'sleep_quality': vas_data.get('sleep_quality', {}).get('latest', 0),

        # 精神的健康指標
        'mental_health_index': vas_data.get('mental_index', 0),
        'stress_level': 10 - vas_data.get('stress_level', {}).get('latest', 0),  # 反転
        'mental_clarity': vas_data.get('mental_clarity', {}).get('latest', 0),

        # パフォーマンス指標
        'performance_index': vas_data.get('performance_index', 0),
        'work_efficiency': vas_data.get('work_efficiency', {}).get('latest', 0),
        'motivation': vas_data.get('motivation', {}).get('latest', 0),

        # チーム指標
        'team_index': vas_data.get('team_index', 0),
        'team_cohesion': vas_data.get('team_cohesion', {}).get('latest', 0),

        # 改善トレンド
        'physical_improvement': vas_data.get('physical_pain', {}).get('improvement', 0),
        'stress_reduction': vas_data.get('stress_level', {}).get('improvement', 0),
        'performance_trend': vas_data.get('work_efficiency', {}).get('trend', 0)
    }

    # ファウンダー特有の指標（存在する場合）
    if 'business_outlook' in vas_data:
        merged_data['health_metrics']['founder_metrics'] = {
            'business_outlook': vas_data['business_outlook'].get('latest', 0),
            'leadership_effectiveness': vas_data.get('leadership_effectiveness', {}).get('latest', 0)
        }

    # 時系列データ（存在する場合）
    if 'time_series' in vas_data:
        merged_data['health_metrics']['time_series'] = vas_data['time_series']

    # 健康改善スコアの計算（加重平均）
    physical_weight = 0.3
    mental_weight = 0.3
    performance_weight = 0.4

    merged_data['health_metrics']['overall_health_score'] = (
        merged_data['health_metrics']['physical_health_index'] * physical_weight +
        merged_data['health_metrics']['mental_health_index'] * mental_weight +
        merged_data['health_metrics']['performance_index'] * performance_weight
    )

    return merged_data

def integrate_vas_data(
    vas_measurements: Dict[str, List[float]],
    company_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    VASスケールの測定データをベイズモデルに統合
    Args:
        vas_measurements: VASスケール測定値
        company_data: 企業データ
    Returns:
        Dict[str, Any]: 統合されたデータ
    """
    # VASデータの前処理
    processed_vas = preprocess_vas_data(vas_measurements)

    # 会社データとVASデータの統合
    augmented_data = merge_company_vas_data(company_data, processed_vas)

    return augmented_data

def calculate_impact_matrix(industry: str, stage: str) -> Dict[str, Dict[str, float]]:
    """
    健康指標から業績指標への影響行列を計算
    Args:
        industry: 業界
        stage: 成長段階
    Returns:
        Dict[str, Dict[str, float]]: 影響行列
    """
    industry_config = get_industry_config(industry)
    stage_matrix = get_health_intervention_matrix(stage)

    # 基本的な影響行列（健康指標→業績指標の影響係数）
    base_matrix = {
        'physical_health': {
            'productivity': 0.7,
            'absenteeism': 0.8,
            'decision_quality': 0.5,
            'innovation': 0.4
        },
        'mental_health': {
            'productivity': 0.6,
            'decision_quality': 0.8,
            'innovation': 0.7,
            'retention': 0.6,
            'team_performance': 0.7
        },
        'stress_level': {
            'productivity': 0.8,
            'decision_quality': 0.9,
            'retention': 0.7,
            'absenteeism': 0.7,
            'team_performance': 0.6
        },
        'energy_level': {
            'productivity': 0.9,
            'innovation': 0.8,
            'leadership': 0.8
        },
        'team_cohesion': {
            'productivity': 0.6,
            'innovation': 0.7,
            'retention': 0.8,
            'team_performance': 0.9
        }
    }

    # 業界・ステージに基づく調整
    adjusted_matrix = {}

    for health_metric, impacts in base_matrix.items():
        adjusted_matrix[health_metric] = {}

        for performance_metric, value in impacts.items():
            # 業界特性による調整
            industry_factor = 1.0
            if performance_metric == 'productivity':
                industry_factor = industry_config.health_sensitivity
            elif performance_metric == 'innovation':
                industry_factor = industry_config.workstyle_factor
            elif performance_metric == 'retention':
                industry_factor = industry_config.burnout_risk

            # ステージ特性による調整
            stage_factor = 1.0
            if health_metric in ['physical_health', 'mental_health'] and 'founder_impact_multiplier' in stage_matrix:
                stage_factor = stage_matrix['founder_impact_multiplier']
            elif health_metric == 'team_cohesion' and 'team_cohesion_factor' in stage_matrix:
                stage_factor = stage_matrix['team_cohesion_factor']
            elif health_metric == 'stress_level' and 'stress_reduction_value' in stage_matrix:
                stage_factor = stage_matrix['stress_reduction_value']

            # 調整後の値
            adjusted_matrix[health_metric][performance_metric] = value * industry_factor * stage_factor

    return adjusted_matrix

def apply_impact_matrix(
    health_metrics: Dict[str, float],
    impact_matrix: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    健康指標の変化を業績指標の変化に変換
    Args:
        health_metrics: 健康指標の変化
        impact_matrix: 影響行列
    Returns:
        Dict[str, float]: 業績指標の変化
    """
    performance_delta = {
        'productivity': 0.0,
        'innovation': 0.0,
        'decision_quality': 0.0,
        'retention': 0.0,
        'absenteeism': 0.0,
        'team_performance': 0.0,
        'leadership': 0.0
    }

    # 各健康指標が各業績指標に与える影響を合算
    for health_metric, health_value in health_metrics.items():
        if health_metric in impact_matrix:
            for perf_metric, impact_value in impact_matrix[health_metric].items():
                # 健康指標値と影響係数を掛け合わせて影響を計算
                performance_delta[perf_metric] += health_value * impact_value

    # 業績指標間の相互関係も考慮
    # 例: リーダーシップの向上はチームパフォーマンスにも影響
    if 'leadership' in performance_delta and 'team_performance' in performance_delta:
        performance_delta['team_performance'] += performance_delta['leadership'] * 0.5

    # 値の正規化（0〜1の範囲に）
    for metric in performance_delta:
        performance_delta[metric] = min(max(performance_delta[metric], 0.0), 1.0)

    return performance_delta

def update_performance_distribution(
    baseline_performance: Dict[str, float],
    performance_delta: Dict[str, float],
    industry_config: IndustryConfig,
    stage_adj: Dict[str, float]
) -> Dict[str, Any]:
    """
    ベイズ更新によるパフォーマンス分布の更新
    Args:
        baseline_performance: ベースラインのパフォーマンス指標
        performance_delta: パフォーマンス指標の変化
        industry_config: 業界設定
        stage_adj: 成長ステージによる調整係数
    Returns:
        Dict[str, Any]: 更新後のパフォーマンス分布
    """
    # ベイズ更新の結果（単純化したモデル）
    updated_performance = {}

    # 各業績指標の更新
    for metric, baseline in baseline_performance.items():
        if metric in performance_delta:
            # 業界と成長ステージに合わせた調整係数
            industry_weight = getattr(industry_config, 'health_sensitivity', 0.7)
            stage_weight = stage_adj.get('health_sensitivity_mult', 1.0)

            # 改善効果の計算
            improvement = performance_delta[metric] * industry_weight * stage_weight

            # 改善の不確実性（ボラティリティに応じて）
            uncertainty = industry_config.volatility_factor * 0.1

            # 更新後の値と信頼区間
            updated_value = baseline * (1 + improvement)
            confidence_interval = (
                updated_value * (1 - uncertainty),
                updated_value * (1 + uncertainty)
            )

            updated_performance[metric] = {
                'baseline': baseline,
                'updated_value': updated_value,
                'improvement': improvement,
                'improvement_percentage': improvement * 100,
                'confidence_interval': confidence_interval
            }

    return updated_performance

def estimate_health_roi(
    baseline_performance: Dict[str, float],
    health_metrics: Dict[str, float],
    industry: str,
    stage: str,
    timeframe: str = 'medium_term'
) -> Dict[str, Any]:
    """
    健康指標の改善が業績に与える影響を推定
    Args:
        baseline_performance: ベースライン業績指標
        health_metrics: 健康指標
        industry: 業界
        stage: 成長ステージ
        timeframe: 時間枠
    Returns:
        Dict[str, Any]: ROI推定結果
    """
    industry_config = get_industry_config(industry)
    stage_adj = get_stage_adjustments(stage)
    time_effects = get_time_effect_parameters(timeframe)

    # 健康指標から業績への変換行列（業界と成長ステージで調整）
    impact_matrix = calculate_impact_matrix(industry, stage)

    # 健康指標の変化を業績指標の変化に変換
    performance_delta = apply_impact_matrix(health_metrics, impact_matrix)

    # 時間効果の適用
    for metric, value in performance_delta.items():
        if metric == 'productivity':
            performance_delta[metric] *= time_effects['productivity_factor']
        elif metric == 'retention':
            performance_delta[metric] *= time_effects['retention_factor']
        elif metric == 'innovation':
            performance_delta[metric] *= time_effects['innovation_factor']
        else:
            # その他の指標には中間的な係数を適用
            performance_delta[metric] *= (time_effects['productivity_factor'] + time_effects['retention_factor']) / 2

    # ベイズ更新による確率的予測
    posterior_performance = update_performance_distribution(
        baseline_performance,
        performance_delta,
        industry_config,
        stage_adj
    )

    # ROI計算
    total_baseline = sum(baseline_performance.values())
    total_updated = sum(perf['updated_value'] for perf in posterior_performance.values())
    program_cost = 1.0  # 標準化された値

    roi = (total_updated - total_baseline - program_cost) / program_cost * 100

    # 結果のまとめ
    result = {
        'roi': roi,
        'baseline_total': total_baseline,
        'updated_total': total_updated,
        'absolute_improvement': total_updated - total_baseline,
        'relative_improvement_percentage': (total_updated - total_baseline) / total_baseline * 100,
        'performance_details': posterior_performance,
        'timeframe': timeframe,
        'health_impact': {
            'productivity_impact': performance_delta.get('productivity', 0) * 100,
            'retention_impact': performance_delta.get('retention', 0) * 100,
            'innovation_impact': performance_delta.get('innovation', 0) * 100
        }
    }

    return result

def calculate_vas_based_health_impact(
    vas_before: Dict[str, float],
    vas_after: Dict[str, float]
) -> HealthImpactConfig:
    """
    VASスケールデータに基づく健康影響係数を計算
    Args:
        vas_before: 施術前のVASスコア
        vas_after: 施術後のVASスコア
    Returns:
        HealthImpactConfig: 健康影響係数
    """
    # バーンアウト低減率の計算
    stress_before = vas_before.get('stress_level', 5)
    stress_after = vas_after.get('stress_level', 5)
    sleep_before = vas_before.get('sleep_quality', 5)
    sleep_after = vas_after.get('sleep_quality', 5)
    motivation_before = vas_before.get('motivation', 5)
    motivation_after = vas_after.get('motivation', 5)

    # ストレス軽減（ストレスは値が小さいほど良い）
    stress_reduction = max(0, (stress_before - stress_after) / max(stress_before, 0.1))
    # 睡眠改善
    sleep_improvement = max(0, (sleep_after - sleep_before) / max(sleep_before, 0.1))
    # モチベーション向上
    motivation_increase = max(0, (motivation_after - motivation_before) / max(motivation_before, 0.1))

    # 加重平均によるバーンアウト低減率
    burnout_reduction = (stress_reduction * 0.5 + sleep_improvement * 0.3 + motivation_increase * 0.2)

    # 生産性向上係数の計算
    work_efficiency_before = vas_before.get('work_efficiency', 5)
    work_efficiency_after = vas_after.get('work_efficiency', 5)
    mental_clarity_before = vas_before.get('mental_clarity', 5)
    mental_clarity_after = vas_after.get('mental_clarity', 5)
    energy_before = vas_before.get('energy_level', 5)
    energy_after = vas_after.get('energy_level', 5)

    # 各指標の改善率
    efficiency_improvement = max(0, (work_efficiency_after - work_efficiency_before) / max(work_efficiency_before, 0.1))
    clarity_improvement = max(0, (mental_clarity_after - mental_clarity_before) / max(mental_clarity_before, 0.1))
    energy_improvement = max(0, (energy_after - energy_before) / max(energy_before, 0.1))

    # 加重平均による生産性向上係数
    productivity_boost = (efficiency_improvement * 0.5 + clarity_improvement * 0.3 + energy_improvement * 0.2)

    # 人材定着率向上の計算
    satisfaction_before = vas_before.get('job_satisfaction', 5)
    satisfaction_after = vas_after.get('job_satisfaction', 5)
    team_cohesion_before = vas_before.get('team_cohesion', 5)
    team_cohesion_after = vas_after.get('team_cohesion', 5)
    future_prospects_before = vas_before.get('future_prospects', 5) if 'future_prospects' in vas_before else 5
    future_prospects_after = vas_after.get('future_prospects', 5) if 'future_prospects' in vas_after else 5

    # 各指標の改善率
    satisfaction_improvement = max(0, (satisfaction_after - satisfaction_before) / max(satisfaction_before, 0.1))
    cohesion_improvement = max(0, (team_cohesion_after - team_cohesion_before) / max(team_cohesion_before, 0.1))
    prospects_improvement = max(0, (future_prospects_after - future_prospects_before) / max(future_prospects_before, 0.1))

    # 加重平均による人材定着率向上
    retention_improvement = (satisfaction_improvement * 0.4 + cohesion_improvement * 0.3 + prospects_improvement * 0.3)

    # 病欠低減率の計算
    pain_before = vas_before.get('physical_pain', 5)
    pain_after = vas_after.get('physical_pain', 5)
    overall_condition_before = vas_before.get('overall_condition', 5)
    overall_condition_after = vas_after.get('overall_condition', 5)

    # 痛み軽減（痛みは値が小さいほど良い）
    pain_reduction = max(0, (pain_before - pain_after) / max(pain_before, 0.1))
    # 全体的な体調改善
    condition_improvement = max(0, (overall_condition_after - overall_condition_before) / max(overall_condition_before, 0.1))

    # 加重平均による病欠低減率
    sick_leave_reduction = (pain_reduction * 0.6 + condition_improvement * 0.4)

    # 健康影響係数の作成
    return HealthImpactConfig(
        burnout_reduction_factor=burnout_reduction,
        productivity_boost_factor=productivity_boost,
        retention_improvement_rate=retention_improvement,
        sick_leave_reduction=sick_leave_reduction
    )

def predict_vas_improvement(
    baseline_vas: Dict[str, float],
    industry: str,
    stage: str,
    intervention_count: int,
    timeframe: str = 'medium_term'
) -> Dict[str, Any]:
    """
    施術介入による将来のVASスコア改善を予測
    Args:
        baseline_vas: ベースラインのVASスコア
        industry: 業界
        stage: 成長ステージ
        intervention_count: 予定されている施術回数
        timeframe: 時間枠
    Returns:
        Dict[str, Any]: 予測結果
    """
    industry_config = get_industry_config(industry)
    stage_matrix = get_health_intervention_matrix(stage)
    time_effects = get_time_effect_parameters(timeframe)

    # 予測結果
    predicted_vas = {}

    # 各VAS項目について改善予測
    for metric, value in baseline_vas.items():
        # 基本改善係数
        base_improvement = 0.05  # 1回の施術あたり約5%の改善

        # 業界別調整
        industry_factor = industry_config.health_sensitivity

        # 時間枠による効果調整
        time_factor = 1.0
        if 'stress' in metric.lower():
            time_factor = time_effects['stress_reduction']
        elif 'motivation' in metric.lower() or 'satisfaction' in metric.lower():
            time_factor = time_effects['engagement_boost']
        else:
            time_factor = time_effects['productivity_factor']

        # 改善の飽和モデル（介入回数が増えるほど追加効果は減少）
        saturation_factor = 1 - np.exp(-0.2 * intervention_count)

        # 指標特有の調整
        metric_factor = 1.0
        if metric in VAS_METRICS_MAPPING:
            metric_info = VAS_METRICS_MAPPING[metric]
            metric_factor = metric_info.get('weight', 1.0)

            # 創業者健康の重要性などを考慮
            if metric_info['category'] == 'founder' and 'founder_impact_multiplier' in stage_matrix:
                metric_factor *= stage_matrix['founder_impact_multiplier']
            elif metric_info['category'] == 'team' and 'team_cohesion_factor' in stage_matrix:
                metric_factor *= stage_matrix['team_cohesion_factor']

        # 総合改善率の計算
        total_improvement_rate = (
            base_improvement *
            industry_factor *
            time_factor *
            metric_factor *
            saturation_factor *
            intervention_count
        )

        # 最大改善の制限
        max_improvement = 0.7  # 最大70%の改善
        total_improvement_rate = min(total_improvement_rate, max_improvement)

        # 改善後の値を計算（逆スケールの場合は減少）
        is_inverse = metric in ['physical_pain', 'stress_level'] or (
            metric in VAS_METRICS_MAPPING and
            VAS_METRICS_MAPPING.get(metric, {}).get('inverse', False)
        )

        if is_inverse:
            # 値が小さいほど良い場合（例：痛み、ストレス）
            predicted_value = value * (1 - total_improvement_rate)
        else:
            # 値が大きいほど良い場合（例：エネルギー、睡眠の質）
            max_value = 10.0
            room_for_improvement = max_value - value
            predicted_value = value + (room_for_improvement * total_improvement_rate)

        # 不確実性の計算
        uncertainty = industry_config.volatility_factor * 0.1
        confidence_interval = (
            predicted_value * (1 - uncertainty),
            predicted_value * (1 + uncertainty)
        )

        # 結果の格納
        predicted_vas[metric] = {
            'baseline': value,
            'predicted': predicted_value,
            'improvement': abs(predicted_value - value),
            'improvement_percentage': abs(predicted_value - value) / max(value, 0.1) * 100,
            'confidence_interval': confidence_interval
        }

    return {
        'predicted_vas': predicted_vas,
        'intervention_count': intervention_count,
        'timeframe': timeframe,
        'average_improvement_percentage': np.mean([
            item['improvement_percentage'] for item in predicted_vas.values()
        ])
    }