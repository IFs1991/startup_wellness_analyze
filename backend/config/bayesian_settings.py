"""
ベイズ分析の設定

業界別、成長ステージ別のベイズ分析パラメータを定義します。
設定値は以下のデータソースに基づいています：

1. CB Insights "State of Venture Capital 2023"
2. PitchBook "Venture Monitor Q4 2023"
3. Crunchbase "Startup Growth Rate Analysis 2020-2023"
4. Y Combinator "Startup Growth Benchmarks"
"""

from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

class Industry(str, Enum):
    """業界の分類"""
    TECH = "tech"          # ソフトウェア・IT
    BIOTECH = "biotech"    # バイオテクノロジー・ヘルスケア
    FINTECH = "fintech"    # 金融テクノロジー
    ECOMMERCE = "ecommerce"  # 電子商取引
    SAAS = "saas"          # SaaS
    GENERAL = "general"    # その他

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
    """
    growth_rate_mean: float
    growth_rate_std: float
    success_rate_alpha: float
    success_rate_beta: float
    volatility_factor: float

# 業界別のデフォルト設定
# 出典: CB Insights "State of Venture Capital 2023" & Y Combinator Growth Benchmarks
INDUSTRY_CONFIGS: Dict[Industry, IndustryConfig] = {
    Industry.TECH: IndustryConfig(
        growth_rate_mean=0.20,    # YC Tech Startups中央値: 20% YoY
        growth_rate_std=0.6,      # Tech業界の標準偏差（CB Insights）
        success_rate_alpha=2.0,    # 成功率約20%に基づく
        success_rate_beta=8.0,     # Series A進出率から算出
        volatility_factor=1.2      # NASDAQ volatility indexベース
    ),
    Industry.BIOTECH: IndustryConfig(
        growth_rate_mean=0.10,    # Healthcare Startups中央値
        growth_rate_std=0.8,      # 長期R&Dによる高不確実性
        success_rate_alpha=1.5,    # FDA承認率を考慮
        success_rate_beta=8.5,     # Phase I-III成功確率から算出
        volatility_factor=1.5      # XBI volatility indexベース
    ),
    Industry.FINTECH: IndustryConfig(
        growth_rate_mean=0.15,    # Fintech Funding Report 2023
        growth_rate_std=0.5,      # 規制環境による安定性
        success_rate_alpha=2.5,    # ネオバンク生存率
        success_rate_beta=7.5,     # 規制対応コスト考慮
        volatility_factor=1.1      # 金融セクター平均
    ),
    Industry.ECOMMERCE: IndustryConfig(
        growth_rate_mean=0.25,    # D2C成長率中央値
        growth_rate_std=0.4,      # 季節性を考慮
        success_rate_alpha=3.0,    # D2C生存率
        success_rate_beta=7.0,     # CAC/LTV比率考慮
        volatility_factor=0.9      # 消費財セクター平均
    ),
    Industry.SAAS: IndustryConfig(
        growth_rate_mean=0.30,    # SaaS Magic Number > 1
        growth_rate_std=0.45,     # ARR予測性考慮
        success_rate_alpha=2.8,    # NRR > 100%
        success_rate_beta=7.2,     # Rule of 40考慮
        volatility_factor=1.0      # SaaS指数ベース
    ),
    Industry.GENERAL: IndustryConfig(
        growth_rate_mean=0.15,    # 全業種平均
        growth_rate_std=0.5,      # 総合的な不確実性
        success_rate_alpha=2.0,    # 平均的な成功確率
        success_rate_beta=8.0,     # 一般的な失敗率
        volatility_factor=1.0      # 市場平均
    )
}

# 成長ステージ別の調整係数
# 出典: PitchBook "Company Stage Success Rates 2023"
STAGE_ADJUSTMENTS: Dict[CompanyStage, Dict[str, float]] = {
    CompanyStage.SEED: {
        'growth_rate_multiplier': 1.5,    # 初期の高成長期待
        'std_multiplier': 1.4,            # 高い不確実性
        'success_alpha_multiplier': 0.7,   # 低い生存率
        'success_beta_multiplier': 1.3     # 高い失敗リスク
    },
    CompanyStage.EARLY: {
        'growth_rate_multiplier': 1.3,    # PMF達成後の成長
        'std_multiplier': 1.2,            # 事業モデル検証中
        'success_alpha_multiplier': 0.8,   # Series A進出率
        'success_beta_multiplier': 1.2     # 初期スケール失敗リスク
    },
    CompanyStage.GROWTH: {
        'growth_rate_multiplier': 1.0,    # 標準的な成長期
        'std_multiplier': 1.0,            # 基準となる不確実性
        'success_alpha_multiplier': 1.0,   # 平均的な成功率
        'success_beta_multiplier': 1.0     # 平均的なリスク
    },
    CompanyStage.EXPANSION: {
        'growth_rate_multiplier': 0.8,    # 成長率の逓減
        'std_multiplier': 0.9,            # 事業安定化
        'success_alpha_multiplier': 1.2,   # 高い生存率
        'success_beta_multiplier': 0.9     # 低い失敗リスク
    },
    CompanyStage.LATE: {
        'growth_rate_multiplier': 0.6,    # 成熟期の成長率
        'std_multiplier': 0.8,            # 最も安定
        'success_alpha_multiplier': 1.3,   # 最高の生存率
        'success_beta_multiplier': 0.8     # 最低の失敗リスク
    }
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

def calculate_adjusted_parameters(
    industry: str,
    stage: str,
    data_size: int
) -> Dict[str, Any]:
    """
    業界と成長ステージに応じたパラメータを計算

    Args:
        industry: 業界名
        stage: 成長ステージ
        data_size: データサイズ

    Returns:
        Dict[str, Any]: 調整済みパラメータ
    """
    industry_config = get_industry_config(industry)
    stage_adj = get_stage_adjustments(stage)
    mcmc_settings = get_mcmc_settings(data_size)

    return {
        'prior_parameters': {
            'mu': industry_config.growth_rate_mean * stage_adj['growth_rate_multiplier'],
            'sigma': industry_config.growth_rate_std * stage_adj['std_multiplier'],
            'alpha': industry_config.success_rate_alpha * stage_adj['success_alpha_multiplier'],
            'beta': industry_config.success_rate_beta * stage_adj['success_beta_multiplier'],
            'volatility': industry_config.volatility_factor
        },
        'mcmc_settings': mcmc_settings
    }