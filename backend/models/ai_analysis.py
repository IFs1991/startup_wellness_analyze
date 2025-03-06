import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Protocol, Type, TypeVar, Union
from functools import wraps
from enum import Enum
from pydantic import BaseModel, Field, validator

from .company import Company
from .financial import FinancialData, FinancialRatios, FinancialGrowth
from .wellness import WellnessAggregateMetrics, WellnessTrend
from backend.analysis import (
    AnalysisType,
    AnalysisService,
    AnalysisResult,
    CorrelationAnalyzer,
    TimeSeriesAnalyzer,
    RegressionAnalyzer,
    ClusterAnalyzer,
    BayesianAnalyzer,
    NetworkAnalyzer,
    ROIAnalyzer,
    TextMiningAnalyzer,
    PCAAnalyzer,
    SurvivalAnalyzer,
    AssociationAnalyzer
)

# ロガーの設定
logger = logging.getLogger(__name__)

# 型変数の定義
T = TypeVar('T')

class DataPreprocessor(Protocol):
    """データ前処理インターフェース"""
    async def preprocess(self, data: Any) -> Any:
        """
        データの前処理を行う

        Args:
            data: 前処理対象のデータ

        Returns:
            前処理済みのデータ
        """
        ...

class AnalysisStrategy(Protocol):
    """分析戦略インターフェース"""
    async def analyze(self, data: Any, params: Dict[str, Any]) -> AnalysisResult:
        """
        分析を実行する

        Args:
            data: 分析対象のデータ
            params: 分析パラメータ

        Returns:
            分析結果
        """
        ...

def monitor_performance(func):
    """パフォーマンスモニタリングデコレータ"""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = await func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        self.logger.info(f"{func.__name__} 実行時間: {execution_time:.2f}秒")
        return result
    return wrapper

class CompanyAnalysisContext(BaseModel):
    """企業分析コンテキストモデル"""
    company: Company
    financial_data: List[FinancialData]
    financial_ratios: List[FinancialRatios]
    financial_growth: List[FinancialGrowth]
    wellness_metrics: List[WellnessAggregateMetrics]
    wellness_trends: List[WellnessTrend]
    analysis_date: datetime = Field(default_factory=datetime.now)

    @validator('company')
    def validate_company(cls, v):
        """企業情報の検証"""
        if not v:
            raise ValueError("企業情報が指定されていません")
        if not v.id:
            raise ValueError("企業IDが指定されていません")
        return v

class AIAnalyzer:
    """AI分析クラス"""
    def __init__(
        self,
        preprocessor: DataPreprocessor,
        correlation_analyzer: Optional[AnalysisService] = None,
        time_series_analyzer: Optional[AnalysisService] = None,
        regression_analyzer: Optional[AnalysisService] = None,
        cluster_analyzer: Optional[AnalysisService] = None,
        bayesian_analyzer: Optional[AnalysisService] = None,
        network_analyzer: Optional[AnalysisService] = None,
        roi_analyzer: Optional[AnalysisService] = None,
        text_mining_analyzer: Optional[AnalysisService] = None,
        pca_analyzer: Optional[AnalysisService] = None,
        survival_analyzer: Optional[AnalysisService] = None,
        association_analyzer: Optional[AnalysisService] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        AIアナライザーの初期化

        Args:
            preprocessor: データ前処理器
            correlation_analyzer: 相関分析サービス
            time_series_analyzer: 時系列分析サービス
            regression_analyzer: 回帰分析サービス
            cluster_analyzer: クラスタ分析サービス
            bayesian_analyzer: ベイズ推論サービス
            network_analyzer: ネットワーク分析サービス
            roi_analyzer: ROI計算サービス
            text_mining_analyzer: テキストマイニングサービス
            pca_analyzer: 主成分分析サービス
            survival_analyzer: 生存時間分析サービス
            association_analyzer: アソシエーション分析サービス
            logger: ロガー
        """
        self.preprocessor = preprocessor
        self.logger = logger or logging.getLogger(__name__)

        # 分析サービスの初期化
        self.analyzers = {
            AnalysisType.CORRELATION: correlation_analyzer or CorrelationAnalyzer(preprocessor),
            AnalysisType.TIME_SERIES: time_series_analyzer or TimeSeriesAnalyzer(preprocessor),
            AnalysisType.REGRESSION: regression_analyzer or RegressionAnalyzer(preprocessor),
            AnalysisType.CLUSTER: cluster_analyzer or ClusterAnalyzer(preprocessor),
            AnalysisType.BAYESIAN: bayesian_analyzer or BayesianAnalyzer(preprocessor),
            AnalysisType.NETWORK: network_analyzer or NetworkAnalyzer(preprocessor),
            AnalysisType.ROI: roi_analyzer or ROIAnalyzer(preprocessor),
            AnalysisType.TEXT_MINING: text_mining_analyzer or TextMiningAnalyzer(preprocessor),
            AnalysisType.PCA: pca_analyzer or PCAAnalyzer(preprocessor),
            AnalysisType.SURVIVAL: survival_analyzer or SurvivalAnalyzer(preprocessor),
            AnalysisType.ASSOCIATION: association_analyzer or AssociationAnalyzer(preprocessor)
        }

    @monitor_performance
    async def analyze_company(
        self,
        context: CompanyAnalysisContext,
        analysis_type: AnalysisType,
        params: Optional[Dict[str, Any]] = None
    ) -> 'AIAnalysisResponse':
        """
        企業の分析を実行する

        Args:
            context: 企業分析コンテキスト
            analysis_type: 分析タイプ
            params: 分析パラメータ

        Returns:
            分析レスポンス

        Raises:
            ValueError: 無効な入力の場合
            Exception: 分析処理中のエラー
        """
        try:
            # 入力バリデーション
            if not context:
                raise ValueError("企業分析コンテキストが指定されていません")

            if not analysis_type:
                raise ValueError("分析タイプが指定されていません")

            # 分析器の取得
            analyzer = self.analyzers.get(analysis_type)
            if not analyzer:
                self.logger.error(f"未サポートの分析タイプ: {analysis_type}")
                raise ValueError(f"未サポートの分析タイプ: {analysis_type}")

            # データの前処理
            self.logger.info(f"企業 {context.company.id} のデータ前処理を開始")
            processed_data = await self.preprocessor.preprocess(context)
            self.logger.info(f"企業 {context.company.id} のデータ前処理が完了")

            # パラメータのバリデーション
            valid_params = await self._validate_params(analysis_type, params or {})

            # 分析の実行
            self.logger.info(f"企業 {context.company.id} の {analysis_type.value} 分析を開始")
            result = await analyzer.analyze(processed_data, valid_params)
            self.logger.info(f"企業 {context.company.id} の {analysis_type.value} 分析が完了")

            # 分析結果からレスポンスを生成
            insights = await self._extract_insights(result)
            recommendations = await self._generate_recommendations(result)
            risk_factors = await self._identify_risks(result)
            opportunity_areas = await self._identify_opportunities(result)

            # レスポンス作成
            response = AIAnalysisResponse(
                company_id=context.company.id,
                analysis_type=analysis_type.value,
                insights=insights,
                recommendations=recommendations,
                risk_factors=risk_factors,
                opportunity_areas=opportunity_areas
            )

            self.logger.info(f"企業 {context.company.id} の分析レスポンスを生成しました")
            return response

        except ValueError as ve:
            self.logger.error(f"バリデーションエラー: {str(ve)}")
            raise
        except Exception as e:
            self.logger.error(f"分析エラー: {str(e)}", exc_info=True)
            raise

    async def _validate_params(self, analysis_type: AnalysisType, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析パラメータのバリデーション

        Args:
            analysis_type: 分析タイプ
            params: 分析パラメータ

        Returns:
            検証済みパラメータ
        """
        validated_params = params.copy()

        # 分析タイプ別のパラメータバリデーション
        if analysis_type == AnalysisType.CORRELATION:
            # 相関分析の場合、metrics（変数ペア）が必要
            if "metrics" not in validated_params or not validated_params["metrics"]:
                self.logger.warning("相関分析にはmetricsパラメータが必要です。デフォルト値を使用します。")
                validated_params["metrics"] = ["revenue", "wellness_score"]

        elif analysis_type == AnalysisType.TIME_SERIES:
            # 時系列分析の場合、time_rangeが必要
            if "time_range" not in validated_params or not validated_params["time_range"]:
                self.logger.warning("時系列分析にはtime_rangeパラメータが必要です。デフォルト値を使用します。")
                validated_params["time_range"] = "1y"  # デフォルトは1年

        elif analysis_type == AnalysisType.BAYESIAN:
            # ベイズ推論の場合のバリデーション
            if "prior" not in validated_params:
                self.logger.warning("ベイズ推論にはpriorパラメータが推奨されます。デフォルト値を使用します。")
                validated_params["prior"] = "portfolio_average"

        elif analysis_type == AnalysisType.ROI:
            # ROI計算の場合のバリデーション
            if "investment_period" not in validated_params:
                self.logger.warning("ROI計算にはinvestment_periodパラメータが必要です。デフォルト値を使用します。")
                validated_params["investment_period"] = "1y"  # デフォルトは1年

        elif analysis_type == AnalysisType.NETWORK:
            # ネットワーク分析の場合のバリデーション
            if "network_depth" not in validated_params:
                self.logger.warning("ネットワーク分析にはnetwork_depthパラメータが推奨されます。デフォルト値を使用します。")
                validated_params["network_depth"] = 2  # デフォルトは2次のつながりまで

        return validated_params

    async def _extract_insights(self, result: AnalysisResult) -> List[Dict[str, Any]]:
        """
        分析結果からインサイトを抽出

        Args:
            result: 分析結果

        Returns:
            インサイトのリスト
        """
        insights = []

        # 基本的なインサイト
        if "explanation" in result.metadata:
            insights.append({
                "type": "insight",
                "content": result.metadata.get("explanation", ""),
                "confidence": result.metadata.get("confidence", 0.0),
                "source": "analysis"
            })

        # トレンドに関するインサイト
        if "trends" in result.metadata:
            for trend in result.metadata["trends"]:
                insights.append({
                    "type": "trend",
                    "content": trend.get("description", ""),
                    "direction": trend.get("direction", ""),
                    "strength": trend.get("strength", 0.0),
                    "source": "trend_analysis"
                })

        # 相関関係に関するインサイト
        if "correlations" in result.metadata:
            for corr in result.metadata["correlations"]:
                insights.append({
                    "type": "correlation",
                    "content": f"{corr['var1']}と{corr['var2']}の間に{corr['strength']}の相関関係があります",
                    "correlation_coefficient": corr.get("coefficient", 0.0),
                    "significance": corr.get("significance", 0.0),
                    "source": "correlation_analysis"
                })

        # ベイズ分析からのインサイト
        if "posterior_distribution" in result.metadata:
            insights.append({
                "type": "bayesian_insight",
                "content": result.metadata.get("bayesian_explanation", ""),
                "credible_interval": result.metadata.get("credible_interval", {}),
                "source": "bayesian_analysis"
            })

        # デフォルトのインサイト（他に何もない場合）
        if not insights:
            insights.append({
                "type": "insight",
                "content": "分析結果からインサイトを抽出できませんでした",
                "confidence": 0.0,
                "source": "default"
            })

        return insights

    async def _generate_recommendations(self, result: AnalysisResult) -> List[Dict[str, Any]]:
        """
        分析結果から推奨アクションを生成

        Args:
            result: 分析結果

        Returns:
            推奨アクションのリスト
        """
        recommendations = []

        # 明示的に推奨されているアクション
        if "recommendations" in result.metadata:
            for rec in result.metadata["recommendations"]:
                if isinstance(rec, tuple) and len(rec) >= 2:
                    action, priority = rec
                    recommendations.append({
                        "type": "recommendation",
                        "action": action,
                        "priority": priority,
                        "rationale": "分析結果に基づく推奨"
                    })
                elif isinstance(rec, dict):
                    recommendations.append({
                        "type": "recommendation",
                        "action": rec.get("action", ""),
                        "priority": rec.get("priority", "medium"),
                        "rationale": rec.get("rationale", "分析結果に基づく推奨")
                    })

        # トレンドに基づく推奨
        if "trends" in result.metadata:
            for trend in result.metadata["trends"]:
                if trend.get("actionable", False) and "recommendation" in trend:
                    recommendations.append({
                        "type": "trend_recommendation",
                        "action": trend["recommendation"],
                        "priority": trend.get("priority", "medium"),
                        "rationale": f"トレンド「{trend.get('description', '')}」に基づく推奨"
                    })

        # ROI最適化に基づく推奨
        if "roi_optimization" in result.metadata:
            for opt in result.metadata["roi_optimization"]:
                recommendations.append({
                    "type": "roi_recommendation",
                    "action": opt.get("action", ""),
                    "priority": opt.get("priority", "high"),
                    "expected_impact": opt.get("expected_impact", 0.0),
                    "rationale": "ROI最適化に基づく推奨"
                })

        return recommendations

    async def _identify_risks(self, result: AnalysisResult) -> Optional[List[Dict[str, Any]]]:
        """
        分析結果からリスク要因を特定

        Args:
            result: 分析結果

        Returns:
            リスク要因のリスト、またはNone
        """
        risks = []

        # 明示的に指定されているリスク
        if "risks" in result.metadata:
            for risk in result.metadata["risks"]:
                if isinstance(risk, tuple) and len(risk) >= 2:
                    factor, severity = risk
                    risks.append({
                        "type": "risk",
                        "factor": factor,
                        "severity": severity,
                        "mitigation": "リスク軽減策を検討してください"
                    })
                elif isinstance(risk, dict):
                    risks.append({
                        "type": "risk",
                        "factor": risk.get("factor", ""),
                        "severity": risk.get("severity", "medium"),
                        "mitigation": risk.get("mitigation", "リスク軽減策を検討してください")
                    })

        # 異常値に基づくリスク
        if "outliers" in result.metadata:
            for outlier in result.metadata["outliers"]:
                risks.append({
                    "type": "outlier_risk",
                    "factor": f"{outlier.get('metric', '')}の異常値",
                    "severity": outlier.get("severity", "medium"),
                    "current_value": outlier.get("value", 0),
                    "expected_range": outlier.get("expected_range", []),
                    "mitigation": "異常値の原因を調査し対応してください"
                })

        # 負の相関に基づくリスク
        if "negative_correlations" in result.metadata:
            for corr in result.metadata["negative_correlations"]:
                if corr.get("coefficient", 0) < -0.5:  # 強い負の相関がある場合
                    risks.append({
                        "type": "correlation_risk",
                        "factor": f"{corr.get('var1', '')}と{corr.get('var2', '')}の間に強い負の相関関係",
                        "severity": "medium",
                        "correlation_coefficient": corr.get("coefficient", 0),
                        "mitigation": "これらの変数間の関係性を詳細に調査してください"
                    })

        return risks if risks else None

    async def _identify_opportunities(self, result: AnalysisResult) -> Optional[List[Dict[str, Any]]]:
        """
        分析結果から機会領域を特定

        Args:
            result: 分析結果

        Returns:
            機会領域のリスト、またはNone
        """
        opportunities = []

        # 明示的に指定されている機会
        if "opportunities" in result.metadata:
            for opp in result.metadata["opportunities"]:
                if isinstance(opp, tuple) and len(opp) >= 2:
                    area, potential = opp
                    opportunities.append({
                        "type": "opportunity",
                        "area": area,
                        "potential": potential,
                        "next_steps": "詳細な実行計画を立てて実施してください"
                    })
                elif isinstance(opp, dict):
                    opportunities.append({
                        "type": "opportunity",
                        "area": opp.get("area", ""),
                        "potential": opp.get("potential", "medium"),
                        "next_steps": opp.get("next_steps", "詳細な実行計画を立てて実施してください")
                    })

        # 正の相関に基づく機会
        if "positive_correlations" in result.metadata:
            for corr in result.metadata["positive_correlations"]:
                if corr.get("coefficient", 0) > 0.7:  # 強い正の相関がある場合
                    opportunities.append({
                        "type": "correlation_opportunity",
                        "area": f"{corr.get('var1', '')}と{corr.get('var2', '')}の相乗効果",
                        "potential": "high",
                        "correlation_coefficient": corr.get("coefficient", 0),
                        "next_steps": "これらの変数間の関係性を活かした取り組みを検討してください"
                    })

        # クラスタ分析に基づく機会
        if "clusters" in result.metadata:
            for cluster in result.metadata["clusters"]:
                if cluster.get("opportunity_score", 0) > 0.7:  # 機会スコアが高いクラスタ
                    opportunities.append({
                        "type": "cluster_opportunity",
                        "area": f"クラスタ「{cluster.get('name', '')}」における機会",
                        "potential": "high",
                        "characteristics": cluster.get("characteristics", []),
                        "next_steps": "このクラスタに特化した施策を検討してください"
                    })

        # ROI最適化に基づく機会
        if "roi_opportunities" in result.metadata:
            for roi_opp in result.metadata["roi_opportunities"]:
                opportunities.append({
                    "type": "roi_opportunity",
                    "area": roi_opp.get("area", ""),
                    "potential": roi_opp.get("potential", "medium"),
                    "expected_roi": roi_opp.get("expected_roi", 0.0),
                    "next_steps": roi_opp.get("next_steps", "ROI向上のための詳細計画を立ててください")
                })

        return opportunities if opportunities else None

class AIAnalysisRequest(BaseModel):
    """AI分析リクエストモデル"""
    company_id: str = Field(..., description="企業ID")
    analysis_type: str = Field(..., description="分析タイプ")
    time_range: Optional[str] = Field(None, description="時間範囲")
    metrics: Optional[List[str]] = Field(None, description="分析対象指標")
    additional_context: Optional[Dict[str, Any]] = Field(None, description="追加コンテキスト")

    @validator('company_id')
    def validate_company_id(cls, v):
        """企業IDの検証"""
        if not v:
            raise ValueError("企業IDが指定されていません")
        return v

    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        """分析タイプの検証"""
        valid_types = [t.value for t in AnalysisType]
        if v not in valid_types:
            raise ValueError(f"無効な分析タイプです。有効な値: {', '.join(valid_types)}")
        return v

    @validator('time_range')
    def validate_time_range(cls, v):
        """時間範囲の検証"""
        if v and not any(v.endswith(unit) for unit in ['d', 'w', 'm', 'y']):
            raise ValueError("時間範囲は'd'(日), 'w'(週), 'm'(月), 'y'(年)のいずれかで終わる必要があります")
        return v

class AIAnalysisResponse(BaseModel):
    """AI分析レスポンスモデル"""
    company_id: str = Field(..., description="企業ID")
    analysis_type: str = Field(..., description="分析タイプ")
    analysis_date: datetime = Field(default_factory=datetime.now)
    insights: List[Dict[str, Any]] = Field(..., description="分析インサイト")
    recommendations: List[Dict[str, Any]] = Field(..., description="推奨アクション")
    risk_factors: Optional[List[Dict[str, Any]]] = Field(None, description="リスク要因")
    opportunity_areas: Optional[List[Dict[str, Any]]] = Field(None, description="機会領域")

    class Config:
        json_schema_extra = {
            "example": {
                "company_id": "company-123",
                "analysis_type": "correlation",
                "analysis_date": "2024-02-15T10:30:00",
                "insights": [
                    {
                        "type": "insight",
                        "content": "健康スコアと生産性の間に強い正の相関関係が見られます",
                        "confidence": 0.85,
                        "source": "correlation_analysis"
                    }
                ],
                "recommendations": [
                    {
                        "type": "recommendation",
                        "action": "健康促進プログラムへの投資を増やす",
                        "priority": "high",
                        "rationale": "生産性向上に直接つながる可能性が高い"
                    }
                ],
                "risk_factors": [
                    {
                        "type": "risk",
                        "factor": "長時間労働による健康スコア低下",
                        "severity": "high",
                        "mitigation": "労働時間管理と休息促進の施策を検討"
                    }
                ],
                "opportunity_areas": [
                    {
                        "type": "opportunity",
                        "area": "チーム間コラボレーションの促進",
                        "potential": "medium",
                        "next_steps": "部門を超えた定期的な交流の場を設ける"
                    }
                ]
            }
        }

class AIAssistantContext(BaseModel):
    """AIアシスタントコンテキストモデル"""
    company_context: CompanyAnalysisContext
    current_analysis: Optional[AIAnalysisResponse] = None
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    user_preferences: Optional[Dict[str, Any]] = Field(None)

    class Config:
        from_attributes = True