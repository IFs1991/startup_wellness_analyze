#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Startup Wellness データ分析システム
チーム・組織分析モジュール (TeamAnalyzer.py)

このモジュールは、投資先企業の経営陣とチームの質、組織的な強みとリスクを評価する機能を提供します。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Iterator
import logging
import gc
import weakref
from functools import lru_cache
import contextlib
from google.cloud import firestore
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import networkx as nx
import matplotlib.pyplot as plt
# wordcloudをtry-exceptで囲んで依存関係エラーを防止
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("wordcloud パッケージがインストールされていません。テキスト視覚化機能は制限されます。")
    WORDCLOUD_AVAILABLE = False

# NLTKリソースのダウンロード（初回実行時に必要）
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# ロギング設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class AnalysisCache:
    """
    分析結果をキャッシュするためのクラス
    """
    def __init__(self, max_size=50):
        self._cache = weakref.WeakValueDictionary()
        self._max_size = max_size
        self._access_count = {}

    def get(self, key):
        """キャッシュからデータを取得"""
        if key in self._cache:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return self._cache[key]
        return None

    def put(self, key, value):
        """キャッシュにデータを格納"""
        # キャッシュが最大サイズに達した場合、最も使用頻度の低いアイテムを削除
        if len(self._cache) >= self._max_size:
            min_key = min(self._access_count.items(), key=lambda x: x[1])[0]
            del self._cache[min_key]
            del self._access_count[min_key]

        self._cache[key] = value
        self._access_count[key] = 1

    def clear(self):
        """キャッシュをクリア"""
        self._cache.clear()
        self._access_count.clear()
        gc.collect()


@contextlib.contextmanager
def managed_df(df, copy=False):
    """
    データフレームの効率的な管理のためのコンテキストマネージャ

    Args:
        df: 管理対象のデータフレーム
        copy: Trueの場合、データフレームのコピーを作成

    Yields:
        管理対象のデータフレーム（またはそのコピー）
    """
    try:
        if copy:
            # コピーが必要な場合のみコピーを作成
            df_copy = df.copy()
            yield df_copy
        else:
            # コピーが不要な場合は参照を返す
            yield df
    finally:
        # コンテキスト終了時の処理
        if copy:
            # 明示的にメモリを解放
            del df_copy
            gc.collect()


@contextlib.contextmanager
def plot_context():
    """
    Matplotlibプロットリソースを安全に管理するためのコンテキストマネージャ
    """
    try:
        # コンテキスト開始時の処理
        yield
    finally:
        # コンテキスト終了時に確実にプロットをクリア
        plt.close('all')
        gc.collect()


class TeamAnalyzer:
    """
    チーム・組織分析を行うクラス

    投資先企業の創業チーム評価、組織成長分析、人材獲得力評価、
    文化・エンゲージメント分析などの機能を提供します。
    """

    def __init__(self, db: Optional[firestore.Client] = None):
        """
        初期化メソッド

        Args:
            db: Firestoreデータベースクライアント（Noneの場合は新規作成）
        """
        try:
            # 環境変数のエラーを回避するための初期化方法
            if db:
                self.db = db
            else:
                try:
                    # service/firestore/client.pyから関数をインポート
                    from service.firestore.client import get_firestore_client
                    self.db = get_firestore_client()
                except Exception as e:
                    # 直接Firebase Adminを使用する
                    try:
                        from firebase_admin import firestore
                        self.db = firestore.client()
                    except Exception as e2:
                        print(f"Firestoreクライアント初期化エラー: {e2}。モックを使用します。")
                        self.db = None
        except Exception as e:
            print(f"TeamAnalyzer初期化エラー: {e}")
            self.db = None

        self.sia = SentimentIntensityAnalyzer()

        # 分析結果キャッシュの初期化
        self._analysis_cache = AnalysisCache(max_size=50)

        # データ格納用の弱参照辞書
        self._temp_data = weakref.WeakValueDictionary()

        # 業界ごとの標準スコア（実際の環境では外部データソースから取得）
        self.industry_benchmarks = {
            "software": {
                "founder_score": 75,
                "leadership_coverage": 70,
                "domain_expertise": 80,
                "execution_score": 75,
                "hiring_velocity": 15.5,  # 月間成長率%
                "turnover_rate": 12.0,    # 年間%
                "culture_score": 72
            },
            "hardware": {
                "founder_score": 70,
                "leadership_coverage": 75,
                "domain_expertise": 85,
                "execution_score": 70,
                "hiring_velocity": 10.2,
                "turnover_rate": 8.5,
                "culture_score": 68
            },
            "biotech": {
                "founder_score": 85,
                "leadership_coverage": 78,
                "domain_expertise": 90,
                "execution_score": 68,
                "hiring_velocity": 8.5,
                "turnover_rate": 7.2,
                "culture_score": 65
            },
            "fintech": {
                "founder_score": 78,
                "leadership_coverage": 72,
                "domain_expertise": 82,
                "execution_score": 76,
                "hiring_velocity": 14.2,
                "turnover_rate": 13.5,
                "culture_score": 70
            },
            "ecommerce": {
                "founder_score": 72,
                "leadership_coverage": 68,
                "domain_expertise": 75,
                "execution_score": 78,
                "hiring_velocity": 16.8,
                "turnover_rate": 15.2,
                "culture_score": 67
            }
        }

        # スタートアップステージごとの期待値
        self.stage_expectations = {
            "seed": {
                "team_completeness": 40,
                "leadership_coverage": 50,
                "hiring_velocity": 20.0,
                "structure_score": 35
            },
            "series_a": {
                "team_completeness": 60,
                "leadership_coverage": 70,
                "hiring_velocity": 15.0,
                "structure_score": 55
            },
            "series_b": {
                "team_completeness": 75,
                "leadership_coverage": 85,
                "hiring_velocity": 12.0,
                "structure_score": 70
            },
            "series_c": {
                "team_completeness": 85,
                "leadership_coverage": 90,
                "hiring_velocity": 8.0,
                "structure_score": 85
            },
            "growth": {
                "team_completeness": 95,
                "leadership_coverage": 95,
                "hiring_velocity": 5.0,
                "structure_score": 90
            }
        }

    def __del__(self):
        """
        デストラクタ - リソースの解放を確実に行う
        """
        self.release_resources()

    def release_resources(self):
        """
        使用したリソースを明示的に解放する
        """
        # キャッシュのクリア
        if hasattr(self, '_analysis_cache'):
            self._analysis_cache.clear()

        # 一時データの解放
        if hasattr(self, '_temp_data'):
            self._temp_data.clear()

        # SIAオブジェクトの解放
        if hasattr(self, 'sia'):
            self.sia = None

        # グラフをクリア
        plt.close('all')

        # ガベージコレクションを呼び出し
        gc.collect()

    def _get_cached_or_compute(self, cache_key, compute_func, *args, **kwargs):
        """
        キャッシュから結果を取得するか、計算して保存する汎用関数

        Args:
            cache_key: キャッシュキー
            compute_func: 結果を計算する関数
            *args, **kwargs: compute_funcに渡す引数

        Returns:
            計算結果
        """
        # キャッシュから結果を取得
        cached_result = self._analysis_cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"キャッシュからデータを取得: {cache_key}")
            return cached_result

        # キャッシュにない場合は計算して保存
        result = compute_func(*args, **kwargs)
        self._analysis_cache.put(cache_key, result)
        return result

    def evaluate_founding_team(self,
                               founder_profiles: List[Dict],
                               company_stage: str,
                               industry: str = "software") -> Dict[str, Any]:
        """
        創業チームの強み・弱み分析を行います。

        Args:
            founder_profiles: 創業者プロフィールのリスト
                各プロフィールは以下のキーを含むべき:
                - name: 名前
                - role: 役職
                - education: 学歴リスト
                - prior_experience: 過去の経験リスト
                - domain_experience_years: 業界経験年数
                - prior_exits: 過去のイグジット数
                - skills: スキルリスト
            company_stage: 会社のステージ ('seed', 'series_a', 'series_b', 'series_c', 'growth')
            industry: 業界カテゴリ

        Returns:
            評価結果を含む辞書
        """
        # キャッシュキーの生成
        cache_key = f"founding_team_{company_stage}_{industry}_{hash(str(founder_profiles))}"

        # 計算関数の定義
        def compute_team_evaluation():
            logger.info(f"創業チーム評価の実行: {len(founder_profiles)}名の創業者, ステージ: {company_stage}")

            # 創業チームの基本情報
            num_founders = len(founder_profiles)
            founder_roles = [f["role"] for f in founder_profiles]

            # 初期スコア設定
            founder_score = 0
            leadership_coverage_score = 0
            domain_expertise_score = 0
            execution_score = 0

            # 1. 創業者のバックグラウンド評価
            # リスト内包表記を使って効率化
            prior_exit_count = sum(f.get("prior_exits", 0) for f in founder_profiles)
            avg_domain_experience = np.mean([f.get("domain_experience_years", 0) for f in founder_profiles])

            # 学歴評価 (学位の種類とレベルでスコア化)
            education_score = 0
            for founder in founder_profiles:
                for edu in founder.get("education", []):
                    if "PhD" in edu or "博士" in edu:
                        education_score += 5
                    elif "MBA" in edu or "Master" in edu or "修士" in edu:
                        education_score += 3
                    elif "Bachelor" in edu or "学士" in edu:
                        education_score += 2
            education_score = min(education_score, 20) / 20 * 100  # 正規化

            # 2. リーダーシップカバレッジ評価
            # 主要な役職カテゴリ
            key_roles = ["CEO", "CTO", "CFO", "COO", "CMO", "CPO"]
            existing_roles = [role for role in founder_roles if any(key in role for key in key_roles)]
            leadership_coverage = len(existing_roles) / len(key_roles)

            # ステージに応じた期待値との比較
            stage_expectation = self.stage_expectations.get(company_stage, self.stage_expectations["seed"])
            expected_leadership = stage_expectation["leadership_coverage"] / 100

            leadership_coverage_relative = leadership_coverage / expected_leadership
            leadership_coverage_score = min(leadership_coverage_relative * 100, 100)

            # 3. ドメイン知識評価
            industry_benchmark = self.industry_benchmarks.get(industry, self.industry_benchmarks["software"])
            domain_expertise_relative = avg_domain_experience / 10  # 10年を満点とする
            domain_expertise_score = min(domain_expertise_relative * 100, 100)

            # ドメイン知識のベンチマーク調整
            domain_expertise_score = domain_expertise_score * 0.7 + (domain_expertise_score / industry_benchmark["domain_expertise"] * 100) * 0.3

            # 4. 実行力評価
            execution_factors = {
                "prior_exits": prior_exit_count * 15,  # 各イグジットで15ポイント
                "team_balance": 100 if len(set(founder_roles)) >= 3 else (len(set(founder_roles)) / 3 * 100),  # チームバランス
                "technical_founder": 100 if any("CTO" in role or "技術" in role or "エンジニア" in role for role in founder_roles) else 0
            }

            execution_score = np.mean(list(execution_factors.values()))

            # 5. チーム完全性評価
            team_completeness = leadership_coverage_score * 0.7 + (num_founders >= 3) * 30  # 創業者3名以上で加点

            # 総合スコア計算
            founder_score = (
                education_score * 0.2 +
                domain_expertise_score * 0.3 +
                execution_score * 0.3 +
                team_completeness * 0.2
            )

            # 業界別ベンチマークとの比較
            benchmark_comparison = {
                "founder_score": founder_score / industry_benchmark["founder_score"] * 100,
                "leadership_coverage": leadership_coverage_score / industry_benchmark["leadership_coverage"] * 100,
                "domain_expertise": domain_expertise_score / industry_benchmark["domain_expertise"] * 100,
                "execution_score": execution_score / industry_benchmark["execution_score"] * 100
            }

            # スキルカバレッジ分析
            # フラット化して効率的に処理
            all_skills = [skill for founder in founder_profiles for skill in founder.get("skills", [])]

            # 重要スキルカテゴリ
            skill_categories = {
                "technical": ["programming", "development", "engineering", "architecture", "プログラミング", "開発", "エンジニアリング"],
                "business": ["sales", "marketing", "strategy", "営業", "マーケティング", "戦略"],
                "financial": ["finance", "accounting", "investment", "財務", "会計", "投資"],
                "operations": ["operations", "management", "logistics", "運営", "管理", "ロジスティクス"],
                "product": ["product", "design", "UX", "製品", "デザイン"]
            }

            skill_coverage = {}
            for category, keywords in skill_categories.items():
                coverage = any(any(keyword.lower() in skill.lower() for keyword in keywords) for skill in all_skills)
                skill_coverage[category] = 100 if coverage else 0

            # 強みと弱みの特定
            strengths = []
            weaknesses = []

            if founder_score >= 75:
                strengths.append("創業チーム全体の質")
            elif founder_score < 50:
                weaknesses.append("創業チーム全体の質")

            if leadership_coverage_score >= 75:
                strengths.append("リーダーシップカバレッジ")
            elif leadership_coverage_score < 50:
                weaknesses.append("リーダーシップカバレッジ")

            if domain_expertise_score >= 75:
                strengths.append("ドメイン知識・経験")
            elif domain_expertise_score < 50:
                weaknesses.append("ドメイン知識・経験")

            if execution_score >= 75:
                strengths.append("実行力・過去の実績")
            elif execution_score < 50:
                weaknesses.append("実行力・過去の実績")

            for category, score in skill_coverage.items():
                if score == 0:
                    weaknesses.append(f"{category}スキルの不足")

            # 結果の返却
            result = {
                "timestamp": datetime.now().isoformat(),
                "company_stage": company_stage,
                "industry": industry,
                "num_founders": num_founders,
                "scores": {
                    "founder_score": round(founder_score, 1),
                    "leadership_coverage_score": round(leadership_coverage_score, 1),
                    "domain_expertise_score": round(domain_expertise_score, 1),
                    "execution_score": round(execution_score, 1),
                    "team_completeness": round(team_completeness, 1)
                },
                "benchmark_comparison": {k: round(v, 1) for k, v in benchmark_comparison.items()},
                "skill_coverage": skill_coverage,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "recommendations": self._generate_team_recommendations(weaknesses, company_stage)
            }

            logger.info(f"創業チーム評価完了: 総合スコア {round(founder_score, 1)}/100")
            return result

        # キャッシュから取得または計算
        return self._get_cached_or_compute(cache_key, compute_team_evaluation)

    def analyze_org_growth(self,
                          employee_data: pd.DataFrame,
                          timeline: str = "1y",
                          company_stage: str = "series_a",
                          industry: str = "software") -> Dict[str, Any]:
        """
        組織成長の健全性評価を行います。

        Args:
            employee_data: 従業員データのDataFrame
                必要なカラム:
                - date: 日付
                - headcount: 従業員数
                - new_hires: 新規採用数
                - departures: 退職者数
                - department: 部署
                - level: レベル（役職）
            timeline: 分析期間 ('3m', '6m', '1y', '2y', '3y')
            company_stage: 会社のステージ
            industry: 業界

        Returns:
            組織成長分析結果を含む辞書
        """
        # キャッシュキーの生成（データフレームのハッシュを含める）
        df_hash = hash(str(employee_data.iloc[0:min(5, len(employee_data))]) + str(len(employee_data)))
        cache_key = f"org_growth_{timeline}_{company_stage}_{industry}_{df_hash}"

        def compute_org_growth():
            logger.info(f"組織成長分析の実行: 期間 {timeline}, ステージ: {company_stage}")

            try:
                # データフレームの参照を使用（コピーを避ける）
                with managed_df(employee_data) as df:
                    # データの前処理
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.sort_values("date")

                    # 分析期間の設定
                    timeline_days = {
                        "3m": 90,
                        "6m": 180,
                        "1y": 365,
                        "2y": 730,
                        "3y": 1095
                    }.get(timeline, 365)

                    # 期間でフィルタリング
                    if "date" in df.columns:
                        end_date = df["date"].max()
                        start_date = end_date - timedelta(days=timeline_days)
                        # inplaceを使用せずに新しい参照を作成
                        period_data = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
                    else:
                        period_data = df

                    # 1. 基本的な成長指標の計算
                    initial_headcount = period_data["headcount"].iloc[0]
                    final_headcount = period_data["headcount"].iloc[-1]

                    # 成長率計算
                    abs_growth = final_headcount - initial_headcount
                    growth_rate = (final_headcount / initial_headcount - 1) * 100 if initial_headcount > 0 else 0

                    # 月間平均成長率の計算
                    months = timeline_days / 30
                    monthly_growth_rate = ((1 + growth_rate / 100) ** (1 / months) - 1) * 100 if growth_rate > 0 else 0

                    # 離職率の計算
                    total_departures = period_data["departures"].sum() if "departures" in period_data.columns else 0
                    avg_headcount = period_data["headcount"].mean()
                    turnover_rate = (total_departures / avg_headcount) * (365 / timeline_days) * 100 if avg_headcount > 0 else 0

                    # 部署別の成長分析
                    dept_growth = None
                    if "department" in period_data.columns:
                        # データフレームの集約処理の効率化
                        dept_aggs = {
                            "headcount": [
                                lambda x: x.iloc[-1] - x.iloc[0],
                                lambda x: x.iloc[-1] / x.iloc[0] * 100 - 100 if x.iloc[0] > 0 else 0
                            ]
                        }

                        if "new_hires" in period_data.columns:
                            dept_aggs["new_hires"] = "sum"
                        if "departures" in period_data.columns:
                            dept_aggs["departures"] = "sum"

                        dept_growth = period_data.groupby("department").agg(dept_aggs)

                        # カラム名の設定
                        dept_growth.columns = [
                            "abs_growth", "percent_growth",
                            "new_hires" if "new_hires" in period_data.columns else None,
                            "departures" if "departures" in period_data.columns else None
                        ]
                        # Noneを除去
                        dept_growth.columns = [c for c in dept_growth.columns if c is not None]
                        dept_growth = dept_growth.reset_index()

                        # 部署別の離職率（効率的な計算）
                        if "departures" in dept_growth.columns:
                            dept_growth["turnover_rate"] = 0.0  # デフォルト値を設定
                            mask = (dept_growth["abs_growth"] + dept_growth["departures"]) > 0
                            if mask.any():
                                dept_growth.loc[mask, "turnover_rate"] = (
                                    dept_growth.loc[mask, "departures"] /
                                    (dept_growth.loc[mask, "abs_growth"] + dept_growth.loc[mask, "departures"])
                                ) * 100

                    # 2. 組織構造の適切性評価
                    structure_score = 0

                    # レベル比率の分析
                    if "level" in period_data.columns:
                        # 効率的なグループ集計
                        level_counts = period_data.groupby("level")["headcount"].last()

                        # 管理職と非管理職の比率
                        management_levels = ["Director", "VP", "C-Level", "Manager", "ディレクター", "マネージャー", "部長", "課長"]
                        ic_levels = ["IC", "Individual Contributor", "Associate", "アソシエイト", "一般社員"]

                        # 効率的な集計
                        mgmt_count = 0
                        ic_count = 0

                        for level in level_counts.index:
                            if any(ml in level for ml in management_levels):
                                mgmt_count += level_counts[level]
                            elif any(il in level for il in ic_levels):
                                ic_count += level_counts[level]

                        # 理想的な比率に基づくスコアリング
                        if ic_count > 0:
                            mgmt_ratio = mgmt_count / ic_count

                            # 理想的な比率は1:7程度（業界・ステージによって異なる）
                            if company_stage in ["seed", "series_a"]:
                                ideal_ratio = 1/10
                            elif company_stage == "series_b":
                                ideal_ratio = 1/8
                            else:
                                ideal_ratio = 1/6

                            ratio_score = 100 - min(abs(mgmt_ratio - ideal_ratio) / ideal_ratio * 100, 100)
                            structure_score += ratio_score * 0.5

                        # 最高レベルの充足度
                        executive_levels = ["C-Level", "VP", "Director", "執行役員", "部長"]
                        has_executives = any(level in level_counts.index for level in executive_levels)
                        structure_score += 50 if has_executives else 0
                    else:
                        structure_score = 50  # データが不足している場合のデフォルト値

                    # 3. 部門バランスの評価
                    dept_balance_score = 0
                    if "department" in period_data.columns and dept_growth is not None and not dept_growth.empty:
                        # 部門名を小文字に変換して重複カウントを避ける
                        dept_names_lower = [d.lower() for d in dept_growth["department"]]

                        # 主要部門の存在確認
                        key_departments = ["Engineering", "Product", "Sales", "Marketing", "Operations", "Finance",
                                        "エンジニアリング", "プロダクト", "営業", "マーケティング", "オペレーション", "財務"]

                        # 効率的なカウント方法
                        key_dept_coverage = sum(1 for key in key_departments if any(key.lower() in dept.lower() for dept in dept_names_lower))

                        # ステージに応じた期待部門数
                        expected_depts = {
                            "seed": 3,
                            "series_a": 4,
                            "series_b": 5,
                            "series_c": 6,
                            "growth": 6
                        }.get(company_stage, 4)

                        dept_coverage_score = min(key_dept_coverage / expected_depts * 100, 100)

                        # 部門間の極端な不均衡をチェック
                        dept_sizes = period_data.groupby("department")["headcount"].last()
                        if len(dept_sizes) >= 2:
                            max_dept_size = dept_sizes.max()
                            min_dept_size = dept_sizes.min()
                            avg_dept_size = dept_sizes.mean()

                            # 極端な不均衡があるかチェック (最大部門が平均の3倍以上、または最小部門が平均の1/3以下)
                            balance_score = 100
                            if max_dept_size > avg_dept_size * 3:
                                balance_score -= 30
                            if min_dept_size < avg_dept_size / 3:
                                balance_score -= 30

                            dept_balance_score = dept_coverage_score * 0.6 + balance_score * 0.4
                        else:
                            dept_balance_score = dept_coverage_score
                    else:
                        dept_balance_score = 50  # データが不足している場合のデフォルト値

                    # 4. ベンチマークとの比較
                    industry_benchmark = self.industry_benchmarks.get(industry, self.industry_benchmarks["software"])
                    stage_expectation = self.stage_expectations.get(company_stage, self.stage_expectations["seed"])

                    benchmark_comparison = {
                        "growth_rate": {
                            "actual": monthly_growth_rate,
                            "benchmark": industry_benchmark["hiring_velocity"],
                            "comparison": (monthly_growth_rate / industry_benchmark["hiring_velocity"]) * 100 if industry_benchmark["hiring_velocity"] > 0 else 0
                        },
                        "turnover_rate": {
                            "actual": turnover_rate,
                            "benchmark": industry_benchmark["turnover_rate"],
                            "comparison": (industry_benchmark["turnover_rate"] / turnover_rate) * 100 if turnover_rate > 0 else 100  # 低いほど良い
                        },
                        "structure_score": {
                            "actual": structure_score,
                            "benchmark": stage_expectation["structure_score"],
                            "comparison": (structure_score / stage_expectation["structure_score"]) * 100 if stage_expectation["structure_score"] > 0 else 0
                        }
                    }

                    # 5. 総合スコアの計算
                    growth_health_score = (
                        (benchmark_comparison["growth_rate"]["comparison"] * 0.4) +
                        (benchmark_comparison["turnover_rate"]["comparison"] * 0.3) +
                        (benchmark_comparison["structure_score"]["comparison"] * 0.3)
                    )

                    # 制限を適用
                    growth_health_score = max(min(growth_health_score, 100), 0)

                    # 6. 強みと弱みの特定
                    strengths = []
                    weaknesses = []

                    # 成長率の評価
                    if benchmark_comparison["growth_rate"]["comparison"] >= 110:
                        strengths.append("業界平均を上回る採用・成長速度")
                    elif benchmark_comparison["growth_rate"]["comparison"] < 70:
                        weaknesses.append("業界平均を下回る採用・成長速度")

                    # 離職率の評価（低いほど良い）
                    if benchmark_comparison["turnover_rate"]["comparison"] >= 110:
                        strengths.append("業界平均を下回る離職率")
                    elif benchmark_comparison["turnover_rate"]["comparison"] < 70:
                        weaknesses.append("業界平均を上回る離職率")

                    # 組織構造の評価
                    if benchmark_comparison["structure_score"]["comparison"] >= 110:
                        strengths.append("ステージに適した組織構造")
                    elif benchmark_comparison["structure_score"]["comparison"] < 70:
                        weaknesses.append("ステージに適していない組織構造")

                    # 部門バランスの評価
                    if dept_balance_score >= 75:
                        strengths.append("バランスの取れた部門構成")
                    elif dept_balance_score < 50:
                        weaknesses.append("部門構成の不均衡")

                    # 結果の返却
                    result = {
                        "timestamp": datetime.now().isoformat(),
                        "company_stage": company_stage,
                        "industry": industry,
                        "timeline": timeline,
                        "headcount": {
                            "initial": int(initial_headcount),
                            "final": int(final_headcount),
                            "absolute_growth": int(abs_growth),
                            "growth_rate_percent": round(growth_rate, 1),
                            "monthly_growth_rate_percent": round(monthly_growth_rate, 1)
                        },
                        "turnover": {
                            "total_departures": int(total_departures),
                            "annual_turnover_rate_percent": round(turnover_rate, 1)
                        },
                        "structure": {
                            "structure_score": round(structure_score, 1),
                            "department_balance_score": round(dept_balance_score, 1)
                        },
                        "benchmark_comparison": {
                            k: {sk: round(sv, 1) if isinstance(sv, float) else sv
                                for sk, sv in v.items()}
                            for k, v in benchmark_comparison.items()
                        },
                        "growth_health_score": round(growth_health_score, 1),
                        "strengths": strengths,
                        "weaknesses": weaknesses,
                        "recommendations": self._generate_org_recommendations(weaknesses, company_stage)
                    }

                    # 部門別成長データを追加（データがある場合）
                    if dept_growth is not None and not dept_growth.empty:
                        # 辞書変換を効率化（列単位で変換）
                        dept_dict_list = []
                        for _, row in dept_growth.iterrows():
                            dept_dict_list.append(dict(row))
                        result["department_growth"] = dept_dict_list

                    logger.info(f"組織成長分析完了: 成長健全性スコア {round(growth_health_score, 1)}/100")
                    return result

            except Exception as e:
                logger.error(f"組織成長分析中にエラーが発生しました: {str(e)}")
                return {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        # キャッシュから取得または計算
        return self._get_cached_or_compute(cache_key, compute_org_growth)

    def measure_culture_strength(self,
                                engagement_data: pd.DataFrame,
                                survey_results: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        文化的一貫性と強さの定量化を行います。

        Args:
            engagement_data: エンゲージメントデータのDataFrame
                必要なカラム:
                - date: 日付
                - employee_id: 従業員ID
                - engagement_score: エンゲージメントスコア (0-100)
                - department: 部署
                - tenure_months: 勤続月数
            survey_results: 文化調査結果のDataFrame (オプション)
                含むと良いカラム:
                - employee_id: 従業員ID
                - question: 質問内容
                - response: 回答（テキストまたは数値）
                - category: 質問カテゴリ

        Returns:
            文化分析結果を含む辞書
        """
        # キャッシュキーの生成（データフレームのハッシュを含める）
        engage_hash = hash(str(engagement_data.iloc[0:min(5, len(engagement_data))]) + str(len(engagement_data)))
        survey_hash = hash(str(survey_results.iloc[0:min(5, len(survey_results))]) + str(len(survey_results))) if survey_results is not None else 0
        cache_key = f"culture_strength_{engage_hash}_{survey_hash}"

        def compute_culture_strength():
            logger.info("文化・エンゲージメント分析の実行")

            try:
                # データフレームの参照を使用
                with managed_df(engagement_data) as eng_data:
                    # 1. 基本的なエンゲージメント統計の計算
                    # NumPyの高速関数を使用
                    engagement_scores = eng_data["engagement_score"].values
                    mean_engagement = np.mean(engagement_scores)
                    median_engagement = np.median(engagement_scores)
                    engagement_std = np.std(engagement_scores)

                    # エンゲージメントスコアの分布（ベクトル化演算）
                    total_rows = len(eng_data)
                    low_mask = engagement_scores < 50
                    medium_mask = (engagement_scores >= 50) & (engagement_scores < 75)
                    high_mask = engagement_scores >= 75

                    low_engagement = np.sum(low_mask) / total_rows * 100
                    medium_engagement = np.sum(medium_mask) / total_rows * 100
                    high_engagement = np.sum(high_mask) / total_rows * 100

                    # 2. 部署間のエンゲージメント一貫性
                    if "department" in eng_data.columns:
                        # 集約処理の効率化
                        dept_engagement = eng_data.groupby("department")["engagement_score"].agg(['mean', 'std']).reset_index()

                        # 部署間の標準偏差（低いほど一貫性が高い）
                        dept_means = dept_engagement["mean"].values
                        interdept_std = np.std(dept_means)

                        # 一貫性スコア（100が最高）
                        consistency_score = max(100 - (interdept_std * 2), 0)

                        # 部署別の分析
                        dept_analysis = [dict(row) for _, row in dept_engagement.iterrows()]
                    else:
                        consistency_score = None
                        dept_analysis = None
                        interdept_std = None

                    # 3. 勤続期間とエンゲージメントの関係
                    if "tenure_months" in eng_data.columns:
                        # 勤続期間でグループ化
                        tenure_bins = [0, 3, 6, 12, 24, 36, float('inf')]
                        tenure_labels = ['0-3m', '3-6m', '6-12m', '1-2y', '2-3y', '3y+']

                        eng_data['tenure_group'] = pd.cut(eng_data['tenure_months'],
                                                            bins=tenure_bins,
                                                            labels=tenure_labels)

                        tenure_engagement = eng_data.groupby('tenure_group')['engagement_score'].agg(['mean', 'count', 'std']).reset_index()

                        # 勤続期間による差異スコア
                        if len(tenure_engagement) > 1:
                            max_diff = tenure_engagement['mean'].max() - tenure_engagement['mean'].min()
                            tenure_impact_score = max(100 - (max_diff * 1.5), 0)
                        else:
                            tenure_impact_score = None

                        tenure_analysis = [dict(row) for _, row in tenure_engagement.iterrows()]
                    else:
                        tenure_impact_score = None
                        tenure_analysis = None

                    # 4. 文化調査の分析（データがある場合）
                    culture_keywords = {}
                    sentiment_scores = {}
                    culture_themes = {}

                    if survey_results is not None and not survey_results.empty:
                        with managed_df(survey_results) as survey_data:
                            # テキスト回答の感情分析
                            if "response" in survey_data.columns and survey_data["response"].dtype == object:
                                # 感情分析（バッチ処理で効率化）
                                sentiment_values = []
                                for response in survey_data['response']:
                                    if isinstance(response, str):
                                        sentiment_values.append(self.sia.polarity_scores(response)['compound'])
                                    else:
                                        sentiment_values.append(0)

                                survey_data['sentiment'] = sentiment_values

                                # カテゴリ別の感情スコア
                                if "category" in survey_data.columns:
                                    sentiment_scores = survey_data.groupby("category")["sentiment"].mean().to_dict()

                                # キーワード抽出（簡易版）
                                all_responses = " ".join(survey_data['response'].dropna().astype(str))
                                # 頻度計算を効率化
                                culture_keywords = {}
                                # 重要キーワードのリスト
                                important_keywords = ["teamwork", "innovation", "communication", "respect",
                                                    "transparency", "balance", "growth", "チームワーク",
                                                    "イノベーション", "コミュニケーション", "尊重", "透明性",
                                                    "バランス", "成長"]

                                for word in important_keywords:
                                    culture_keywords[word] = all_responses.lower().count(word.lower())

                                # 上位5つのキーワードを抽出
                                culture_keywords = dict(sorted(culture_keywords.items(), key=lambda x: x[1], reverse=True)[:5])

                            # 文化テーマの抽出（カテゴリがある場合）
                            if "category" in survey_data.columns:
                                category_scores = survey_data.groupby("category").agg({
                                    "response": lambda x: x.mean() if x.dtype in [np.float64, np.int64] else None
                                }).dropna()

                                if not category_scores.empty:
                                    culture_themes = category_scores["response"].to_dict()

                    # 5. 文化強度スコアの計算
                    # エンゲージメントの平均値（40%）
                    engagement_component = min(mean_engagement, 100) * 0.4

                    # 一貫性（部署間の差異が小さいほど良い）（30%）
                    consistency_component = consistency_score * 0.3 if consistency_score is not None else 20

                    # 勤続期間の影響（小さいほど良い）（30%）
                    tenure_component = tenure_impact_score * 0.3 if tenure_impact_score is not None else 20

                    culture_strength_score = engagement_component + consistency_component + tenure_component

                    # 6. 強みと弱みの特定
                    strengths = []
                    weaknesses = []

                    if mean_engagement >= 75:
                        strengths.append("高いエンゲージメントスコア")
                    elif mean_engagement < 60:
                        weaknesses.append("低いエンゲージメントスコア")

                    if consistency_score is not None:
                        if consistency_score >= 80:
                            strengths.append("部署間の一貫したエンゲージメント")
                        elif consistency_score < 60:
                            weaknesses.append("部署間のエンゲージメント格差")

                    if tenure_impact_score is not None:
                        if tenure_impact_score >= 80:
                            strengths.append("勤続期間に関わらず一貫したエンゲージメント")
                        elif tenure_impact_score < 60:
                            weaknesses.append("勤続期間によるエンゲージメント低下")

                    if high_engagement < 30:
                        weaknesses.append("高エンゲージメント層の不足")

                    if low_engagement > 25:
                        weaknesses.append("低エンゲージメント層の比率が高い")

                    # 7. 結果の返却
                    result = {
                        "timestamp": datetime.now().isoformat(),
                        "engagement_stats": {
                            "mean": round(mean_engagement, 1),
                            "median": round(median_engagement, 1),
                            "std_dev": round(engagement_std, 1),
                            "distribution": {
                                "low_percent": round(low_engagement, 1),
                                "medium_percent": round(medium_engagement, 1),
                                "high_percent": round(high_engagement, 1)
                            }
                        },
                        "culture_strength_score": round(culture_strength_score, 1),
                        "strengths": strengths,
                        "weaknesses": weaknesses,
                        "recommendations": self._generate_culture_recommendations(weaknesses)
                    }

                    # 部署別分析（データがある場合）
                    if dept_analysis:
                        result["department_analysis"] = {
                            "consistency_score": round(consistency_score, 1),
                            "interdepartment_std_dev": round(interdept_std, 1),
                            "department_details": dept_analysis
                        }

                    # 勤続期間別分析（データがある場合）
                    if tenure_analysis:
                        result["tenure_analysis"] = {
                            "tenure_impact_score": round(tenure_impact_score, 1) if tenure_impact_score else None,
                            "tenure_details": tenure_analysis
                        }

                    # 文化調査分析（データがある場合）
                    if culture_keywords or sentiment_scores or culture_themes:
                        result["culture_survey_analysis"] = {
                            "key_themes": culture_keywords,
                            "category_sentiment": {k: round(v, 2) for k, v in sentiment_scores.items()},
                            "culture_dimensions": {k: round(v, 1) for k, v in culture_themes.items()}
                        }

                    logger.info(f"文化・エンゲージメント分析完了: 文化強度スコア {round(culture_strength_score, 1)}/100")
                    return result

            except Exception as e:
                logger.error(f"文化・エンゲージメント分析中にエラーが発生しました: {str(e)}")
                return {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        # キャッシュから取得または計算
        return self._get_cached_or_compute(cache_key, compute_culture_strength)

    def analyze_hiring_effectiveness(self,
                                    hiring_data: pd.DataFrame,
                                    performance_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        人材獲得力の評価を行います。

        Args:
            hiring_data: 採用データのDataFrame
                必要なカラム:
                - date: 採用日
                - position: 役職
                - department: 部署
                - time_to_fill: 採用所要日数
                - source: 採用ソース
                - level: 役職レベル
            performance_data: 採用後のパフォーマンスデータ (オプション)
                含むと良いカラム:
                - employee_id: 従業員ID
                - hire_date: 採用日
                - performance_score: パフォーマンススコア
                - retention_days: 在籍日数

        Returns:
            人材獲得力分析結果を含む辞書
        """
        # キャッシュキーの生成（データフレームのハッシュを含める）
        hiring_hash = hash(str(hiring_data.iloc[0:min(5, len(hiring_data))]) + str(len(hiring_data)))
        perf_hash = 0
        if performance_data is not None and not performance_data.empty:
            perf_hash = hash(str(performance_data.iloc[0:min(5, len(performance_data))]) + str(len(performance_data)))
        cache_key = f"hiring_effectiveness_{hiring_hash}_{perf_hash}"

        def compute_hiring_effectiveness():
            logger.info("人材獲得力分析の実行")

            try:
                # 効率的なデータフレーム処理
                with managed_df(hiring_data) as hire_df:
                    # 1. 採用効率の基本指標
                    avg_time_to_fill = hire_df["time_to_fill"].mean() if "time_to_fill" in hire_df.columns else None
                    median_time_to_fill = hire_df["time_to_fill"].median() if "time_to_fill" in hire_df.columns else None

                    # 採用ソース分析
                    source_analysis = None
                    if "source" in hire_df.columns:
                        # 効率的な集計
                        source_distribution = hire_df["source"].value_counts().to_dict()

                        source_effectiveness = None
                        if "time_to_fill" in hire_df.columns:
                            # 効率的なグループ集計
                            source_effectiveness = hire_df.groupby("source")["time_to_fill"].mean().reset_index()
                            source_effectiveness = [dict(row) for _, row in source_effectiveness.iterrows()]

                        source_analysis = {
                            "distribution": source_distribution,
                            "effectiveness": source_effectiveness
                        }

                    # 2. 部署・レベル別採用分析
                    dept_hiring_analysis = None
                    if "department" in hire_df.columns and "time_to_fill" in hire_df.columns:
                        # 効率的な集計処理
                        dept_hiring = hire_df.groupby("department").agg({
                            "position": "count",
                            "time_to_fill": ["mean", "median"]
                        })
                        dept_hiring.columns = ["position_count", "avg_time_to_fill", "median_time_to_fill"]
                        dept_hiring = dept_hiring.reset_index()

                        dept_hiring_analysis = [dict(row) for _, row in dept_hiring.iterrows()]

                    level_hiring_analysis = None
                    if "level" in hire_df.columns and "time_to_fill" in hire_df.columns:
                        # 効率的な集計処理
                        level_hiring = hire_df.groupby("level").agg({
                            "position": "count",
                            "time_to_fill": ["mean", "median"]
                        })
                        level_hiring.columns = ["position_count", "avg_time_to_fill", "median_time_to_fill"]
                        level_hiring = level_hiring.reset_index()

                        level_hiring_analysis = [dict(row) for _, row in level_hiring.iterrows()]

                    # 3. 採用の質分析（パフォーマンスデータがある場合）
                    quality_metrics = None

                    if performance_data is not None and not performance_data.empty:
                        with managed_df(performance_data) as perf_df:
                            # 新規採用のパフォーマンス
                            avg_performance = perf_df["performance_score"].mean() if "performance_score" in perf_df.columns else None

                            # 在籍率
                            retention_metrics = None
                            if "retention_days" in perf_df.columns:
                                # ベクトル化演算による効率的な計算
                                retention_days = perf_df["retention_days"].values
                                total_rows = len(perf_df)

                                retention_90d = np.sum(retention_days >= 90) / total_rows * 100
                                retention_180d = np.sum(retention_days >= 180) / total_rows * 100
                                retention_365d = np.sum(retention_days >= 365) / total_rows * 100

                                retention_metrics = {
                                    "90_day_retention_percent": round(retention_90d, 1),
                                    "180_day_retention_percent": round(retention_180d, 1),
                                    "365_day_retention_percent": round(retention_365d, 1)
                                }

                            quality_metrics = {
                                "average_performance": round(avg_performance, 1) if avg_performance is not None else None,
                                "retention": retention_metrics
                            }

                    # 4. 採用力スコアの計算
                    hiring_effectiveness_score = 0
                    components = []

                    # 採用所要時間（所要時間が短いほど良い - 業界平均60日と仮定）
                    if avg_time_to_fill is not None:
                        time_to_fill_benchmark = 60  # 業界平均の仮定値
                        time_component_score = max(100 - ((avg_time_to_fill - 30) / time_to_fill_benchmark * 100), 0)
                        hiring_effectiveness_score += time_component_score * 0.4
                        components.append(("time_to_fill", time_component_score))
                    else:
                        hiring_effectiveness_score += 25  # データがない場合のデフォルト

                    # 採用経路多様性（複数の採用ソースがあるほど良い）
                    if source_analysis is not None:
                        source_count = len(source_analysis["distribution"])
                        source_diversity_score = min(source_count / 5 * 100, 100)  # 5種類以上で満点
                        hiring_effectiveness_score += source_diversity_score * 0.2
                        components.append(("source_diversity", source_diversity_score))
                    else:
                        hiring_effectiveness_score += 10  # データがない場合のデフォルト

                    # 採用の質（パフォーマンスと在籍率）
                    if quality_metrics is not None:
                        quality_score = 0
                        component_count = 0

                        if quality_metrics["average_performance"] is not None:
                            perf_score = quality_metrics["average_performance"]
                            quality_score += perf_score
                            component_count += 1

                        if quality_metrics["retention"] is not None:
                            retention_365 = quality_metrics["retention"]["365_day_retention_percent"]
                            retention_score = retention_365  # 1年後の在籍率をそのままスコアとして使用
                            quality_score += retention_score
                            component_count += 1

                        if component_count > 0:
                            avg_quality_score = quality_score / component_count
                            hiring_effectiveness_score += avg_quality_score * 0.4
                            components.append(("quality", avg_quality_score))
                        else:
                            hiring_effectiveness_score += 20  # データがない場合のデフォルト
                    else:
                        hiring_effectiveness_score += 20  # データがない場合のデフォルト

                    # 5. 強みと弱みの特定
                    strengths = []
                    weaknesses = []

                    if avg_time_to_fill is not None:
                        if avg_time_to_fill < 45:
                            strengths.append("迅速な採用プロセス")
                        elif avg_time_to_fill > 75:
                            weaknesses.append("採用プロセスに時間がかかりすぎている")

                    if source_analysis is not None:
                        source_count = len(source_analysis["distribution"])
                        if source_count >= 4:
                            strengths.append("多様な採用チャネルの活用")
                        elif source_count <= 2:
                            weaknesses.append("採用チャネルの多様性不足")

                    if quality_metrics is not None and quality_metrics["retention"] is not None:
                        retention_365 = quality_metrics["retention"]["365_day_retention_percent"]
                        if retention_365 >= 85:
                            strengths.append("高い1年後在籍率")
                        elif retention_365 < 70:
                            weaknesses.append("低い1年後在籍率")

                    if quality_metrics is not None and quality_metrics["average_performance"] is not None:
                        avg_perf = quality_metrics["average_performance"]
                        if avg_perf >= 75:
                            strengths.append("採用人材の高いパフォーマンス")
                        elif avg_perf < 60:
                            weaknesses.append("採用人材のパフォーマンス不足")

                    # 6. 結果の返却
                    result = {
                        "timestamp": datetime.now().isoformat(),
                        "hiring_metrics": {
                            "avg_time_to_fill_days": round(avg_time_to_fill, 1) if avg_time_to_fill is not None else None,
                            "median_time_to_fill_days": round(median_time_to_fill, 1) if median_time_to_fill is not None else None
                        },
                        "hiring_effectiveness_score": round(hiring_effectiveness_score, 1),
                        "score_components": {name: round(score, 1) for name, score in components},
                        "strengths": strengths,
                        "weaknesses": weaknesses,
                        "recommendations": self._generate_hiring_recommendations(weaknesses)
                    }

                    # 採用ソース分析（データがある場合）
                    if source_analysis:
                        result["source_analysis"] = source_analysis

                    # 部署・レベル別分析（データがある場合）
                    if dept_hiring_analysis:
                        result["department_hiring_analysis"] = dept_hiring_analysis

                    if level_hiring_analysis:
                        result["level_hiring_analysis"] = level_hiring_analysis

                    # 採用の質分析（データがある場合）
                    if quality_metrics:
                        result["quality_metrics"] = quality_metrics

                    logger.info(f"人材獲得力分析完了: 採用効果スコア {round(hiring_effectiveness_score, 1)}/100")
                    return result

            except Exception as e:
                logger.error(f"人材獲得力分析中にエラーが発生しました: {str(e)}")
                return {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        # キャッシュから取得または計算
        return self._get_cached_or_compute(cache_key, compute_hiring_effectiveness)

    @contextlib.contextmanager
    def _managed_graph(self):
        """グラフオブジェクトの管理コンテキスト"""
        G = nx.Graph()
        try:
            yield G
        finally:
            # 明示的にグラフリソースを解放
            G.clear()
            del G
            # プロットをクリーンアップ
            plt.close('all')
            gc.collect()

    def generate_org_network_graph(self,
                                  interaction_data: pd.DataFrame,
                                  employee_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        組織ネットワーク分析を行い、チーム間の連携状況を可視化します。

        Args:
            interaction_data: 従業員間の相互作用データ
                必要なカラム:
                - from_id: 送信者ID
                - to_id: 受信者ID
                - weight: 相互作用の強さ（オプション）
                - type: 相互作用の種類（オプション）
            employee_data: 従業員データ（オプション）
                含むと良いカラム:
                - employee_id: 従業員ID
                - department: 部署
                - level: レベル
                - tenure_months: 勤続月数

        Returns:
            ネットワーク分析結果を含む辞書
        """
        # キャッシュキーの生成（データフレームのハッシュを含める）
        interact_hash = hash(str(interaction_data.iloc[0:min(5, len(interaction_data))]) + str(len(interaction_data)))
        emp_hash = 0
        if employee_data is not None and not employee_data.empty:
            emp_hash = hash(str(employee_data.iloc[0:min(5, len(employee_data))]) + str(len(employee_data)))
        cache_key = f"network_graph_{interact_hash}_{emp_hash}"

        def compute_network_graph():
            logger.info("組織ネットワーク分析の実行")

            try:
                # コンテキストマネージャでグラフリソースを管理
                with self._managed_graph() as G, managed_df(interaction_data) as interact_df:
                    emp_df = None
                    if employee_data is not None and not employee_data.empty:
                        emp_df = employee_data  # 参照のみ使用

                    # 従業員データがある場合はノード情報を追加
                    if emp_df is not None:
                        for _, row in emp_df.iterrows():
                            G.add_node(row["employee_id"],
                                    department=row.get("department", "Unknown"),
                                    level=row.get("level", "Unknown"),
                                    tenure_months=row.get("tenure_months", 0))

                    # エッジの追加（バッチ処理を使用）
                    edges_to_add = []
                    for _, row in interact_df.iterrows():
                        from_id = row["from_id"]
                        to_id = row["to_id"]
                        weight = row.get("weight", 1.0)

                        if from_id != to_id:  # 自己ループを除外
                            edges_to_add.append((from_id, to_id, weight))

                    # バッチでエッジを追加
                    G.add_weighted_edges_from([(u, v, w) for u, v, w in edges_to_add])

                    # 1. 基本的なネットワーク指標の計算
                    if G.number_of_nodes() > 0:
                        degrees = [d for n, d in G.degree()]
                        avg_degree = np.mean(degrees) if degrees else 0
                    else:
                        avg_degree = 0

                    density = nx.density(G)

                    # 例外処理を追加
                    try:
                        avg_clustering = nx.average_clustering(G)
                    except:
                        avg_clustering = 0

                    try:
                        # 最大連結成分のみで計算
                        largest_cc = max(nx.connected_components(G), key=len)
                        largest_subgraph = G.subgraph(largest_cc)
                        avg_shortest_path = nx.average_shortest_path_length(largest_subgraph)
                    except:
                        # グラフが連結でない場合
                        avg_shortest_path = None

                    # 2. 中心性指標の計算
                    try:
                        # 効率化のため、主要な中心性指標のみ計算
                        degree_centrality = nx.degree_centrality(G)

                        # 大規模グラフの場合は近似アルゴリズムを使用
                        if G.number_of_nodes() > 1000:
                            betweenness_centrality = nx.approximation.betweenness_centrality(G)
                        else:
                            betweenness_centrality = nx.betweenness_centrality(G)

                        closeness_centrality = nx.closeness_centrality(G)

                        # 上位5人の影響力者を特定（効率的なソート）
                        influencers = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                        bridges = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                    except:
                        degree_centrality = {}
                        betweenness_centrality = {}
                        closeness_centrality = {}
                        influencers = []
                        bridges = []

                    # 3. コミュニティ検出
                    try:
                        # 大規模グラフの場合はより効率的なアルゴリズムを使用
                        if G.number_of_nodes() > 1000:
                            # ラベル伝播法（効率的）
                            communities = list(nx.algorithms.community.label_propagation_communities(G))
                        else:
                            communities = list(nx.algorithms.community.greedy_modularity_communities(G))

                        num_communities = len(communities)
                        community_sizes = [len(c) for c in communities]
                    except:
                        num_communities = 0
                        community_sizes = []

                    # 4. 部門間連携分析（従業員データがある場合）
                    dept_connectivity = None

                    if emp_df is not None and "department" in emp_df.columns:
                        # 部署間の接続を分析
                        dept_edges = {}

                        # 効率的に部署間エッジを集計
                        for u, v, w in G.edges(data=True):
                            if u in G.nodes and v in G.nodes:
                                u_dept = G.nodes[u].get("department", "Unknown")
                                v_dept = G.nodes[v].get("department", "Unknown")

                                if u_dept != v_dept:  # 部署間の接続のみを考慮
                                    dept_pair = tuple(sorted([u_dept, v_dept]))

                                    if dept_pair not in dept_edges:
                                        dept_edges[dept_pair] = 0

                                    dept_edges[dept_pair] += w.get("weight", 1.0)

                        # 部署間連携の強さをリスト形式に変換
                        dept_connections = [{"dept_a": a, "dept_b": b, "connection_strength": strength}
                                        for (a, b), strength in dept_edges.items()]

                        # 各部署のネットワーク内での中心性
                        dept_graph = nx.Graph()
                        for a, b in dept_edges.keys():
                            if a not in dept_graph:
                                dept_graph.add_node(a)
                            if b not in dept_graph:
                                dept_graph.add_node(b)
                            dept_graph.add_edge(a, b)

                        try:
                            dept_centrality = nx.degree_centrality(dept_graph)
                            dept_connectivity = {
                                "connections": dept_connections,
                                "department_centrality": {k: round(v, 3) for k, v in dept_centrality.items()}
                            }
                        except:
                            dept_connectivity = {
                                "connections": dept_connections,
                                "department_centrality": {}
                            }

                    # 5. 孤立した従業員の特定
                    isolates = list(nx.isolates(G))
                    isolation_ratio = len(isolates) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0

                    # 6. 組織ネットワーク健全性スコアの計算
                    network_health_score = 0
                    components = []

                    # 密度（高いほど良い）
                    density_score = min(density * 200, 100)  # 密度0.5以上で満点
                    network_health_score += density_score * 0.25
                    components.append(("density", density_score))

                    # クラスタリング係数（高いほど良い）
                    clustering_score = min(avg_clustering * 100, 100)
                    network_health_score += clustering_score * 0.25
                    components.append(("clustering", clustering_score))

                    # 平均最短経路（短いほど良い、3~4が理想的）
                    if avg_shortest_path is not None:
                        path_score = 100 - min(max(avg_shortest_path - 3, 0) * 20, 100)
                        network_health_score += path_score * 0.25
                        components.append(("path_length", path_score))
                    else:
                        network_health_score += 15  # グラフが連結でない場合

                    # 孤立率（低いほど良い）
                    isolation_score = 100 - min(isolation_ratio * 100, 100)
                    network_health_score += isolation_score * 0.25
                    components.append(("isolation", isolation_score))

                    # 7. 強みと弱みの特定
                    strengths = []
                    weaknesses = []

                    if density >= 0.3:
                        strengths.append("高い組織内連携密度")
                    elif density < 0.1:
                        weaknesses.append("組織内連携の不足")

                    if avg_clustering >= 0.6:
                        strengths.append("強いチーム内結束")
                    elif avg_clustering < 0.3:
                        weaknesses.append("チーム内結束の弱さ")

                    if avg_shortest_path is not None:
                        if avg_shortest_path <= 3.5:
                            strengths.append("効率的な情報伝達経路")
                        elif avg_shortest_path > 5:
                            weaknesses.append("情報伝達経路の長さ")

                    if isolation_ratio <= 0.05:
                        strengths.append("孤立した従業員の少なさ")
                    elif isolation_ratio > 0.15:
                        weaknesses.append("孤立した従業員の多さ")

                    if dept_connectivity is not None and len(dept_connectivity["connections"]) >= 5:
                        strengths.append("活発な部署間連携")
                    elif dept_connectivity is not None and len(dept_connectivity["connections"]) < 3:
                        weaknesses.append("部署間連携の不足")

                    # 8. 結果の返却
                    result = {
                        "timestamp": datetime.now().isoformat(),
                        "network_metrics": {
                            "nodes": G.number_of_nodes(),
                            "edges": G.number_of_edges(),
                            "avg_degree": round(avg_degree, 2),
                            "density": round(density, 3),
                            "avg_clustering": round(avg_clustering, 3),
                            "avg_shortest_path": round(avg_shortest_path, 2) if avg_shortest_path is not None else None
                        },
                        "communities": {
                            "count": num_communities,
                            "sizes": community_sizes
                        },
                        "key_members": {
                            "influencers": [{"id": id, "score": round(score, 3)} for id, score in influencers],
                            "bridges": [{"id": id, "score": round(score, 3)} for id, score in bridges]
                        },
                        "isolation": {
                            "isolated_count": len(isolates),
                            "isolation_ratio": round(isolation_ratio, 3)
                        },
                        "network_health_score": round(network_health_score, 1),
                        "score_components": {name: round(score, 1) for name, score in components},
                        "strengths": strengths,
                        "weaknesses": weaknesses,
                        "recommendations": self._generate_network_recommendations(weaknesses)
                    }

                    # 部署間連携分析（データがある場合）
                    if dept_connectivity:
                        result["department_connectivity"] = dept_connectivity

                    logger.info(f"組織ネットワーク分析完了: ネットワーク健全性スコア {round(network_health_score, 1)}/100")
                    return result

            except Exception as e:
                logger.error(f"組織ネットワーク分析中にエラーが発生しました: {str(e)}")
                return {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        # キャッシュから取得または計算
        return self._get_cached_or_compute(cache_key, compute_network_graph)

    # 推奨生成関数をLRUキャッシュでメモ化
    @lru_cache(maxsize=32)
    def _generate_team_recommendations(self, weaknesses_tuple: Tuple[str, ...], company_stage: str) -> List[str]:
        """
        チーム評価の弱みに基づいて推奨アクションを生成します。

        Args:
            weaknesses_tuple: 特定された弱みのタプル（リストからの変換が必要）
            company_stage: 会社のステージ

        Returns:
            推奨アクションのリスト
        """
        # タプルをリストに変換（LRUキャッシュはハッシュ可能な型を引数に要求）
        weaknesses = list(weaknesses_tuple)

        recommendations = []

        for weakness in weaknesses:
            if "リーダーシップカバレッジ" in weakness:
                if company_stage in ["seed", "series_a"]:
                    recommendations.append("主要な役職（CTO、CFO、CMOなど）について、正式な採用が難しい場合はアドバイザーやフラクショナルエグゼクティブの活用を検討する")
                else:
                    recommendations.append("現在不足している主要役職（CTO、CFO、CMO、COOなど）の採用を優先し、リーダーシップチームを完成させる")

            if "ドメイン知識" in weakness:
                recommendations.append("業界経験豊富なアドバイザーを役員会やアドバイザリーボードに追加し、ドメイン知識のギャップを埋める")
                recommendations.append("創業チームに対する業界特化型のメンタリングやトレーニングプログラムを導入する")

            if "実行力" in weakness:
                recommendations.append("特定の業界で実績のあるCOOまたは経験豊富な事業開発責任者の採用を優先する")
                recommendations.append("OKR（目標と主要結果）フレームワークを導入し、透明性と説明責任を高める")

            if "チーム全体の質" in weakness:
                if company_stage in ["seed", "series_a"]:
                    recommendations.append("質の高いエグゼクティブの採用とリテンションを専門とするタレントアドバイザーとの連携")
                else:
                    recommendations.append("経験豊富な経営陣メンバーを追加採用し、チームの強化と多様化を図る")

        # デフォルトの推奨事項
        if not recommendations:
            recommendations = [
                "創業チームの強みと弱みのバランスを定期的に評価し、弱みを補完するための採用計画を策定する",
                "現在のリーダーシップの強みを最大限に活かしつつ、不足している専門知識は外部から取り入れる戦略を検討する"
            ]

        return recommendations[:3]  # 最大3つの推奨事項に制限

    # 他の_generate_recommendations関数も同様に最適化（リストをタプルに変換してキャッシュ対応）
    def _generate_org_recommendations(self, weaknesses: List[str], company_stage: str) -> List[str]:
        return self._generate_org_recommendations_cached(tuple(weaknesses), company_stage)

    @lru_cache(maxsize=32)
    def _generate_org_recommendations_cached(self, weaknesses_tuple: Tuple[str, ...], company_stage: str) -> List[str]:
        """
        組織評価の弱みに基づいて推奨アクションを生成します。

        Args:
            weaknesses_tuple: 特定された弱みのタプル
            company_stage: 会社のステージ

        Returns:
            推奨アクションのリスト
        """
        weaknesses = list(weaknesses_tuple)
        recommendations = []

        for weakness in weaknesses:
            if "採用・成長速度" in weakness:
                if company_stage in ["seed", "series_a"]:
                    recommendations.append("採用プロセスの最適化と複数の採用チャネルの活用によって、採用パイプラインを強化する")
                else:
                    recommendations.append("内部の採用チームを強化するか、専門の採用エージェンシーとのパートナーシップを検討する")

            if "離職率" in weakness:
                recommendations.append("従業員の定着率向上のためのオンボーディングプロセスの改善と早期警告システムの導入")
                recommendations.append("従業員のエンゲージメントと満足度を高めるためのプログラムや福利厚生の見直し")

            if "組織構造" in weakness:
                if company_stage in ["seed", "series_a"]:
                    recommendations.append("現段階では柔軟性を保ちつつも、成長に合わせた組織構造の段階的な計画を策定する")
                else:
                    recommendations.append("現在の組織構造を見直し、スケーラビリティと効率性のバランスを考慮した再設計を行う")

            if "部門構成の不均衡" in weakness:
                recommendations.append("各部門の最適な人員配置を再評価し、事業戦略に基づいたリソース配分計画を策定する")
                recommendations.append("不足している部門や過剰に肥大化している部門を特定し、バランスの取れた組織構造へと調整する")

        # デフォルトの推奨事項
        if not recommendations:
            recommendations = [
                "事業の成長計画に合わせた組織構造と人員配置の計画を定期的に見直し、最適化する",
                "従業員のエンゲージメントと定着率を高めるための施策を継続的に実施する"
            ]

        return recommendations[:3]  # 最大3つの推奨事項に制限

    def _generate_culture_recommendations(self, weaknesses: List[str]) -> List[str]:
        return self._generate_culture_recommendations_cached(tuple(weaknesses))

    @lru_cache(maxsize=32)
    def _generate_culture_recommendations_cached(self, weaknesses_tuple: Tuple[str, ...]) -> List[str]:
        """
        文化評価の弱みに基づいて推奨アクションを生成します。

        Args:
            weaknesses_tuple: 特定された弱みのタプル

        Returns:
            推奨アクションのリスト
        """
        weaknesses = list(weaknesses_tuple)
        recommendations = []

        for weakness in weaknesses:
            if "エンゲージメントスコア" in weakness:
                recommendations.append("定期的な1on1ミーティングとフィードバックループの確立によるエンゲージメント向上")
                recommendations.append("従業員のモチベーションドライバーを特定し、個別のキャリア開発計画を作成する")

            if "部署間のエンゲージメント格差" in weakness:
                recommendations.append("低エンゲージメント部署に対する重点的な改善計画の実施と、高エンゲージメント部署のベストプラクティス共有")
                recommendations.append("部署間の交流を促進するクロスファンクショナルプロジェクトの推進")

            if "勤続期間によるエンゲージメント低下" in weakness:
                recommendations.append("長期勤続者向けの新たな挑戦や成長機会の創出とキャリアパスの再設計")
                recommendations.append("勤続期間に応じたエンゲージメント施策の導入（長期インセンティブ、サバティカル制度など）")

            if "高エンゲージメント層の不足" in weakness:
                recommendations.append("組織の目的とビジョンの再定義と浸透活動の強化")
                recommendations.append("従業員が情熱を持って取り組める「イノベーションタイム」や特別プロジェクトの導入")

            if "低エンゲージメント層の比率が高い" in weakness:
                recommendations.append("早期の介入プログラムと低エンゲージメント従業員に対する個別サポートの提供")
                recommendations.append("匿名フィードバックの仕組みを導入し、従業員の率直な意見を収集・対応する")

        # デフォルトの推奨事項
        if not recommendations:
            recommendations = [
                "企業文化の強みを特定し、それらを強化・発展させるための具体的な施策を実施する",
                "定期的な文化サーベイと1on1ミーティングを通じて、文化的な課題を早期に発見し対応する体制を整える"
            ]

        return recommendations[:3]  # 最大3つの推奨事項に制限

    def _generate_hiring_recommendations(self, weaknesses: List[str]) -> List[str]:
        return self._generate_hiring_recommendations_cached(tuple(weaknesses))

    @lru_cache(maxsize=32)
    def _generate_hiring_recommendations_cached(self, weaknesses_tuple: Tuple[str, ...]) -> List[str]:
        """
        採用力評価の弱みに基づいて推奨アクションを生成します。

        Args:
            weaknesses_tuple: 特定された弱みのタプル

        Returns:
            推奨アクションのリスト
        """
        weaknesses = list(weaknesses_tuple)
        recommendations = []

        for weakness in weaknesses:
            if "採用プロセスに時間がかかりすぎ" in weakness:
                recommendations.append("採用プロセスの各ステップを分析し、ボトルネックを特定して最適化する")
                recommendations.append("採用基準と評価プロセスを明確化し、意思決定までの時間を短縮する")

            if "採用チャネルの多様性不足" in weakness:
                recommendations.append("リファラル採用プログラム、業界固有のジョブボード、テック・ミートアップなど、多様な採用チャネルの開拓")
                recommendations.append("パッシブ候補者へのリーチを強化するためのエンプロイヤーブランディング戦略の構築")

            if "低い1年後在籍率" in weakness:
                recommendations.append("オンボーディングプロセスの改善と90日間の初期定着プログラムの導入")
                recommendations.append("採用と実際の職務内容のミスマッチを防ぐための、より正確な職務記述と期待値設定")

            if "採用人材のパフォーマンス不足" in weakness:
                recommendations.append("採用基準と評価方法の見直しによる、より予測性の高い選考プロセスの確立")
                recommendations.append("テクニカルスキルと文化適合性の両方を適切に評価できる構造化面接の導入")

        # デフォルトの推奨事項
        if not recommendations:
            recommendations = [
                "データドリブンな採用プロセスの確立と継続的な最適化",
                "候補者体験の向上と長期的な人材パイプラインの構築"
            ]

        return recommendations[:3]  # 最大3つの推奨事項に制限

    def _generate_network_recommendations(self, weaknesses: List[str]) -> List[str]:
        return self._generate_network_recommendations_cached(tuple(weaknesses))

    @lru_cache(maxsize=32)
    def _generate_network_recommendations_cached(self, weaknesses_tuple: Tuple[str, ...]) -> List[str]:
        """
        ネットワーク評価の弱みに基づいて推奨アクションを生成します。

        Args:
            weaknesses_tuple: 特定された弱みのタプル

        Returns:
            推奨アクションのリスト
        """
        weaknesses = list(weaknesses_tuple)
        recommendations = []

        for weakness in weaknesses:
            if "組織内連携の不足" in weakness:
                recommendations.append("クロスファンクショナルプロジェクトチームの結成による部門間の協力促進")
                recommendations.append("全社的な情報共有ミーティングやオープンスペースの導入による交流機会の創出")

            if "チーム内結束の弱さ" in weakness:
                recommendations.append("チームビルディング活動とチーム単位での目標設定の強化")
                recommendations.append("チーム内の信頼構築ワークショップと定期的なチームレトロスペクティブの実施")

            if "情報伝達経路の長さ" in weakness:
                recommendations.append("組織階層の見直しと意思決定プロセスの合理化")
                recommendations.append("「スキップレベル」ミーティングの導入による階層を超えた直接的なコミュニケーションの促進")

            if "孤立した従業員の多さ" in weakness:
                recommendations.append("メンターシッププログラムの導入と新入社員の組織ネットワークへの積極的な統合支援")
                recommendations.append("バディシステムやアフィニティグループの設立による所属意識の醸成")

            if "部署間連携の不足" in weakness:
                recommendations.append("部署間連携を促進するインセンティブや評価指標の導入")
                recommendations.append("定期的な部署間ローテーションやジョブシャドウイングプログラムの実施")

        # デフォルトの推奨事項
        if not recommendations:
            recommendations = [
                "組織内のコミュニケーションチャネルの多様化と情報の透明性向上",
                "公式・非公式な交流機会の創出による自然なネットワーク形成の促進"
            ]

        return recommendations[:3]  # 最大3つの推奨事項に制限