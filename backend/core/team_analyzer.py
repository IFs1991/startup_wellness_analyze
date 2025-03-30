# -*- coding: utf-8 -*-
"""
チーム分析モジュール
投資先企業のチーム構成や組織文化を分析する機能を提供します。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import networkx as nx
from datetime import datetime, timedelta
import logging
from google.cloud import firestore
import uuid

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 分析エンジンのインポート
from analysis.Team_Analyzer import TeamAnalyzer as AnalysisTeamEngine

# Firestoreクライアントのインポート
from service.firestore.client import get_firestore_client

class TeamAnalyzer:
    """
    チーム分析を行うためのコアクラス

    このクラスは分析結果の永続化と取得を担当し、
    実際の分析ロジックはanalysis.Team_Analyzerに委譲します。
    """

    def __init__(self):
        """TeamAnalyzerの初期化"""
        # 分析エンジンの初期化
        self._analysis_engine = AnalysisTeamEngine()

        # Firestoreコレクション参照の設定
        self._db = get_firestore_client()
        self._team_analysis_collection = self._db.collection('team_analyses')

        # スコアマッピングなどの定数は分析エンジンから取得
        self.founder_experience_scores = self._analysis_engine.founder_experience_scores
        self.industry_knowledge_scores = self._analysis_engine.industry_knowledge_scores
        self.team_completeness_scores = self._analysis_engine.team_completeness_scores
        self.stage_appropriateness_scores = self._analysis_engine.stage_appropriateness_scores

        logger.info("TeamAnalyzerが初期化されました")

    # 分析ロジックをanalysis.Team_Analyzerに委譲するメソッド
    async def evaluate_founding_team(self, founder_profiles, company_stage="seed", industry="software"):
        """
        創業チームの質を評価する

        Args:
            founder_profiles: 創業者のプロフィールデータ
            company_stage: 企業のステージ
            industry: 業界

        Returns:
            dict: 創業チーム評価結果
        """
        try:
            # 分析エンジンに処理を委譲
            results = self._analysis_engine.evaluate_founding_team(
                founder_profiles,
                company_stage=company_stage,
                industry=industry
            )

            # メタデータを追加
            results["analyzed_at"] = datetime.now().isoformat()
            results["analysis_type"] = "founding_team"

            return results

        except Exception as e:
            logger.error(f"創業チーム評価中にエラーが発生しました: {str(e)}")
            raise

    async def analyze_org_growth(self, employee_data, timeline="1y",
                                company_stage="series_a", industry="software"):
        """
        組織の成長を分析する

        Args:
            employee_data: 従業員データ
            timeline: 分析期間
            company_stage: 企業のステージ
            industry: 業界

        Returns:
            dict: 組織成長分析結果
        """
        try:
            # 分析エンジンに処理を委譲
            results = self._analysis_engine.analyze_org_growth(
                employee_data,
                timeline=timeline,
                company_stage=company_stage,
                industry=industry
            )

            # メタデータを追加
            results["analyzed_at"] = datetime.now().isoformat()
            results["analysis_type"] = "org_growth"

            return results

        except Exception as e:
            logger.error(f"組織成長分析中にエラーが発生しました: {str(e)}")
            raise

    async def measure_culture_strength(self, engagement_data, survey_results=None):
        """
        企業文化の強さを測定する

        Args:
            engagement_data: エンゲージメントデータ
            survey_results: サーベイ結果データ (オプション)

        Returns:
            dict: 企業文化強度分析結果
        """
        try:
            # 分析エンジンに処理を委譲
            results = self._analysis_engine.measure_culture_strength(
                engagement_data,
                survey_results=survey_results
            )

            # メタデータを追加
            results["analyzed_at"] = datetime.now().isoformat()
            results["analysis_type"] = "culture_strength"

            return results

        except Exception as e:
            logger.error(f"企業文化分析中にエラーが発生しました: {str(e)}")
            raise

    async def analyze_hiring_effectiveness(self, hiring_data, tenure_data=None):
        """
        採用効果を分析する

        Args:
            hiring_data: 採用データ
            tenure_data: 在職期間データ (オプション)

        Returns:
            dict: 採用効果分析結果
        """
        try:
            # 分析エンジンに処理を委譲
            results = self._analysis_engine.analyze_hiring_effectiveness(
                hiring_data,
                tenure_data=tenure_data
            )

            # メタデータを追加
            results["analyzed_at"] = datetime.now().isoformat()
            results["analysis_type"] = "hiring_effectiveness"

            return results

        except Exception as e:
            logger.error(f"採用効果分析中にエラーが発生しました: {str(e)}")
            raise

    async def generate_org_network(self, employee_data, interaction_data=None):
        """
        組織ネットワークグラフを生成する

        Args:
            employee_data: 従業員データ
            interaction_data: インタラクションデータ (オプション)

        Returns:
            dict: 組織ネットワーク分析結果
        """
        try:
            # 分析エンジンに処理を委譲
            results = self._analysis_engine.generate_org_network(
                employee_data,
                interaction_data=interaction_data
            )

            # メタデータを追加
            results["analyzed_at"] = datetime.now().isoformat()
            results["analysis_type"] = "org_network"

            return results

        except Exception as e:
            logger.error(f"組織ネットワーク分析中にエラーが発生しました: {str(e)}")
            raise

    async def provide_team_recommendations(self, team_data, identified_weaknesses=None):
        """
        チーム改善のための推奨事項を提供する

        Args:
            team_data: チームデータ
            identified_weaknesses: 特定された弱点 (オプション)

        Returns:
            dict: 推奨事項
        """
        try:
            # 分析エンジンに処理を委譲
            results = self._analysis_engine.provide_team_recommendations(
                team_data,
                identified_weaknesses=identified_weaknesses
            )

            # メタデータを追加
            results["analyzed_at"] = datetime.now().isoformat()
            results["analysis_type"] = "team_recommendations"

            return results

        except Exception as e:
            logger.error(f"チーム推奨事項の生成中にエラーが発生しました: {str(e)}")
            raise

    # Firestoreとの連携を担当する新メソッド
    async def save_founding_team_analysis(self, company_id: str, analysis_results: Dict[str, Any],
                                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        創業チーム分析結果をFirestoreに保存する

        Args:
            company_id: 企業ID
            analysis_results: 分析結果
            metadata: 追加のメタデータ

        Returns:
            str: 保存された分析のID
        """
        try:
            # 保存用データの準備
            analysis_id = str(uuid.uuid4())
            now = datetime.now()

            # 保存するデータの構築
            analysis_data = {
                "id": analysis_id,
                "company_id": company_id,
                "analysis_type": "founding_team",
                "results": analysis_results,
                "metadata": metadata or {},
                "created_at": now,
                "updated_at": now
            }

            # Firestoreに保存
            await self._team_analysis_collection.document(analysis_id).set(analysis_data)
            logger.info(f"創業チーム分析結果を保存しました。ID: {analysis_id}")

            return analysis_id

        except Exception as e:
            logger.error(f"創業チーム分析結果の保存中にエラーが発生しました: {str(e)}")
            raise

    async def save_org_growth_analysis(self, company_id: str, analysis_results: Dict[str, Any],
                                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        組織成長分析結果をFirestoreに保存する

        Args:
            company_id: 企業ID
            analysis_results: 分析結果
            metadata: 追加のメタデータ

        Returns:
            str: 保存された分析のID
        """
        try:
            # 保存用データの準備
            analysis_id = str(uuid.uuid4())
            now = datetime.now()

            # 保存するデータの構築
            analysis_data = {
                "id": analysis_id,
                "company_id": company_id,
                "analysis_type": "org_growth",
                "results": analysis_results,
                "metadata": metadata or {},
                "created_at": now,
                "updated_at": now
            }

            # Firestoreに保存
            await self._team_analysis_collection.document(analysis_id).set(analysis_data)
            logger.info(f"組織成長分析結果を保存しました。ID: {analysis_id}")

            return analysis_id

        except Exception as e:
            logger.error(f"組織成長分析結果の保存中にエラーが発生しました: {str(e)}")
            raise

    async def save_culture_strength_analysis(self, company_id: str, analysis_results: Dict[str, Any],
                                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        企業文化強度分析結果をFirestoreに保存する

        Args:
            company_id: 企業ID
            analysis_results: 分析結果
            metadata: 追加のメタデータ

        Returns:
            str: 保存された分析のID
        """
        try:
            # 保存用データの準備
            analysis_id = str(uuid.uuid4())
            now = datetime.now()

            # 保存するデータの構築
            analysis_data = {
                "id": analysis_id,
                "company_id": company_id,
                "analysis_type": "culture_strength",
                "results": analysis_results,
                "metadata": metadata or {},
                "created_at": now,
                "updated_at": now
            }

            # Firestoreに保存
            await self._team_analysis_collection.document(analysis_id).set(analysis_data)
            logger.info(f"企業文化強度分析結果を保存しました。ID: {analysis_id}")

            return analysis_id

        except Exception as e:
            logger.error(f"企業文化強度分析結果の保存中にエラーが発生しました: {str(e)}")
            raise

    async def save_other_team_analysis(self, company_id: str, analysis_type: str,
                                     analysis_results: Dict[str, Any],
                                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        その他のチーム分析結果をFirestoreに保存する

        Args:
            company_id: 企業ID
            analysis_type: 分析タイプ
            analysis_results: 分析結果
            metadata: 追加のメタデータ

        Returns:
            str: 保存された分析のID
        """
        try:
            # 保存用データの準備
            analysis_id = str(uuid.uuid4())
            now = datetime.now()

            # 保存するデータの構築
            analysis_data = {
                "id": analysis_id,
                "company_id": company_id,
                "analysis_type": analysis_type,
                "results": analysis_results,
                "metadata": metadata or {},
                "created_at": now,
                "updated_at": now
            }

            # Firestoreに保存
            await self._team_analysis_collection.document(analysis_id).set(analysis_data)
            logger.info(f"{analysis_type}分析結果を保存しました。ID: {analysis_id}")

            return analysis_id

        except Exception as e:
            logger.error(f"{analysis_type}分析結果の保存中にエラーが発生しました: {str(e)}")
            raise

    async def get_team_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        チーム分析結果をIDで取得する

        Args:
            analysis_id: 分析ID

        Returns:
            Optional[Dict[str, Any]]: 分析結果、存在しない場合はNone
        """
        try:
            # Firestoreから取得
            doc_ref = self._team_analysis_collection.document(analysis_id)
            doc = await doc_ref.get()

            if not doc.exists:
                logger.warning(f"指定されたIDの分析結果が見つかりません: {analysis_id}")
                return None

            analysis_data = doc.to_dict()
            logger.info(f"チーム分析結果を取得しました。ID: {analysis_id}")

            return analysis_data

        except Exception as e:
            logger.error(f"チーム分析結果の取得中にエラーが発生しました: {str(e)}")
            raise