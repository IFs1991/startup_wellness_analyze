# -*- coding: utf-8 -*-
"""
財務分析モジュール（コア実装）
投資先企業の包括的な財務状況と成長性を評価するAPIとデータ永続化機能を提供します。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging
from google.cloud import firestore
import uuid

# 純粋な分析ロジックをインポート
from analysis.FinancialAnalyzer import FinancialAnalyzer as AnalysisFinancialEngine

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class FinancialAnalyzer:
    """
    財務分析モジュール（コア実装）

    投資先企業の包括的な財務状況と成長性を評価するためのAPIを提供し、
    結果をFirestoreに永続化します。純粋な分析ロジックはanalysisディレクトリの
    FinancialAnalyzerに委譲します。
    """

    def __init__(self, db: Optional[firestore.Client] = None):
        """
        FinancialAnalyzerの初期化

        Parameters
        ----------
        db : Optional[firestore.Client]
            Firestoreクライアントのインスタンス
        """
        self.db = db
        self._analysis_engine = AnalysisFinancialEngine()
        logger.info("Core FinancialAnalyzer initialized")

    async def calculate_burn_rate(self,
                          financial_data: pd.DataFrame,
                          period: str = 'monthly',
                          cash_column: str = 'cash_balance',
                          expense_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        キャッシュバーン率とランウェイ（資金枯渇までの期間）を計算（非同期APIラッパー）

        Parameters
        ----------
        financial_data : pd.DataFrame
            財務データ（日付インデックス付き）
        period : str, optional
            'monthly'または'quarterly'（デフォルト: 'monthly'）
        cash_column : str, optional
            現金残高のカラム名（デフォルト: 'cash_balance'）
        expense_columns : List[str], optional
            費用項目のカラム名リスト。指定がなければ現金残高の変化から計算

        Returns
        -------
        Dict[str, Any]
            バーン率、ランウェイ、関連指標を含む辞書
        """
        try:
            # 分析エンジンの同期メソッドを呼び出し
            results = self._analysis_engine.calculate_burn_rate(
                financial_data,
                period=period,
                cash_column=cash_column,
                expense_columns=expense_columns
            )

            # 追加のメタデータを付与
            results["analysis_timestamp"] = datetime.now().isoformat()
            results["analysis_type"] = "burn_rate"

            return results

        except Exception as e:
            logger.error(f"バーンレート計算エラー: {str(e)}")
            raise

    async def analyze_unit_economics(self,
                            revenue_data: pd.DataFrame,
                            customer_data: pd.DataFrame,
                            cost_data: pd.DataFrame) -> Dict[str, Any]:
        """
        ユニットエコノミクス分析を実行（非同期APIラッパー）

        Parameters
        ----------
        revenue_data : pd.DataFrame
            収益データ
        customer_data : pd.DataFrame
            顧客データ
        cost_data : pd.DataFrame
            コストデータ

        Returns
        -------
        Dict[str, Any]
            CAC、LTV、LTV/CAC比率などの指標を含む辞書
        """
        try:
            # 分析エンジンの同期メソッドを呼び出し
            results = self._analysis_engine.analyze_unit_economics(
                revenue_data,
                customer_data,
                cost_data
            )

            # 追加のメタデータを付与
            results["analysis_timestamp"] = datetime.now().isoformat()
            results["analysis_type"] = "unit_economics"

            return results

        except Exception as e:
            logger.error(f"ユニットエコノミクス分析エラー: {str(e)}")
            raise

    async def evaluate_growth_metrics(self,
                             financial_data: pd.DataFrame,
                             benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        成長指標を評価（非同期APIラッパー）

        Parameters
        ----------
        financial_data : pd.DataFrame
            時系列財務データ
        benchmark_data : Optional[pd.DataFrame], optional
            ベンチマークとなる同業他社や市場平均データ

        Returns
        -------
        Dict[str, Any]
            成長指標評価結果を含む辞書
        """
        try:
            # 分析エンジンの同期メソッドを呼び出し
            results = self._analysis_engine.evaluate_growth_metrics(
                financial_data,
                benchmark_data
            )

            # 追加のメタデータを付与
            results["analysis_timestamp"] = datetime.now().isoformat()
            results["analysis_type"] = "growth_metrics"

            return results

        except Exception as e:
            logger.error(f"成長指標評価エラー: {str(e)}")
            raise

    async def save_burn_rate_analysis(self,
                              company_id: str,
                              analysis_results: Dict[str, Any],
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        バーンレート分析結果をFirestoreに保存

        Parameters
        ----------
        company_id : str
            企業ID
        analysis_results : Dict[str, Any]
            バーンレート分析結果
        metadata : Optional[Dict[str, Any]], optional
            追加のメタデータ

        Returns
        -------
        str
            分析結果のドキュメントID
        """
        if self.db is None:
            logger.warning("Firestoreクライアントが設定されていないため、分析結果は保存されません")
            return str(uuid.uuid4())  # ダミーIDを返す

        try:
            # 分析結果にメタデータを統合
            document_data = {
                "company_id": company_id,
                "analysis_type": "burn_rate",
                "results": analysis_results,
                "created_at": datetime.now(),
                "metadata": metadata or {}
            }

            # Firestoreに保存
            collection_path = f"companies/{company_id}/financial_analyses"
            doc_ref = self.db.collection(collection_path).document()
            doc_ref.set(document_data)

            logger.info(f"バーンレート分析結果をFirestoreに保存しました。ID: {doc_ref.id}")
            return doc_ref.id

        except Exception as e:
            logger.error(f"バーンレート分析結果の保存エラー: {str(e)}")
            raise

    async def save_unit_economics_analysis(self,
                                   company_id: str,
                                   analysis_results: Dict[str, Any],
                                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        ユニットエコノミクス分析結果をFirestoreに保存

        Parameters
        ----------
        company_id : str
            企業ID
        analysis_results : Dict[str, Any]
            ユニットエコノミクス分析結果
        metadata : Optional[Dict[str, Any]], optional
            追加のメタデータ

        Returns
        -------
        str
            分析結果のドキュメントID
        """
        if self.db is None:
            logger.warning("Firestoreクライアントが設定されていないため、分析結果は保存されません")
            return str(uuid.uuid4())  # ダミーIDを返す

        try:
            # 分析結果にメタデータを統合
            document_data = {
                "company_id": company_id,
                "analysis_type": "unit_economics",
                "results": analysis_results,
                "created_at": datetime.now(),
                "metadata": metadata or {}
            }

            # Firestoreに保存
            collection_path = f"companies/{company_id}/financial_analyses"
            doc_ref = self.db.collection(collection_path).document()
            doc_ref.set(document_data)

            logger.info(f"ユニットエコノミクス分析結果をFirestoreに保存しました。ID: {doc_ref.id}")
            return doc_ref.id

        except Exception as e:
            logger.error(f"ユニットエコノミクス分析結果の保存エラー: {str(e)}")
            raise

    async def save_growth_metrics_analysis(self,
                                   company_id: str,
                                   analysis_results: Dict[str, Any],
                                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        成長指標分析結果をFirestoreに保存

        Parameters
        ----------
        company_id : str
            企業ID
        analysis_results : Dict[str, Any]
            成長指標分析結果
        metadata : Optional[Dict[str, Any]], optional
            追加のメタデータ

        Returns
        -------
        str
            分析結果のドキュメントID
        """
        if self.db is None:
            logger.warning("Firestoreクライアントが設定されていないため、分析結果は保存されません")
            return str(uuid.uuid4())  # ダミーIDを返す

        try:
            # 分析結果にメタデータを統合
            document_data = {
                "company_id": company_id,
                "analysis_type": "growth_metrics",
                "results": analysis_results,
                "created_at": datetime.now(),
                "metadata": metadata or {}
            }

            # Firestoreに保存
            collection_path = f"companies/{company_id}/financial_analyses"
            doc_ref = self.db.collection(collection_path).document()
            doc_ref.set(document_data)

            logger.info(f"成長指標分析結果をFirestoreに保存しました。ID: {doc_ref.id}")
            return doc_ref.id

        except Exception as e:
            logger.error(f"成長指標分析結果の保存エラー: {str(e)}")
            raise

    async def get_financial_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        保存された財務分析結果を取得

        Parameters
        ----------
        analysis_id : str
            分析結果のドキュメントID

        Returns
        -------
        Optional[Dict[str, Any]]
            分析結果のデータ、見つからない場合はNone
        """
        if self.db is None:
            logger.warning("Firestoreクライアントが設定されていないため、分析結果は取得できません")
            return None

        try:
            # すべての企業の財務分析コレクションを探索する必要があるため
            # 先にキャッシュやインデックスを活用して検索を最適化するべき
            # 実用的にはインデックス付きのコレクションを使用して効率化すべき

            # 簡易な実装（本番では最適化が必要）
            all_companies = self.db.collection("companies").stream()

            for company in all_companies:
                analyses_path = f"companies/{company.id}/financial_analyses"
                analysis_doc = self.db.collection(analyses_path).document(analysis_id).get()

                if analysis_doc.exists:
                    analysis_data = analysis_doc.to_dict()
                    analysis_data["id"] = analysis_doc.id
                    return analysis_data

            # 見つからない場合
            return None

        except Exception as e:
            logger.error(f"財務分析結果の取得エラー: {str(e)}")
            raise