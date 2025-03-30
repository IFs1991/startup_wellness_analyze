# -*- coding: utf-8 -*-
"""
市場・競合分析モジュール（コア実装）
投資先企業の市場ポジションと競争環境を分析するAPIとデータ永続化機能を提供します。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
from google.cloud import firestore
import uuid

# 純粋な分析ロジックをインポート
from analysis.MarketAnalyzer import MarketAnalyzer as AnalysisMarketEngine

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class MarketAnalyzer:
    """
    市場・競合分析モジュール（コア実装）

    投資先企業の市場ポジションと競争環境を分析するためのAPIを提供し、
    結果をFirestoreに永続化します。純粋な分析ロジックはanalysisディレクトリの
    MarketAnalyzerに委譲します。
    """

    def __init__(self, db: Optional[firestore.Client] = None):
        """
        MarketAnalyzerの初期化

        Parameters
        ----------
        db : Optional[firestore.Client]
            Firestoreクライアントのインスタンス
        """
        self.db = db
        self._analysis_engine = AnalysisMarketEngine()
        logger.info("Core MarketAnalyzer initialized")

    async def estimate_market_size(self,
                           market_data: Dict[str, Any],
                           growth_factors: Optional[Dict[str, float]] = None,
                           projection_years: int = 5) -> Dict[str, Any]:
        """
        TAM/SAM/SOMの推定と予測を行う（非同期APIラッパー）

        Parameters
        ----------
        market_data : Dict[str, Any]
            市場データ {'tam': 値, 'sam': 値, 'som': 値, 'year': 年}
        growth_factors : Dict[str, float], optional
            成長係数 {'tam_growth': 率, 'sam_growth': 率, 'som_growth': 率}
        projection_years : int, optional
            予測年数（デフォルト: 5）

        Returns
        -------
        Dict[str, Any]
            TAM/SAM/SOMの現在値と将来予測
        """
        try:
            # 分析エンジンの同期メソッドを呼び出し
            results = self._analysis_engine.estimate_market_size(
                market_data,
                growth_factors=growth_factors,
                projection_years=projection_years
            )

            # 追加のメタデータを付与
            results["analysis_timestamp"] = datetime.now().isoformat()
            results["analysis_type"] = "market_size"

            return results

        except Exception as e:
            logger.error(f"市場規模推定エラー: {str(e)}")
            raise

    async def create_competitive_map(self,
                             competitor_data: Dict[str, Any],
                             dimensions: List[str],
                             focal_company: Optional[str] = None) -> Dict[str, Any]:
        """
        競合マッピングを実行する（非同期APIラッパー）

        Parameters
        ----------
        competitor_data : Dict[str, Any]
            競合企業データ
        dimensions : List[str]
            マッピングの軸となる指標
        focal_company : str, optional
            中心となる企業名（通常は自社）

        Returns
        -------
        Dict[str, Any]
            競合マッピング結果と分析
        """
        try:
            # 競合データをDataFrameに変換
            competitor_df = pd.DataFrame(competitor_data)

            # 分析エンジンの同期メソッドを呼び出し
            results = self._analysis_engine.create_competitive_map(
                competitor_df,
                dimensions=dimensions,
                focal_company=focal_company
            )

            # 追加のメタデータを付与
            results["analysis_timestamp"] = datetime.now().isoformat()
            results["analysis_type"] = "competitive_map"

            return results

        except Exception as e:
            logger.error(f"競合マッピングエラー: {str(e)}")
            raise

    async def track_market_trends(self,
                          keyword_list: List[str],
                          date_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        市場トレンドを追跡する（非同期APIラッパー）

        Parameters
        ----------
        keyword_list : List[str]
            追跡するキーワードリスト
        date_range : Dict[str, str], optional
            分析期間（'start': 開始日, 'end': 終了日）

        Returns
        -------
        Dict[str, Any]
            キーワードのトレンド分析結果
        """
        try:
            # 分析エンジンの使用可能なメソッドを確認
            keyword_data = {"keywords": keyword_list}

            if hasattr(self._analysis_engine, "track_market_trends"):
                results = self._analysis_engine.track_market_trends(
                    keyword_list,
                    date_range=date_range
                )
            elif hasattr(self._analysis_engine, "analyze_market_trends"):
                results = self._analysis_engine.analyze_market_trends(
                    keyword_data,
                    date_range=tuple(date_range.values()) if date_range else None
                )
            else:
                raise ValueError("市場トレンド分析メソッドが見つかりません")

            # 追加のメタデータを付与
            results["analysis_timestamp"] = datetime.now().isoformat()
            results["analysis_type"] = "market_trends"

            return results

        except Exception as e:
            logger.error(f"市場トレンド分析エラー: {str(e)}")
            raise

    async def save_market_size_analysis(self,
                               company_id: str,
                               analysis_results: Dict[str, Any],
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        市場規模分析結果をFirestoreに保存

        Parameters
        ----------
        company_id : str
            企業ID
        analysis_results : Dict[str, Any]
            市場規模分析結果
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
                "analysis_type": "market_size",
                "results": analysis_results,
                "created_at": datetime.now(),
                "metadata": metadata or {}
            }

            # Firestoreに保存
            collection_path = f"companies/{company_id}/market_analyses"
            doc_ref = self.db.collection(collection_path).document()
            doc_ref.set(document_data)

            logger.info(f"市場規模分析結果をFirestoreに保存しました。ID: {doc_ref.id}")
            return doc_ref.id

        except Exception as e:
            logger.error(f"市場規模分析結果の保存エラー: {str(e)}")
            raise

    async def save_competitive_map_analysis(self,
                                   company_id: str,
                                   analysis_results: Dict[str, Any],
                                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        競合マッピング分析結果をFirestoreに保存

        Parameters
        ----------
        company_id : str
            企業ID
        analysis_results : Dict[str, Any]
            競合マッピング分析結果
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
                "analysis_type": "competitive_map",
                "results": analysis_results,
                "created_at": datetime.now(),
                "metadata": metadata or {}
            }

            # Firestoreに保存
            collection_path = f"companies/{company_id}/market_analyses"
            doc_ref = self.db.collection(collection_path).document()
            doc_ref.set(document_data)

            logger.info(f"競合マッピング分析結果をFirestoreに保存しました。ID: {doc_ref.id}")
            return doc_ref.id

        except Exception as e:
            logger.error(f"競合マッピング分析結果の保存エラー: {str(e)}")
            raise

    async def save_market_trends_analysis(self,
                                 company_id: str,
                                 analysis_results: Dict[str, Any],
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        市場トレンド分析結果をFirestoreに保存

        Parameters
        ----------
        company_id : str
            企業ID
        analysis_results : Dict[str, Any]
            市場トレンド分析結果
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
                "analysis_type": "market_trends",
                "results": analysis_results,
                "created_at": datetime.now(),
                "metadata": metadata or {}
            }

            # Firestoreに保存
            collection_path = f"companies/{company_id}/market_analyses"
            doc_ref = self.db.collection(collection_path).document()
            doc_ref.set(document_data)

            logger.info(f"市場トレンド分析結果をFirestoreに保存しました。ID: {doc_ref.id}")
            return doc_ref.id

        except Exception as e:
            logger.error(f"市場トレンド分析結果の保存エラー: {str(e)}")
            raise

    async def get_market_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        保存された市場分析結果を取得

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
            # すべての企業の市場分析コレクションを探索する必要があるため
            # 先にキャッシュやインデックスを活用して検索を最適化するべき
            # 実用的にはインデックス付きのコレクションを使用して効率化すべき

            # 簡易な実装（本番では最適化が必要）
            all_companies = self.db.collection("companies").stream()

            for company in all_companies:
                analyses_path = f"companies/{company.id}/market_analyses"
                analysis_doc = self.db.collection(analyses_path).document(analysis_id).get()

                if analysis_doc.exists:
                    analysis_data = analysis_doc.to_dict()
                    analysis_data["id"] = analysis_doc.id
                    return analysis_data

            # 見つからない場合
            return None

        except Exception as e:
            logger.error(f"市場分析結果の取得エラー: {str(e)}")
            raise