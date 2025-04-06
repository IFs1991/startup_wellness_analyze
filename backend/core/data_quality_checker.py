# -*- coding: utf-8 -*-
"""
データ品質管理モジュール
データの品質を監視・管理し、結果をFirestoreに保存します。

Features:
    - データの整合性チェック
    - 品質メトリクスの計算
    - チェック結果の永続化
"""
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from firebase_admin import firestore
from service.firestore.client import FirestoreService, StorageError  # 絶対インポートに修正
from .generative_ai_manager import GenerativeAIManager



# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class DataQualityError(Exception):
    """データ品質チェック関連のエラー"""
    pass

class DataQualityChecker:
    """
    データの品質をチェックし、結果をFirestoreに保存するクラス
    """
    def __init__(self, firestore_service: FirestoreService, generative_ai_manager: GenerativeAIManager):
        """
        初期化
        Args:
            firestore_service: FirestoreServiceのインスタンス
            generative_ai_manager: GenerativeAIManagerのインスタンス
        """
        self.firestore_service = firestore_service
        self.generative_ai_manager = generative_ai_manager
        self.collection_name = "data_quality_reports"
        self.history_collection_suffix = "_history" # 履歴用サブコレクション

    async def check(self, data: pd.DataFrame, company_id: Optional[str] = None) -> Dict[str, Any]:
        """データ品質をチェックし、レポートを生成する

        Args:
            data (pd.DataFrame): チェック対象のデータフレーム
            company_id (Optional[str]): 企業ID (指定された場合)

        Returns:
            Dict[str, Any]: データ品質レポート
        """
        logger.info(f"データ品質チェックを開始: {company_id if company_id else '全体'}")
        report = {
            "id": f"dq_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "companyId": company_id,
            "timestamp": datetime.now(),
            "overallScore": 0.0,
            "metrics": [],
            "issues": [],
            "recommendations": []
        }

        try:
            # 1. 完全性チェック (欠損値)
            completeness_metrics, completeness_issues = self._check_completeness(data)
            report["metrics"].extend(completeness_metrics)
            report["issues"].extend(completeness_issues)

            # 2. 有効性チェック (データ型、範囲)
            validity_metrics, validity_issues = self._check_validity(data)
            report["metrics"].extend(validity_metrics)
            report["issues"].extend(validity_issues)

            # 3. 一貫性チェック (矛盾するデータ)
            consistency_metrics, consistency_issues = self._check_consistency(data)
            report["metrics"].extend(consistency_metrics)
            report["issues"].extend(consistency_issues)

            # 4. 適時性チェック (データの古さ - 仮実装)
            timeliness_metrics, timeliness_issues = self._check_timeliness(data)
            report["metrics"].extend(timeliness_metrics)
            report["issues"].extend(timeliness_issues)

            # 全体スコアの計算 (仮: 各メトリクスの平均値)
            if report["metrics"]:
                report["overallScore"] = sum(m['value'] for m in report["metrics"]) / len(report["metrics"])

            # AIによる推奨事項生成
            if report["issues"]:
                report["recommendations"] = await self.generate_recommendations(report["issues"])

            # レポートをFirestoreに保存
            await self.save_report(report)
            # レポート履歴をFirestoreに保存
            await self.save_report_history(report)

            logger.info(f"データ品質チェック完了: {report['id']}")
            return report

        except Exception as e:
            logger.error(f"データ品質チェック中にエラー発生: {str(e)}")
            raise

    def _check_completeness(self, data: pd.DataFrame) -> (List[Dict], List[Dict]):
        """完全性 (欠損値) チェック"""
        metrics = []
        issues = []
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        completeness_score = 100 * (1 - missing_cells / total_cells) if total_cells > 0 else 100

        metrics.append({
            "name": "Completeness",
            "value": round(completeness_score, 2),
            "threshold": 95.0, # 例: 閾値
            "status": "good" if completeness_score >= 95.0 else ("warning" if completeness_score >= 80.0 else "critical"),
            "description": "データの欠損値の割合",
            "lastUpdated": datetime.now()
        })

        if missing_cells > 0:
            missing_details = data.isnull().sum()
            missing_fields = missing_details[missing_details > 0].index.tolist()
            issues.append({
                "type": "missing",
                "severity": "medium" if completeness_score < 95 else "high",
                "description": f"合計 {missing_cells} 個の欠損値 ({round(100*missing_cells/total_cells, 1)}%) が検出されました。",
                "affectedFields": missing_fields,
                "recommendations": ["欠損値を補完するか、データ収集プロセスを見直してください。"]
            })
        return metrics, issues

    def _check_validity(self, data: pd.DataFrame) -> (List[Dict], List[Dict]):
        """有効性 (データ型、範囲) チェック - 仮実装"""
        # ここに具体的なチェックロジックを実装 (例: 数値範囲、カテゴリ値リスト)
        metrics = []
        issues = []
        # 仮のスコア
        validity_score = 98.0
        metrics.append({
            "name": "Validity",
            "value": validity_score,
            "threshold": 98.0,
            "status": "good" if validity_score >= 98.0 else "warning",
            "description": "データ型や値の範囲の正しさ",
            "lastUpdated": datetime.now()
        })
        # 仮のイシュー
        # issues.append({...})
        return metrics, issues

    def _check_consistency(self, data: pd.DataFrame) -> (List[Dict], List[Dict]):
        """一貫性 (矛盾するデータ) チェック - 仮実装"""
        # ここに具体的なチェックロジックを実装 (例: 複数列間の矛盾)
        metrics = []
        issues = []
        # 仮のスコア
        consistency_score = 99.0
        metrics.append({
            "name": "Consistency",
            "value": consistency_score,
            "threshold": 97.0,
            "status": "good" if consistency_score >= 97.0 else "warning",
            "description": "データ間の論理的な一貫性",
            "lastUpdated": datetime.now()
        })
        # 仮のイシュー
        # issues.append({...})
        return metrics, issues

    def _check_timeliness(self, data: pd.DataFrame) -> (List[Dict], List[Dict]):
        """適時性 (データの古さ) チェック - 仮実装"""
        # ここに具体的なチェックロジックを実装 (例: タイムスタンプ列の確認)
        metrics = []
        issues = []
        # 仮のスコア
        timeliness_score = 95.0
        metrics.append({
            "name": "Timeliness",
            "value": timeliness_score,
            "threshold": 90.0,
            "status": "good" if timeliness_score >= 90.0 else "warning",
            "description": "データの鮮度、更新頻度",
            "lastUpdated": datetime.now()
        })
        # 仮のイシュー
        # issues.append({...})
        return metrics, issues

    async def save_report(self, report: Dict[str, Any]):
        """データ品質レポートをFirestoreに保存する"""
        try:
            report_id = report["id"]
            # datetimeオブジェクトをISO形式文字列に変換
            report_to_save = self._serialize_datetimes(report)
            await self.firestore_service.set_document(self.collection_name, report_id, report_to_save)
            logger.info(f"データ品質レポートを保存しました: {report_id}")
        except Exception as e:
            logger.error(f"レポートの保存中にエラー発生: {str(e)}")

    async def save_report_history(self, report: Dict[str, Any]):
        """データ品質レポートの履歴をFirestoreに保存する"""
        try:
            company_id = report.get("companyId")
            if not company_id:
                logger.warning("履歴保存のためにはcompanyIdが必要です。スキップします。")
                return

            history_doc_id = f"{company_id}_{report['timestamp'].strftime('%Y%m%d%H%M%S')}"
            parent_collection = self.collection_name
            parent_doc_id = company_id # 企業ごとのレポート履歴を管理する場合
            sub_collection = self.history_collection_suffix

            # Firestoreのサブコレクションパスを構築
            history_path = f"{parent_collection}/{parent_doc_id}/{sub_collection}"

            # datetimeオブジェクトをISO形式文字列に変換
            report_to_save = self._serialize_datetimes(report)

            # サブコレクションにドキュメントを追加
            # FirestoreServiceにサブコレクションへの書き込みメソッドが必要
            # await self.firestore_service.add_document_to_subcollection(parent_collection, parent_doc_id, sub_collection, history_doc_id, report_to_save)
            # 仮実装: 通常のコレクションに追加
            await self.firestore_service.set_document(f"{history_path}", history_doc_id, report_to_save)

            logger.info(f"データ品質レポート履歴を保存しました: {history_path}/{history_doc_id}")
        except Exception as e:
            logger.error(f"レポート履歴の保存中にエラー発生: {str(e)}")

    async def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """指定されたIDのデータ品質レポートを取得する"""
        try:
            report_data = await self.firestore_service.get_document(self.collection_name, report_id)
            return self._deserialize_datetimes(report_data) if report_data else None
        except Exception as e:
            logger.error(f"レポートの取得中にエラー発生 ({report_id}): {str(e)}")
            return None

    async def get_report_history(self, company_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """指定された企業のデータ品質レポート履歴を取得する"""
        try:
            history_path = f"{self.collection_name}/{company_id}/{self.history_collection_suffix}"
            # FirestoreServiceにサブコレクションからの取得メソッドが必要
            # history_data = await self.firestore_service.get_documents_from_subcollection(
            #     self.collection_name, company_id, self.history_collection_suffix, order_by="timestamp", direction="DESCENDING", limit=limit
            # )
            # 仮実装: 通常のコレクションからクエリ (要調整)
            query = {
                "collection": history_path,
                "filters": [],
                "orderBy": {"field": "timestamp", "direction": "DESCENDING"},
                "limit": limit
            }
            # FirestoreServiceにクエリ実行メソッドが必要と仮定
            history_data_raw = await self.firestore_service.query_documents(**query)

            # timestamp フィールドでソート (FirestoreServiceがソートしない場合)
            history_data_raw.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)

            # 逆シリアライズ
            history_data = [self._deserialize_datetimes(doc) for doc in history_data_raw]

            logger.info(f"企業 {company_id} のレポート履歴を {len(history_data)} 件取得しました")
            return history_data
        except Exception as e:
            logger.error(f"レポート履歴の取得中にエラー発生 ({company_id}): {str(e)}")
            return []

    async def generate_recommendations(self, issues: List[Dict]) -> List[Dict]:
        """AIを使用してデータ品質改善の推奨事項を生成する"""
        if not self.generative_ai_manager:
            logger.warning("GenerativeAIManagerが初期化されていません。推奨事項生成をスキップします。")
            return []

        try:
            # AIへの入力プロンプトを作成
            prompt = "以下のデータ品質の問題点に基づいて、具体的な改善策を優先度順に提案してください。\n\n問題点リスト:\n"
            for i, issue in enumerate(issues):
                prompt += f"{i+1}. タイプ: {issue.get('type', 'N/A')}, "
                prompt += f"深刻度: {issue.get('severity', 'N/A')}, "
                prompt += f"説明: {issue.get('description', 'N/A')}\n"
                if 'affectedFields' in issue:
                    prompt += f"  影響を受けるフィールド: {', '.join(issue['affectedFields'])}\n"

            prompt += "\n改善策 (優先度高 > 中 > 低):
1. [具体的なアクション1]
2. [具体的なアクション2]
..."

            logger.info("AIによる改善推奨事項の生成を開始")
            ai_response = await self.generative_ai_manager.generate_text(prompt)
            raw_recommendations = ai_response.get("text", "")
            logger.info(f"AIレスポンス受信: {raw_recommendations[:100]}...")

            # AIの応答をパースして構造化 (簡単な実装)
            recommendations = []
            lines = raw_recommendations.strip().split('\n')
            priority_map = {"高": "high", "中": "medium", "低": "low"}
            current_priority = "medium" # デフォルト

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 優先度を判定 (簡易)
                if "優先度高" in line or "high priority" in line.lower():
                    current_priority = "high"
                elif "優先度中" in line or "medium priority" in line.lower():
                    current_priority = "medium"
                elif "優先度低" in line or "low priority" in line.lower():
                    current_priority = "low"

                # 番号付きリストの項目を抽出 (簡易)
                if line.lstrip().startswith(('1.', '2.', '3.', '4.', '5.', '- ', '* ')):
                    desc = line.lstrip('12345.-* ') # 番号や記号を除去
                    recommendations.append({
                        "priority": current_priority,
                        "description": desc,
                        "impact": "(AIによる自動生成)", # 必要に応じて詳細化
                        "effort": "medium" # 必要に応じて詳細化
                    })

            if not recommendations and raw_recommendations: # パース失敗時
                 recommendations.append({
                        "priority": "medium",
                        "description": raw_recommendations, # 生のテキストを入れる
                        "impact": "(AIによる自動生成)",
                        "effort": "medium"
                    })

            logger.info(f"{len(recommendations)} 件の推奨事項を生成しました")
            return recommendations

        except Exception as e:
            logger.error(f"AIによる推奨事項生成中にエラー発生: {str(e)}")
            return [] # エラー時は空リストを返す

    def _serialize_datetimes(self, data: Dict) -> Dict:
        """辞書内のdatetimeオブジェクトをISO文字列に変換（Firestore保存用）"""
        serialized = {}
        for key, value in data.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_datetimes(value)
            elif isinstance(value, list):
                serialized[key] = [self._serialize_datetimes(item) if isinstance(item, dict) else item for item in value]
            else:
                serialized[key] = value
        return serialized

    def _deserialize_datetimes(self, data: Dict) -> Dict:
        """辞書内のISO文字列をdatetimeオブジェクトに変換（Firestore取得後）"""
        deserialized = {}
        for key, value in data.items():
            if isinstance(value, str):
                try:
                    # ISO 8601形式の文字列をdatetimeオブジェクトに変換
                    dt_obj = datetime.fromisoformat(value)
                    deserialized[key] = dt_obj
                except (ValueError, TypeError):
                    # 変換できない場合はそのまま
                    deserialized[key] = value
            elif isinstance(value, dict):
                deserialized[key] = self._deserialize_datetimes(value)
            elif isinstance(value, list):
                deserialized[key] = [self._deserialize_datetimes(item) if isinstance(item, dict) else item for item in value]
            else:
                deserialized[key] = value
        return deserialized

    async def run_auto_fix(self, data: pd.DataFrame, company_id: Optional[str] = None) -> pd.DataFrame:
        """データ品質の問題を自動修正する (仮実装)

        Args:
            data (pd.DataFrame): 修正対象のデータフレーム
            company_id (Optional[str]): 企業ID

        Returns:
            pd.DataFrame: 修正後のデータフレーム
        """
        logger.info(f"データ品質の自動修正を開始: {company_id if company_id else '全体'}")
        fixed_data = data.copy()

        try:
            # 例: 欠損値を平均値で補完 (数値列のみ)
            numeric_cols = fixed_data.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                if fixed_data[col].isnull().any():
                    mean_val = fixed_data[col].mean()
                    fixed_data[col].fillna(mean_val, inplace=True)
                    logger.debug(f"列 '{col}' の欠損値を平均値 ({mean_val}) で補完しました")

            # 他の自動修正ルールを追加...

            logger.info(f"データ品質の自動修正が完了: {company_id if company_id else '全体'}")
            # 修正後のデータで再度品質チェックを行い、レポートを更新
            await self.check(fixed_data, company_id)
            return fixed_data

        except Exception as e:
            logger.error(f"自動修正中にエラー発生: {str(e)}")
            return data # エラー時は元のデータを返す

    async def get_quality_config(self, company_id: Optional[str] = None) -> Dict[str, Any]:
        """データ品質チェックの設定を取得する (仮実装)

        Args:
            company_id (Optional[str]): 企業ID (企業固有設定の場合)

        Returns:
            Dict[str, Any]: データ品質設定
        """
        # ここでFirestoreなどから設定を読み込むロジックを実装
        # デフォルト設定を返す仮実装
        default_config = {
            "thresholds": {
                "Completeness": 95.0,
                "Validity": 98.0,
                "Consistency": 97.0,
                "Timeliness": 90.0
            },
            "rules": {
                "check_missing_values": {"enabled": True, "parameters": {}},
                "check_data_types": {"enabled": True, "parameters": {}},
                # ... 他のルール
            },
            "autoFix": {
                "enabled": False,
                "rules": ["fill_missing_mean"] # 自動修正ルール
            }
        }
        logger.info(f"データ品質設定を取得 ({company_id if company_id else 'デフォルト'})")
        return default_config

    async def update_quality_config(self, config: Dict[str, Any], company_id: Optional[str] = None):
        """データ品質チェックの設定を更新する (仮実装)

        Args:
            config (Dict[str, Any]): 新しい設定
            company_id (Optional[str]): 企業ID (企業固有設定の場合)
        """
        # ここでFirestoreなどに設定を保存するロジックを実装
        logger.info(f"データ品質設定を更新 ({company_id if company_id else 'デフォルト'}) : {config}")
        # 例: await self.firestore_service.set_document("dq_config", company_id or "default", config)
        pass