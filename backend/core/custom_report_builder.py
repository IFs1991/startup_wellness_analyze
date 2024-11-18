# -*- coding: utf-8 -*-
"""
レポートカスタマイズサービス
対象読者 (VC, 経営陣, 従業員) 別のレポートカスタマイズと永続化を行います。
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import asyncio
import firebase_admin
from firebase_admin import firestore

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# カスタム例外クラス
class ReportError(Exception):
    """レポート生成に関するエラー"""
    pass

# カスタムレポートビルダークラス
class CustomReportBuilder:
    """
    対象読者に応じてレポート内容をカスタマイズし、Firestoreに保存するクラスです。
    """
    def __init__(self):
        """
        初期化メソッド
        """
        try:
            # firebase-adminの初期化（まだ初期化されていない場合）
            if not firebase_admin._apps:
                firebase_admin.initialize_app()
            self.db = firestore.client()
            self.collection_name = 'custom_reports'
            logger.info("CustomReportBuilder initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CustomReportBuilder: {str(e)}")
            raise

    async def build_and_save_report(
        self,
        data: Dict[str, Any],
        target_audience: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        カスタマイズされたレポートを作成しFirestoreに保存します。

        Args:
            data (Dict[str, Any]): レポートに含めるデータ
            target_audience (str): 対象読者 ('VC', '経営陣', '従業員')
            user_id (str): レポート作成者のID
            metadata (Optional[Dict[str, Any]]): 追加のメタデータ

        Returns:
            str: 生成されたレポートのドキュメントID

        Raises:
            ReportError: レポート生成または保存に失敗した場合
            ValueError: 無効な対象読者が指定された場合
        """
        try:
            logger.info(f"Building report for target audience: {target_audience}")

            # 対象読者の検証
            valid_audiences = {'VC', '経営陣', '従業員'}
            if target_audience not in valid_audiences:
                raise ValueError(f"Invalid target audience. Must be one of: {valid_audiences}")

            # レポートの構築
            report_content = await self._build_report_content(data, target_audience)

            # レポートドキュメントの作成
            report_doc = {
                'content': report_content,
                'target_audience': target_audience,
                'user_id': user_id,
                'created_at': datetime.now(),
                'status': 'completed',
                'metadata': metadata or {}
            }

            # Firestoreへの保存
            doc_ref = self.db.collection(self.collection_name).document()

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: doc_ref.set(report_doc))

            logger.info(f"Successfully saved report with ID: {doc_ref.id}")
            return doc_ref.id

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            error_msg = f"Error building and saving report: {str(e)}"
            logger.error(error_msg)
            raise ReportError(error_msg) from e

    async def _build_report_content(
        self,
        data: Dict[str, Any],
        target_audience: str
    ) -> Dict[str, Any]:
        """
        対象読者に応じたレポート内容を構築します。

        Args:
            data (Dict[str, Any]): レポートデータ
            target_audience (str): 対象読者

        Returns:
            Dict[str, Any]: カスタマイズされたレポート内容
        """
        try:
            if target_audience == 'VC':
                return {
                    'title': 'Investment Performance Report',
                    'sections': [
                        {
                            'title': 'Financial Metrics',
                            'content': self._extract_financial_metrics(data)
                        },
                        {
                            'title': 'Market Analysis',
                            'content': self._extract_market_analysis(data)
                        },
                        {
                            'title': 'Growth Projections',
                            'content': self._extract_growth_projections(data)
                        }
                    ]
                }

            elif target_audience == '経営陣':
                return {
                    'title': 'Executive Performance Report',
                    'sections': [
                        {
                            'title': 'Operational Metrics',
                            'content': self._extract_operational_metrics(data)
                        },
                        {
                            'title': 'Strategic Initiatives',
                            'content': self._extract_strategic_initiatives(data)
                        },
                        {
                            'title': 'Resource Allocation',
                            'content': self._extract_resource_allocation(data)
                        }
                    ]
                }

            else:  # target_audience == '従業員'
                return {
                    'title': 'Team Performance Report',
                    'sections': [
                        {
                            'title': 'Team Metrics',
                            'content': self._extract_team_metrics(data)
                        },
                        {
                            'title': 'Project Status',
                            'content': self._extract_project_status(data)
                        },
                        {
                            'title': 'Development Goals',
                            'content': self._extract_development_goals(data)
                        }
                    ]
                }

        except Exception as e:
            error_msg = f"Error building report content: {str(e)}"
            logger.error(error_msg)
            raise ReportError(error_msg) from e

    async def get_report(self, report_id: str) -> Dict[str, Any]:
        """
        保存されたレポートを取得します。

        Args:
            report_id (str): レポートのドキュメントID

        Returns:
            Dict[str, Any]: レポートデータ

        Raises:
            ReportError: レポートの取得に失敗した場合
        """
        try:
            doc_ref = self.db.collection(self.collection_name).document(report_id)

            loop = asyncio.get_event_loop()
            doc = await loop.run_in_executor(None, doc_ref.get)

            if not doc.exists:
                raise ReportError(f"Report with ID {report_id} not found")

            data = doc.to_dict()
            if data is None:
                raise ReportError(f"Report with ID {report_id} has no data")

            logger.info(f"Successfully retrieved report with ID: {report_id}")
            return data

        except Exception as e:
            error_msg = f"Error retrieving report: {str(e)}"
            logger.error(error_msg)
            raise ReportError(error_msg) from e

    def _extract_financial_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """財務指標を抽出"""
        return {
            'revenue': data.get('revenue'),
            'growth_rate': data.get('growth_rate'),
            'burn_rate': data.get('burn_rate'),
            'runway': data.get('runway')
        }

    def _extract_market_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """市場分析を抽出"""
        return {
            'market_size': data.get('market_size'),
            'market_share': data.get('market_share'),
            'competitors': data.get('competitors')
        }

    def _extract_growth_projections(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """成長予測を抽出"""
        return {
            'projected_revenue': data.get('projected_revenue'),
            'growth_targets': data.get('growth_targets'),
            'expansion_plans': data.get('expansion_plans')
        }

    def _extract_operational_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """運用指標を抽出"""
        return {
            'operational_efficiency': data.get('operational_efficiency'),
            'resource_utilization': data.get('resource_utilization'),
            'quality_metrics': data.get('quality_metrics')
        }

    def _extract_strategic_initiatives(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """戦略的施策を抽出"""
        return {
            'current_initiatives': data.get('current_initiatives'),
            'progress_metrics': data.get('progress_metrics'),
            'key_milestones': data.get('key_milestones')
        }

    def _extract_resource_allocation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """リソース配分を抽出"""
        return {
            'budget_allocation': data.get('budget_allocation'),
            'team_distribution': data.get('team_distribution'),
            'resource_constraints': data.get('resource_constraints')
        }

    def _extract_team_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """チーム指標を抽出"""
        return {
            'team_performance': data.get('team_performance'),
            'productivity_metrics': data.get('productivity_metrics'),
            'collaboration_scores': data.get('collaboration_scores')
        }

    def _extract_project_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """プロジェクト状況を抽出"""
        return {
            'ongoing_projects': data.get('ongoing_projects'),
            'project_health': data.get('project_health'),
            'deadline_status': data.get('deadline_status')
        }

    def _extract_development_goals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """開発目標を抽出"""
        return {
            'individual_goals': data.get('individual_goals'),
            'team_goals': data.get('team_goals'),
            'learning_objectives': data.get('learning_objectives')
        }

def get_custom_report_builder() -> CustomReportBuilder:
    """
    CustomReportBuilderのインスタンスを取得します。

    Returns:
        CustomReportBuilder: 初期化済みのCustomReportBuilderインスタンス
    """
    return CustomReportBuilder()

# エントリーポイント（テスト用）
if __name__ == "__main__":
    async def main():
        # レポートビルダーの初期化
        report_builder = get_custom_report_builder()

        # テストデータ
        test_data = {
            'revenue': 1000000,
            'growth_rate': 0.25,
            'market_size': '10B',
        }

        # レポートの生成と保存
        try:
            report_id = await report_builder.build_and_save_report(
                data=test_data,
                target_audience='VC',
                user_id='test_user',
                metadata={'version': '1.0'}
            )
            print(f"Created report with ID: {report_id}")

            # レポートの取得
            report = await report_builder.get_report(report_id)
            print("Retrieved report:", report)

        except Exception as e:
            print(f"Error: {str(e)}")

    # 非同期メインの実行
    asyncio.run(main())