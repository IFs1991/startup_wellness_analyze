# -*- coding: utf-8 -*-
"""
レポートカスタマイズサービス
対象読者 (VC, 経営陣, 従業員) 別のレポートカスタマイズと永続化を行います。
"""
from typing import Dict, Any, Optional, List, Union, cast
from datetime import datetime, timedelta
import logging
import asyncio
import firebase_admin
from firebase_admin import firestore
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import io
import base64
from PIL import Image

# ロギングの設定
# モジュールレベルのロガー設定は維持しつつ、クラス内でも使用できるようにします
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
        ロガー、Firestoreクライアント、コレクション名を設定します
        """
        try:
            # インスタンス変数としてロガーを設定
            self.logger = logging.getLogger(__name__)

            # firebase-adminの初期化（まだ初期化されていない場合）
            if not firebase_admin._apps:
                firebase_admin.initialize_app()
            self.db = firestore.client()
            self.collection_name = 'custom_reports'
            self.logger.info("CustomReportBuilder initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize CustomReportBuilder: {str(e)}")
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
            data: レポートに含めるデータ
            target_audience: 対象読者 ('VC', '経営陣', '従業員')
            user_id: レポート作成者のID
            metadata: 追加のメタデータ

        Returns:
            生成されたレポートのドキュメントID

        Raises:
            ReportError: レポート生成または保存に失敗した場合
            ValueError: 無効な対象読者が指定された場合
        """
        try:
            self.logger.info(f"Building report for target audience: {target_audience}")

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

            self.logger.info(f"Successfully saved report with ID: {doc_ref.id}")
            return doc_ref.id

        except ValueError as e:
            self.logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            error_msg = f"Error building and saving report: {str(e)}"
            self.logger.error(error_msg)
            raise ReportError(error_msg) from e

    async def _build_report_content(
        self,
        data: Dict[str, Any],
        target_audience: str
    ) -> Dict[str, Any]:
        """レポート内容を構築するプライベートメソッド"""
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
            self.logger.error(error_msg)
            raise ReportError(error_msg) from e

    # データ抽出メソッド群...
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

    async def build_correlation_report(
        self,
        correlation_data: Dict[str, Any],
        target_audience: str
    ) -> Dict[str, Any]:
        """相関分析レポートを生成"""
        try:
            loop = asyncio.get_event_loop()
            # 基本的な相関情報の抽出
            significant_correlations = await loop.run_in_executor(
                None,
                self._extract_significant_correlations,
                correlation_data['correlation_matrix'],
                0.3
            )

            # 対象読者に応じたレポートの構築
            report = {
                'summary': await loop.run_in_executor(
                    None,
                    self._generate_correlation_summary,
                    significant_correlations
                ),
                'key_findings': await loop.run_in_executor(
                    None,
                    self._extract_key_findings,
                    significant_correlations
                ),
                'recommendations': await loop.run_in_executor(
                    None,
                    self._generate_recommendations,
                    significant_correlations,
                    target_audience
                ),
                'visualizations': await loop.run_in_executor(
                    None,
                    self._create_correlation_visualizations,
                    correlation_data['correlation_matrix']
                )
            }

            return report

        except Exception as e:
            self.logger.error(f"相関分析レポート生成中にエラーが発生: {str(e)}")
            raise ReportError(f"レポート生成中にエラーが発生: {str(e)}")

    def _extract_significant_correlations(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """重要な相関関係を抽出"""
        significant_corrs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                try:
                    # 相関係数を適切に取得し型変換
                    corr_value = correlation_matrix.iloc[i, j]

                    # numpy型からfloatへの変換
                    if isinstance(corr_value, (np.number, pd.Series)):
                        corr = float(np.asarray(corr_value).astype(np.float64))
                    elif isinstance(corr_value, (bytes, bytearray, memoryview)):
                        # バイナリデータの場合はスキップ
                        continue
                    elif pd.isna(corr_value):
                        # 欠損値の場合はスキップ
                        continue
                    else:
                        try:
                            # その他の型の場合は文字列経由で変換を試みる
                            corr = float(str(corr_value))
                        except (ValueError, TypeError):
                            # 変換できない場合はスキップ
                            continue

                    if abs(corr) >= threshold:
                        significant_corrs.append({
                            'var1': str(correlation_matrix.columns[i]),
                            'var2': str(correlation_matrix.columns[j]),
                            'correlation': corr,
                            'strength': self._get_correlation_strength(corr)
                        })
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"相関係数の変換中にエラー: {str(e)}")
                    continue

        return significant_corrs

    def _generate_correlation_summary(
        self,
        correlations: List[Dict[str, Any]]
    ) -> str:
        """相関分析の要約を生成"""
        strong_corrs = sum(1 for c in correlations if c['strength'] == '強い')
        moderate_corrs = sum(1 for c in correlations if c['strength'] == '中程度')

        return (
            f"分析により{len(correlations)}個の重要な相関関係が発見されました。"
            f"そのうち、{strong_corrs}個が強い相関、"
            f"{moderate_corrs}個が中程度の相関を示しています。"
        )

    def _extract_key_findings(
        self,
        correlations: List[Dict[str, Any]]
    ) -> List[str]:
        """主要な発見事項を抽出"""
        findings = []
        for corr in correlations:
            direction = '正の' if corr['correlation'] > 0 else '負の'
            findings.append(
                f"{corr['var1']}と{corr['var2']}の間に"
                f"{direction}{corr['strength']}相関"
                f"（相関係数: {corr['correlation']:.2f}）が見られます"
            )
        return findings

    def _get_correlation_strength(self, correlation: float) -> str:
        """
        相関係数の強さを判定

        Args:
            correlation: 相関係数（float型）

        Returns:
            str: 相関の強さの説明（'強い', '中程度', '弱い'）
        """
        abs_corr = abs(float(correlation))  # 明示的にfloat型に変換
        if abs_corr >= 0.7:
            return '強い'
        elif abs_corr >= 0.4:
            return '中程度'
        else:
            return '弱い'

    def _create_correlation_network(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        相関ネットワークグラフを生成します。
        視認性を重視した実装となっています。

        Args:
            correlation_matrix: 相関行列
            threshold: 表示する相関の閾値（デフォルト: 0.5）

        Returns:
            Dict[str, Any]: エンコードされたネットワークグラフデータ
        """
        try:
            # NetworkXグラフの作成
            G = nx.Graph()

            # ノードの追加
            for col in correlation_matrix.columns:
                G.add_node(str(col))  # 文字列型に変換

            # エッジの追加（閾値以上の相関のみ）
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    try:
                        # 相関係数を適切に取得し型変換
                        corr_value = correlation_matrix.iloc[i, j]

                        # numpy型からfloatへの変換
                        if isinstance(corr_value, (np.number, pd.Series)):
                            corr = float(np.asarray(corr_value).astype(np.float64))
                        elif isinstance(corr_value, (bytes, bytearray, memoryview)):
                            # バイナリデータの場合はスキップ
                            continue
                        elif pd.isna(corr_value):
                            # 欠損値の場合はスキップ
                            continue
                        else:
                            try:
                                # その他の型の場合は文字列経由で変換を試みる
                                corr = float(str(corr_value))
                            except (ValueError, TypeError):
                                # 変換できない場合はスキップ
                                continue

                        if abs(corr) >= threshold:
                            G.add_edge(
                                str(correlation_matrix.columns[i]),
                                str(correlation_matrix.columns[j]),
                                weight=abs(corr)
                            )
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"相関係数の変換中にエラー: {str(e)}")
                        continue

            # レイアウトの計算（より安定したレイアウトのために重み付けを使用）
            pos = nx.spring_layout(G, k=2.0, iterations=50)

            # Plotlyでネットワークグラフを作成
            edge_x = []
            edge_y = []
            edge_text = []
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_text.append(f'相関係数: {edge[2]["weight"]:.2f}')

            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_text = list(G.nodes())

            # グラフの作成
            fig = go.Figure()

            # エッジの追加（相関の強さに応じて線の太さと色を変える）
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(
                    width=2,
                    color='rgba(100,100,100,0.8)'
                ),
                hoverinfo='text',
                text=edge_text,
                mode='lines'
            ))

            # ノードの追加
            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="top center",
                marker=dict(
                    size=20,
                    color='#1f77b4',
                    line=dict(width=2, color='white')
                )
            ))

            # レイアウトの設定
            fig.update_layout(
                title='相関ネットワーク図',
                showlegend=False,
                width=1000,
                height=800,
                plot_bgcolor='white',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(l=50, r=50, t=50, b=50)
            )

            # 画像として保存
            img_bytes = fig.to_image(format="png")

            # Base64エンコード
            encoded_image = base64.b64encode(img_bytes).decode('utf-8')

            return {
                'type': 'network',
                'format': 'png',
                'data': encoded_image
            }

        except Exception as e:
            self.logger.error(f"ネットワークグラフ生成中にエラー: {str(e)}")
            return {
                'type': 'network',
                'error': str(e)
            }

    def _generate_recommendations(
        self,
        correlations: List[Dict[str, Any]],
        target_audience: str
    ) -> List[str]:
        """
        対象読者に応じた推奨事項を生成します。

        Args:
            correlations: 重要な相関関係のリスト
            target_audience: 対象読者

        Returns:
            List[str]: 推奨事項のリスト
        """
        if target_audience == 'VC':
            return self._generate_vc_recommendations(correlations)
        elif target_audience == '経営陣':
            return self._generate_management_recommendations(correlations)
        else:
            return self._generate_employee_recommendations(correlations)

    def _generate_vc_recommendations(
        self,
        correlations: List[Dict[str, Any]]
    ) -> List[str]:
        """VCむけの推奨事項を生成"""
        recommendations = []
        for corr in correlations:
            if abs(float(corr['correlation'])) >= 0.7:
                recommendations.append(
                    f"{corr['var1']}と{corr['var2']}の強い関連性は"
                    "投資判断の重要な指標となり得ます"
                )
        return recommendations

    def _generate_management_recommendations(
        self,
        correlations: List[Dict[str, Any]]
    ) -> List[str]:
        """経営陣むけの推奨事項を生成"""
        recommendations = []
        for corr in correlations:
            if abs(float(corr['correlation'])) >= 0.5:
                recommendations.append(
                    f"{corr['var1']}と{corr['var2']}の関連性に基づき、"
                    "戦略的な施策の検討が推奨されます"
                )
        return recommendations

    def _generate_employee_recommendations(
        self,
        correlations: List[Dict[str, Any]]
    ) -> List[str]:
        """従業員むけの推奨事項を生成"""
        recommendations = []
        for corr in correlations:
            if abs(float(corr['correlation'])) >= 0.4:
                recommendations.append(
                    f"{corr['var1']}と{corr['var2']}の関連性を"
                    "日常業務に活かすことが推奨されます"
                )
        return recommendations

    def _create_correlation_visualizations(
        self,
        correlation_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        相関関係の可視化を生成します。
        ネットワークグラフのみを生成し、視認性の高い形式で提供します。

        Args:
            correlation_matrix: 相関行列

        Returns:
            Dict[str, Any]: エンコードされた可視化データ
        """
        try:
            return self._create_correlation_network(correlation_matrix)
        except Exception as e:
            self.logger.error(f"相関の可視化生成中にエラー: {str(e)}")
            return {
                'type': 'network',
                'error': str(e)
            }

    async def get_report(self, report_id: str) -> Dict[str, Any]:
        """
        Firestoreからレポートを取得します。

        Args:
            report_id: レポートのドキュメントID

        Returns:
            Dict[str, Any]: レポートのデータ

        Raises:
            ReportError: レポートの取得に失敗した場合
        """
        try:
            # Firestoreのドキュメント参照を取得
            doc_ref = self.db.collection(self.collection_name).document(report_id)
            doc = doc_ref.get()

            if not doc.exists:
                raise ReportError(f"Report with ID {report_id} not found")

            # ドキュメントデータを辞書形式で取得
            report_data = doc.to_dict()
            if report_data is None:
                raise ReportError(f"Report data is empty for ID {report_id}")

            return report_data

        except Exception as e:
            error_msg = f"レポート取得中にエラーが発生: {str(e)}"
            self.logger.error(error_msg)
            raise ReportError(error_msg) from e

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
            'team_performance': {
                'efficiency': 0.85,
                'satisfaction': 0.9
            }
        }

        # レポートの生成と保存
        try:
            # VCむけレポートの生成
            vc_report_id = await report_builder.build_and_save_report(
                data=test_data,
                target_audience='VC',
                user_id='test_user',
                metadata={'version': '1.0', 'type': 'quarterly'}
            )
            print(f"Created VC report with ID: {vc_report_id}")

            # 経営陣むけレポートの生成
            management_report_id = await report_builder.build_and_save_report(
                data=test_data,
                target_audience='経営陣',
                user_id='test_user',
                metadata={'version': '1.0', 'type': 'monthly'}
            )
            print(f"Created management report with ID: {management_report_id}")

            # レポートの取得と内容確認
            vc_report = await report_builder.get_report(vc_report_id)
            print("\nVC Report Content:", vc_report)

            management_report = await report_builder.get_report(management_report_id)
            print("\nManagement Report Content:", management_report)

        except Exception as e:
            print(f"Error during test: {str(e)}")

    # 非同期メインの実行
    asyncio.run(main())