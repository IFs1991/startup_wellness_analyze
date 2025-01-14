# -*- coding: utf-8 -*-
"""
インタラクティブな可視化サービス
Plotlyを使用したインタラクティブなグラフの生成と、
Firestoreでの永続化を提供します。
"""
from typing import Dict, Any, Optional, List, Union, cast
import plotly.express as px
from plotly.graph_objs._figure import Figure
import pandas as pd
from datetime import datetime
import logging
from firebase_admin import firestore
import json

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class VisualizationError(Exception):
    """可視化処理に関するエラー"""
    pass

class InteractiveVisualizer:
    """
    Plotlyを使用したインタラクティブなグラフを作成し、
    Firestoreと連携するクラスです。
    """
    def __init__(self, db: Any):
        """
        初期化
        Args:
            db: Firestoreクライアントインスタンス
        """
        self.db = cast(firestore.firestore.Client, db)
        self.collection_name = 'visualizations'
        logger.info("Interactive Visualizer initialized")

    async def create_interactive_line_chart(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        インタラクティブな線グラフを作成しFirestoreに保存します。

        Args:
            data: グラフ化するデータ
            x_col: X軸のカラム名
            y_col: Y軸のカラム名
            title: グラフのタイトル
            user_id: ユーザーID
            metadata: 追加のメタデータ

        Returns:
            作成されたグラフの情報（ID、URL等）を含む辞書

        Raises:
            VisualizationError: グラフ作成または保存に失敗した場合
        """
        try:
            # グラフの作成
            fig = px.line(data, x=x_col, y=y_col, title=title)

            # Plotlyの図をJSONに変換
            graph_json = self._figure_to_json(fig)

            # Firestoreに保存するデータの作成
            visualization_data = {
                'type': 'line_chart',
                'title': title,
                'data': {
                    'x_col': x_col,
                    'y_col': y_col,
                    'graph_json': graph_json
                },
                'user_id': user_id,
                'created_at': datetime.now(),
                'metadata': metadata or {}
            }

            # Firestoreに保存
            doc_ref = self.db.collection(self.collection_name).document()
            doc_ref.set(visualization_data)

            result = {
                'visualization_id': doc_ref.id,
                'title': title,
                'created_at': visualization_data['created_at'],
                'type': 'line_chart'
            }

            logger.info(f"Successfully created line chart: {doc_ref.id}")
            return result

        except Exception as e:
            error_msg = f"Error creating line chart: {str(e)}"
            logger.error(error_msg)
            raise VisualizationError(error_msg) from e

    async def get_visualization(
        self,
        visualization_id: str
    ) -> Dict[str, Any]:
        """
        保存された可視化を取得します。

        Args:
            visualization_id: 可視化のID

        Returns:
            可視化データを含む辞書

        Raises:
            VisualizationError: データの取得に失敗した場合
        """
        try:
            doc_ref = self.db.collection(self.collection_name).document(visualization_id)
            doc = doc_ref.get()

            if not doc.exists:
                raise VisualizationError(f"Visualization {visualization_id} not found")

            data = doc.to_dict()

            # グラフJSONからFigureオブジェクトを再構築
            if data and 'data' in data and 'graph_json' in data['data']:
                graph_json = data['data']['graph_json']
                fig = self._json_to_figure(graph_json)
                data['figure'] = fig

            logger.info(f"Successfully retrieved visualization: {visualization_id}")
            return data if data else {}

        except Exception as e:
            error_msg = f"Error retrieving visualization: {str(e)}"
            logger.error(error_msg)
            raise VisualizationError(error_msg) from e

    def _figure_to_json(self, fig: Figure) -> str:
        """
        PlotlyのFigureオブジェクトをJSON文字列に変換します。
        """
        try:
            return json.dumps(fig.to_dict())
        except Exception as e:
            raise VisualizationError(f"Error converting figure to JSON: {str(e)}") from e

    def _json_to_figure(self, json_str: str) -> Figure:
        """
        JSON文字列からPlotlyのFigureオブジェクトを再構築します。
        """
        try:
            figure_dict = json.loads(json_str)
            return Figure(figure_dict)
        except Exception as e:
            raise VisualizationError(f"Error converting JSON to figure: {str(e)}") from e

    async def delete_visualization(
        self,
        visualization_id: str,
        user_id: str
    ) -> None:
        """
        保存された可視化を削除します。

        Args:
            visualization_id: 削除する可視化のID
            user_id: 削除を要求するユーザーのID

        Raises:
            VisualizationError: 削除に失敗した場合
        """
        try:
            doc_ref = self.db.collection(self.collection_name).document(visualization_id)
            doc = doc_ref.get()

            if not doc.exists:
                raise VisualizationError(f"Visualization {visualization_id} not found")

            data = doc.to_dict()
            if data and data.get('user_id') != user_id:
                raise VisualizationError("Permission denied")

            doc_ref.delete()
            logger.info(f"Successfully deleted visualization: {visualization_id}")

        except Exception as e:
            error_msg = f"Error deleting visualization: {str(e)}"
            logger.error(error_msg)
            raise VisualizationError(error_msg) from e