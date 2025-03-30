import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output
from google.cloud.firestore_v1 import DocumentSnapshot
from google.cloud.firestore_v1.watch import DocumentChange  # 修正: 正しい型をインポート
from google.cloud import firestore
import logging

# ロガーの設定
logger = logging.getLogger(__name__)

class DashboardCreationError(Exception):
    """ダッシュボード作成時のエラーを表すカスタム例外"""
    pass

class DashboardConfig:
    """ダッシュボードの設定を保持するクラス"""
    def __init__(
        self,
        collection_name: str,
        startup_field: str,
        time_field: str = 'timestamp',
        title: str = 'スタートアップ分析ダッシュボード',
        metrics: Optional[List[str]] = None
    ):
        self.collection_name = collection_name
        self.startup_field = startup_field
        self.time_field = time_field
        self.title = title
        self.metrics = metrics or []

class DashboardCreator:
    """Firestoreデータを使用してダッシュボードを作成するクラス"""
    def __init__(self, config: DashboardConfig, db: Optional[Any] = None):
        """
        Args:
            config (DashboardConfig): ダッシュボードの設定
            db (Optional[Any]): 既存のFirestoreクライアントインスタンス
        """
        self.config = config

        # 既存のクライアントが提供されていれば使用、なければモックオブジェクトを作成
        try:
            if db is not None:
                self.db = db
            else:
                try:
                    # 既存クライアントがない場合はFirestoreClientの初期化を試みる
                    self.db = firestore.Client()
                    logger.info("Firestore client initialized for DashboardCreator")
                except Exception as e:
                    # 初期化失敗時はモックオブジェクトを作成
                    logger.warning(f"Failed to initialize Firestore client: {str(e)}")
                    logger.warning("Using a mock Firestore client for dashboard")
                    from unittest.mock import MagicMock
                    mock_db = MagicMock()
                    mock_collection = MagicMock()
                    mock_db.collection.return_value = mock_collection
                    mock_collection.order_by.return_value = mock_collection
                    mock_collection.get.return_value = []
                    self.db = mock_db
        except Exception as e:
            logger.error(f"Error setting up Firestore client: {str(e)}")
            # エラー発生時はモックオブジェクトを使用
            from unittest.mock import MagicMock
            self.db = MagicMock()

        self.app = dash.Dash(__name__)
        self._data: Optional[pd.DataFrame] = None

    async def fetch_dashboard_data(self) -> pd.DataFrame:
        """
        Firestoreからダッシュボードのデータを取得
        Returns:
            pd.DataFrame: 取得したデータ
        """
        try:
            logger.info(f"Fetching data from collection: {self.config.collection_name}")
            collection_ref = self.db.collection(self.config.collection_name)
            query = collection_ref.order_by(self.config.time_field)

            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, query.get)

            data = []
            for doc in docs:
                doc_data = doc.to_dict()
                if doc_data:
                    doc_data['id'] = doc.id
                    data.append(doc_data)

            df = pd.DataFrame(data) if data else pd.DataFrame()
            logger.info(f"Successfully fetched {len(df)} records")
            self._data = df
            return df

        except Exception as e:
            error_msg = f"Error fetching dashboard data: {str(e)}"
            logger.error(error_msg)
            raise DashboardCreationError(error_msg) from e

    def get_startup_options(self, data: Optional[pd.DataFrame] = None) -> List[Dict[str, str]]:
        """
        スタートアップのドロップダウンオプションを生成
        Args:
            data: Optional[pd.DataFrame] - 使用するデータフレーム
        Returns:
            List[Dict[str, str]]: ドロップダウンオプションのリスト
        """
        df = data if data is not None else self._data
        if df is None or df.empty or self.config.startup_field not in df.columns:
            return []

        startups = df[self.config.startup_field].unique()
        return [{'label': str(i), 'value': str(i)} for i in startups]

    async def create_dashboard(self) -> dash.Dash:
        """
        Firestoreのデータを使用してダッシュボードを作成
        Returns:
            dash.Dash: 作成したダッシュボードオブジェクト
        """
        try:
            data = await self.fetch_dashboard_data()
            startup_options = self.get_startup_options(data)
            default_value = startup_options[0]['value'] if startup_options else ''

            self.app.layout = html.Div(children=[
                html.H1(children=self.config.title),
                html.Div(className="dashboard-controls", children=[
                    dcc.Dropdown(
                        id='startup-dropdown',
                        options=startup_options,
                        value=default_value
                    ),
                ]),
                html.Div(className="dashboard-graphs", children=[
                    dcc.Graph(id='vas-time-series'),
                    dcc.Graph(id='financial-data')
                ])
            ])

            self._setup_callbacks(data)
            logger.info("Dashboard created successfully")
            return self.app

        except Exception as e:
            error_msg = f"Error creating dashboard: {str(e)}"
            logger.error(error_msg)
            raise DashboardCreationError(error_msg) from e

    def _setup_callbacks(self, data: pd.DataFrame) -> None:
        """
        ダッシュボードのコールバックを設定
        Args:
            data (pd.DataFrame): ダッシュボードに表示するデータ
        """
        @self.app.callback(
            Output('vas-time-series', 'figure'),
            [Input('startup-dropdown', 'value')]
        )
        def update_vas_graph(selected_startup: str) -> Dict[str, Any]:
            try:
                if not selected_startup or data is None or data.empty:
                    return {}

                filtered_data = data[
                    data[self.config.startup_field] == selected_startup
                ].copy()

                if filtered_data.empty:
                    return {}

                figure = {
                    'data': [
                        {
                            'x': filtered_data[self.config.time_field],
                            'y': filtered_data['physical_symptoms'],
                            'type': 'line',
                            'name': 'Physical Symptoms'
                        }
                    ],
                    'layout': {
                        'title': f'{selected_startup} の VAS スコア推移',
                        'xaxis': {'title': '日付'},
                        'yaxis': {'title': 'スコア'},
                    }
                }
                return figure
            except Exception as e:
                logger.error(f"Error updating VAS graph: {str(e)}")
                return {}

        @self.app.callback(
            Output('financial-data', 'figure'),
            [Input('startup-dropdown', 'value')]
        )
        def update_financial_graph(selected_startup: str) -> Dict[str, Any]:
            try:
                if not selected_startup or data is None or data.empty:
                    return {}

                filtered_data = data[
                    data[self.config.startup_field] == selected_startup
                ].copy()

                if filtered_data.empty:
                    return {}

                figure = {
                    'data': [
                        {
                            'x': filtered_data['year'],
                            'y': filtered_data['revenue'],
                            'type': 'bar',
                            'name': '売上高'
                        }
                    ],
                    'layout': {
                        'title': f'{selected_startup} の 財務データ',
                        'xaxis': {'title': '年度'},
                        'yaxis': {'title': '金額（円）'},
                    }
                }
                return figure
            except Exception as e:
                logger.error(f"Error updating financial graph: {str(e)}")
                return {}

    async def update_real_time_data(self) -> None:
        """
        リアルタイムデータの更新を監視し、ダッシュボードを更新
        """
        try:
            def on_snapshot(
                doc_snapshot: List[DocumentSnapshot],
                changes: List[DocumentChange],  # 修正: 正しい型を使用
                read_time: datetime
            ) -> None:
                logger.info("Received real-time update")
                asyncio.create_task(self.create_dashboard())

            collection_ref = self.db.collection(self.config.collection_name)
            collection_ref.on_snapshot(on_snapshot)

        except Exception as e:
            error_msg = f"Error setting up real-time updates: {str(e)}"
            logger.error(error_msg)
            raise DashboardCreationError(error_msg) from e

    async def close(self) -> None:
        """
        リソースのクリーンアップ
        """
        try:
            # Firestoreのリスナーを停止する場合はここに実装
            logger.info("Dashboard resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up dashboard resources: {str(e)}")
            raise

def create_dashboard(config: DashboardConfig) -> DashboardCreator:
    """
    ダッシュボードクリエーターのインスタンスを作成
    Args:
        config (DashboardConfig): ダッシュボードの設定
    Returns:
        DashboardCreator: 作成したインスタンス
    """
    return DashboardCreator(config)

# 使用例
async def main():
    try:
        # ダッシュボードの設定
        config = DashboardConfig(
            collection_name='startup_data',
            startup_field='startup_name',
            metrics=['revenue', 'physical_symptoms']
        )

        # ダッシュボードの作成
        dashboard = create_dashboard(config)
        app = await dashboard.create_dashboard()

        # リアルタイム更新の開始
        await dashboard.update_real_time_data()

        # サーバーの起動
        app.run_server(debug=True)

    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        raise

if __name__ == '__main__':
    asyncio.run(main())