from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class GraphConfig(BaseModel):
    """グラフ設定のスキーマ"""
    graph_type: str
    title: str
    data_source: str
    x_axis: str
    y_axis: str
    filters: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None

class DashboardConfig(BaseModel):
    """ダッシュボード設定のスキーマ"""
    title: str
    layout: str
    graphs: List[GraphConfig]
    refresh_interval: Optional[int] = None
    filters: Optional[Dict[str, Any]] = None

class VisualizationResponse(BaseModel):
    """可視化レスポンスのスキーマ"""
    id: str
    user_id: str
    visualization_type: str
    config: Dict[str, Any]
    created_at: str
    updated_at: Optional[str] = None