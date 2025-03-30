import logging
import base64
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import asyncio
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from dotenv import load_dotenv
from utils.gemini_wrapper import GeminiWrapper

# .envファイルを読み込む
load_dotenv()

logger = logging.getLogger(__name__)

class GeminiChartGenerator:
    """
    Google Gemini APIを活用してデータ可視化を生成するクラス。
    複雑なデータ可視化ライブラリ（matplotlib、seabornなど）の依存性を排除し、
    自然言語プロンプトでチャートを生成します。
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        GeminiChartGeneratorの初期化

        Args:
            api_key: Gemini APIキー。Noneの場合は環境変数から取得
        """
        # APIキーの取得優先順位: 引数 > 環境変数GEMINI_API_KEY
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables or .env file")
            raise ValueError("GEMINI_API_KEY is required. Please set it in .env file or provide as an argument.")

        self.gemini_wrapper = GeminiWrapper(api_key=self.api_key)
        self.cache_dir = Path("./cache/visualizations")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def generate_chart(self,
                           data: Dict[str, Any],
                           chart_type: str,
                           title: str,
                           description: Optional[str] = None,
                           width: int = 800,
                           height: int = 500,
                           theme: str = "professional",
                           use_cache: bool = True) -> Dict[str, Any]:
        """
        データセットから指定されたタイプのチャートを生成。

        Args:
            data: 可視化するデータ
            chart_type: チャートタイプ（bar、line、scatter、pie、heatmapなど）
            title: チャートのタイトル
            description: チャートの説明（オプション）
            width: 画像の幅
            height: 画像の高さ
            theme: テーマ（professional、dark、light、modernなど）
            use_cache: キャッシュを使用するかどうか

        Returns:
            生成された画像情報（base64エンコードデータを含む）

        Raises:
            Exception: チャート生成に失敗した場合
        """
        cache_key = self._generate_cache_key(data, chart_type, title, width, height, theme)
        cache_path = self.cache_dir / f"{cache_key}.png"

        # キャッシュチェック
        if use_cache and cache_path.exists():
            logger.info(f"Using cached visualization: {cache_path}")
            with open(cache_path, "rb") as f:
                image_data = f.read()
            return {
                "success": True,
                "image_data": base64.b64encode(image_data).decode("utf-8"),
                "format": "png",
                "width": width,
                "height": height,
                "cached": True
            }

        try:
            # Gemini APIを使用して画像を生成
            image_bytes = await self.gemini_wrapper.generate_visualization(
                data=data,
                chart_type=chart_type,
                title=title,
                description=description,
                width=width,
                height=height,
                theme=theme
            )

            # キャッシュに保存
            with open(cache_path, "wb") as f:
                f.write(image_bytes)

            return {
                "success": True,
                "image_data": base64.b64encode(image_bytes).decode("utf-8"),
                "format": "png",
                "width": width,
                "height": height,
                "cached": False
            }
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def generate_multiple_charts(self,
                                     chart_configs: List[Dict[str, Any]],
                                     use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        複数のチャートを並行して生成

        Args:
            chart_configs: チャート設定のリスト
            use_cache: キャッシュを使用するかどうか

        Returns:
            生成されたチャート情報のリスト
        """
        tasks = []
        for config in chart_configs:
            task = self.generate_chart(
                data=config.get("data", {}),
                chart_type=config.get("chart_type", "bar"),
                title=config.get("title", "Chart"),
                description=config.get("description"),
                width=config.get("width", 800),
                height=config.get("height", 500),
                theme=config.get("theme", "professional"),
                use_cache=use_cache
            )
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def generate_dashboard(self,
                               dashboard_data: Dict[str, Any],
                               title: str,
                               layout: Optional[List[Dict[str, Any]]] = None,
                               width: int = 1200,
                               height: int = 800,
                               theme: str = "professional") -> Dict[str, Any]:
        """
        複数のチャートを含むダッシュボードを生成

        Args:
            dashboard_data: ダッシュボード用データ
            title: ダッシュボードのタイトル
            layout: チャートのレイアウト設定
            width: ダッシュボードの幅
            height: ダッシュボードの高さ
            theme: テーマ

        Returns:
            生成されたダッシュボード情報（HTML形式）
        """
        try:
            # ダッシュボード用のレイアウトがない場合はデフォルトを生成
            if not layout:
                layout = self._generate_default_layout(dashboard_data)

            # 各チャートを生成
            chart_results = await self.generate_multiple_charts(
                [item["chart_config"] for item in layout if "chart_config" in item],
                use_cache=True
            )

            # チャート結果をレイアウトに組み込む
            for i, result in enumerate(chart_results):
                if i < len(layout) and "chart_config" in layout[i]:
                    layout[i]["chart_result"] = result

            # ダッシュボードHTMLを生成
            dashboard_html = self._generate_dashboard_html(
                title=title,
                layout=layout,
                width=width,
                height=height,
                theme=theme
            )

            return {
                "success": True,
                "html": dashboard_html,
                "width": width,
                "height": height
            }
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _generate_cache_key(self,
                           data: Dict[str, Any],
                           chart_type: str,
                           title: str,
                           width: int,
                           height: int,
                           theme: str) -> str:
        """
        キャッシュキーを生成

        Args:
            data: 可視化データ
            chart_type: チャートタイプ
            title: タイトル
            width: 幅
            height: 高さ
            theme: テーマ

        Returns:
            キャッシュキー文字列
        """
        import hashlib
        # データとパラメータをJSON化して一貫したハッシュを生成
        key_data = {
            "data": data,
            "chart_type": chart_type,
            "title": title,
            "width": width,
            "height": height,
            "theme": theme
        }
        json_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()

    def _generate_default_layout(self, dashboard_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        ダッシュボードの基本レイアウトを生成

        Args:
            dashboard_data: ダッシュボードデータ

        Returns:
            レイアウト設定のリスト
        """
        layout = []

        # データからチャートを自動生成
        for key, value in dashboard_data.items():
            if isinstance(value, dict) and len(value) > 0:
                # 辞書データがあればバーチャートかラインチャートを追加
                chart_type = "line" if any(isinstance(v, list) for v in value.values()) else "bar"
                layout.append({
                    "chart_config": {
                        "data": value,
                        "chart_type": chart_type,
                        "title": key.replace("_", " ").title(),
                        "width": 600,
                        "height": 400
                    },
                    "position": {"x": 0, "y": 0, "width": 6, "height": 4}
                })

        return layout

    def _generate_dashboard_html(self,
                                title: str,
                                layout: List[Dict[str, Any]],
                                width: int,
                                height: int,
                                theme: str) -> str:
        """
        ダッシュボードのHTML表現を生成

        Args:
            title: ダッシュボードのタイトル
            layout: レイアウト設定
            width: 幅
            height: 高さ
            theme: テーマ

        Returns:
            HTML文字列
        """
        # HTMLテンプレート
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: {self._get_theme_background(theme)};
                    color: {self._get_theme_text_color(theme)};
                }}
                .dashboard-container {{
                    width: {width}px;
                    min-height: {height}px;
                    display: grid;
                    grid-template-columns: repeat(12, 1fr);
                    grid-auto-rows: minmax(100px, auto);
                    gap: 20px;
                }}
                .dashboard-title {{
                    grid-column: 1 / -1;
                    text-align: center;
                    font-size: 24px;
                    margin-bottom: 20px;
                }}
                .chart-container {{
                    background-color: {self._get_theme_card_background(theme)};
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    padding: 15px;
                    display: flex;
                    flex-direction: column;
                }}
                .chart-title {{
                    font-size: 16px;
                    margin-bottom: 10px;
                    color: {self._get_theme_title_color(theme)};
                }}
                .chart-image {{
                    flex: 1;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
                .chart-image img {{
                    max-width: 100%;
                    max-height: 100%;
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-title">{title}</div>
            <div class="dashboard-container">
        """

        # 各チャートのHTMLを追加
        for item in layout:
            if "chart_result" in item and item["chart_result"].get("success", False):
                chart_result = item["chart_result"]
                position = item.get("position", {"x": 0, "y": 0, "width": 6, "height": 4})

                # グリッド位置の計算
                grid_col_start = position.get("x", 0) + 1
                grid_col_end = grid_col_start + position.get("width", 6)
                grid_row_start = position.get("y", 0) + 1
                grid_row_end = grid_row_start + position.get("height", 4)

                chart_title = item.get("chart_config", {}).get("title", "Chart")
                image_data = chart_result.get("image_data", "")

                html += f"""
                <div class="chart-container" style="grid-column: {grid_col_start} / {grid_col_end}; grid-row: {grid_row_start} / {grid_row_end};">
                    <div class="chart-title">{chart_title}</div>
                    <div class="chart-image">
                        <img src="data:image/png;base64,{image_data}" alt="{chart_title}">
                    </div>
                </div>
                """

        # HTMLを閉じる
        html += """
            </div>
        </body>
        </html>
        """

        return html

    def _get_theme_background(self, theme: str) -> str:
        """テーマに応じた背景色を取得"""
        themes = {
            "professional": "#f5f7fa",
            "dark": "#1e1e1e",
            "light": "#ffffff",
            "modern": "#f0f2f5"
        }
        return themes.get(theme, themes["professional"])

    def _get_theme_card_background(self, theme: str) -> str:
        """テーマに応じたカード背景色を取得"""
        themes = {
            "professional": "#ffffff",
            "dark": "#2d2d2d",
            "light": "#f9f9f9",
            "modern": "#ffffff"
        }
        return themes.get(theme, themes["professional"])

    def _get_theme_text_color(self, theme: str) -> str:
        """テーマに応じたテキスト色を取得"""
        themes = {
            "professional": "#333333",
            "dark": "#e0e0e0",
            "light": "#333333",
            "modern": "#2c3e50"
        }
        return themes.get(theme, themes["professional"])

    def _get_theme_title_color(self, theme: str) -> str:
        """テーマに応じたタイトル色を取得"""
        themes = {
            "professional": "#2c3e50",
            "dark": "#ffffff",
            "light": "#2c3e50",
            "modern": "#1a365d"
        }
        return themes.get(theme, themes["professional"])