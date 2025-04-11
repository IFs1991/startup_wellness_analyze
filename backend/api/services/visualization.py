"""
可視化サービス

このモジュールは可視化機能の中核となるサービスを提供します。
チャート、グラフ、ダッシュボードの生成と管理を担当します。
"""

import logging
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from fastapi import Depends, HTTPException
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from api.core.config import Settings, get_settings
from api.utils.storage import StorageService, get_storage_service

# カスタム例外クラス
class ChartGenerationError(Exception):
    """チャート生成中に発生したエラー"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

class VisualizationService:
    """可視化サービス

    チャートとダッシュボードの生成と管理を行うサービスクラス
    """

    def __init__(
        self,
        settings: Settings,
        storage_service: StorageService
    ):
        """可視化サービスの初期化

        Args:
            settings: アプリケーション設定
            storage_service: ストレージサービス
        """
        self.settings = settings
        self.storage_service = storage_service
        self.output_dir = Path(settings.temp_dir) / "visualization"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    async def generate_chart(
        self,
        config: Dict[str, Any],
        data: Dict[str, Any],
        format: str = "png",
        template_id: Optional[str] = None,
        user_id: Optional[str] = None,
        layout: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """チャートを生成する

        Args:
            config: チャート設定
            data: チャートデータ
            format: 出力フォーマット
            template_id: テンプレートID
            user_id: ユーザーID
            layout: レイアウト設定

        Returns:
            生成されたチャート情報

        Raises:
            ChartGenerationError: チャート生成に失敗した場合
        """
        try:
            self.logger.info(f"チャート生成開始: {config.get('chart_type')} - {config.get('title')}")

            # チャート生成の実装
            chart_id = str(uuid.uuid4())

            # データをDataFrameに変換
            df = self._prepare_data(data)

            # Matplotlibを使用してチャートを生成
            fig, ax = plt.subplots(figsize=(10, 6))

            chart_type = config.get('chart_type', 'bar').lower()

            if chart_type == 'bar':
                self._create_bar_chart(df, ax, config)
            elif chart_type == 'line':
                self._create_line_chart(df, ax, config)
            elif chart_type == 'pie':
                self._create_pie_chart(df, ax, config)
            elif chart_type == 'scatter':
                self._create_scatter_chart(df, ax, config)
            else:
                raise ChartGenerationError(f"未対応のチャートタイプ: {chart_type}")

            # タイトルと軸ラベルの設定
            plt.title(config.get('title', ''))
            if config.get('x_axis_label'):
                plt.xlabel(config.get('x_axis_label'))
            if config.get('y_axis_label'):
                plt.ylabel(config.get('y_axis_label'))

            # 色設定
            if config.get('color_scheme') and config.get('color_scheme') != 'default':
                try:
                    plt.set_cmap(config.get('color_scheme'))
                except:
                    self.logger.warning(f"指定されたカラースキームが見つかりません: {config.get('color_scheme')}")

            # ファイル保存
            output_path = self.output_dir / f"{chart_id}.{format}"
            plt.savefig(output_path, format=format, bbox_inches='tight', dpi=300)
            plt.close(fig)

            # ストレージに保存
            storage_path = f"visualization/{user_id if user_id else 'anonymous'}/{chart_id}.{format}"
            public_url = await self.storage_service.upload_file(
                file_path=str(output_path),
                storage_path=storage_path,
                content_type=f"image/{format}"
            )

            # サムネイル生成（オプション）
            thumbnail_url = None
            if format != "svg":
                # サムネイル用に小さいサイズで再生成
                thumb_fig, thumb_ax = plt.subplots(figsize=(3, 2))
                if chart_type == 'bar':
                    self._create_bar_chart(df, thumb_ax, config, is_thumbnail=True)
                elif chart_type == 'line':
                    self._create_line_chart(df, thumb_ax, config, is_thumbnail=True)
                elif chart_type == 'pie':
                    self._create_pie_chart(df, thumb_ax, config, is_thumbnail=True)
                elif chart_type == 'scatter':
                    self._create_scatter_chart(df, thumb_ax, config, is_thumbnail=True)

                thumb_path = self.output_dir / f"{chart_id}_thumb.{format}"
                plt.savefig(thumb_path, format=format, bbox_inches='tight', dpi=150)
                plt.close(thumb_fig)

                # サムネイルをストレージに保存
                thumb_storage_path = f"visualization/{user_id if user_id else 'anonymous'}/{chart_id}_thumb.{format}"
                thumbnail_url = await self.storage_service.upload_file(
                    file_path=str(thumb_path),
                    storage_path=thumb_storage_path,
                    content_type=f"image/{format}"
                )

            # メタデータを保存
            metadata = {
                "created_at": datetime.now().isoformat(),
                "user_id": user_id,
                "chart_type": chart_type,
                "title": config.get('title'),
                "template_id": template_id,
                "format": format
            }

            # メタデータをストレージに保存
            metadata_path = f"visualization/{user_id if user_id else 'anonymous'}/{chart_id}.json"
            metadata_file = self.output_dir / f"{chart_id}.json"
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)

            await self.storage_service.upload_file(
                file_path=str(metadata_file),
                storage_path=metadata_path,
                content_type="application/json"
            )

            # 結果を返す
            return {
                "chart_id": chart_id,
                "url": public_url,
                "format": format,
                "thumbnail_url": thumbnail_url,
                "metadata": metadata,
                "file_path": str(output_path)
            }

        except ChartGenerationError as e:
            self.logger.error(f"チャート生成エラー: {str(e)}")
            raise
        except Exception as e:
            self.logger.exception(f"予期せぬエラー: {str(e)}")
            raise ChartGenerationError(message=f"チャート生成中にエラーが発生しました: {str(e)}")

    async def generate_multiple_charts(
        self,
        chart_requests: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """複数のチャートを生成する

        Args:
            chart_requests: チャート生成リクエストのリスト
            user_id: ユーザーID

        Returns:
            生成されたチャート情報のリスト

        Raises:
            ChartGenerationError: チャート生成に失敗した場合
        """
        results = []

        for i, request in enumerate(chart_requests):
            try:
                chart_result = await self.generate_chart(
                    config=request.get('config', {}),
                    data=request.get('data', {}),
                    format=request.get('format', 'png'),
                    template_id=request.get('template_id'),
                    user_id=user_id,
                    layout=request.get('layout')
                )
                results.append(chart_result)

            except Exception as e:
                self.logger.error(f"チャート {i+1}/{len(chart_requests)} の生成に失敗しました: {str(e)}")
                # エラーが発生しても処理を続行し、他のチャート生成を試みる
                results.append({
                    "error": str(e),
                    "chart_id": f"error_{uuid.uuid4()}",
                    "status": "failed"
                })

        return results

    async def generate_dashboard(
        self,
        title: str,
        sections: List[Dict[str, Any]],
        chart_ids: List[str],
        description: Optional[str] = None,
        theme: str = "light",
        format: str = "pdf",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """ダッシュボードを生成する

        Args:
            title: ダッシュボードのタイトル
            sections: セクション情報
            chart_ids: 含まれるチャートのID一覧
            description: ダッシュボードの説明
            theme: テーマ
            format: 出力フォーマット
            user_id: ユーザーID

        Returns:
            生成されたダッシュボード情報

        Raises:
            ChartGenerationError: ダッシュボード生成に失敗した場合
        """
        try:
            self.logger.info(f"ダッシュボード生成開始: {title}")
            dashboard_id = str(uuid.uuid4())

            # HTMLテンプレートの作成（実際の実装ではテンプレートエンジンを使用するべき）
            html_content = f"""
            <!DOCTYPE html>
            <html lang="ja">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{title}</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: {self._get_theme_color(theme, 'background')};
                        color: {self._get_theme_color(theme, 'text')};
                    }}
                    .dashboard-header {{
                        text-align: center;
                        margin-bottom: 30px;
                        padding-bottom: 20px;
                        border-bottom: 1px solid {self._get_theme_color(theme, 'border')};
                    }}
                    .dashboard-section {{
                        margin-bottom: 30px;
                        padding: 20px;
                        background-color: {self._get_theme_color(theme, 'panel')};
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    .section-title {{
                        margin-top: 0;
                        color: {self._get_theme_color(theme, 'heading')};
                        border-bottom: 1px solid {self._get_theme_color(theme, 'border')};
                        padding-bottom: 10px;
                    }}
                    .charts-container {{
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: center;
                        gap: 20px;
                    }}
                    .chart-wrapper {{
                        text-align: center;
                        margin-bottom: 20px;
                    }}
                    img {{
                        max-width: 100%;
                        height: auto;
                        border: 1px solid {self._get_theme_color(theme, 'border')};
                        border-radius: 5px;
                    }}
                </style>
            </head>
            <body>
                <div class="dashboard-header">
                    <h1>{title}</h1>
                    {f"<p>{description}</p>" if description else ""}
                    <p>生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            """

            # 各セクションとチャートの追加
            for section in sections:
                section_title = section.get("title", "無題のセクション")
                html_content += f"""
                <div class="dashboard-section">
                    <h2 class="section-title">{section_title}</h2>
                    <div class="charts-container">
                """

                # セクション内のチャートを追加
                for chart_index in section.get("charts", []):
                    try:
                        if chart_index < 0 or chart_index >= len(chart_ids):
                            self.logger.warning(f"インデックス {chart_index} が範囲外です")
                            continue

                        chart_id = chart_ids[chart_index]

                        # チャート画像のパスを取得
                        chart_info = await self.get_chart_info(chart_id, user_id)
                        if not chart_info:
                            self.logger.warning(f"チャート {chart_id} が見つかりません")
                            continue

                        html_content += f"""
                        <div class="chart-wrapper">
                            <img src="{chart_info.get('url')}" alt="Chart">
                            <p>{chart_info.get('metadata', {}).get('title', 'チャート')}</p>
                        </div>
                        """
                    except Exception as e:
                        self.logger.error(f"チャート {chart_id} の処理中にエラーが発生しました: {str(e)}")
                        # エラーが発生しても処理を続行

                html_content += """
                    </div>
                </div>
                """

            # HTMLの終了タグ
            html_content += """
            </body>
            </html>
            """

            # 出力ファイルの保存
            output_path = self.output_dir / f"{dashboard_id}.{format}"

            if format.lower() == "pdf":
                # HTMLからPDFへの変換（実際の実装ではweasyprint、wkhtmltopdfなどを使用）
                # ここでは簡易的な実装として、HTMLをファイルに保存し、ダミーPDFとして扱う
                html_path = self.output_dir / f"{dashboard_id}.html"
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)

                # 本来はここでHTML→PDF変換を行う
                # 例: await self._convert_html_to_pdf(html_path, output_path)

                # ダミー実装: HTMLをそのまま使用
                output_path = html_path
                format = "html"  # フォーマットを強制的にHTMLに変更

            else:
                # HTMLとして保存
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)

            # ストレージに保存
            storage_path = f"visualization/{user_id if user_id else 'anonymous'}/dashboard_{dashboard_id}.{format}"
            public_url = await self.storage_service.upload_file(
                file_path=str(output_path),
                storage_path=storage_path,
                content_type=f"{'text/html' if format == 'html' else 'application/pdf'}"
            )

            # 結果を返す
            return {
                "dashboard_id": dashboard_id,
                "url": public_url,
                "format": format,
                "chart_ids": chart_ids,
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "user_id": user_id,
                    "title": title,
                    "sections_count": len(sections),
                    "charts_count": len(chart_ids),
                    "theme": theme
                },
                "file_path": str(output_path)
            }

        except Exception as e:
            self.logger.exception(f"ダッシュボード生成エラー: {str(e)}")
            raise ChartGenerationError(message=f"ダッシュボード生成中にエラーが発生しました: {str(e)}")

    async def get_chart_info(
        self,
        chart_id: str,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """チャート情報を取得する

        Args:
            chart_id: チャートID
            user_id: ユーザーID

        Returns:
            チャート情報またはNone
        """
        try:
            # チャートファイルのパス
            file_path = await self.get_chart_file_path(chart_id, user_id)
            if not file_path or not os.path.exists(file_path):
                return None

            # ファイル形式の取得
            format = os.path.splitext(file_path)[1][1:]

            # ストレージURLの構築
            storage_path = f"visualization/{user_id if user_id else 'anonymous'}/{chart_id}.{format}"
            public_url = await self.storage_service.get_public_url(storage_path)

            # メタデータの読み込み
            metadata = {}
            metadata_file = self.output_dir / f"{chart_id}.json"
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, 'r') as f:
                    try:
                        metadata = json.load(f)
                    except:
                        self.logger.warning(f"メタデータファイルの読み込みに失敗しました: {metadata_file}")

            return {
                "chart_id": chart_id,
                "url": public_url,
                "format": format,
                "file_path": file_path,
                "metadata": metadata
            }
        except Exception as e:
            self.logger.error(f"チャート情報の取得中にエラーが発生しました: {str(e)}")
            return None

    async def get_chart_file_path(
        self,
        chart_id: str,
        user_id: Optional[str] = None
    ) -> Optional[str]:
        """チャートファイルパスを取得する

        Args:
            chart_id: チャートID
            user_id: ユーザーID

        Returns:
            チャートファイルパスまたはNone
        """
        try:
            # サポートされている形式でファイルを探す
            for format in ['png', 'svg', 'pdf']:
                file_path = self.output_dir / f"{chart_id}.{format}"
                if os.path.exists(file_path):
                    return str(file_path)

            # 見つからない場合はストレージから検索（実装例）
            # ここでは簡略化のため未実装

            return None
        except Exception as e:
            self.logger.error(f"チャートファイルパスの取得中にエラーが発生しました: {str(e)}")
            return None

    def _prepare_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """チャートデータの準備

        データをPandasのデータフレームに変換します。

        Args:
            data: チャートデータ

        Returns:
            整形されたデータフレーム
        """
        try:
            # データをデータフレームに変換
            labels = data.get('labels', [])
            datasets = data.get('datasets', [])

            # データセットごとにデータを準備
            df_data = {}

            # ラベルをインデックスとして設定
            df_data['label'] = labels

            # 各データセットの値を追加
            for ds in datasets:
                ds_label = ds.get('label', f'dataset_{len(df_data)}')
                ds_data = ds.get('data', [])

                # データが不足している場合は0で補完
                if len(ds_data) < len(labels):
                    ds_data.extend([0] * (len(labels) - len(ds_data)))
                # データが多すぎる場合は切り捨て
                elif len(ds_data) > len(labels):
                    ds_data = ds_data[:len(labels)]

                df_data[ds_label] = ds_data

            # データフレームの作成
            df = pd.DataFrame(df_data)
            df.set_index('label', inplace=True)

            return df
        except Exception as e:
            self.logger.error(f"データ準備中にエラーが発生しました: {str(e)}")
            raise ChartGenerationError(message=f"データ準備中にエラーが発生しました: {str(e)}")

    def _create_bar_chart(self, df: pd.DataFrame, ax, config: Dict[str, Any], is_thumbnail: bool = False):
        """棒グラフの作成"""
        if is_thumbnail:
            # サムネイル用の簡略化された描画
            df.plot(kind='bar', ax=ax, legend=False)
            ax.set_xticklabels([])
            ax.set_ylabel('')
        else:
            # 通常のバーチャート描画
            df.plot(kind='bar', ax=ax, legend=config.get('show_legend', True))
            if config.get('show_legend', True):
                ax.legend(loc='best', frameon=True)

            # 値ラベルの表示（オプション）
            if config.get('show_values', False):
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.1f')

    def _create_line_chart(self, df: pd.DataFrame, ax, config: Dict[str, Any], is_thumbnail: bool = False):
        """折れ線グラフの作成"""
        if is_thumbnail:
            # サムネイル用の簡略化された描画
            df.plot(kind='line', ax=ax, legend=False)
            ax.set_xticklabels([])
            ax.set_ylabel('')
        else:
            # 通常の折れ線グラフ描画
            df.plot(kind='line', ax=ax, marker='o', legend=config.get('show_legend', True))
            if config.get('show_legend', True):
                ax.legend(loc='best', frameon=True)

            # グリッド表示（オプション）
            if config.get('show_grid', True):
                ax.grid(True, linestyle='--', alpha=0.7)

    def _create_pie_chart(self, df: pd.DataFrame, ax, config: Dict[str, Any], is_thumbnail: bool = False):
        """円グラフの作成"""
        # 円グラフはデータセットが1つのみ対応
        if len(df.columns) > 0:
            column = df.columns[0]

            if is_thumbnail:
                # サムネイル用の簡略化された描画
                df[column].plot(kind='pie', ax=ax, legend=False, autopct='', pctdistance=1.1)
                ax.set_ylabel('')
            else:
                # 通常の円グラフ描画
                df[column].plot(
                    kind='pie',
                    ax=ax,
                    autopct='%1.1f%%',
                    shadow=config.get('shadow', False),
                    explode=[0.05] * len(df) if config.get('explode', False) else None,
                    startangle=config.get('start_angle', 90),
                    legend=config.get('show_legend', False)
                )
                ax.set_ylabel('')

                # タイトル位置調整
                plt.subplots_adjust(top=0.85)

    def _create_scatter_chart(self, df: pd.DataFrame, ax, config: Dict[str, Any], is_thumbnail: bool = False):
        """散布図の作成"""
        if len(df.columns) >= 2:
            # 最初の2つのデータセットを使用
            x_col = df.columns[0]
            y_col = df.columns[1]

            if is_thumbnail:
                # サムネイル用の簡略化された描画
                ax.scatter(df[x_col], df[y_col])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                # 通常の散布図描画
                if config.get('use_seaborn', False):
                    # Seabornの拡張機能を使用
                    sns.scatterplot(
                        x=x_col,
                        y=y_col,
                        data=df.reset_index(),
                        hue=df.index.name if config.get('color_by_label', False) else None,
                        size=df.columns[2] if len(df.columns) > 2 and config.get('size_by_third_value', False) else None,
                        ax=ax
                    )
                else:
                    # Matplotlibの標準散布図
                    scatter = ax.scatter(df[x_col], df[y_col])

                    # ポイントにラベル表示（オプション）
                    if config.get('show_point_labels', False):
                        for i, label in enumerate(df.index):
                            ax.annotate(label, (df[x_col].iloc[i], df[y_col].iloc[i]))

    def _get_theme_color(self, theme: str, element: str) -> str:
        """テーマカラーの取得

        Args:
            theme: テーマ名
            element: 要素名

        Returns:
            カラーコード
        """
        themes = {
            "light": {
                "background": "#ffffff",
                "text": "#333333",
                "heading": "#222222",
                "panel": "#f9f9f9",
                "border": "#dddddd"
            },
            "dark": {
                "background": "#222222",
                "text": "#eeeeee",
                "heading": "#ffffff",
                "panel": "#333333",
                "border": "#555555"
            },
            "blue": {
                "background": "#f0f8ff",
                "text": "#333333",
                "heading": "#1e6eb7",
                "panel": "#e6f2ff",
                "border": "#b3d1ff"
            }
        }

        # テーマが存在しない場合はlightをデフォルトとして使用
        if theme not in themes:
            theme = "light"

        # 要素が存在しない場合はデフォルト値を返す
        return themes[theme].get(element, "#333333")

# サービス注入用の依存関数
def get_visualization_service(
    settings: Settings = Depends(get_settings),
    storage_service: StorageService = Depends(get_storage_service)
) -> VisualizationService:
    """可視化サービスの取得

    Args:
        settings: アプリケーション設定
        storage_service: ストレージサービス

    Returns:
        VisualizationService: 可視化サービスのインスタンス
    """
    return VisualizationService(settings=settings, storage_service=storage_service)