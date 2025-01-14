# -*- coding: utf-8 -*-
"""
グラフ生成モジュール
Firestoreのデータを使用して様々な種類のグラフを生成します。
非同期処理とエラーハンドリングに対応し、
ログ機能とタイプヒントを備えています。
"""
import logging
from typing import Optional, Dict, Any, List, Tuple, Literal, Union
import asyncio
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
from fastapi import HTTPException
import numpy as np
from typing_extensions import TypeAlias

# 型定義
StyleType = Literal['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']
PlotType = Literal['bar', 'box']
ArrayLike: TypeAlias = Union[List[float], np.ndarray, pd.Series]

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class GraphGenerationError(Exception):
    """グラフ生成に関するエラー"""
    pass

class GraphGenerator:
    """
    Firestoreのデータを使用してグラフを生成するクラス
    """
    def __init__(self):
        """
        グラフ生成の初期設定を行います
        """
        # グラフのスタイル設定
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12

        # デフォルトのカラーパレット
        self.color_palette = sns.color_palette("husl", 8)
        sns.set_palette(self.color_palette)

    async def generate_time_series(
        self,
        data: List[Dict[str, Any]],
        time_column: str,
        value_column: str,
        title: str,
        **kwargs: Any
    ) -> Tuple[Figure, str]:
        """
        時系列データのグラフを生成します

        Args:
            data: Firestoreから取得したデータのリスト
            time_column: 時間軸として使用するカラム名
            value_column: 値として使用するカラム名
            title: グラフのタイトル
            **kwargs: その他のグラフオプション

        Returns:
            Tuple[Figure, str]: 生成されたグラフとファイル名
        """
        try:
            logger.info(f"Generating time series graph for {value_column}")

            # DataFrameの作成と前処理
            df = pd.DataFrame(data)
            df[time_column] = pd.to_datetime(df[time_column])
            df = df.sort_values(time_column)

            # グラフの生成
            fig = plt.figure()
            sns.lineplot(
                data=df,
                x=time_column,
                y=value_column,
                marker='o',
                **kwargs
            )

            # グラフの設定
            plt.title(title)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # ファイル名の生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"time_series_{timestamp}.png"

            logger.info(f"Successfully generated time series graph: {filename}")
            return fig, filename

        except Exception as e:
            error_msg = f"Error generating time series graph: {str(e)}"
            logger.error(error_msg)
            raise GraphGenerationError(error_msg)

    async def generate_distribution(
        self,
        data: List[Dict[str, Any]],
        value_column: str,
        title: str,
        bins: Union[ArrayLike, int] = 30,
        **kwargs: Any
    ) -> Tuple[Figure, str]:
        """
        分布を可視化するヒストグラムを生成します

        Args:
            data: Firestoreから取得したデータのリスト
            value_column: 分布を表示する値のカラム名
            title: グラフのタイトル
            bins: ヒストグラムの区間数または区間の配列
            **kwargs: その他のグラフオプション

        Returns:
            Tuple[Figure, str]: 生成されたグラフとファイル名
        """
        try:
            logger.info(f"Generating distribution graph for {value_column}")

            # DataFrameの作成
            df = pd.DataFrame(data)

            # グラフの生成
            fig = plt.figure()
            sns.histplot(
                data=df,
                x=value_column,
                bins=bins,
                kde=True,
                **kwargs
            )

            # グラフの設定
            plt.title(title)
            plt.tight_layout()

            # ファイル名の生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"distribution_{timestamp}.png"

            logger.info(f"Successfully generated distribution graph: {filename}")
            return fig, filename

        except Exception as e:
            error_msg = f"Error generating distribution graph: {str(e)}"
            logger.error(error_msg)
            raise GraphGenerationError(error_msg)

    async def generate_comparison(
        self,
        data: List[Dict[str, Any]],
        category_column: str,
        value_column: str,
        title: str,
        plot_type: PlotType = 'bar',
        **kwargs: Any
    ) -> Tuple[Figure, str]:
        """
        カテゴリ間の比較グラフを生成します

        Args:
            data: Firestoreから取得したデータのリスト
            category_column: カテゴリを表すカラム名
            value_column: 値を表すカラム名
            title: グラフのタイトル
            plot_type: グラフの種類 ('bar' または 'box')
            **kwargs: その他のグラフオプション

        Returns:
            Tuple[Figure, str]: 生成されたグラフとファイル名
        """
        try:
            logger.info(f"Generating comparison graph for {category_column} vs {value_column}")

            # DataFrameの作成
            df = pd.DataFrame(data)

            # グラフの生成
            fig = plt.figure()
            if plot_type == 'bar':
                sns.barplot(
                    data=df,
                    x=category_column,
                    y=value_column,
                    **kwargs
                )
            elif plot_type == 'box':
                sns.boxplot(
                    data=df,
                    x=category_column,
                    y=value_column,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")

            # グラフの設定
            plt.title(title)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # ファイル名の生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_{timestamp}.png"

            logger.info(f"Successfully generated comparison graph: {filename}")
            return fig, filename

        except Exception as e:
            error_msg = f"Error generating comparison graph: {str(e)}"
            logger.error(error_msg)
            raise GraphGenerationError(error_msg)

    def set_style(
        self,
        style: StyleType = 'whitegrid',
        palette: Optional[str] = None,
        figure_size: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        グラフのスタイルをカスタマイズします

        Args:
            style: seabornのスタイル名
            palette: カラーパレット名
            figure_size: グラフサイズ (width, height)
        """
        try:
            sns.set_style(style)
            if palette:
                self.color_palette = sns.color_palette(palette, 8)
                sns.set_palette(self.color_palette)
            plt.rcParams['figure.figsize'] = figure_size
            logger.info("Successfully updated graph style settings")

        except Exception as e:
            error_msg = f"Error setting graph style: {str(e)}"
            logger.error(error_msg)
            raise GraphGenerationError(error_msg)

    async def close(self) -> None:
        """
        リソースのクリーンアップを行います
        """
        try:
            plt.close('all')
            logger.info("Successfully cleaned up graph resources")

        except Exception as e:
            logger.error(f"Error closing graph resources: {str(e)}")
            raise