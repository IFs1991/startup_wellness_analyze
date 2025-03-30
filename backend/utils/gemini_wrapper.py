import os
import logging
import asyncio
from typing import Dict, Any, Optional, Union, List
from dotenv import load_dotenv

import google.generativeai as genai
# 不要なインポートを削除し、型ヒントでの使用に変更
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from google.generativeai import GenerationConfig

# .envファイルを読み込む
load_dotenv()

logger = logging.getLogger(__name__)

class GeminiWrapper:
    """
    Google Gemini APIを使用するためのラッパークラス。
    データ可視化、レポート生成、テキスト分析など様々な機能で利用される共通クラス。
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = 'gemini-1.5-pro'):
        """
        Gemini APIラッパーの初期化

        Args:
            api_key: Gemini APIキー。Noneの場合は環境変数から取得
            model_name: 使用するモデル名（デフォルト: 'gemini-1.5-pro'）
        """
        # APIキーの取得優先順位: 引数 > 環境変数GEMINI_API_KEY
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables or .env file")
            raise ValueError("GEMINI_API_KEY is required. Please set it in .env file or provide as an argument.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    async def generate_content_async(self,
                               prompt: Union[str, List[Dict[str, Any]]],
                               generation_config: Optional[Dict[str, Any]] = None,
                               safety_settings: Optional[Dict[str, Any]] = None) -> Any:
        """
        Gemini APIを使用してコンテンツを非同期で生成

        Args:
            prompt: テキストまたはマルチモーダルのプロンプト
            generation_config: 生成設定パラメータ
            safety_settings: セーフティ設定パラメータ

        Returns:
            生成されたレスポンス

        Raises:
            Exception: API呼び出しに失敗した場合
        """
        try:
            # 非同期処理に変換
            loop = asyncio.get_event_loop()
            config = None
            if generation_config:
                config = GenerationConfig(**generation_config)

            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=config,
                    safety_settings=safety_settings
                )
            )
            return response
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def generate_content(self,
                        prompt: Union[str, List[Dict[str, Any]]],
                        generation_config: Optional[Dict[str, Any]] = None,
                        safety_settings: Optional[Dict[str, Any]] = None) -> Any:
        """
        Gemini APIを使用してコンテンツを同期的に生成

        Args:
            prompt: テキストまたはマルチモーダルのプロンプト
            generation_config: 生成設定パラメータ
            safety_settings: セーフティ設定パラメータ

        Returns:
            生成されたレスポンス

        Raises:
            Exception: API呼び出しに失敗した場合
        """
        try:
            config = None
            if generation_config:
                config = GenerationConfig(**generation_config)

            response = self.model.generate_content(
                prompt,
                generation_config=config,
                safety_settings=safety_settings
            )
            return response
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    async def generate_visualization(self,
                                    data: Dict[str, Any],
                                    chart_type: str,
                                    title: str,
                                    description: Optional[str] = None,
                                    width: int = 800,
                                    height: int = 500,
                                    theme: str = "professional") -> bytes:
        """
        データ可視化画像を生成

        Args:
            data: 可視化するデータ
            chart_type: グラフのタイプ（bar, line, scatter, pie, heatmapなど）
            title: グラフのタイトル
            description: グラフの説明（オプション）
            width: 画像の幅
            height: 画像の高さ
            theme: テーマ（professional, dark, light, modernなど）

        Returns:
            生成された画像データ（バイト形式）

        Raises:
            Exception: 可視化の生成に失敗した場合
        """
        try:
            # データをJSON文字列に変換
            import json
            data_str = json.dumps(data, ensure_ascii=False)

            # プロンプトの構築
            prompt = f"""
            データを元に{chart_type}グラフを生成してください。

            タイトル: {title}

            説明: {description or 'なし'}

            データ: {data_str}

            以下の要件に従ってください:
            - グラフの種類: {chart_type}
            - サイズ: {width}x{height}ピクセル
            - テーマ: {theme}
            - プロフェッショナルな見た目で、見やすく整理されたグラフにすること
            - 日本語でラベルを表示すること
            - 適切な色使いとコントラストで見やすさを確保すること

            画像のみを返してください。説明は不要です。
            """

            response = await self.generate_content_async(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 2048,
                }
            )

            # 画像データを取得
            if hasattr(response, 'parts') and len(response.parts) > 0:
                for part in response.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        return part.inline_data.data

            raise ValueError("画像データがレスポンスに含まれていません")
        except Exception as e:
            logger.error(f"Visualization generation error: {e}")
            raise

    async def generate_report_html(self,
                                 template_id: str,
                                 company_data: Dict[str, Any],
                                 period: str,
                                 include_sections: List[str],
                                 customization: Optional[Dict[str, Any]] = None) -> str:
        """
        レポートのHTML形式を生成

        Args:
            template_id: レポートテンプレートID
            company_data: 企業データ
            period: レポート期間
            include_sections: 含めるセクションのリスト
            customization: カスタマイズパラメータ（オプション）

        Returns:
            生成されたHTMLコンテンツ

        Raises:
            Exception: レポート生成に失敗した場合
        """
        try:
            import json
            data_str = json.dumps(company_data, ensure_ascii=False)
            sections_str = ", ".join(include_sections)
            custom_str = json.dumps(customization or {}, ensure_ascii=False)

            prompt = f"""
            以下のデータに基づいて企業ウェルネス分析レポートのHTMLを生成してください。

            テンプレートID: {template_id}
            対象期間: {period}
            含めるセクション: {sections_str}
            カスタマイズ設定: {custom_str}

            企業データ: {data_str}

            以下の要件に従ってください:
            - レスポンスとして有効なHTML形式のみを返すこと
            - モダンでプロフェッショナルなデザインにすること
            - 指定されたセクションのみを含めること
            - データの可視化と説明を適切に組み合わせること
            - レスポンシブデザインにすること
            - CSSはインラインで含めること
            - 外部依存関係は使用しないこと

            HTMLコードのみを返してください。説明は不要です。
            """

            response = await self.generate_content_async(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 8192,
                }
            )

            # テキストを取得
            return response.text
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            raise

    async def analyze_text(self,
                         text: str,
                         analysis_type: str,
                         parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        テキスト分析を実行

        Args:
            text: 分析対象のテキスト
            analysis_type: 分析タイプ（'sentiment', 'summary', 'keywords', 'entities'など）
            parameters: 追加分析パラメータ（オプション）

        Returns:
            分析結果の辞書

        Raises:
            Exception: テキスト分析に失敗した場合
        """
        try:
            params_str = ""
            if parameters:
                import json
                params_str = json.dumps(parameters, ensure_ascii=False)

            prompt = f"""
            以下のテキストを分析してください。

            分析タイプ: {analysis_type}
            {f"パラメータ: {params_str}" if params_str else ""}

            テキスト:
            {text}

            結果はJSON形式で返してください。説明は不要です。
            """

            response = await self.generate_content_async(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 4096,
                }
            )

            # JSONレスポンスをパース
            import json
            try:
                result = json.loads(response.text)
                return result
            except json.JSONDecodeError:
                # JSONパースに失敗した場合、テキストをそのまま返す
                return {"raw_response": response.text}

        except Exception as e:
            logger.error(f"Text analysis error: {e}")
            raise