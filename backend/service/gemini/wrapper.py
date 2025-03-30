# -*- coding: utf-8 -*-
"""
Gemini APIラッパー
Google Gemini AIモデルへのアクセスを提供し、レポート生成や分析機能を実装します。
"""

import logging
import os
import time
import json
import re
import asyncio
from typing import Dict, Any, List, Optional, Callable
import requests
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class GeminiWrapper:
    """
    Google Gemini API のラッパークラス
    AI生成コンテンツの取得やHTMLレポート生成機能を提供
    """

    # Gemini APIのエンドポイント
    API_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        """
        Gemini API ラッパーの初期化

        Args:
            api_key: Gemini APIキー。指定しない場合は環境変数から取得
            model: 使用するGeminiモデル名
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini APIキーが必要です。環境変数または初期化時に指定してください。")

        self.model = model
        logger.info(f"GeminiWrapper initialized with model: {model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RequestException)
    )
    async def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        """
        テキスト生成リクエストを送信

        Args:
            prompt: 生成プロンプト
            temperature: 生成の多様性（0.0-1.0）

        Returns:
            生成されたテキスト
        """
        try:
            url = f"{self.API_URL}/models/{self.model}:generateContent?key={self.api_key}"

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": 8192,
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    }
                ]
            }

            # 非同期実行のためにループ取得
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(url, json=payload)
            )

            response.raise_for_status()
            result = response.json()

            if "candidates" in result and result["candidates"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                logger.error(f"予期しないAPIレスポンス形式: {result}")
                raise ValueError("APIからのレスポンスが予期しない形式です")

        except RequestException as e:
            logger.error(f"API呼び出しエラー: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"テキスト生成中の予期しないエラー: {str(e)}")
            raise ValueError(f"テキスト生成中にエラーが発生しました: {str(e)}")

    async def generate_report_html(
        self,
        template_id: str,
        company_data: Dict[str, Any],
        period: str,
        include_sections: List[str],
        customization: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        レポートのHTML生成

        Args:
            template_id: テンプレートID
            company_data: 企業データ
            period: レポート期間
            include_sections: 含めるセクション
            customization: カスタマイズ設定

        Returns:
            生成されたHTMLコンテンツ
        """
        try:
            # データのシリアライズ
            data_json = json.dumps(company_data, ensure_ascii=False, indent=2)
            sections_json = json.dumps(include_sections, ensure_ascii=False)
            customization_json = "{}" if customization is None else json.dumps(
                customization, ensure_ascii=False, indent=2
            )

            # プロンプトの作成
            prompt = f"""
            # レポート生成タスク

            ## テンプレートID
            {template_id}

            ## 対象期間
            {period}

            ## 企業データ
            ```json
            {data_json}
            ```

            ## 含めるセクション
            ```json
            {sections_json}
            ```

            ## カスタマイズ設定
            ```json
            {customization_json}
            ```

            # 指示
            上記の情報を元に、プロフェッショナルなレポートをHTML形式で生成してください。
            以下の要件を満たすようにしてください：

            1. 完全なHTML文書（DOCTYPE宣言、head、bodyタグなど含む）を生成すること
            2. レスポンシブデザインのためのCSSを含めること
            3. 企業データに基づいた分析と洞察を盛り込むこと
            4. グラフや表の表現にはChart.jsを使用すること
            5. 「含めるセクション」に指定されたセクションのみを含めること
            6. 適切な見出し構造、フォントスタイル、余白を使用して読みやすさを確保すること
            7. プリント時に最適化されたスタイルも含めること

            出力はHTML形式のみとし、Markdownや説明は含めないでください。
            HTMLタグのみのレスポンスをお願いします。
            """

            # Geminiを使ってHTML生成
            html_content = await self.generate_text(prompt, temperature=0.2)

            # 前後の余分なマークダウンやコードブロックを取り除く
            html_content = re.sub(r'^```html\s*', '', html_content)
            html_content = re.sub(r'\s*```$', '', html_content)

            return html_content

        except Exception as e:
            logger.error(f"レポートHTML生成中のエラー: {str(e)}")
            raise ValueError(f"レポートの生成中にエラーが発生しました: {str(e)}")

    async def analyze_data(
        self,
        data: Dict[str, Any],
        analysis_type: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        データの分析を実行

        Args:
            data: 分析対象データ
            analysis_type: 分析タイプ（"sentiment", "trends", "comparison" など）
            context: 追加のコンテキスト情報

        Returns:
            分析結果
        """
        try:
            # データとコンテキストの準備
            data_json = json.dumps(data, ensure_ascii=False, indent=2)
            context_text = context or "追加のコンテキスト情報はありません。"

            # 分析タイプに応じたプロンプト生成
            prompt_templates = {
                "sentiment": f"""
                # 感情分析タスク

                以下のデータに対して感情分析を行い、結果をJSON形式で返してください。
                ポジティブ/ネガティブの度合い、主要な感情カテゴリ、および信頼度スコアを含めてください。

                ## データ
                ```json
                {data_json}
                ```

                ## コンテキスト
                {context_text}

                ## 出力形式
                {{
                  "sentiment_score": float,  // -1.0 (非常にネガティブ) から 1.0 (非常にポジティブ)
                  "sentiment_label": string,  // "positive", "negative", "neutral"
                  "confidence": float,  // 0.0 から 1.0
                  "primary_emotions": [string],  // 主要な感情（"joy", "anger" など）
                  "analysis_summary": string  // 分析の要約
                }}
                """,

                "trends": f"""
                # トレンド分析タスク

                以下の時系列データに対してトレンド分析を行い、結果をJSON形式で返してください。
                上昇/下降トレンド、季節性、異常値、および将来予測を含めてください。

                ## データ
                ```json
                {data_json}
                ```

                ## コンテキスト
                {context_text}

                ## 出力形式
                {{
                  "overall_trend": string,  // "rising", "falling", "stable"
                  "trend_strength": float,  // 0.0 から 1.0
                  "seasonality_detected": boolean,
                  "anomalies": [{{ "point": string, "value": number, "description": string }}],
                  "future_prediction": string,
                  "analysis_summary": string
                }}
                """,

                "comparison": f"""
                # 比較分析タスク

                以下のデータセットを比較分析し、結果をJSON形式で返してください。
                主要な差異、類似点、および総合評価を含めてください。

                ## データ
                ```json
                {data_json}
                ```

                ## コンテキスト
                {context_text}

                ## 出力形式
                {{
                  "key_differences": [{{ "category": string, "description": string }}],
                  "key_similarities": [{{ "category": string, "description": string }}],
                  "statistical_significance": boolean,
                  "overall_assessment": string,
                  "analysis_summary": string
                }}
                """
            }

            # 分析タイプの存在確認
            if analysis_type not in prompt_templates:
                raise ValueError(f"サポートされていない分析タイプです: {analysis_type}")

            # 分析の実行
            prompt = prompt_templates[analysis_type]
            response_text = await self.generate_text(prompt, temperature=0.1)

            # JSONレスポンスの抽出
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text

            # 余分な文字を取り除いてJSONとしてパース
            json_str = re.sub(r'^```json\s*', '', json_str)
            json_str = re.sub(r'\s*```$', '', json_str)

            result = json.loads(json_str)
            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSONパースエラー: {str(e)}")
            raise ValueError(f"AIからの応答をJSONとして解析できませんでした: {str(e)}")
        except Exception as e:
            logger.error(f"データ分析中のエラー: {str(e)}")
            raise ValueError(f"データ分析中にエラーが発生しました: {str(e)}")