from typing import Dict, Any, Optional
import openai
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.system_prompt = """
あなたは企業のウェルネス分析アシスタントです。
与えられた企業データに基づいて、以下の観点から分析と提案を行ってください：

1. 従業員のパフォーマンスと健康状態の相関
2. 財務指標との関連性
3. 改善のための具体的な提案
4. リスク要因の特定と対策

企業データには以下の情報が含まれています：
- basic_info: 企業の基本情報
- employee_performance: 従業員のパフォーマンスデータ（直近10件）
- financial_data: 財務データ（直近4四半期）
- analysis_results: 過去の分析結果（直近5件）

回答は具体的で実用的なものにしてください。
"""

    async def get_response(
        self,
        message: str,
        company_data: Dict[str, Any],
        model: str = "gpt-4"
    ) -> str:
        """企業データを考慮してAIレスポンスを生成"""
        try:
            # 企業データを文字列に変換
            company_context = json.dumps(company_data, ensure_ascii=False, default=str)

            # メッセージを構築
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "system", "content": f"企業データ: {company_context}"},
                {"role": "user", "content": message}
            ]

            # OpenAI APIを呼び出し
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0
            )

            # レスポンスを取得
            ai_response = response.choices[0].message.content

            # レスポンスをログに記録
            logger.info(f"AI Response generated for company data at {datetime.now()}")

            return ai_response

        except Exception as e:
            logger.error(f"AIレスポンスの生成中にエラーが発生しました: {str(e)}")
            raise Exception(f"AIレスポンスの生成に失敗しました: {str(e)}")