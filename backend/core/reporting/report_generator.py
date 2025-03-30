import os
import json
import time
import tempfile
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from log import logger

class ReportGenerator:
    """
    Gemini APIを使用して企業データと分析結果からレポートを生成するクラス
    オンデマンド方式で、リクエスト時のみレポートを生成
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        レポートジェネレーターの初期化

        Args:
            api_key: Gemini APIキー。Noneの場合は環境変数から取得
        """
        # APIキーの取得優先順位: 引数 > 環境変数GEMINI_API_KEY
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables or .env file")
            raise ValueError("GEMINI_API_KEY is required. Please set it in .env file or provide as an argument.")

        self.gemini_wrapper = GeminiWrapper(api_key=self.api_key)

        # レポート保存ディレクトリの設定
        self.reports_dir = Path("./storage/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # キャッシュディレクトリの設定
        self.cache_dir = Path("./storage/reports/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def generate_and_save_report(
        self,
        report_id: str,
        template_id: str,
        company_data: Dict[str, Any],
        period: str,
        include_sections: List[str],
        customization: Optional[Dict[str, Any]] = None,
        format: str = "pdf",
        cache_key: Optional[str] = None
    ):
        """
        レポートを生成して保存するメインメソッド

        Args:
            report_id: レポートの一意ID
            template_id: 使用するテンプレートID
            company_data: 企業データ
            period: レポート期間
            include_sections: 含めるセクションのリスト
            customization: カスタマイズ設定
            format: 出力フォーマット
            cache_key: キャッシュキー
        """
        try:
            # 企業データと分析結果を準備
            report_data = await self._prepare_report_data(company_data, period, include_sections)

            # HTMLコンテンツを生成
            html_content = await self.gemini_wrapper.generate_report_html(
                template_id=template_id,
                company_data=report_data["company_data"],
                analysis_results=report_data["analysis_results"],
                customization=customization or {}
            )

            # HTMLファイルとして保存
            report_path = self.reports_dir / f"{report_id}.html"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            # キャッシュに保存（もし有効なら）
            if cache_key:
                cache_path = self.cache_dir / f"{cache_key}.html"
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(html_content)

            # PDFへの変換（もし要求されていれば）
            if format == "pdf":
                pdf_path = self.reports_dir / f"{report_id}.pdf"
                await self._convert_html_to_pdf(html_content, str(pdf_path))

                # キャッシュに保存（もし有効なら）
                if cache_key:
                    cache_pdf_path = self.cache_dir / f"{cache_key}.pdf"
                    await self._convert_html_to_pdf(html_content, str(cache_pdf_path))

            # レポート生成完了を記録
            status_path = self.reports_dir / f"{report_id}.status"
            with open(status_path, "w") as f:
                f.write(json.dumps({
                    "status": "completed",
                    "timestamp": time.time(),
                    "format": format
                }))

            logger.info(f"Report generation completed: {report_id}")

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            # エラー状態を記録
            status_path = self.reports_dir / f"{report_id}.status"
            with open(status_path, "w") as f:
                f.write(json.dumps({
                    "status": "error",
                    "timestamp": time.time(),
                    "error": str(e)
                }))

    async def _prepare_report_data(
        self,
        company_data: Dict[str, Any],
        period: str,
        include_sections: List[str]
    ) -> Dict[str, Any]:
        """
        レポートに必要なデータを準備する

        Args:
            company_data: 企業の基本データ
            period: 分析期間
            include_sections: 含めるセクション

        Returns:
            レポート生成に必要なデータ
        """
        # 企業の分析結果を取得（ここではデータベースアクセスなどの処理を省略）
        # 実際の実装では、企業IDや期間に基づいてデータベースから最新の分析結果を取得

        analysis_results = {
            "wellness_score": 85,  # 例: ウェルネススコア
            "financial_metrics": {
                "revenue_growth": 12.5,
                "profit_margin": 8.3,
                "employee_productivity": 92.1
            },
            "wellness_metrics": {
                "work_life_balance": 78,
                "employee_engagement": 82,
                "burnout_risk": 34
            },
            "correlation_analysis": {
                "wellness_vs_performance": 0.72,
                "engagement_vs_productivity": 0.68,
                "balance_vs_retention": 0.81
            },
            "recommendations": [
                {
                    "area": "work_life_balance",
                    "suggestion": "リモートワークオプションの拡充を検討",
                    "expected_impact": "従業員満足度15%向上の可能性"
                },
                {
                    "area": "burnout_prevention",
                    "suggestion": "マネージャー向けのバーンアウト検知トレーニングを実施",
                    "expected_impact": "離職率5%低減の可能性"
                }
            ]
        }

        # 指定されたセクションのみを含める
        filtered_results = {}
        for section in include_sections:
            if section in analysis_results:
                filtered_results[section] = analysis_results[section]

        return {
            "company_data": company_data,
            "analysis_results": filtered_results,
            "period": period
        }

    async def _convert_html_to_pdf(self, html_content: str, output_path: str):
        """
        HTMLコンテンツをPDFに変換

        Args:
            html_content: 変換するHTMLコンテンツ
            output_path: 出力PDFパス
        """
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp:
            temp.write(html_content.encode('utf-8'))
            temp_path = temp.name

        try:
            # Puppeteerを使用してHTMLをPDFに変換
            cmd = [
                'node',
                'backend/scripts/html_to_pdf.js',
                temp_path,
                output_path
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"PDF conversion failed: {stderr.decode()}")
                raise RuntimeError(f"PDF conversion failed: {stderr.decode()}")

            logger.info(f"PDF generated successfully: {output_path}")

        finally:
            # 一時ファイルを削除
            if os.path.exists(temp_path):
                os.unlink(temp_path)