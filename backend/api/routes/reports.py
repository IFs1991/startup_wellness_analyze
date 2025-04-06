import logging
import os
import uuid
import json
import tempfile
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
import hashlib

from backend.utils.gemini_wrapper import GeminiWrapper
from backend.core.config import Settings, get_settings

# backend/.envファイルを読み込む
# プロジェクトルートからのパスを構築
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
ENV_PATH = os.path.join(ROOT_DIR, 'backend', '.env')
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    # フォールバック: 現在のディレクトリ相対パスで試す
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.abspath(os.path.join(current_dir, '../..'))
    ENV_PATH = os.path.join(backend_dir, '.env')
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/reports", tags=["reports"])

# レポート生成リクエスト用モデル
class ReportRequest(BaseModel):
    template_id: str = Field(..., description="レポートテンプレートID")
    company_data: Dict[str, Any] = Field(..., description="企業データ")
    period: str = Field(..., description="レポート期間")
    include_sections: List[str] = Field(..., description="含めるセクション")
    customization: Optional[Dict[str, Any]] = Field(None, description="カスタマイズ設定")
    format: str = Field("pdf", description="出力フォーマット ('pdf', 'html')")

# レポート生成レスポンスモデル
class ReportResponse(BaseModel):
    success: bool = Field(..., description="成功したかどうか")
    report_url: Optional[str] = Field(None, description="生成されたレポートURL")
    report_id: Optional[str] = Field(None, description="レポートID")
    message: Optional[str] = Field(None, description="メッセージ")
    error: Optional[str] = Field(None, description="エラーメッセージ（失敗時）")

# HTMLからPDFに変換する関数
async def html_to_pdf(html_content: str, output_path: str):
    """
    HTMLコンテンツをPDFに変換する
    Puppeteerを使用して変換するため、Node.jsとPuppeteerが必要

    Args:
        html_content: PDF化するHTMLコンテンツ
        output_path: 出力PDFのパス
    """
    try:
        # 一時HTMLファイルに書き出し
        with NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as temp_html:
            temp_html_path = temp_html.name
            temp_html.write(html_content)

        # puppeteerスクリプトで実行するPDFコンバータスクリプトのパス
        converter_script = Path(__file__).parent.parent.parent / "utils" / "pdf_converter.js"

        # puppeteerを使ってPDF変換
        import asyncio
        process = await asyncio.create_subprocess_exec(
            "node",
            str(converter_script),
            temp_html_path,
            output_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        # 一時HTMLファイルを削除
        os.unlink(temp_html_path)

        if process.returncode != 0:
            logger.error(f"PDF conversion failed: {stderr.decode()}")
            raise RuntimeError(f"PDF変換に失敗しました: {stderr.decode()}")

        logger.info(f"PDF generated successfully: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"HTML to PDF conversion error: {e}")
        raise

# レポート生成クラス
class ReportGenerator:
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
        self.reports_dir = Path("./reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    async def generate_report(self,
                            template_id: str,
                            company_data: Dict[str, Any],
                            period: str,
                            include_sections: List[str],
                            customization: Optional[Dict[str, Any]] = None,
                            format: str = "pdf") -> Dict[str, Any]:
        """
        レポートを生成する

        Args:
            template_id: レポートテンプレートID
            company_data: 企業データ
            period: レポート期間
            include_sections: 含めるセクション
            customization: カスタマイズ設定
            format: 出力フォーマット

        Returns:
            生成結果情報
        """
        try:
            # HTMLコンテンツを生成
            html_content = await self.gemini_wrapper.generate_report_html(
                template_id=template_id,
                company_data=company_data,
                period=period,
                include_sections=include_sections,
                customization=customization
            )

            # レポートID生成（企業名と期間から）
            import hashlib
            company_name = company_data.get("company_name", "company")
            sanitized_name = "".join(c if c.isalnum() else "_" for c in company_name)
            report_id = f"{sanitized_name}_{period}_{hashlib.md5(html_content[:100].encode()).hexdigest()[:8]}"

            if format.lower() == "html":
                # HTMLとして保存
                html_path = self.reports_dir / f"{report_id}.html"
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)

                return {
                    "success": True,
                    "format": "html",
                    "report_id": report_id,
                    "file_path": str(html_path)
                }
            else:
                # PDFに変換して保存
                pdf_path = self.reports_dir / f"{report_id}.pdf"
                await html_to_pdf(html_content, str(pdf_path))

                return {
                    "success": True,
                    "format": "pdf",
                    "report_id": report_id,
                    "file_path": str(pdf_path)
                }
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Gemini APIを使ったレポートジェネレーター取得用の依存関数
def get_report_generator(settings: Settings = Depends(get_settings)):
    """設定からレポートジェネレーターインスタンスを取得"""
    try:
        # 設定からGemini APIキーを取得
        api_key = settings.gemini_api_key
        if not api_key:
            logger.warning("Gemini API key not found in settings, trying environment variables")

        # ReportGeneratorインスタンスを作成
        # ここでは明示的にAPIキーを渡し、クラス内で.envや環境変数からも読み込まれる
        return ReportGenerator(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize report generator: {e}")
        raise HTTPException(status_code=500, detail="レポート生成サービスの初期化に失敗しました")

@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    request: ReportRequest,
    background_tasks: BackgroundTasks,
    report_generator: ReportGenerator = Depends(get_report_generator),
    settings: Settings = Depends(get_settings)
):
    """
    企業データと分析結果に基づいてレポートを生成する
    オンデマンド方式: リクエストがあった場合のみ生成
    """
    try:
        # リクエストから企業データと分析結果を取得
        company_data = request.company_data

        # レポートIDの生成（一意の識別子）
        report_id = f"report_{uuid.uuid4().hex}"

        # レポート生成とキャッシュの確認
        cache_key = hashlib.md5(
            f"{request.template_id}_{json.dumps(company_data, sort_keys=True)}_{request.period}".encode()
        ).hexdigest()

        # キャッシュディレクトリのチェック
        cache_dir = Path("./storage/reports/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.{request.format}"

        # キャッシュが有効かつファイルが存在する場合はキャッシュを返す
        if settings.report_cache_enabled and cache_path.exists():
            # キャッシュの有効期限をチェック（デフォルト24時間）
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age < settings.report_cache_ttl:
                logger.info(f"Returning cached report: {cache_key}")
                return ReportResponse(
                    success=True,
                    report_id=report_id,
                    report_url=f"/api/v1/reports/download/{cache_key}.{request.format}",
                    message="レポートがキャッシュから取得されました"
                )

        # レポート生成をバックグラウンドタスクとして実行
        background_tasks.add_task(
            report_generator.generate_and_save_report,
            report_id=report_id,
            template_id=request.template_id,
            company_data=company_data,
            period=request.period,
            include_sections=request.include_sections,
            customization=request.customization,
            format=request.format,
            cache_key=cache_key
        )

        return ReportResponse(
            success=True,
            report_id=report_id,
            message="レポート生成を開始しました。完了したらダウンロードできます。"
        )
    except Exception as e:
        logger.error(f"Failed to start report generation: {e}")
        return ReportResponse(
            success=False,
            error=f"レポート生成の開始に失敗しました: {str(e)}"
        )

@router.post("/generate-async", response_model=ReportResponse)
async def generate_report_async(
    request: ReportRequest,
    background_tasks: BackgroundTasks,
    report_generator: ReportGenerator = Depends(get_report_generator)
):
    """
    企業データからレポートを非同期で生成するエンドポイント
    """
    try:
        # 非同期処理用の情報を事前に生成
        import hashlib
        company_name = request.company_data.get("company_name", "company")
        sanitized_name = "".join(c if c.isalnum() else "_" for c in company_name)
        temp_hash = hashlib.md5(f"{company_name}_{request.period}_{request.template_id}".encode()).hexdigest()[:8]
        report_id = f"{sanitized_name}_{request.period}_{temp_hash}"

        # バックグラウンドタスクとしてレポート生成を追加
        background_tasks.add_task(
            report_generator.generate_report,
            template_id=request.template_id,
            company_data=request.company_data,
            period=request.period,
            include_sections=request.include_sections,
            customization=request.customization,
            format=request.format
        )

        # 予想されるレスポンスURLを構築
        format = request.format.lower()
        report_url = f"/api/v1/reports/download/{report_id}.{format}"

        return {
            "success": True,
            "report_url": report_url,
            "report_id": report_id,
            "message": "レポート生成を開始しました。数分後に指定されたURLでダウンロードできるようになります。"
        }
    except Exception as e:
        logger.error(f"Async report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"非同期レポート生成の開始中にエラーが発生しました: {str(e)}")

@router.get("/download/{filename}")
async def download_report(
    filename: str,
    report_generator: ReportGenerator = Depends(get_report_generator)
):
    """
    生成済みレポートをダウンロードするエンドポイント
    """
    file_path = report_generator.reports_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="レポートが見つかりません。まだ生成中か、存在しない可能性があります。")

    # ファイル拡張子に基づいてContent-Typeを設定
    content_type_map = {
        ".pdf": "application/pdf",
        ".html": "text/html"
    }
    extension = file_path.suffix.lower()
    media_type = content_type_map.get(extension, "application/octet-stream")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )

@router.get("/templates", response_model=List[Dict[str, Any]])
async def list_report_templates():
    """
    利用可能なレポートテンプレート一覧を取得するエンドポイント
    """
    # テンプレート情報のハードコーディング（実際にはデータベースから取得する）
    templates = [
        {
            "id": "quarterly_wellness",
            "name": "四半期ウェルネスレポート",
            "description": "企業の四半期ごとのウェルネス指標をまとめたレポート",
            "available_sections": [
                "executive_summary",
                "wellness_metrics",
                "department_comparison",
                "trend_analysis",
                "recommendations"
            ],
            "thumbnail_url": "/static/templates/quarterly_wellness.png"
        },
        {
            "id": "annual_wellness",
            "name": "年間ウェルネスレポート",
            "description": "年間を通した企業ウェルネス指標の詳細分析",
            "available_sections": [
                "executive_summary",
                "annual_metrics",
                "quarterly_comparison",
                "department_analysis",
                "financial_correlation",
                "benchmark_comparison",
                "recommendations",
                "future_outlook"
            ],
            "thumbnail_url": "/static/templates/annual_wellness.png"
        },
        {
            "id": "wellness_financial",
            "name": "ウェルネスと財務パフォーマンス分析",
            "description": "ウェルネス指標と財務業績の相関関係を分析するレポート",
            "available_sections": [
                "executive_summary",
                "wellness_overview",
                "financial_overview",
                "correlation_analysis",
                "department_breakdown",
                "roi_analysis",
                "recommendations"
            ],
            "thumbnail_url": "/static/templates/wellness_financial.png"
        }
    ]

    return templates