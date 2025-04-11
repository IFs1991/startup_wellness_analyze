"""
このモジュールは非推奨となりました。
代わりに`backend.api.routers.reports`を使用してください。
このファイルは後方互換性のために残されています。
"""

import logging
import os
import uuid
import json
import tempfile
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Response, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
import hashlib
from fastapi.routing import APIRoute
import warnings

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
warnings.warn(
    "routes.reports モジュールは非推奨です。代わりに routers.reports を使用してください。",
    DeprecationWarning,
    stacklevel=2
)

# 元のルーター
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

# リダイレクト処理を行うクラス
class RedirectRoute(APIRoute):
    def __init__(self, *args, **kwargs):
        self.redirect_path_prefix = kwargs.pop("redirect_path_prefix", "/api/reports")
        super().__init__(*args, **kwargs)

    async def handle(self, request: Request) -> RedirectResponse:
        # 元のパスからプレフィックスを除去し、新しいパスを作成
        path = request.url.path
        new_path = path.replace("/api/v1/reports", self.redirect_path_prefix)

        # クエリパラメータを維持
        if request.url.query:
            new_path = f"{new_path}?{request.url.query}"

        logger.info(f"レポートリクエストをリダイレクト: {path} -> {new_path}")
        return RedirectResponse(url=new_path, status_code=307)

# 元のエンドポイントパターンに対応するリダイレクトルートを作成
@router.api_route("/generate", methods=["POST"], response_class=RedirectResponse, route_class=RedirectRoute)
async def redirect_generate_report(request: Request):
    return {}

@router.api_route("/generate-async", methods=["POST"], response_class=RedirectResponse, route_class=RedirectRoute)
async def redirect_generate_report_async(request: Request):
    return {}

@router.api_route("/download/{filename}", methods=["GET"], response_class=RedirectResponse, route_class=RedirectRoute)
async def redirect_download_report(request: Request, filename: str):
    return {}

@router.api_route("/templates", methods=["GET"], response_class=RedirectResponse, route_class=RedirectRoute)
async def redirect_list_report_templates(request: Request):
    return {}

# すべてのHTTPメソッドに対応する汎用的なキャッチオールルート
@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"], response_class=RedirectResponse, route_class=RedirectRoute)
async def redirect_all(request: Request, path: str):
    return {}