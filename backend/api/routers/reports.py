# -*- coding: utf-8 -*-
"""
レポート生成API
企業データに基づいたレポートを生成・管理するエンドポイントを提供します。
Gemini APIを活用してインテリジェントなレポート生成を行います。
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
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from tempfile import NamedTemporaryFile
import hashlib

from service.gemini.wrapper import GeminiWrapper
from core.config import get_settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

router = APIRouter(
    prefix="/api/reports",
    tags=["reports"],
    responses={404: {"description": "Not found"}},
)

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
            logger.error(f"PDF変換失敗: {stderr.decode()}")
            raise RuntimeError(f"PDF変換に失敗しました: {stderr.decode()}")

        logger.info(f"PDF生成成功: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"HTML→PDF変換エラー: {e}")
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
            logger.warning("GEMINI_API_KEYが環境変数または.envファイルに見つかりません")
            raise ValueError("GEMINI_API_KEYが必要です。.envファイルに設定するか引数として提供してください。")

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
            logger.error(f"レポート生成失敗: {e}")
            return {
                "success": False,
                "error": str(e)
            }

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
        レポートを非同期で生成し保存する（バックグラウンドタスク用）

        Args:
            report_id: レポートID
            template_id: テンプレートID
            company_data: 企業データ
            period: 期間
            include_sections: 含めるセクション
            customization: カスタマイズ設定
            format: フォーマット
            cache_key: キャッシュキー
        """
        try:
            result = await self.generate_report(
                template_id=template_id,
                company_data=company_data,
                period=period,
                include_sections=include_sections,
                customization=customization,
                format=format
            )

            if not result["success"]:
                logger.error(f"バックグラウンドでのレポート生成に失敗: {result.get('error')}")
                return

            # キャッシュキーが提供されている場合はキャッシュに保存
            if cache_key:
                cache_dir = Path("./storage/reports/cache")
                cache_dir.mkdir(parents=True, exist_ok=True)

                cache_path = cache_dir / f"{cache_key}.{format}"

                # ファイルをキャッシュディレクトリにコピー
                import shutil
                shutil.copy2(result["file_path"], str(cache_path))
                logger.info(f"キャッシュに保存しました: {cache_path}")

            logger.info(f"レポート生成完了 (ID: {report_id})")
        except Exception as e:
            logger.error(f"バックグラウンドレポート生成エラー: {e}")

# Gemini APIを使ったレポートジェネレーター取得用の依存関数
def get_report_generator():
    """設定からレポートジェネレーターインスタンスを取得"""
    try:
        # 設定からGemini APIキーを取得
        settings = get_settings()
        api_key = settings.gemini_api_key
        if not api_key:
            logger.warning("設定からGemini APIキーが見つかりません。環境変数を試します")

        # ReportGeneratorインスタンスを作成
        # ここでは明示的にAPIキーを渡し、クラス内で.envや環境変数からも読み込まれる
        return ReportGenerator(api_key=api_key)
    except Exception as e:
        logger.error(f"レポートジェネレーターの初期化に失敗: {e}")
        raise HTTPException(status_code=500, detail="レポート生成サービスの初期化に失敗しました")

@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    request: ReportRequest,
    background_tasks: BackgroundTasks,
    report_generator: ReportGenerator = Depends(get_report_generator)
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
        settings = get_settings()
        if settings.report_cache_enabled and cache_path.exists():
            # キャッシュの有効期限をチェック（デフォルト24時間）
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age < settings.report_cache_ttl:
                logger.info(f"キャッシュからレポートを返します: {cache_key}")
                return ReportResponse(
                    success=True,
                    report_id=report_id,
                    report_url=f"/api/reports/download/{cache_key}.{request.format}",
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
        logger.error(f"レポート生成リクエスト処理エラー: {e}")
        return ReportResponse(
            success=False,
            error=f"レポート生成リクエストの処理中にエラーが発生しました: {str(e)}"
        )

@router.get("/download/{filename}")
async def download_report(filename: str):
    """
    生成されたレポートをダウンロードする

    Args:
        filename: ダウンロードするファイル名
    """
    try:
        # キャッシュディレクトリをチェック
        cache_dir = Path("./storage/reports/cache")
        cache_path = cache_dir / filename

        # レポートディレクトリをチェック
        reports_dir = Path("./reports")
        report_path = reports_dir / filename

        # ファイルが存在する場所を特定
        if cache_path.exists():
            path = cache_path
        elif report_path.exists():
            path = report_path
        else:
            raise HTTPException(status_code=404, detail="レポートが見つかりません")

        # Content-Typeを決定
        if filename.endswith(".pdf"):
            media_type = "application/pdf"
        elif filename.endswith(".html"):
            media_type = "text/html"
        else:
            media_type = "application/octet-stream"

        return FileResponse(
            path=str(path),
            media_type=media_type,
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"レポートのダウンロード中にエラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=f"レポートのダウンロード中にエラーが発生しました: {str(e)}")

@router.get("/templates")
async def list_report_templates():
    """
    利用可能なレポートテンプレート一覧を取得する
    """
    try:
        # テンプレートデータの定義（実際の環境では外部から読み込む）
        templates = [
            {
                "id": "quarterly_financial",
                "name": "四半期財務レポート",
                "description": "四半期ごとの財務状況の詳細分析",
                "sections": ["財務概要", "収益分析", "コスト分析", "キャッシュフロー", "予測"]
            },
            {
                "id": "wellness_standard",
                "name": "スタンダードウェルネスレポート",
                "description": "スタートアップ企業の健康状態の標準レポート",
                "sections": ["組織健全性", "コミュニケーション分析", "ストレス指標", "モチベーション傾向", "改善提案"]
            },
            {
                "id": "vc_investment",
                "name": "VC投資分析レポート",
                "description": "投資判断のための包括的なスタートアップ分析",
                "sections": ["市場機会", "チーム評価", "技術評価", "財務見通し", "リスク分析", "投資提案"]
            }
        ]

        return templates

    except Exception as e:
        logger.error(f"テンプレート一覧の取得中にエラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail="テンプレート一覧の取得に失敗しました")