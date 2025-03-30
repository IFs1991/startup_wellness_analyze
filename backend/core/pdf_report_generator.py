# -*- coding: utf-8 -*-
"""
PDF レポート自動生成サービス
ReportLabを使用してPDFレポートを生成し、FirestoreおよびCloud Storageと連携します。
"""
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
from typing import Dict, Any, Optional, Union
from datetime import datetime
import logging
from pathlib import Path
import asyncio
import firebase_admin
from firebase_admin import firestore
from google.cloud import storage
from pydantic import BaseModel
import jinja2

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class PDFGenerationError(Exception):
    """PDFレポート生成に関するエラー"""
    pass

class ReportMetadata(BaseModel):
    """レポートのメタデータモデル"""
    report_type: str
    user_id: str
    created_at: datetime
    storage_path: str
    filename: str
    metadata: Optional[Dict[str, Any]] = None

# シンプルなPDFレポートジェネレーター（main.pyから使用されるクラス）
class PDFReportGenerator:
    """
    シンプルなPDFレポート生成クラス
    """
    def __init__(self, output_dir: Optional[str] = "data/reports"):
        """
        初期化

        Args:
            output_dir: レポートの出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PDFReportGenerator initialized with output directory: {output_dir}")

    async def generate(self, data: Dict[str, Any], report_id: str) -> str:
        """
        レポートデータからPDFを生成し、ファイルシステムに保存

        Args:
            data: レポートデータ
            report_id: レポートID

        Returns:
            保存したPDFファイルのパス
        """
        try:
            # PDFファイルパスを作成
            pdf_filename = f"{report_id}.pdf"
            pdf_path = self.output_dir / pdf_filename

            # PDFを生成
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)

            # スタイル設定
            styles = getSampleStyleSheet()
            title_style = styles['Heading1']
            normal_style = styles['Normal']

            # ドキュメント要素
            elements = []

            # タイトル
            title = data.get('title', 'スタートアップ分析レポート')
            elements.append(Paragraph(title, title_style))
            elements.append(Spacer(1, 12))

            # サブタイトル
            subtitle = data.get('subtitle', f'レポート ID: {report_id}')
            elements.append(Paragraph(subtitle, styles['Heading2']))
            elements.append(Spacer(1, 12))

            # レポート内容
            for section_name, section_data in data.items():
                if section_name in ['title', 'subtitle']:
                    continue

                # セクションヘッダー
                elements.append(Paragraph(section_name, styles['Heading2']))
                elements.append(Spacer(1, 6))

                # セクションの内容を処理
                if isinstance(section_data, dict):
                    # 辞書の場合はテーブルとして表示
                    table_data = [['項目', '値']]
                    for key, value in section_data.items():
                        table_data.append([key, str(value)])

                    table = Table(table_data, colWidths=[200, 300])
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    elements.append(table)

                elif isinstance(section_data, list):
                    # リストの場合は箇条書きとして表示
                    for item in section_data:
                        if isinstance(item, dict):
                            # 辞書のリストの場合はテーブルとして表示
                            if len(section_data) > 0:
                                keys = list(item.keys())
                                table_data = [keys]  # ヘッダー行
                                for row_dict in section_data:
                                    table_data.append([str(row_dict.get(k, '')) for k in keys])

                                table = Table(table_data)
                                table.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                                ]))
                                elements.append(table)
                                break  # 一度だけテーブルを作成
                        else:
                            # 通常の値のリストの場合
                            bullet_text = f"• {item}"
                            elements.append(Paragraph(bullet_text, normal_style))
                else:
                    # 単純な値の場合はテキストとして表示
                    elements.append(Paragraph(str(section_data), normal_style))

                elements.append(Spacer(1, 12))

            # フッター
            elements.append(Spacer(1, 20))
            elements.append(Paragraph(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))

            # PDFビルド
            doc.build(elements)

            # ファイルに保存
            with open(pdf_path, 'wb') as f:
                f.write(buffer.getvalue())

            logger.info(f"PDF report successfully generated and saved to {pdf_path}")
            return str(pdf_path)

        except Exception as e:
            error_msg = f"Error generating PDF report: {str(e)}"
            logger.error(error_msg)
            raise PDFGenerationError(error_msg) from e

class PDFReportService:
    """
    分析結果をPDFレポートとして生成し、FirestoreとCloud Storageに保存するサービス
    """
    def __init__(self, db: Any, storage_client: storage.Client, bucket_name: str, templates_dir: Optional[str] = None):
        """
        サービスを初期化します

        Args:
            db: Firestoreクライアント
            storage_client: Cloud Storageクライアント
            bucket_name: Cloud Storageバケット名
            templates_dir: テンプレートディレクトリのパス（オプション）
        """
        self.db = db
        self.storage_client = storage_client
        self.bucket_name = bucket_name

        # テンプレートエンジンのセットアップ
        if templates_dir:
            self.template_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(templates_dir),
                autoescape=jinja2.select_autoescape(['html', 'xml'])
            )
        else:
            # デフォルトのテンプレートディレクトリ
            default_templates_dir = Path(__file__).parent.parent / 'templates'
            default_templates_dir.mkdir(exist_ok=True)
            self.template_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(default_templates_dir),
                autoescape=jinja2.select_autoescape(['html', 'xml'])
            )

        logger.info("PDF Report Service initialized")

    async def generate_and_save_report(
        self,
        data: Dict[str, Any],
        user_id: str,
        report_type: str = "wellness",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ReportMetadata:
        """
        データからPDFレポートを生成し、FirestoreとCloud Storageに保存します

        Args:
            data: レポートに含めるデータ
            user_id: ユーザーID
            report_type: レポートの種類
            metadata: レポートのメタデータ

        Returns:
            保存されたレポートのメタデータ
        """
        try:
            logger.info(f"Generating {report_type} report for user {user_id}")

            # PDFを生成
            pdf_content = await self._generate_report(data)

            # ファイル名とパスを生成
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{report_type}_{user_id}_{timestamp}.pdf"
            storage_path = f"reports/{user_id}/{filename}"

            # Cloud Storageに保存
            await self._save_to_storage(pdf_content.getvalue(), storage_path)

            # メタデータを作成
            report_metadata = ReportMetadata(
                report_type=report_type,
                user_id=user_id,
                created_at=datetime.now(),
                storage_path=storage_path,
                filename=filename,
                metadata=metadata
            )

            # Firestoreにメタデータを保存
            await self._save_metadata(report_metadata)

            logger.info(f"Successfully generated and saved report {filename}")
            return report_metadata

        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            logger.error(error_msg)
            raise PDFGenerationError(error_msg) from e

    async def generate_template_report(
        self,
        data: Dict[str, Any],
        template_name: str,
        user_id: str,
        report_type: str = "template_report",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ReportMetadata:
        """
        テンプレートベースのPDFレポートを生成し、FirestoreとCloud Storageに保存します
        HTML/CSSの代わりにReportLabのPlatypusを使用します

        Args:
            data: レポートに含めるデータ
            template_name: 使用するテンプレート名
            user_id: ユーザーID
            report_type: レポートの種類
            metadata: レポートのメタデータ

        Returns:
            保存されたレポートのメタデータ
        """
        try:
            logger.info(f"Generating template-based {report_type} report for user {user_id}")

            # テンプレートの内容をレンダリング（テキスト形式）
            template = self.template_env.get_template(template_name)
            template_content = template.render(**data)

            # PDFバッファを作成
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)

            # スタイルの設定
            styles = getSampleStyleSheet()
            title_style = styles['Heading1']
            normal_style = styles['Normal']

            # レポート要素のリスト
            elements = []

            # タイトルの追加
            title = data.get('title', 'レポート')
            elements.append(Paragraph(title, title_style))
            elements.append(Spacer(1, 12))

            # コンテンツの追加
            for section_title, section_content in data.get('sections', {}).items():
                elements.append(Paragraph(section_title, styles['Heading2']))
                elements.append(Spacer(1, 6))

                if isinstance(section_content, str):
                    elements.append(Paragraph(section_content, normal_style))
                elif isinstance(section_content, dict):
                    for key, value in section_content.items():
                        text = f"{key}: {value}"
                        elements.append(Paragraph(text, normal_style))
                elif isinstance(section_content, list):
                    data_rows = []
                    # ヘッダーがある場合はそれを使用
                    if section_content and isinstance(section_content[0], dict):
                        headers = list(section_content[0].keys())
                        data_rows.append(headers)
                        for item in section_content:
                            data_rows.append([str(item.get(h, '')) for h in headers])
                    else:
                        for item in section_content:
                            data_rows.append([str(item)])

                    if data_rows:
                        t = Table(data_rows)
                        t.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        elements.append(t)

                elements.append(Spacer(1, 12))

            # フッターの追加
            elements.append(Spacer(1, 20))
            footer_text = f"生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}"
            elements.append(Paragraph(footer_text, normal_style))

            # PDFの生成
            doc.build(elements)
            buffer.seek(0)

            # ファイル名とパスを生成
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{report_type}_{user_id}_{timestamp}.pdf"
            storage_path = f"reports/{user_id}/{filename}"

            # Cloud Storageに保存
            await self._save_to_storage(buffer.getvalue(), storage_path)

            # メタデータを作成
            report_metadata = ReportMetadata(
                report_type=report_type,
                user_id=user_id,
                created_at=datetime.now(),
                storage_path=storage_path,
                filename=filename,
                metadata={
                    **(metadata or {}),
                    "template_used": template_name
                }
            )

            # Firestoreにメタデータを保存
            await self._save_metadata(report_metadata)

            logger.info(f"Successfully generated and saved template-based report {filename}")
            return report_metadata

        except Exception as e:
            error_msg = f"Error generating template-based report: {str(e)}"
            logger.error(error_msg)
            raise PDFGenerationError(error_msg) from e

    async def _generate_report(self, data: Dict[str, Any]) -> io.BytesIO:
        """
        データからPDFレポートを生成します（ReportLabを使用）

        Args:
            data: レポートに含めるデータ

        Returns:
            生成されたPDFのバイトコンテンツ
        """
        try:
            # PDFバッファを作成
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer)

            # タイトルページ
            c.setFont("Helvetica-Bold", 24)
            c.drawString(100, 750, "Startup Wellness Analysis")

            c.setFont("Helvetica", 12)
            c.drawString(100, 700, f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # データを追加
            y_position = 650
            for key, value in data.items():
                if y_position < 100:  # ページの下端に達したら新しいページを開始
                    c.showPage()
                    y_position = 750

                c.setFont("Helvetica-Bold", 14)
                c.drawString(100, y_position, f"{key}:")
                y_position -= 20

                c.setFont("Helvetica", 12)
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        c.drawString(120, y_position, f"{sub_key}: {sub_value}")
                        y_position -= 15
                else:
                    c.drawString(120, y_position, f"{value}")

                y_position -= 30

            # PDFを保存
            c.showPage()
            c.save()
            buffer.seek(0)

            return buffer

        except Exception as e:
            error_msg = f"Error generating PDF report: {str(e)}"
            logger.error(error_msg)
            raise PDFGenerationError(error_msg) from e

    async def _save_to_storage(self, content: Union[bytes, io.BytesIO], storage_path: str) -> None:
        """
        コンテンツをCloud Storageに保存します

        Args:
            content: 保存するコンテンツ
            storage_path: 保存先のパス
        """
        try:
            # BytesIOからバイト列に変換（必要な場合）
            if isinstance(content, io.BytesIO):
                bytes_content = content.getvalue()
            else:
                bytes_content = content

            # バケットとBlobを取得
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(storage_path)

            # コンテンツをアップロード
            blob.upload_from_string(
                bytes_content,
                content_type='application/pdf'
            )

            logger.info(f"Successfully saved PDF to {storage_path}")

        except Exception as e:
            error_msg = f"Error saving to Cloud Storage: {str(e)}"
            logger.error(error_msg)
            raise PDFGenerationError(error_msg) from e

    async def _save_metadata(self, metadata: ReportMetadata) -> None:
        """
        レポートのメタデータをFirestoreに保存します

        Args:
            metadata: レポートのメタデータ
        """
        try:
            # コレクションとドキュメントへの参照を取得
            reports_ref = self.db.collection('reports')

            # メタデータを辞書に変換
            metadata_dict = metadata.dict()

            # Firestoreに保存
            reports_ref.add(metadata_dict)

            logger.info(f"Successfully saved report metadata to Firestore")

        except Exception as e:
            error_msg = f"Error saving metadata to Firestore: {str(e)}"
            logger.error(error_msg)
            raise PDFGenerationError(error_msg) from e

def get_pdf_report_service(
    db: Any,
    storage_client: storage.Client,
    bucket_name: str,
    templates_dir: Optional[str] = None
) -> PDFReportService:
    """
    PDFレポートサービスのインスタンスを取得します

    Args:
        db: Firestoreクライアント
        storage_client: Cloud Storageクライアント
        bucket_name: Cloud Storageバケット名
        templates_dir: テンプレートディレクトリのパス（オプション）

    Returns:
        PDFReportServiceのインスタンス
    """
    return PDFReportService(db, storage_client, bucket_name, templates_dir)