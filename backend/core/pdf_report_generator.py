# -*- coding: utf-8 -*-
"""
PDF レポート自動生成サービス
ReportLabを使用してPDFレポートを生成し、FirestoreおよびCloud Storageと連携します。
"""
from reportlab.pdfgen import canvas
import io
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path
import asyncio
import firebase_admin
from firebase_admin import firestore
from google.cloud import storage
from pydantic import BaseModel

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

class PDFReportService:
    """
    分析結果をPDFレポートとして生成し、FirestoreとCloud Storageに保存するサービス
    """
    def __init__(self, db: Any, storage_client: storage.Client, bucket_name: str):
        """
        サービスを初期化します

        Args:
            db: Firestoreクライアント
            storage_client: Cloud Storageクライアント
            bucket_name: Cloud Storageバケット名
        """
        self.db = db
        self.storage_client = storage_client
        self.bucket_name = bucket_name
        logger.info("PDF Report Service initialized")

    async def generate_and_save_report(
        self,
        data: Dict[str, Any],
        user_id: str,
        report_type: str = "wellness",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ReportMetadata:
        """
        PDFレポートを生成し、Cloud StorageとFirestoreに保存します

        Args:
            data: レポートに含めるデータ
            user_id: レポート生成を要求したユーザーのID
            report_type: レポートの種類
            metadata: 追加のメタデータ

        Returns:
            ReportMetadata: 保存されたレポートのメタデータ
        """
        try:
            # PDFの生成
            pdf_buffer = await self._generate_report(data)

            # ファイル名の生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_type}_{timestamp}.pdf"
            storage_path = f"reports/{user_id}/{filename}"

            # Cloud Storageへの保存
            await self._save_to_storage(pdf_buffer.getvalue(), storage_path)

            # メタデータの作成
            report_metadata = ReportMetadata(
                report_type=report_type,
                user_id=user_id,
                created_at=datetime.now(),
                storage_path=storage_path,
                filename=filename,
                metadata=metadata
            )

            # Firestoreへのメタデータ保存
            await self._save_metadata(report_metadata)

            logger.info(f"Successfully generated and saved report: {filename}")
            return report_metadata

        except Exception as e:
            error_msg = f"Error generating and saving report: {str(e)}"
            logger.error(error_msg)
            raise PDFGenerationError(error_msg) from e

    async def _generate_report(self, data: Dict[str, Any]) -> io.BytesIO:
        """
        PDFレポートを生成します

        Args:
            data: レポートに含めるデータ

        Returns:
            io.BytesIO: 生成されたPDFのバイトストリーム
        """
        try:
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer)

            # レポートのヘッダー
            c.drawString(100, 750, "Startup Wellness プログラム レポート")
            c.drawString(100, 730, f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # レポートの内容
            y_position = 700
            for key, value in data.items():
                c.drawString(100, y_position, f"{key}: {str(value)}")
                y_position -= 20

            c.save()
            buffer.seek(0)
            return buffer

        except Exception as e:
            error_msg = f"Error generating PDF report: {str(e)}"
            logger.error(error_msg)
            raise PDFGenerationError(error_msg) from e

    async def _save_to_storage(self, content: bytes, storage_path: str) -> None:
        """
        PDFをCloud Storageに保存します

        Args:
            content: PDFのバイトデータ
            storage_path: 保存先のパス
        """
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(storage_path)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: blob.upload_from_string(
                    content,
                    content_type='application/pdf'
                )
            )

            logger.info(f"Successfully uploaded PDF to {storage_path}")

        except Exception as e:
            error_msg = f"Error uploading PDF to Cloud Storage: {str(e)}"
            logger.error(error_msg)
            raise PDFGenerationError(error_msg) from e

    async def _save_metadata(self, metadata: ReportMetadata) -> None:
        """
        レポートのメタデータをFirestoreに保存します

        Args:
            metadata: 保存するメタデータ
        """
        try:
            collection_ref = self.db.collection('pdf_reports')
            doc_ref = collection_ref.document()

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: doc_ref.set(metadata.dict())
            )

            logger.info(f"Successfully saved report metadata to Firestore")

        except Exception as e:
            error_msg = f"Error saving metadata to Firestore: {str(e)}"
            logger.error(error_msg)
            raise PDFGenerationError(error_msg) from e

def get_pdf_report_service(
    db: Any,
    storage_client: storage.Client,
    bucket_name: str
) -> PDFReportService:
    """
    PDFReportServiceのインスタンスを取得します

    Args:
        db: Firestoreクライアント
        storage_client: Cloud Storageクライアント
        bucket_name: Cloud Storageバケット名

    Returns:
        PDFReportService: 初期化されたサービスインスタンス
    """
    return PDFReportService(db, storage_client, bucket_name)