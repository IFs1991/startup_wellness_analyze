# -*- coding: utf-8 -*-
"""
業績データリポジトリモジュール
業績関連データのデータアクセスを提供します。
"""
from typing import List, Dict, Any, Optional, Tuple, Union, cast
from datetime import datetime, date, timedelta
import os
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc, extract
from ..models_sql import (
    MonthlyBusinessPerformance, UploadedDocument,
    DocumentExtractionResult
)
from ..repository import Repository, EntityNotFoundException, ValidationException


class BusinessPerformanceRepository(Repository):
    """
    業績データリポジトリクラス
    業績データの検索、保存、更新、削除機能を提供します。
    """
    def __init__(self, session: Session):
        """初期化"""
        self.session = session

    async def find_by_id(self, report_id: int) -> MonthlyBusinessPerformance:
        """
        IDによる業績データの検索

        Args:
            report_id: レポートID

        Returns:
            MonthlyBusinessPerformance: 業績データレコード

        Raises:
            EntityNotFoundException: レコードが見つからない場合
        """
        result = self.session.query(MonthlyBusinessPerformance).filter(
            MonthlyBusinessPerformance.report_id == report_id
        ).first()

        if not result:
            raise EntityNotFoundException(f"業績レコードが見つかりません: ID {report_id}")

        return result

    async def find_by_company_id(
        self,
        company_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 12,
        offset: int = 0
    ) -> List[MonthlyBusinessPerformance]:
        """
        企業IDによる業績データの検索

        Args:
            company_id: 企業ID
            start_date: 開始日（デフォルト: なし）
            end_date: 終了日（デフォルト: なし）
            limit: 結果の最大数（デフォルト: 12）
            offset: 結果のオフセット（デフォルト: 0）

        Returns:
            List[MonthlyBusinessPerformance]: 業績データレコードのリスト
        """
        query = self.session.query(MonthlyBusinessPerformance).filter(
            MonthlyBusinessPerformance.company_id == company_id
        )

        if start_date:
            query = query.filter(MonthlyBusinessPerformance.report_month >= start_date)

        if end_date:
            query = query.filter(MonthlyBusinessPerformance.report_month <= end_date)

        query = query.order_by(desc(MonthlyBusinessPerformance.report_month))

        if limit:
            query = query.limit(limit)

        if offset:
            query = query.offset(offset)

        return query.all()

    async def find_by_month(
        self,
        year: int,
        month: int,
        company_id: Optional[str] = None
    ) -> List[MonthlyBusinessPerformance]:
        """
        年月による業績データの検索

        Args:
            year: 年
            month: 月
            company_id: 企業ID（デフォルト: なし）

        Returns:
            List[MonthlyBusinessPerformance]: 業績データレコードのリスト
        """
        # 開始日と終了日を計算
        start_date = date(year, month, 1)

        # 月の末日を計算
        if month == 12:
            end_date = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = date(year, month + 1, 1) - timedelta(days=1)

        query = self.session.query(MonthlyBusinessPerformance).filter(
            extract('year', MonthlyBusinessPerformance.report_month) == year,
            extract('month', MonthlyBusinessPerformance.report_month) == month
        )

        if company_id:
            query = query.filter(MonthlyBusinessPerformance.company_id == company_id)

        query = query.order_by(MonthlyBusinessPerformance.company_id)

        return query.all()

    async def create(self, data: Dict[str, Any]) -> MonthlyBusinessPerformance:
        """
        業績データの作成

        Args:
            data: 業績データの辞書

        Returns:
            MonthlyBusinessPerformance: 作成された業績データレコード

        Raises:
            ValidationException: バリデーションエラーの場合
        """
        # 必須フィールドの検証
        required_fields = ['company_id', 'report_month']

        for field in required_fields:
            if field not in data:
                raise ValidationException(f"必須フィールドがありません: {field}")

        # 数値フィールドの検証
        numeric_fields = ['revenue', 'expenses', 'profit_margin', 'headcount', 'new_clients', 'turnover_rate']

        for field in numeric_fields:
            if field in data and data[field] is not None:
                if field in ['profit_margin', 'turnover_rate'] and (data[field] < 0 or data[field] > 100):
                    raise ValidationException(f"{field}は0から100の間である必要があります")
                elif field in ['revenue', 'expenses'] and data[field] < 0:
                    raise ValidationException(f"{field}は0以上である必要があります")
                elif field in ['headcount', 'new_clients'] and data[field] < 0:
                    raise ValidationException(f"{field}は0以上である必要があります")

        # 重複レコードのチェック
        existing = self.session.query(MonthlyBusinessPerformance).filter(
            MonthlyBusinessPerformance.company_id == data['company_id'],
            MonthlyBusinessPerformance.report_month == data['report_month']
        ).first()

        if existing:
            raise ValidationException("この月のレコードはすでに存在します")

        # レコードの作成
        record = MonthlyBusinessPerformance(**data)
        self.session.add(record)
        self.session.flush()

        return record

    async def update(self, report_id: int, data: Dict[str, Any]) -> MonthlyBusinessPerformance:
        """
        業績データの更新

        Args:
            report_id: レポートID
            data: 更新データの辞書

        Returns:
            MonthlyBusinessPerformance: 更新された業績データレコード

        Raises:
            EntityNotFoundException: レコードが見つからない場合
            ValidationException: バリデーションエラーの場合
        """
        record = await self.find_by_id(report_id)

        # 数値フィールドの検証
        numeric_fields = ['revenue', 'expenses', 'profit_margin', 'headcount', 'new_clients', 'turnover_rate']

        for field in numeric_fields:
            if field in data and data[field] is not None:
                if field in ['profit_margin', 'turnover_rate'] and (data[field] < 0 or data[field] > 100):
                    raise ValidationException(f"{field}は0から100の間である必要があります")
                elif field in ['revenue', 'expenses'] and data[field] < 0:
                    raise ValidationException(f"{field}は0以上である必要があります")
                elif field in ['headcount', 'new_clients'] and data[field] < 0:
                    raise ValidationException(f"{field}は0以上である必要があります")

        # フィールドの更新
        for key, value in data.items():
            if hasattr(record, key):
                setattr(record, key, value)

        # 自動的にupdated_atが更新される

        return record

    async def delete(self, report_id: int) -> bool:
        """
        業績データの削除

        Args:
            report_id: レポートID

        Returns:
            bool: 削除が成功した場合はTrue

        Raises:
            EntityNotFoundException: レコードが見つからない場合
        """
        record = await self.find_by_id(report_id)

        self.session.delete(record)

        return True

    async def save_uploaded_document(
        self,
        company_id: str,
        file_path: str,
        original_filename: str,
        file_type: str,
        file_size: int,
        content_type: str,
        uploaded_by: str
    ) -> UploadedDocument:
        """
        アップロードされたドキュメントの保存

        Args:
            company_id: 企業ID
            file_path: ファイルパス
            original_filename: 元のファイル名
            file_type: ファイルタイプ（'pdf'/'csv'など）
            file_size: ファイルサイズ（バイト）
            content_type: コンテンツタイプ
            uploaded_by: アップロード実行者ID

        Returns:
            UploadedDocument: 保存されたドキュメント情報
        """
        document = UploadedDocument(
            company_id=company_id,
            file_name=os.path.basename(file_path),
            original_file_name=original_filename,
            file_type=file_type,
            file_size=file_size,
            upload_path=file_path,
            content_type=content_type,
            processing_status='pending',
            uploaded_by=uploaded_by
        )

        self.session.add(document)
        self.session.flush()

        return document

    async def get_document_by_id(self, document_id: int) -> UploadedDocument:
        """
        IDによるドキュメントの取得

        Args:
            document_id: ドキュメントID

        Returns:
            UploadedDocument: ドキュメント情報

        Raises:
            EntityNotFoundException: ドキュメントが見つからない場合
        """
        document = self.session.query(UploadedDocument).filter(
            UploadedDocument.document_id == document_id
        ).first()

        if not document:
            raise EntityNotFoundException(f"ドキュメントが見つかりません: ID {document_id}")

        return document

    async def get_documents_by_company(
        self,
        company_id: str,
        file_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[UploadedDocument]:
        """
        企業IDによるドキュメントの取得

        Args:
            company_id: 企業ID
            file_type: ファイルタイプでフィルタリング（デフォルト: なし）
            status: 処理ステータスでフィルタリング（デフォルト: なし）
            limit: 結果の最大数（デフォルト: 50）
            offset: 結果のオフセット（デフォルト: 0）

        Returns:
            List[UploadedDocument]: ドキュメント情報のリスト
        """
        query = self.session.query(UploadedDocument).filter(
            UploadedDocument.company_id == company_id
        )

        if file_type:
            query = query.filter(UploadedDocument.file_type == file_type)

        if status:
            query = query.filter(UploadedDocument.processing_status == status)

        query = query.order_by(desc(UploadedDocument.created_at))

        if limit:
            query = query.limit(limit)

        if offset:
            query = query.offset(offset)

        return query.all()

    async def update_document_status(
        self,
        document_id: int,
        status: str,
        error_details: Optional[str] = None
    ) -> UploadedDocument:
        """
        ドキュメントのステータス更新

        Args:
            document_id: ドキュメントID
            status: 新しいステータス
            error_details: エラー詳細（デフォルト: なし）

        Returns:
            UploadedDocument: 更新されたドキュメント情報

        Raises:
            EntityNotFoundException: ドキュメントが見つからない場合
        """
        document = await self.get_document_by_id(document_id)

        document.processing_status = status

        if status == 'completed' or status == 'failed':
            document.processed_at = datetime.now()

        if error_details:
            document.error_details = error_details

        self.session.flush()

        return document

    async def save_extraction_result(
        self,
        document_id: int,
        extracted_data: Dict[str, Any],
        confidence_score: Optional[float] = None,
        report_id: Optional[int] = None
    ) -> DocumentExtractionResult:
        """
        抽出結果の保存

        Args:
            document_id: ドキュメントID
            extracted_data: 抽出されたデータ
            confidence_score: 信頼度スコア（デフォルト: なし）
            report_id: 関連する業績レポートID（デフォルト: なし）

        Returns:
            DocumentExtractionResult: 保存された抽出結果

        Raises:
            EntityNotFoundException: ドキュメントが見つからない場合
        """
        # ドキュメントの存在チェック
        await self.get_document_by_id(document_id)

        # レポートIDが指定されている場合はその存在をチェック
        if report_id:
            await self.find_by_id(report_id)

        result = DocumentExtractionResult(
            document_id=document_id,
            report_id=report_id,
            extracted_data=extracted_data,
            confidence_score=confidence_score,
            review_status='pending'
        )

        self.session.add(result)
        self.session.flush()

        return result

    async def get_extraction_result(self, result_id: int) -> DocumentExtractionResult:
        """
        IDによる抽出結果の取得

        Args:
            result_id: 結果ID

        Returns:
            DocumentExtractionResult: 抽出結果

        Raises:
            EntityNotFoundException: 結果が見つからない場合
        """
        result = self.session.query(DocumentExtractionResult).filter(
            DocumentExtractionResult.result_id == result_id
        ).first()

        if not result:
            raise EntityNotFoundException(f"抽出結果が見つかりません: ID {result_id}")

        return result

    async def get_extraction_results_by_document(
        self,
        document_id: int
    ) -> List[DocumentExtractionResult]:
        """
        ドキュメントIDによる抽出結果の取得

        Args:
            document_id: ドキュメントID

        Returns:
            List[DocumentExtractionResult]: 抽出結果のリスト
        """
        return self.session.query(DocumentExtractionResult).filter(
            DocumentExtractionResult.document_id == document_id
        ).order_by(desc(DocumentExtractionResult.created_at)).all()

    async def update_extraction_review(
        self,
        result_id: int,
        review_status: str,
        reviewed_by: str,
        review_notes: Optional[str] = None
    ) -> DocumentExtractionResult:
        """
        抽出結果のレビュー更新

        Args:
            result_id: 結果ID
            review_status: レビューステータス
            reviewed_by: レビュー者ID
            review_notes: レビューノート（デフォルト: なし）

        Returns:
            DocumentExtractionResult: 更新された抽出結果

        Raises:
            EntityNotFoundException: 結果が見つからない場合
            ValidationException: 無効なレビューステータスの場合
        """
        result = await self.get_extraction_result(result_id)

        valid_statuses = ['reviewed', 'accepted', 'rejected']
        if review_status not in valid_statuses:
            raise ValidationException(f"無効なレビューステータス。有効な値: {', '.join(valid_statuses)}")

        result.review_status = review_status
        result.reviewed_by = reviewed_by
        result.reviewed_at = datetime.now()

        if review_notes:
            result.review_notes = review_notes

        self.session.flush()

        return result