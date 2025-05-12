# -*- coding: utf-8 -*-
"""
ドキュメント処理サービス
PDF/CSVファイルからの業績データ抽出・処理機能を提供します。
"""
import os
import logging
import asyncio
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO
from datetime import datetime, date
import json
import csv
import io
import re
from pathlib import Path

from ..repositories.business_performance_repository import BusinessPerformanceRepository
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DocumentProcessingError(Exception):
    """ドキュメント処理に関するエラー"""
    pass

class DocumentProcessingService:
    """
    ドキュメント処理サービス
    PDF/CSVファイルからの業績データ抽出と処理を管理します。
    """
    def __init__(self, session: Session):
        """
        初期化

        Args:
            session: データベースセッション
        """
        self.session = session
        self.repository = BusinessPerformanceRepository(session)

    async def process_uploaded_document(
        self,
        document_id: int,
        save_extracted_data: bool = True,
        auto_create_report: bool = False
    ) -> Dict[str, Any]:
        """
        アップロードされたドキュメントの処理

        Args:
            document_id: ドキュメントID
            save_extracted_data: 抽出データを保存するフラグ
            auto_create_report: 業績レポートを自動作成するフラグ

        Returns:
            Dict[str, Any]: 処理結果

        Raises:
            DocumentProcessingError: 処理中のエラー
        """
        try:
            # ドキュメント情報の取得
            document = await self.repository.get_document_by_id(document_id)

            # ステータスの更新
            await self.repository.update_document_status(
                document_id, 'processing'
            )

            # ファイルタイプによる処理の分岐
            file_type = document.file_type.lower()
            extracted_data = {}
            confidence_score = None

            if file_type == 'pdf':
                extracted_data, confidence_score = await self._process_pdf(document)
            elif file_type == 'csv':
                extracted_data = await self._process_csv(document)
            else:
                raise DocumentProcessingError(f"サポートされていないファイルタイプ: {file_type}")

            # 抽出結果の保存
            extraction_result = None
            report_id = None

            if save_extracted_data:
                extraction_result = await self.repository.save_extraction_result(
                    document_id=document_id,
                    extracted_data=extracted_data,
                    confidence_score=confidence_score
                )

                # 業績レポートの自動作成
                if auto_create_report and extracted_data:
                    report_data = self._transform_to_report_data(
                        extracted_data, document.company_id
                    )

                    # レポートの作成または更新
                    try:
                        # 既存レポートの確認
                        existing_reports = await self.repository.find_by_company_id(
                            company_id=document.company_id,
                            start_date=report_data['report_month'],
                            end_date=report_data['report_month']
                        )

                        if existing_reports:
                            # 更新
                            report = await self.repository.update(
                                existing_reports[0].report_id, report_data
                            )
                            report_id = report.report_id
                        else:
                            # 新規作成
                            report = await self.repository.create(report_data)
                            report_id = report.report_id

                        # 抽出結果とレポートの関連付け
                        if extraction_result:
                            extraction_result.report_id = report_id
                            self.session.flush()

                    except Exception as e:
                        logger.error(f"レポート作成エラー: {str(e)}")
                        # レポート作成エラーはスローせず、処理を続行

            # ステータスの更新
            await self.repository.update_document_status(
                document_id, 'completed'
            )

            return {
                'document_id': document_id,
                'status': 'completed',
                'extraction_result_id': extraction_result.result_id if extraction_result else None,
                'report_id': report_id,
                'extracted_data': extracted_data,
                'confidence_score': confidence_score
            }

        except Exception as e:
            error_msg = f"ドキュメント処理エラー: {str(e)}"
            logger.error(error_msg)

            # ステータスの更新
            try:
                await self.repository.update_document_status(
                    document_id, 'failed', error_details=error_msg
                )
            except Exception as update_error:
                logger.error(f"ステータス更新エラー: {str(update_error)}")

            raise DocumentProcessingError(error_msg)

    async def _process_pdf(
        self,
        document
    ) -> Tuple[Dict[str, Any], Optional[float]]:
        """
        PDFファイルからのデータ抽出

        Args:
            document: ドキュメント情報

        Returns:
            Tuple[Dict[str, Any], Optional[float]]: 抽出データと信頼度スコア

        Raises:
            DocumentProcessingError: 抽出中のエラー
        """
        try:
            # ここにPDF処理コードを実装
            # 実際の実装では、PyPDF2、pdfplumber、OCRエンジンなどを使用

            # 簡易的な実装（実際のプロジェクトでは置き換える）
            logger.info(f"PDFファイル処理: {document.file_name}")

            # ファイルの存在確認
            file_path = document.upload_path
            if not os.path.exists(file_path):
                raise DocumentProcessingError(f"ファイルが見つかりません: {file_path}")

            # ここでは、実際のPDF処理の代わりにダミーデータを返す
            # トレーニングセット用の簡易実装

            # ファイル名から年月を推測
            file_name = os.path.basename(file_path)

            # YYYY-MM形式の検索
            year_month_pattern = r'(\d{4})[-_](\d{1,2})'
            match = re.search(year_month_pattern, file_name)

            if match:
                year = int(match.group(1))
                month = int(match.group(2))

                # 現在の日付より未来の場合は無効として扱う
                current_date = datetime.now().date()
                if date(year, month, 1) > current_date:
                    year = current_date.year
                    month = current_date.month
            else:
                # デフォルト値として前月を使用
                current_date = datetime.now().date()
                if current_date.month == 1:
                    year = current_date.year - 1
                    month = 12
                else:
                    year = current_date.year
                    month = current_date.month - 1

            # ダミーデータの作成
            extracted_data = {
                'year': year,
                'month': month,
                'revenue': 10000000,  # 1000万円
                'expenses': 8000000,  # 800万円
                'profit_margin': 20.0,  # 20%
                'headcount': 50,
                'new_clients': 3,
                'turnover_rate': 5.0,  # 5%
                'extra_data': {
                    'page_count': 5,
                    'detected_tables': 3,
                    'detected_fields': 15
                }
            }

            # 信頼度スコア（実際の実装では、抽出品質に基づいて計算）
            confidence_score = 0.85

            return extracted_data, confidence_score

        except Exception as e:
            raise DocumentProcessingError(f"PDF処理エラー: {str(e)}")

    async def _process_csv(self, document) -> Dict[str, Any]:
        """
        CSVファイルからのデータ抽出

        Args:
            document: ドキュメント情報

        Returns:
            Dict[str, Any]: 抽出データ

        Raises:
            DocumentProcessingError: 抽出中のエラー
        """
        try:
            logger.info(f"CSVファイル処理: {document.file_name}")

            # ファイルの存在確認
            file_path = document.upload_path
            if not os.path.exists(file_path):
                raise DocumentProcessingError(f"ファイルが見つかりません: {file_path}")

            # CSVファイルを読み込み
            data = {}
            headers = []
            rows = []

            with open(file_path, 'r', encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file)
                headers = next(csv_reader)  # ヘッダー行

                for row in csv_reader:
                    if len(row) == len(headers):
                        rows.append(row)

            # ヘッダーと行データを設定
            data['headers'] = headers
            data['rows'] = rows

            # ヘッダーの分析とマッピング
            # カラム名の類似性に基づいて必要なフィールドを特定
            field_mapping = self._analyze_csv_headers(headers)

            # マッピングに基づいてデータを抽出
            extracted_data = self._extract_from_csv_rows(rows, field_mapping)

            # 全体のデータを返す
            return {
                'data': data,
                'field_mapping': field_mapping,
                'extracted': extracted_data
            }

        except Exception as e:
            raise DocumentProcessingError(f"CSV処理エラー: {str(e)}")

    def _analyze_csv_headers(self, headers: List[str]) -> Dict[str, int]:
        """
        CSVヘッダーの分析とマッピング

        Args:
            headers: ヘッダー行

        Returns:
            Dict[str, int]: フィールド名とカラムインデックスのマッピング
        """
        mapping = {}

        # マッピングルール - フィールド名と対応する可能性のあるカラム名のリスト
        mapping_rules = {
            'year': ['年', '年度', 'year', '年次'],
            'month': ['月', '月次', 'month'],
            'revenue': ['売上', '売上高', '収入', 'revenue', '売上金額'],
            'expenses': ['費用', '経費', '支出', 'expenses', 'コスト'],
            'profit_margin': ['利益率', '粗利率', 'profit margin', '利益率（%）'],
            'headcount': ['従業員数', '社員数', '人数', 'headcount', '人員'],
            'new_clients': ['新規顧客', '新規取引先', '新規獲得数', 'new clients'],
            'turnover_rate': ['離職率', '退職率', 'turnover rate', '離職率（%）']
        }

        # ヘッダーのマッピング
        for field, possible_names in mapping_rules.items():
            for i, header in enumerate(headers):
                # 完全一致または部分一致で検索
                if header in possible_names or any(name in header for name in possible_names):
                    mapping[field] = i
                    break

        return mapping

    def _extract_from_csv_rows(
        self,
        rows: List[List[str]],
        field_mapping: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        CSVデータからの値抽出

        Args:
            rows: CSV行データ
            field_mapping: フィールドマッピング

        Returns:
            Dict[str, Any]: 抽出データ
        """
        # データが空の場合
        if not rows:
            return {}

        # 通常は最初の行のデータを使用
        row = rows[0]

        # 抽出データの初期化
        extracted = {}

        # 現在の日付を取得
        current_date = datetime.now().date()

        # 年月の設定
        year = current_date.year
        month = current_date.month

        # マッピングに従ってデータを抽出
        for field, index in field_mapping.items():
            if index < len(row) and row[index]:
                # 文字列をクリーンアップ（数値以外の文字を除去）
                value = re.sub(r'[^\d.-]', '', row[index]) if row[index] else ''

                if field in ['year', 'month', 'headcount', 'new_clients']:
                    # 整数フィールド
                    try:
                        extracted[field] = int(float(value)) if value else None
                    except ValueError:
                        extracted[field] = None
                elif field in ['revenue', 'expenses']:
                    # 通貨フィールド
                    try:
                        extracted[field] = float(value) if value else None
                    except ValueError:
                        extracted[field] = None
                elif field in ['profit_margin', 'turnover_rate']:
                    # パーセンテージフィールド
                    try:
                        # パーセント表記（例：20%）の場合、数値に変換
                        value = value.replace('%', '')
                        extracted[field] = float(value) if value else None
                    except ValueError:
                        extracted[field] = None
                else:
                    # その他のフィールド
                    extracted[field] = row[index]

        # 年月が抽出されなかった場合、ファイル名から推測
        if 'year' not in extracted or not extracted['year']:
            extracted['year'] = year
        if 'month' not in extracted or not extracted['month']:
            extracted['month'] = month

        return extracted

    def _transform_to_report_data(
        self,
        extracted_data: Dict[str, Any],
        company_id: str
    ) -> Dict[str, Any]:
        """
        抽出データから業績レポートデータへの変換

        Args:
            extracted_data: 抽出データ
            company_id: 企業ID

        Returns:
            Dict[str, Any]: 業績レポートデータ
        """
        report_data = {
            'company_id': company_id
        }

        # PDFからの抽出データの場合
        if 'year' in extracted_data and 'month' in extracted_data:
            year = extracted_data.get('year')
            month = extracted_data.get('month')
            report_data['report_month'] = date(year, month, 1)

        # CSVからの抽出データの場合
        elif 'extracted' in extracted_data and isinstance(extracted_data['extracted'], dict):
            extracted = extracted_data['extracted']
            year = extracted.get('year')
            month = extracted.get('month')

            if year and month:
                report_data['report_month'] = date(year, month, 1)
            else:
                # デフォルト値として前月を使用
                current_date = datetime.now().date()
                if current_date.month == 1:
                    report_data['report_month'] = date(current_date.year - 1, 12, 1)
                else:
                    report_data['report_month'] = date(current_date.year, current_date.month - 1, 1)

            # 他のフィールドをコピー
            for field in ['revenue', 'expenses', 'profit_margin', 'headcount', 'new_clients', 'turnover_rate']:
                if field in extracted and extracted[field] is not None:
                    report_data[field] = extracted[field]

        # その他のケース（デフォルト値）
        else:
            # デフォルト値として前月を使用
            current_date = datetime.now().date()
            if current_date.month == 1:
                report_data['report_month'] = date(current_date.year - 1, 12, 1)
            else:
                report_data['report_month'] = date(current_date.year, current_date.month - 1, 1)

        # PDFの直接抽出データをコピー
        for field in ['revenue', 'expenses', 'profit_margin', 'headcount', 'new_clients', 'turnover_rate']:
            if field in extracted_data and extracted_data[field] is not None:
                report_data[field] = extracted_data[field]

        # 抽出元をノートとして記録
        report_data['notes'] = f"自動抽出データ ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"

        return report_data

    async def get_document_data(self, document_id: int) -> Dict[str, Any]:
        """
        ドキュメントの詳細情報を取得

        Args:
            document_id: ドキュメントID

        Returns:
            Dict[str, Any]: ドキュメント情報と抽出結果

        Raises:
            DocumentProcessingError: 処理中のエラー
        """
        try:
            # ドキュメント情報の取得
            document = await self.repository.get_document_by_id(document_id)

            # 抽出結果の取得
            extraction_results = await self.repository.get_extraction_results_by_document(document_id)

            # レスポンスの構築
            result = {
                'document': {
                    'document_id': document.document_id,
                    'company_id': document.company_id,
                    'file_name': document.file_name,
                    'original_file_name': document.original_file_name,
                    'file_type': document.file_type,
                    'file_size': document.file_size,
                    'upload_path': document.upload_path,
                    'processing_status': document.processing_status,
                    'processed_at': document.processed_at.isoformat() if document.processed_at else None,
                    'error_details': document.error_details,
                    'uploaded_by': document.uploaded_by,
                    'created_at': document.created_at.isoformat(),
                    'updated_at': document.updated_at.isoformat()
                },
                'extraction_results': []
            }

            # 抽出結果の追加
            for extraction_result in extraction_results:
                result['extraction_results'].append({
                    'result_id': extraction_result.result_id,
                    'document_id': extraction_result.document_id,
                    'report_id': extraction_result.report_id,
                    'extracted_data': extraction_result.extracted_data,
                    'confidence_score': float(extraction_result.confidence_score) if extraction_result.confidence_score else None,
                    'review_status': extraction_result.review_status,
                    'reviewed_by': extraction_result.reviewed_by,
                    'reviewed_at': extraction_result.reviewed_at.isoformat() if extraction_result.reviewed_at else None,
                    'review_notes': extraction_result.review_notes,
                    'created_at': extraction_result.created_at.isoformat(),
                    'updated_at': extraction_result.updated_at.isoformat()
                })

            return result

        except Exception as e:
            raise DocumentProcessingError(f"ドキュメントデータ取得エラー: {str(e)}")

    async def review_extraction_result(
        self,
        result_id: int,
        review_status: str,
        reviewed_by: str,
        review_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        抽出結果のレビュー更新

        Args:
            result_id: 結果ID
            review_status: レビューステータス
            reviewed_by: レビュー者ID
            review_notes: レビューノート

        Returns:
            Dict[str, Any]: 更新結果

        Raises:
            DocumentProcessingError: 処理中のエラー
        """
        try:
            # レビューの更新
            updated_result = await self.repository.update_extraction_review(
                result_id=result_id,
                review_status=review_status,
                reviewed_by=reviewed_by,
                review_notes=review_notes
            )

            return {
                'result_id': updated_result.result_id,
                'document_id': updated_result.document_id,
                'review_status': updated_result.review_status,
                'reviewed_by': updated_result.reviewed_by,
                'reviewed_at': updated_result.reviewed_at.isoformat()
            }

        except Exception as e:
            raise DocumentProcessingError(f"レビュー更新エラー: {str(e)}")

    async def batch_process_pending_documents(
        self,
        company_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        保留中ドキュメントの一括処理

        Args:
            company_id: 企業ID（デフォルト: すべての企業）
            limit: 一度に処理する最大数（デフォルト: 10）

        Returns:
            List[Dict[str, Any]]: 処理結果のリスト
        """
        try:
            # 保留中ドキュメントの取得
            documents = []
            if company_id:
                documents = await self.repository.get_documents_by_company(
                    company_id=company_id,
                    status='pending',
                    limit=limit
                )
            else:
                # 特定の企業IDが指定されていない場合、一部のみを実装
                # 実際の実装では、全企業の保留中ドキュメントを取得する処理が必要
                pass

            # 処理結果
            results = []

            # 各ドキュメントを処理
            for document in documents:
                try:
                    result = await self.process_uploaded_document(
                        document_id=document.document_id,
                        save_extracted_data=True,
                        auto_create_report=True
                    )
                    results.append({
                        'document_id': document.document_id,
                        'status': 'success',
                        'result': result
                    })
                except Exception as e:
                    logger.error(f"ドキュメント処理エラー: ID {document.document_id}, {str(e)}")
                    results.append({
                        'document_id': document.document_id,
                        'status': 'error',
                        'error': str(e)
                    })

            return results

        except Exception as e:
            raise DocumentProcessingError(f"一括処理エラー: {str(e)}")