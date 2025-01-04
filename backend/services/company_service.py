from typing import Dict, Any, Optional
from google.cloud import firestore
import logging

logger = logging.getLogger(__name__)

class CompanyService:
    def __init__(self):
        self.db = firestore.Client()
        self.companies_ref = self.db.collection('companies')

    async def get_company_details(self, company_id: str) -> Optional[Dict[str, Any]]:
        """
        企業の詳細情報を取得します。
        以下の情報を含みます：
        - 基本情報
        - 従業員パフォーマンス
        - 財務データ
        - 分析結果
        """
        try:
            # 基本情報の取得
            company_doc = self.companies_ref.document(company_id).get()
            if not company_doc.exists:
                return None

            company_data = company_doc.to_dict()

            # 従業員パフォーマンスデータの取得
            performance_ref = self.companies_ref.document(company_id).collection('employee_performance')
            performance_docs = performance_ref.order_by('date', direction=firestore.Query.DESCENDING).limit(10).stream()
            performance_data = [doc.to_dict() for doc in performance_docs]

            # 財務データの取得
            financial_ref = self.companies_ref.document(company_id).collection('financial_data')
            financial_docs = financial_ref.order_by('date', direction=firestore.Query.DESCENDING).limit(4).stream()
            financial_data = [doc.to_dict() for doc in financial_docs]

            # 分析結果の取得
            analysis_ref = self.companies_ref.document(company_id).collection('analysis_results')
            analysis_docs = analysis_ref.order_by('created_at', direction=firestore.Query.DESCENDING).limit(5).stream()
            analysis_data = [doc.to_dict() for doc in analysis_docs]

            # データを統合
            company_context = {
                "basic_info": company_data,
                "employee_performance": performance_data,
                "financial_data": financial_data,
                "analysis_results": analysis_data,
                "last_updated": firestore.SERVER_TIMESTAMP
            }

            return company_context

        except Exception as e:
            logger.error(f"企業データの取得中にエラーが発生しました: {str(e)}")
            raise Exception(f"企業データの取得に失敗しました: {str(e)}")