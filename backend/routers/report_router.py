from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List
from datetime import datetime
from backend.src.database.firestore.client import FirestoreClient

router = APIRouter()
firestore_client = FirestoreClient()

@router.get("/")
async def get_reports() -> List[Dict[str, Any]]:
    """レポート一覧を取得"""
    try:
        reports = await firestore_client.query_documents(
            collection='reports',
            order_by=('created_at', 'desc')
        )
        return reports
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{report_id}")
async def get_report(report_id: str) -> Dict[str, Any]:
    """レポートの詳細を取得"""
    try:
        report = await firestore_client.get_document(
            collection='reports',
            doc_id=report_id
        )
        if not report:
            raise HTTPException(status_code=404, detail="レポートが見つかりません")
        return report
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_report(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """レポートを作成"""
    try:
        report_id = report_data.get('id')
        if not report_id:
            raise HTTPException(status_code=400, detail="レポートIDは必須です")

        report_data['created_at'] = datetime.utcnow()
        report_data['updated_at'] = datetime.utcnow()

        await firestore_client.create_document(
            collection='reports',
            doc_id=report_id,
            data=report_data
        )

        return {
            "status": "success",
            "report_id": report_id
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{report_id}")
async def update_report(report_id: str, report_data: Dict[str, Any]) -> Dict[str, Any]:
    """レポートを更新"""
    try:
        report_data['updated_at'] = datetime.utcnow()
        await firestore_client.update_document(
            collection='reports',
            doc_id=report_id,
            data=report_data
        )
        return {
            "status": "success",
            "report_id": report_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_report(report_id: str) -> None:
    """レポートを削除"""
    try:
        await firestore_client.delete_document(
            collection='reports',
            doc_id=report_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))