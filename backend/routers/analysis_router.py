from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime

from auth import get_current_user
from schemas import AnalysisRequest, AnalysisResponse
from src.database.firestore.client import FirestoreClient

# 分析モジュールのインポート
from analysis.association_analyzer import AssociationAnalyzer
from analysis.bayesian_analyzer import BayesianAnalyzer
from analysis.cluster_analyzer import ClusterAnalyzer
from analysis.correlation_analysis import CorrelationAnalyzer
from analysis.pca_analyzer import PCAAnalyzer
from analysis.regression_analyzer import RegressionAnalyzer
from analysis.survival_analyzer import SurvivalAnalyzer
from analysis.text_miner import TextMiner
from analysis.time_series_analyzer import TimeSeriesAnalyzer

router = APIRouter()
firestore_client = FirestoreClient()

@router.post("/association", status_code=status.HTTP_201_CREATED)
async def run_association_analysis(
    analysis_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """アソシエーション分析を実行する"""
    try:
        # 分析IDの生成または取得
        analysis_id = analysis_data.get('id', f"analysis_{datetime.utcnow().timestamp()}")

        # 分析メタデータの設定
        analysis_metadata = {
            "id": analysis_id,
            "type": "association",
            "status": "processing",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "user_id": current_user.get("uid"),
            "params": analysis_data.get("params", {})
        }

        # Firestoreに分析情報を保存
        await firestore_client.create_document(
            collection='analyses',
            doc_id=analysis_id,
            data=analysis_metadata
        )

        # バックグラウンドで分析を実行
        background_tasks.add_task(
            _run_association_analysis,
            analysis_id=analysis_id,
            data=analysis_data.get("data", {}),
            params=analysis_data.get("params", {})
        )

        return {
            "status": "success",
            "message": "分析が開始されました",
            "analysis_id": analysis_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分析の開始に失敗しました: {str(e)}"
        )

@router.post("/regression", status_code=status.HTTP_201_CREATED)
async def run_regression_analysis(
    analysis_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """回帰分析を実行する"""
    try:
        # 分析IDの生成または取得
        analysis_id = analysis_data.get('id', f"analysis_{datetime.utcnow().timestamp()}")

        # 分析メタデータの設定
        analysis_metadata = {
            "id": analysis_id,
            "type": "regression",
            "status": "processing",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "user_id": current_user.get("uid"),
            "params": analysis_data.get("params", {})
        }

        # Firestoreに分析情報を保存
        await firestore_client.create_document(
            collection='analyses',
            doc_id=analysis_id,
            data=analysis_metadata
        )

        # バックグラウンドで分析を実行
        background_tasks.add_task(
            _run_regression_analysis,
            analysis_id=analysis_id,
            data=analysis_data.get("data", {}),
            params=analysis_data.get("params", {})
        )

        return {
            "status": "success",
            "message": "分析が開始されました",
            "analysis_id": analysis_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分析の開始に失敗しました: {str(e)}"
        )

@router.post("/cluster", status_code=status.HTTP_201_CREATED)
async def run_cluster_analysis(
    analysis_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """クラスター分析を実行する"""
    try:
        # 分析IDの生成または取得
        analysis_id = analysis_data.get('id', f"analysis_{datetime.utcnow().timestamp()}")

        # 分析メタデータの設定
        analysis_metadata = {
            "id": analysis_id,
            "type": "cluster",
            "status": "processing",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "user_id": current_user.get("uid"),
            "params": analysis_data.get("params", {})
        }

        # Firestoreに分析情報を保存
        await firestore_client.create_document(
            collection='analyses',
            doc_id=analysis_id,
            data=analysis_metadata
        )

        # バックグラウンドで分析を実行
        background_tasks.add_task(
            _run_cluster_analysis,
            analysis_id=analysis_id,
            data=analysis_data.get("data", {}),
            params=analysis_data.get("params", {})
        )

        return {
            "status": "success",
            "message": "分析が開始されました",
            "analysis_id": analysis_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分析の開始に失敗しました: {str(e)}"
        )

@router.get("/{analysis_id}")
async def get_analysis_result(
    analysis_id: str,
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """分析結果を取得する"""
    try:
        analysis = await firestore_client.get_document(
            collection='analyses',
            doc_id=analysis_id
        )

        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="指定された分析が見つかりません"
            )

        return analysis
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分析結果の取得に失敗しました: {str(e)}"
        )

@router.get("/")
async def list_analyses(
    analysis_type: Optional[str] = None,
    limit: int = 10,
    current_user = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """ユーザーの分析一覧を取得する"""
    try:
        filters = [("user_id", "==", current_user.get("uid"))]

        if analysis_type:
            filters.append(("type", "==", analysis_type))

        analyses = await firestore_client.query_documents(
            collection='analyses',
            filters=filters,
            order_by=('created_at', 'desc'),
            limit=limit
        )

        return analyses
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分析一覧の取得に失敗しました: {str(e)}"
        )

@router.delete("/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """分析を削除する"""
    try:
        # 分析の存在確認とユーザー確認
        analysis = await firestore_client.get_document(
            collection='analyses',
            doc_id=analysis_id
        )

        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="指定された分析が見つかりません"
            )

        if analysis.get("user_id") != current_user.get("uid"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="この分析を削除する権限がありません"
            )

        # 分析の削除
        await firestore_client.delete_document(
            collection='analyses',
            doc_id=analysis_id
        )

        return {
            "status": "success",
            "message": "分析が削除されました"
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分析の削除に失敗しました: {str(e)}"
        )

# バックグラウンドタスク用の関数
async def _run_association_analysis(analysis_id: str, data: Dict[str, Any], params: Dict[str, Any]):
    """アソシエーション分析をバックグラウンドで実行"""
    try:
        # 分析ステータスを更新
        await firestore_client.update_document(
            collection='analyses',
            doc_id=analysis_id,
            data={"status": "processing"}
        )

        # 分析の実行
        analyzer = AssociationAnalyzer()
        result = await analyzer.analyze(
            data=data.get("dataset"),
            min_support=params.get("min_support", 0.1),
            metric=params.get("metric", "lift"),
            min_threshold=params.get("min_threshold", 1.0)
        )

        # 結果の保存
        await firestore_client.update_document(
            collection='analyses',
            doc_id=analysis_id,
            data={
                "status": "completed",
                "result": result,
                "updated_at": datetime.utcnow()
            }
        )
    except Exception as e:
        # エラー情報の保存
        await firestore_client.update_document(
            collection='analyses',
            doc_id=analysis_id,
            data={
                "status": "failed",
                "error": str(e),
                "updated_at": datetime.utcnow()
            }
        )

async def _run_regression_analysis(analysis_id: str, data: Dict[str, Any], params: Dict[str, Any]):
    """回帰分析をバックグラウンドで実行"""
    try:
        # 分析ステータスを更新
        await firestore_client.update_document(
            collection='analyses',
            doc_id=analysis_id,
            data={"status": "processing"}
        )

        # 分析の実行
        analyzer = RegressionAnalyzer()
        result = await analyzer.analyze(
            data=data.get("dataset"),
            target=params.get("target"),
            features=params.get("features"),
            model_type=params.get("model_type", "linear")
        )

        # 結果の保存
        await firestore_client.update_document(
            collection='analyses',
            doc_id=analysis_id,
            data={
                "status": "completed",
                "result": result,
                "updated_at": datetime.utcnow()
            }
        )
    except Exception as e:
        # エラー情報の保存
        await firestore_client.update_document(
            collection='analyses',
            doc_id=analysis_id,
            data={
                "status": "failed",
                "error": str(e),
                "updated_at": datetime.utcnow()
            }
        )

async def _run_cluster_analysis(analysis_id: str, data: Dict[str, Any], params: Dict[str, Any]):
    """クラスター分析をバックグラウンドで実行"""
    try:
        # 分析ステータスを更新
        await firestore_client.update_document(
            collection='analyses',
            doc_id=analysis_id,
            data={"status": "processing"}
        )

        # 分析の実行
        analyzer = ClusterAnalyzer()
        result = await analyzer.analyze(
            data=data.get("dataset"),
            n_clusters=params.get("n_clusters", 3),
            algorithm=params.get("algorithm", "kmeans"),
            features=params.get("features")
        )

        # 結果の保存
        await firestore_client.update_document(
            collection='analyses',
            doc_id=analysis_id,
            data={
                "status": "completed",
                "result": result,
                "updated_at": datetime.utcnow()
            }
        )
    except Exception as e:
        # エラー情報の保存
        await firestore_client.update_document(
            collection='analyses',
            doc_id=analysis_id,
            data={
                "status": "failed",
                "error": str(e),
                "updated_at": datetime.utcnow()
            }
        )